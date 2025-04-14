import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from model.operators import *
from model.spectral import *
from scipy.ndimage import gaussian_filter
from torch.utils.checkpoint import checkpoint
import os
import re

class neuralFSI(nn.Module):
  def __init__(self, params):
    super(neuralFSI, self).__init__()

    self.params = params

    self.encoder = nn.ModuleDict({
      "flow" : nn.Sequential(nn.Linear(params['flow_net']['inNodeFeatures'] + params['time_embedding_dim'], params['flow_net']['nNodeFeatEmbedding'])),
      "memb" : nn.Sequential(nn.Linear(params['memb_net']['inNodeFeatures'] + params['time_embedding_dim'], params['memb_net']['nNodeFeatEmbedding']),
                             nn.ReLU(),
                             nn.Linear(params['memb_net']['nNodeFeatEmbedding'], params['memb_net']['nNodeFeatEmbedding']),
                             nn.ReLU())
    })

    # self.layer_memb = membGNO(
    #   params['memb_net']['inNodeFeatures'], 
    #   params['memb_net']['nNodeFeatEmbedding'], 
    #   params['memb_net']['nEdgeFeatures'], 
    #   params['memb_net']['ker_width'],
    #   params['memb_net']['nlayers']
    #   )

    self.layers = nn.ModuleList([FlowEvolve(params) for _ in range(params['nlayers'])])


    self.decoder = nn.ModuleDict({
      "flow" : nn.Sequential(nn.Linear(params['flow_net']['nNodeFeatEmbedding'],params['flow_net']['outNodeFeatures'])),
      "memb" : nn.Sequential(nn.Linear(params['memb_net']['nNodeFeatEmbedding'],params['memb_net']['inNodeFeatures']))
    })

  def forward(self, batch):
    nBatches = batch['tau'].shape[0]
    nFlowNodes = batch['flow'].x.shape[0]//nBatches
    nMembNodes = batch['memb'].x.shape[0]//nBatches

    time_embeddings = get_timestep_embedding(batch['tau'], self.params['time_embedding_dim'], structure="ordered") #shape = (nBatches, time_embedding_dim)

    #interleave_repeat to implement repeat time embedding properly. so that tau can be mapped to all batches and then to all nodes.
    y_memb = self.encoder["memb"]( torch.cat([batch['memb'].y, time_embeddings.repeat_interleave(nMembNodes, dim=0)], axis=1 ) ) #For now i just train to obtain the latent embedding of membrane nodes features at next timestep
    x_flow = self.encoder["flow"]( torch.cat([batch['flow'].x, time_embeddings.repeat_interleave(nFlowNodes, dim=0)], axis=1 ) )

    flow_edge_index = {
      "flow_to_flow"  : batch['flow','to','flow'].edge_index,
      "memb_to_flow"  : batch['memb','to','flow'].edge_index
    }

    flow_edge_attr = {
      "flow_to_flow"  : batch['flow','to','flow'].edge_attr,
      "memb_to_flow"  : batch['memb','to','flow'].edge_attr
    }

    node_feat = {
      "flow" : x_flow,
      "memb" : y_memb
    }

    tau = batch['tau'].view(-1,1).repeat_interleave(nFlowNodes, dim=0)
    for layer in self.layers:
      y_flow = layer(node_feat, flow_edge_index, flow_edge_attr, tau)

    y_memb = self.decoder["memb"]( y_memb )
    y_flow = self.decoder["flow"]( y_flow + x_flow )

    #z_f_new = 0.5 * z_f + 0.5 * self.dec_f(...)  # Momentum preservation
    return y_flow, y_memb

class FlowEvolve(nn.Module):
  def __init__(self, params):
    super(FlowEvolve, self).__init__()
    self.layer_flow = FlowGNO(params)

    self.time_condition = nn.ModuleDict({
      "flow" : time_conditioning(params['flow_net']['nNodeFeatEmbedding'])#,
      # "memb" : time_conditioning(params['memb_net']['nNodeFeatEmbedding'])
    })

  def forward(self, node_feat, edge_index, edge_attr, tau):
    # GNO step
    gno_out = F.relu(self.layer_flow(node_feat, edge_index, edge_attr))

    # Time conditioning
    time_out = self.time_condition["flow"](gno_out, tau)

    # Residual connection is already implemented in gno_out W*input + intraMessagePassing + crossMessagePassing
    return F.silu(time_out)  # Swish for smoothness, maybe Relu but have to see 

class FlowGNO(nn.Module):
  def __init__(self, params) -> None:
    super(FlowGNO, self).__init__()

    #initialize kernel
    kernel = DenseNet([params['flow_net']['nEdgeFeatures'], 
                       params['flow_net']['ker_width'], 
                       params['flow_net']['ker_width'], 
                       params['flow_net']['nNodeFeatEmbedding']**2], 
                       torch.nn.ReLU)

    #initializeGNO - Flow Flow
    self.intraMessageLayer = intraMessagePassing(params['flow_net']['nNodeFeatEmbedding'], 
                                                 params['flow_net']['nNodeFeatEmbedding'], 
                                                 kernel)
    
    #initializeGNO - Mwmbrane Flow
    self.crossMessageLayer = crossAttnMessagePassing(params['memb_net']['nNodeFeatEmbedding'], 
                                                     params['flow_net']['nNodeFeatEmbedding'], 
                                                     params['attn_dim'])

  def forward(self,x, edge_index, edge_attr):
    x_cross = self.crossMessageLayer(x['flow'], x['memb'], edge_index['memb_to_flow'], edge_attr['memb_to_flow'])
    x_intra = self.intraMessageLayer(x['flow'], edge_index['flow_to_flow'], edge_attr['flow_to_flow']) 
    # Residual connection is already implemented in intramessage passing W*input + intraMessagePassing + crossMessagePassing
    return x_intra + x_cross


# class FlowGNO(nn.Module):
#     def __init__(self, params) -> None:
#         super(FlowGNO, self).__init__()

#         self.kernel = DenseNet([params['flow_net']['nEdgeFeatures'], 
#                                 params['flow_net']['ker_width'], 
#                                 params['flow_net']['ker_width'], 
#                                 params['flow_net']['nNodeFeatEmbedding']**2], 
#                                torch.nn.ReLU)

#         self.intraMessageLayer = intraMessagePassing(params['flow_net']['nNodeFeatEmbedding'], 
#                                                     params['flow_net']['nNodeFeatEmbedding'], 
#                                                     self.kernel)

#     def forward(self, x, edge_index, edge_attr):
#         with torch.profiler.record_function("DenseNet_Forward"):
#             # We can't directly profile self.kernel(x) since it's called in intraMessagePassing
#             x_intra = self.intraMessageLayer(x['flow'], edge_index['flow_to_flow'], edge_attr['flow_to_flow'])
#         with torch.profiler.record_function("DenseNet_Backward"):
#             if x_intra.requires_grad:
#                 x_intra.retain_grad()  # Ensure gradients are tracked for profiling
#         return x_intra