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
    
    self.layer_flow = FlowGNO(params)

    self.time_condition = nn.ModuleDict({
      "flow" : time_conditioning(params['flow_net']['nNodeFeatEmbedding'])#,
      # "memb" : time_conditioning(params['memb_net']['nNodeFeatEmbedding'])
    })

    self.decoder = nn.ModuleDict({
      "flow" : nn.Sequential(nn.Linear(params['flow_net']['nNodeFeatEmbedding'],params['flow_net']['inNodeFeatures'])),
      "memb" : nn.Sequential(nn.Linear(params['memb_net']['nNodeFeatEmbedding'],params['memb_net']['inNodeFeatures']))
    })

  def forward(self, batch):
    nFlowNodes = batch['flow'].x.shape[0]
    nMembNodes = batch['memb'].x.shape[0]

    time_embeddings = get_timestep_embedding(batch['tau'], self.params['time_embedding_dim'], structure="ordered")

    #need to implement repeat time embedding properly. so that tau can be mapped to all batches and then to all nodes.
    #refer to logic used in flow_training code
    
    x_memb = self.encoder["memb"]( torch.cat([batch['memb'].y, time_embeddings.repeat(nMembNodes,1)], axis=1 ) ) #For now i just train to obtain the latent embedding of membrane nodes features at next timestep
    x_flow = self.encoder["flow"]( torch.cat([batch['flow'].x, time_embeddings.repeat(nFlowNodes,1)], axis=1 ) )

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
      "memb" : x_memb
    }

    for i in range(self.params['nlayers']):
      x_flow = F.relu(self.layer_flow(node_feat, flow_edge_index, flow_edge_attr))
      x_flow = x_flow + self.time_condition["flow"](x_flow, batch['tau'])

    x_memb = self.decoder["memb"]( x_memb )
    x_flow = self.decoder["flow"]( x_flow )

    #z_f_new = 0.5 * z_f + 0.5 * self.dec_f(...)  # Momentum preservation
    return x_flow, x_memb
  

class FlowGNO(nn.Module):
  def __init__(self, params) -> None:
    super(FlowGNO, self).__init__()

    #initialize kernel
    kernel = DenseNet([params['flow_net']['nEdgeFeatures'], 
                       params['flow_net']['ker_width'], 
                       params['flow_net']['ker_width'], 
                       params['flow_net']['nNodeFeatEmbedding']**2], 
                       torch.nn.ReLU)

    #initializeGNO
    self.intraMessageLayer = intraMessagePassing(params['flow_net']['nNodeFeatEmbedding'], 
                                                 params['flow_net']['nNodeFeatEmbedding'], 
                                                 kernel)
    
    self.crossMessageLayer = crossAttnMessagePassing(params['memb_net']['nNodeFeatEmbedding'], 
                                                     params['flow_net']['nNodeFeatEmbedding'], 
                                                     params['attn_dim'])

  def forward(self,x, edge_index, edge_attr):
    x_cross = self.crossMessageLayer(x['flow'], x['memb'], edge_index['memb_to_flow'], edge_attr['memb_to_flow'])
    x_intra = self.intraMessageLayer(x['flow'], edge_index['flow_to_flow'], edge_attr['flow_to_flow']) 
    return x_intra + x_cross

