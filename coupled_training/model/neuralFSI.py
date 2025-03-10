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
    
    self.encoder = nn.ModuleDict({
      "flow" : nn.Sequential(nn.Linear(params['flow_net']['inNodeFeatures'], params['flow_net']['nNodeFeatEmbedding']),
                             nn.ReLU()),
      
      "memb" : nn.Sequential(nn.Linear(params['memb_net']['inNodeFeatures'], params['memb_net']['nNodeFeatEmbedding']),
                             nn.ReLU())

    })

    # self.layer_memb = GNO(
    #   params['memb_net']['inNodeFeatures'], 
    #   params['memb_net']['nNodeFeatEmbedding'], 
    #   params['memb_net']['nEdgeFeatures'], 
    #   params['memb_net']['ker_width'],
    #   params['memb_net']['nlayers']
    #   )
    
    # self.layer_flow = GNO(
    #   params['flow_net']['inNodeFeatures'], 
    #   params['flow_net']['nNodeFeatEmbedding'], 
    #   params['flow_net']['nEdgeFeatures'], 
    #   params['flow_net']['ker_width'],
    #   params['flow_net']['nlayers']
    #   )
    
    # self.crossAttention = FlashCrossAttention(
    #   params['memb_net']['inNodeFeatures'], 
    #   params['flow_net']['nEdgeFeatures'], 
    #   hidden_dim=params['attn_dim']
    #   )
    
    self.decoder = nn.ModuleDict({
      "flow" : nn.Sequential(nn.Linear(params['flow_net']['nNodeFeatEmbedding'],params['flow_net']['inNodeFeatures']),
                             nn.ReLU()),
      "memb" : nn.Sequential(nn.Linear(params['memb_net']['nNodeFeatEmbedding'],params['memb_net']['inNodeFeatures']),
                             nn.ReLU())

    })


  def forward(self, batch):

    x_memb = self.encoder["memb"]( batch['memb'].y ) #For now i just train to obtain the latent embedding of membrane nodes features at next timestep
    x_flow = self.encoder["flow"]( batch['flow'].x )

    # # for i in range(self.nlayers):


    x_memb = self.decoder["memb"](x_memb)
    x_flow = self.decoder["flow"](x_flow)

    return x_memb, x_flow
  

class GNO(nn.Module):
  def __init__(self, inNodeFeatures, nNodeFeatEmbedding, nEdgeFeatures, ker_width, nlayers) -> None:
    super(GNO, self).__init__()

    self.nlayers = nlayers

    #Projecting node feature to higher dimensional embeddings
    self.feature_embedding = nn.Linear(inNodeFeatures, nNodeFeatEmbedding)

    #initialize kernel
    kernel = DenseNet([nEdgeFeatures, ker_width, ker_width, nNodeFeatEmbedding**2], torch.nn.ReLU)

    #initializeGNO
    self.layer = NNConv(nNodeFeatEmbedding, nNodeFeatEmbedding, kernel, aggr='mean')

    #bringing higher-dimensional embedding to original feature dimension
    self.inv_embedding = nn.Linear(nNodeFeatEmbedding, inNodeFeatures)

  def forward(self,x, edge_index, edge_attr):

    x = self.feature_embedding(x)

    for i in range(self.nlayers):
      x = F.relu(self.layer(x, edge_index, edge_attr))

    x = self.inv_embedding(x)

    return x

