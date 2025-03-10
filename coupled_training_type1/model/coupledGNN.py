import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from model.GKN import *
from model.spectral import *
from scipy.ndimage import gaussian_filter
import os
import re

class CoupledGNO(nn.Module):
  def __init__(self, params):
    super(CoupledGNO, self).__init__()
    
    # self.layer_memb = GNO(
    #   params['memb_net']['inNodeFeatures'], 
    #   params['memb_net']['nNodeFeatEmbedding'], 
    #   params['memb_net']['nEdgeFeatures'], 
    #   params['memb_net']['ker_width'],
    #   params['memb_net']['nlayers']
    #   )
    
    self.layer_flow = GNO(
      params['flow_net']['inNodeFeatures'], 
      params['flow_net']['nNodeFeatEmbedding'], 
      params['flow_net']['nEdgeFeatures'], 
      params['flow_net']['ker_width'],
      params['flow_net']['nlayers']
      )
    
    self.crossAttention = FlashCrossAttention(
      params['memb_net']['inNodeFeatures'], 
      params['flow_net']['nEdgeFeatures'], 
      hidden_dim=params['attn_dim']
      )
    
  def forward(self, data_memb, data_flow):

    x_memb = self.layer_memb(data_memb.x, data_memb.edge_index, data_memb.edge_attr) #membrane output
    # x_memb = data_memb.y
    cross_attn_output = self.crossAttention(data_flow.edge_attr, x_memb) #this gives the updated edge_features for flow predictions
    # flow_edge_attr = torch.concat((cross_attn_output, data_flow.edge_attr),axis=1)
    x_flow = self.layer_flow(data_flow.x, data_flow.edge_index, cross_attn_output) #flow output

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

