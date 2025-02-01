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
    
    self.layer_memb = GNO(
      params['memb']['inNodeFeatures'], 
      params['memb']['nNodeFeatEmbedding'], 
      params['memb']['nEdgeFeatures'], 
      params['memb']['ker_width'],
      params['memb']['nlayers']
      )
    
    self.layer_flow = GNO(
      params['flow']['inNodeFeatures'], 
      params['flow']['nNodeFeatEmbedding'], 
      params['flow']['nEdgeFeatures']+params['attn_dim'], 
      params['flow']['ker_width'],
      params['flow']['nlayers']
      )
    
    self.crossAttention = FlashCrossAttention(
      params['memb']['nNodeFeatEmbedding'], 
      params['flow']['nEdgeFeatures'], 
      hidden_dim=params['attn_dim']
      )
    
  def forward(self, data):
    x, edge_index, edge_attr, batch, ptr = data.x, data.edge_index, data.edge_attr, data.batch, data.ptr
    # print(x.shape, edge_attr.shape, edge_index.shape) #torch.Size([10542, 38]) torch.Size([71562, 12]) torch.Size([2, 71562])

    x_memb = self.layer_memb(x, edge_index, edge_attr)
    cross_attn_output = self.crossAttention(edge_attr, x_memb)
    x_flow = self.layer_flow(x, edge_index, cross_attn_output)

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

