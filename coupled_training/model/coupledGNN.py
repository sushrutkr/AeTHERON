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
  def __init__(self, inNodeFeatures, nNodeFeatEmbedding,  nEdgeFeatures, ker_width, attn_dim, nConvolutions, ntsteps, nModesFNO=1, activation=nn.SiLU(), num_modes=2, time_emb_dim=32):
    super(CoupledGNO, self).__init__()
    
    self.layer_memb = GNO(inNodeFeatures, nNodeFeatEmbedding, nEdgeFeatures, ker_width)
    self.layer_memb = GNO(inNodeFeatures, nNodeFeatEmbedding, nEdgeFeatures, ker_width)
    
    self.crossAttention = CrossAttention(nEdgeFeatures[1], nNodeFeatEmbedding[0], hidden_dim=attn_dim)
    
  def forward(self, data):
    x, edge_index, edge_attr, batch, ptr = data.x, data.edge_index, data.edge_attr, data.batch, data.ptr
    # print(x.shape, edge_attr.shape, edge_index.shape) #torch.Size([10542, 38]) torch.Size([71562, 12]) torch.Size([2, 71562])

    x = self.feature_embedding(x)

    for i in range(self.nConvolutions):
      x = F.relu(self.layer(x, edge_index, edge_attr))

    x = self.inv_embedding(x)
    return x
  

class GNO(nn.Module):
  def __init__(self, inNodeFeatures, nNodeFeatEmbedding, nEdgeFeatures, ker_width) -> None:
    super(GNO, self).__init__()

    #Projecting node feature to higher dimensional embeddings
    self.feature_embedding = nn.Linear(inNodeFeatures, nNodeFeatEmbedding)

    #initialize kernel
    kernel = DenseNet([nEdgeFeatures, ker_width, ker_width, nNodeFeatEmbedding**2], torch.nn.ReLU)

    #initializeGNO
    self.layer = NNConv(nNodeFeatEmbedding, nNodeFeatEmbedding, kernel, aggr='mean')

    #bringing higher-dimensional embedding to original feature dimension
    self.inv_embedding = nn.Linear(nNodeFeatEmbedding, inNodeFeatures)

  def forward(self,data):
    x, edge_index, edge_attr, batch, ptr = data.x, data.edge_index, data.edge_attr, data.batch, data.ptr

    for i in range(self.nConvolutions):
      x = F.relu(self.layer(x, edge_index, edge_attr))

    x = self.inv_embedding(x)

