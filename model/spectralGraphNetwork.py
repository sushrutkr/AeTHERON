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

class SpecGNO(nn.Module):
  def __init__(self, inNodeFeatures, nNodeFeatEmbedding, ker_width, nConvolutions, nEdgeFeatures, ntsteps, nModesFNO=1, activation=nn.SiLU(), num_modes=2, time_emb_dim=32):
    super(SpecGNO, self).__init__()

    self.nConvolutions = nConvolutions
    self.ntsteps = ntsteps
    self.time_emb_dim = time_emb_dim
    self.nNodeFeatEmbedding = nNodeFeatEmbedding
    self.inNodeFeatures = inNodeFeatures

    #Projecting node feature to higher dimensional embeddings
    self.feature_embedding = nn.Linear(inNodeFeatures+time_emb_dim, nNodeFeatEmbedding)

    #initialize kernel
    kernel = DenseNet([nEdgeFeatures, ker_width, ker_width, nNodeFeatEmbedding**2], torch.nn.ReLU)

    #initializeGNO
    self.layer = NNConv(nNodeFeatEmbedding, nNodeFeatEmbedding, kernel, aggr='mean')

    #initialize spectral layers for features, here we might not need node update
    self.time_conv_modules = nn.ModuleList() 
    self.time_conv_x_modules = nn.ModuleList()
    for _ in range(self.nConvolutions):
      self.time_conv_modules.append(TimeConv(nNodeFeatEmbedding, nNodeFeatEmbedding, nModesFNO, activation, with_nin=False))
      self.time_conv_x_modules.append(TimeConv_x(2, 2, num_modes, activation, with_nin=False))

    #bringing higher-dimensional embedding to original feature dimension
    self.inv_embedding = nn.Linear(nNodeFeatEmbedding, inNodeFeatures)
    
  def forward(self, data):
    x, edge_index, edge_attr, batch, ptr = data.x, data.edge_index, data.edge_attr, data.batch, data.ptr
    # print(x.shape, edge_attr.shape, edge_index.shape) #torch.Size([3514, 6]) torch.Size([23854, 12]) torch.Size([2, 23854])
    
    num_graphs = len(ptr) - 1 # or number of batches
    num_nodes = x.shape[0]
    num_edges = edge_index.shape[1]
    
    time_emb = get_timestep_embedding(torch.arange(self.ntsteps).to(x), embedding_dim=self.time_emb_dim, max_positions=10000)

    x = x.repeat(self.ntsteps,1)
    time_emb_repeated = time_emb.unsqueeze(1).repeat(1, num_nodes, 1).view(-1, self.time_emb_dim)
    x = torch.cat((x, time_emb_repeated), dim=1)

    edge_index = edge_index.repeat(1, self.ntsteps)
    offsets = torch.arange(self.ntsteps, device=edge_index.device).repeat_interleave(num_edges) * num_nodes
    edge_index += offsets.unsqueeze(0)

    edge_attr = edge_attr.repeat(self.ntsteps, 1)

    # print(x.shape, edge_attr.shape, edge_index.shape) #torch.Size([10542, 38]) torch.Size([71562, 12]) torch.Size([2, 71562])

    x = self.feature_embedding(x)

    for i in range(self.nConvolutions):
      # edge_attr = self.time_conv_modules[i](edge_attr)
      x = self.time_conv_modules[i](x.view(self.ntsteps, num_nodes, self.nNodeFeatEmbedding)).view(self.ntsteps*num_nodes, self.nNodeFeatEmbedding)
      x = F.relu(self.layer(x, edge_index, edge_attr))

    x = self.inv_embedding(x)

    x = x.view(self.ntsteps, num_graphs, num_nodes//num_graphs, self.inNodeFeatures)
    x = x.transpose(0, 1).reshape(-1, num_nodes // num_graphs, self.inNodeFeatures)

    return x
  