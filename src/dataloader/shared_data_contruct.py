import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import BallTree
import os
import re

class generateSharedData():
  def __init__(self, flow_graph, eulerian_domain, memb_graph, radius):
    # I am expanding dims because there is just one feature for now 
    self.graph1 = np.expand_dims(flow_graph, axis=1)
    self.graph2 = memb_graph
    self.radius = radius

    #since I have two domains, I am fetching eulerian domain details
    #the lagrangian domain details are contained in memb_graph features [0:3]
    self.flow_grid = eulerian_domain


  @staticmethod
  def ball_connectivity(pos1, pos2, r, self_loop=False):
    tree = BallTree(pos2, leaf_size=2)  # O(N2 log N2)
    ind = tree.query_radius(pos1, r=r)  # O(N1 log N2)

    row = []
    col = []
    for i, neighbors in enumerate(ind):
      for neighbor in neighbors:
        if i != neighbor:  # Jus to avoid self-connection, not used in present case
          row.append(i)
          col.append(neighbor)
    
    edge_index = np.vstack((row, col))
    return edge_index
  
  def computeSharedEdgeIndex(self):

    # finding memb nodes surrouding all fluid nodes so membrane to flow message can be generated.
    self.edge_index_mf = generateSharedData.ball_connectivity(self.graph2[:,0:3], self.flow_grid, self.radius, self_loop=False)
    # now I have to make few correction because the first
    #for now I am assuming mf connection can be obtained from fm, could be true  
    self.edge_index_fm = self.edge_index_mf[[1, 0], :]

    return {
        ('flow', 'to', 'membrane'): self.edge_index_fm,
        ('membrane', 'to', 'flow'): self.edge_index_mf
    }
    
  def computeSharedEdgeAttr(self,edge_index):    
    i = edge_index[1] #Because 1st is the reciever flow node
    j = edge_index[0] #Because 0th is the sender flow node

    edge_attr_mf = np.vstack([
            self.flow_grid[i, 0] - self.graph2[j, 0],
            self.flow_grid[i, 1] - self.graph2[j, 1],
            self.flow_grid[i, 2] - self.graph2[j, 2]
        ]).T 
    
    edge_attr_fm = edge_attr_mf

    return {
        ('flow', 'to', 'membrane'): edge_attr_fm,
        ('membrane', 'to', 'flow'): edge_attr_mf
    }