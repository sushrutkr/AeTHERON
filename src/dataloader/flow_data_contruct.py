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

class generateDatasetFluid:
  def __init__(self,pathName,splitLen=1):
    self.data = np.load(pathName+'data.npy')
    self.nx = self.data.shape[2]
    self.ny = self.data.shape[3]
    self.nz = self.data.shape[4]
    self.ntsteps = self.data.shape[0]
    self.split = splitLen
    self.combined_data()

  def combined_data(self):
    # self.combinedData = np.zeros(shape=(self.ntsteps, self.nx, self.ny, self.nz))
    # self.combinedData[0,:,:,:] = self.data[0,0,:,:,:]
    # for i in range(self.data.shape[0]):
    #   self.combinedData[i+1,:,:,:] = self.data[i,1,:,:,:]    
    self.combinedData = self.data[:,3:7,:,:,:] # u,v,w,p
    return self.combinedData
  
  def get_grid_coords(self):
    #return x, y, z
    return self.data[0,0,:,:,:], self.data[0,1,:,:,:], self.data[0,2,:,:,:]
  
  def scaling(self,scaler):
    self.scaler = scaler
    shape = self.combinedData.shape

    #Fortran type orderring for correct reshape
    self.combinedData = self.combinedData.reshape(shape[0], -1, order='F')
    self.combinedData = self.scaler.fit_transform(self.combinedData)
    return self.scaler, self.combinedData
  
  def splitDataset(self):
    num_splits = self.ntsteps - self.split + 1
    numNodes = self.combinedData.shape[1]
    self.SplitData = np.zeros((num_splits, self.split, numNodes))

    for i in range(num_splits):
      #here, num_split is new data size and for each data point I extract pair from that timestep
      #corresponding to that data point to 1 before num split. 
      #then in the code I use the next timestep data.
      self.SplitData[i,:,:] = self.combinedData[i:i+self.split,:]

    return self.SplitData

class RectilinearMeshGenerator(object):
  def __init__(self, real_space, reference_coords, data):
    super(RectilinearMeshGenerator, self).__init__()

    self.d = len(real_space)
    self.data = data
    self.nx = real_space[0].shape[0] #just getting # points from x arrays
    self.ny = real_space[0].shape[1]
    self.nz = real_space[0].shape[2]

    #flatten order 'F' helps to visualize point in paraview
    self.grid = np.vstack([real_space[0].flatten(order='F') - reference_coords[0], 
                           real_space[1].flatten(order='F') - reference_coords[1], 
                           real_space[2].flatten(order='F') - reference_coords[2]]).T
    
    # print("Grid : ", self.grid.shape, self.grid[53223,:])
    
  def ball_connectivity_old(self, r):
    #computationaly inefficient complex as it creates a full matrix first and then sample
    pwd = sklearn.metrics.pairwise_distances(self.grid)
    self.edge_index = np.vstack(np.where(pwd <= r))
    self.n_edges = self.edge_index.shape[1]

    return self.edge_index
  
  def ball_connectivity(self, r):
    tree = BallTree(self.grid, leaf_size=2)  # O(NlogN)
    ind = tree.query_radius(self.grid, r=r)  # O(logN)
    
    row = []
    col = []
    for i, neighbors in enumerate(ind):
      for neighbor in neighbors:
        # if i != neighbor:  # Jus to avoid self-connection, not used in present case
          row.append(i)
          col.append(neighbor)
  
    self.edge_index = np.vstack((row, col))
    self.n_edges = self.edge_index.shape[1] #shape = (2,N)

    return self.edge_index

  def get_grid(self):
    return self.grid
  
  def attributes(self,k):
    # edge_attr = np.zeros((self.n_edges, 8))
    # for n, (i,j) in enumerate(self.edge_index.transpose()):
    #     edge_attr[n, :] = np.concatenate((self.grid[i, :], self.grid[j, :], [self.data[k, 0, i]], [self.data[k, 0, j]]))
    i = self.edge_index[0]
    j = self.edge_index[1]

    # Gather grid and data values
    grid_i = self.grid[i]
    grid_j = self.grid[j]
    data_i = self.data[k, 0, i].reshape(-1, 1)
    data_j = self.data[k, 0, j].reshape(-1, 1)

    edge_attr = np.concatenate((grid_i, grid_j, data_i, data_j), axis=1)

    return edge_attr
  
  def get_grid_coords(self):
    return self.grid
  


