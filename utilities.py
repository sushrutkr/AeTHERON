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

class generateDatasetMembrane:
  def __init__(self, ninit, nend, ngap, splitLen=1, folder="./"):
    self.nodes = []
    self.elem = []
    self.ninit = ninit
    self.nend = nend 
    self.ngap = ngap
    self.split = splitLen
    self.ntsteps = int(((nend - ninit) / ngap) + 1)
    self.folder = folder
    fnameMesh = os.path.join(folder, "marker.{:>07d}.dat".format(ninit))
    self.nNodes, self.nElem = generateDatasetMembrane.obtainNnodesAndElem(fnameMesh)
    self.AllNodes = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
    self.AllVel = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
    self.AllElem = np.zeros(shape=(self.nElem, 3, self.ntsteps))
    self.SplitNodes = np.array([])
    self.SplitVel = np.array([])
    self.SplitElem = np.array([])
    self.compileData()
    self.splitData()

  @staticmethod
  def obtainNnodesAndElem(fnameMesh):
    with open(fnameMesh) as f:
      for i, line in enumerate(f):
          if i == 1:
            string = f.readline()
          elif i > 1:
            break
    temp = re.findall(r'\d+', string)
    res = list(map(int, temp))
    nNodes = res[0]
    nElem = res[1]
    return nNodes, nElem

  def readFiles(self, fnameMesh):
    self.nNodes, self.nElem = generateDatasetMembrane.obtainNnodesAndElem(fnameMesh)
    self.nodes = np.genfromtxt(fnameMesh, skip_header=3, skip_footer=self.nElem)
    self.elem = np.genfromtxt(fnameMesh, skip_header=3 + self.nNodes, dtype=int)
    return
  
  def compileData(self):
    l = 0
    for k in range(self.ninit, self.nend + self.ngap, self.ngap):
      fnameMesh = os.path.join(self.folder, "marker.{:>07d}.dat".format(k))
      self.readFiles(fnameMesh)
      self.AllNodes[:, :, l] = self.nodes[:, 0:3]
      self.AllVel[:, :, l] = self.nodes[:, 3:6]
      self.AllElem[:, :, l] = self.elem[:, 0:3]
      l += 1

    self.AllElem = np.array(self.AllElem, dtype=int)

  def splitData(self):
    numNodes, coords, ntsteps = self.AllNodes.shape
    num_splits = ntsteps - self.split + 1
    
    self.SplitNodes = np.zeros((num_splits, numNodes, self.split, coords))
    self.SplitVel = np.zeros((num_splits, numNodes, self.split, coords))
    
    for i in range(num_splits):
      self.SplitNodes[i] = self.AllNodes[:, :, i:i+self.split].transpose(0, 2, 1)
      self.SplitVel[i] = self.AllVel[:, :, i:i+self.split].transpose(0, 2, 1)

    
    self.SplitElem = self.AllElem[:, :, 0]

  def get_output_split(self):
    return self.SplitNodes, self.SplitVel, self.SplitElem
  
  def get_output_full(self):
    return self.AllNodes, self.AllVel, self.AllElem

class generateDatasetFluid:
  def __init__(self,pathName,splitLen=1):
    self.data = np.load(pathName+'data.npy')
    self.nx = self.data.shape[2]
    self.ny = self.data.shape[3]
    self.nz = self.data.shape[4]
    self.ntsteps = self.data.shape[0] + 1
    self.split = splitLen
    self.combined_data()

  def combined_data(self):
    self.combinedData = np.zeros(shape=(self.ntsteps, self.nx, self.ny, self.nz))
    self.combinedData[0,:,:,:] = self.data[0,0,:,:,:]
    for i in range(self.data.shape[0]):
      self.combinedData[i+1,:,:,:] = self.data[i,1,:,:,:]    

    return self.combinedData
  
  def scaling(self,scaler):
    self.scaler = scaler
    shape = self.combinedData.shape
    self.combinedData = self.combinedData.reshape(shape[0], -1)
    self.combinedData = self.scaler.fit_transform(self.combinedData)
    return self.scaler, self.combinedData
  
  def splitDataset(self):
    num_splits = self.ntsteps - self.split + 1
    numNodes = self.combinedData.shape[1]
    features = 1
    self.SplitData = np.zeros((num_splits, self.split, numNodes))

    for i in range(num_splits):
      self.SplitData[i,:,:] = self.combinedData[i:i+self.split,:]

    return self.SplitData

  

class unstructMeshGenerator():
  def __init__(self,nodes,vel,elem):
    self.nodes = nodes #[ntsteps/batches, nNodes, input-output, features]
    self.elem = elem #[nElem, connections]
    self.vel = vel
    self.nNodes = len(self.nodes[0,:,0,0])
    self.nElem = len(self.elem[:,0])

  def build_grid(self,k):
    return torch.tensor(self.nodes[k,:,0,:], dtype=torch.float32)

  def getEdgeAttr(self,r):
    coords = self.nodes[0,:,0,:]
    pwd = sklearn.metrics.pairwise_distances(coords)
    self.edge_index = np.vstack(np.where(pwd <= r))
    self.n_edges = self.edge_index.shape[1]
    
    return torch.tensor(self.edge_index, dtype=torch.long)
  
  def attributes(self,k):
    edge_attr = np.zeros((self.n_edges, 12))
    for n, (i,j) in enumerate(self.edge_index.transpose()):
        edge_attr[n,:] = np.array([
                                    self.nodes[k,i,0,0], self.nodes[k,i,0,1], self.nodes[k,i,0,2], 
                                    self.nodes[k,j,0,0], self.nodes[k,j,0,1], self.nodes[k,j,0,2],
                                    self.vel[k,i,0,0], self.vel[k,i,0,1], self.vel[k,i,0,2],
                                    self.vel[k,j,0,1], self.vel[k,j,0,1], self.vel[k,j,0,2]
                                    ])

    return torch.tensor(edge_attr, dtype=torch.float)
  
  def getInputOutput(self,k):
    input = np.concatenate((self.nodes[k, :, 0, :], self.vel[k, :, 0, :]), axis=1)
    output = np.concatenate((self.nodes[k, :, :, :], self.vel[k, :, :, :]), axis=2)
    output = np.transpose(output, (1, 0, 2))

    return torch.tensor(input, dtype=torch.float32), torch.tensor(output, dtype=torch.float32)
  
class DenseNet(torch.nn.Module):
  def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
    super(DenseNet, self).__init__()

    self.n_layers = len(layers) - 1

    assert self.n_layers >= 1

    self.layers = nn.ModuleList()

    for j in range(self.n_layers):
      self.layers.append(nn.Linear(layers[j], layers[j+1]))

      if j != self.n_layers - 1:
        if normalize:
          self.layers.append(nn.BatchNorm1d(layers[j+1]))

        self.layers.append(nonlinearity())

    if out_nonlinearity is not None:
      self.layers.append(out_nonlinearity())

  def forward(self, x):
    for _, l in enumerate(self.layers):
      x = l(x)

    return x


class CartesianMeshGenerator(object):
  def __init__(self, real_space, mesh_size,data):
    super(CartesianMeshGenerator, self).__init__()

    self.d = len(real_space)
    self.s = mesh_size[0]
    self.data = data
    self.nx = mesh_size[0]
    self.ny = mesh_size[1]
    self.nz = mesh_size[2]

    assert len(mesh_size) == self.d

    if self.d == 1:
      self.n = mesh_size[0]
      self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
    else:
      self.n = 1
      grids = []
      for j in range(self.d):
        grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
        self.n *= mesh_size[j]

      self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids, indexing='ij')]).T
    
  def ball_connectivity_old(self, r):
    #computationaly inefficient complex as it creates a full matrix first and then sample
    pwd = sklearn.metrics.pairwise_distances(self.grid)
    self.edge_index = np.vstack(np.where(pwd <= r))
    self.n_edges = self.edge_index.shape[1]

    return torch.tensor(self.edge_index, dtype=torch.long)
  
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
    self.n_edges = self.edge_index.shape[1]

    return torch.tensor(self.edge_index)

  def get_grid(self):
    return torch.tensor(self.grid, dtype=torch.float32)
  
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

    return torch.tensor(edge_attr, dtype=torch.float32)
