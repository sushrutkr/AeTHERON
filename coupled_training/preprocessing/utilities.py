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
    assert self.ntsteps - splitLen > 0
    self.folder = folder
    fnameMesh = os.path.join(folder, "marker.{:>07d}.dat".format(ninit))
    self.nNodes, self.nElem = generateDatasetMembrane.obtainNnodesAndElem(fnameMesh)
    self.AllNodes = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
    self.AllVel = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
    self.AllForce = np.zeros(shape=(self.nNodes, 3, self.ntsteps))
    self.AllElem = np.zeros(shape=(self.nElem, 3, self.ntsteps))
    self.SplitNodes = np.array([])
    self.SplitVel = np.array([])
    self.SplitElem = np.array([])
    self.BCNodes = self.get_bc_node_info()
    self.compileData()
    self.calculate_point_mass(thickness=0.02,density=1)
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
  
  def calculate_point_mass(self, thickness, density):
    self.pointMass = np.zeros(shape=(self.nNodes))

    edges_AB = self.nodes[self.elem[:,0]-1,0:3] - self.nodes[self.elem[:,1]-1,0:3]
    edges_AC = self.nodes[self.elem[:,0]-1,0:3] - self.nodes[self.elem[:,2]-1,0:3]
    normal = np.cross(edges_AB, edges_AC)/2
    area = np.linalg.norm(normal, axis=1)

    for iElem in range(self.nElem):
      mass = area[iElem]*thickness*density
      for i in range(3):
        nodeIndex = self.elem[iElem,i] - 1
        self.pointMass[nodeIndex] += mass/3
     
    return
  
  def get_bc_node_info(self):
    nodeIDs = np.genfromtxt(self.folder+"dirichlet_bc_nodes.csv", skip_header=1)
    return nodeIDs
  
  def compileData(self):
    l = 0
    for k in range(self.ninit, self.nend + self.ngap, self.ngap):
      fnameMesh = os.path.join(self.folder, "marker.{:>07d}.dat".format(k))
      self.readFiles(fnameMesh)
      self.AllNodes[:, :, l] = self.nodes[:, 0:3]
      self.AllVel[:, :, l] = self.nodes[:, 3:6]
      self.AllForce[:, :, l] = self.nodes[:, 7:10]
      self.AllElem[:, :, l] = self.elem[:, 0:3]
      l += 1

    self.AllElem = np.array(self.AllElem, dtype=int)

  def splitData(self):
    numNodes, coords, ntsteps = self.AllNodes.shape
    num_splits = ntsteps - self.split + 1
    
    self.SplitNodes = np.zeros((num_splits, numNodes, self.split, coords))
    self.SplitVel = np.zeros((num_splits, numNodes, self.split, coords))
    self.SplitExtForce = np.zeros((num_splits, numNodes, self.split, coords))
    
    for i in range(num_splits):
      self.SplitNodes[i] = self.AllNodes[:, :, i:i+self.split].transpose(0, 2, 1)
      self.SplitVel[i] = self.AllVel[:, :, i:i+self.split].transpose(0, 2, 1)
      self.SplitExtForce[i] = self.AllForce[:, :, i:i+self.split].transpose(0, 2, 1)
    
    self.SplitElem = self.AllElem[:, :, 0]

  def get_output_split(self):
    return self.SplitNodes, self.SplitVel, self.SplitExtForce, self.SplitElem, np.expand_dims(self.pointMass, axis=1), self.BCNodes
  
  def get_output_full(self):
    return self.AllNodes, self.AllVel, self.AllElem

class unstructMeshGenerator():
  def __init__(self,nodes,vel,forceExt,pointMass,elem,bc_nodes):
    self.nodes = nodes #[ntsteps/batches, nNodes, input-output, features]
    self.elem = elem #[nElem, connections]
    self.vel = vel
    self.forceExt = forceExt
    self.pointMass = pointMass
    self.nNodes = len(self.nodes[0,:,0,0])
    self.nElem = len(self.elem[:,0])
    self.bc_nodes = bc_nodes.flatten().astype(int)

  def build_grid(self,k):
    return self.nodes[k,:,0,:]

  def getEdgeAttr(self,r):
    coords = self.nodes[0,:,0,:]
    pwd = sklearn.metrics.pairwise_distances(coords)
    self.edge_index = np.vstack(np.where(pwd <= r))
    self.n_edges = self.edge_index.shape[1]
    
    return self.edge_index
  
  def attributes(self,k):
    #shape of node data (e.g. coords) - (100, 1757, 2, 3)
    edge_attr = np.zeros((self.n_edges, 7))
    for n, (i,j) in enumerate(self.edge_index.transpose()):
        edge_attr[n,:] = np.array([
                                    self.nodes[k,i,0,0] - self.nodes[k,j,0,0], 
                                    self.nodes[k,i,0,1] - self.nodes[k,j,0,1], 
                                    self.nodes[k,i,0,2] - self.nodes[k,j,0,2],
                                    np.sqrt((self.nodes[k,i,0,0] - self.nodes[k,j,0,0])**2 +
                                            (self.nodes[k,i,0,1] - self.nodes[k,j,0,1])**2 +
                                            (self.nodes[k,i,0,2] - self.nodes[k,j,0,2])**2),
                                    self.vel[k,i,0,0] - self.vel[k,j,0,0], 
                                    self.vel[k,i,0,1] - self.vel[k,j,0,1], 
                                    self.vel[k,i,0,2] - self.vel[k,j,0,2]
                                    ])

    return edge_attr
  
  def getInputOutput(self,k):
    input = np.concatenate((self.nodes[k, :, 0, :], 
                            self.vel[k, :, 0, :], 
                            self.forceExt[k, :, 0, :],
                            self.pointMass[:]),
                            axis=1)
    output = np.concatenate((self.nodes[k, :, 1:, :], self.vel[k, :, 1:, :], self.forceExt[k, :, 1:, :]), axis=2) #1: is to extract next timestep data
    output = np.transpose(output, (1, 0, 2))

    bc = np.ones(shape=(self.nNodes,7)) # 7 - [flag, 3 coords, 3 vels], no need for force as that's going to come from flow
    bc[self.bc_nodes,0] = 0 #to nullify any preditions at those nodes
    #--- Since the dimen for output is output shape :  (1, 1757, 9), i have to use first index otherwise it shoudl not be use
    bc[self.bc_nodes,1:] = output[0,self.bc_nodes,0:6]

    return input, output, bc
  
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
    self.combinedData = self.data[:,3,:,:,:] # as 3rd data is for vorticity
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
  

class DummyScaler:
    def fit(self, data):
        return self

    def transform(self, data):
        return data

    def fit_transform(self, data):
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        return data