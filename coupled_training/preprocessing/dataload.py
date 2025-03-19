import os
import random
from timeit import default_timer
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from preprocessing.utilities import *
from model.neuralFSI import *
# from model_standalone.spectralGraphNetwork import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import HeteroData
import json

	
def dataloader_memb(folder, ninit, nend, ngap, ntsteps=1):
	data = generateDatasetMembrane(ninit, nend, ngap, splitLen=ntsteps, folder=folder)
	nodes, vel, forceExt, elem, pointMass, bc_nodes = data.get_output_split()

	nodes -= 20 #since 20 is center of rotation

	scaler = StandardScaler()

	mesh = unstructMeshGenerator(nodes=nodes, vel=vel, forceExt=forceExt, pointMass=pointMass, elem=elem, bc_nodes=bc_nodes)

	return mesh, scaler

def dataloader_flow(folder, ntsteps=1):
	"""
	1. Define the bound of spatial domain. Currently assuming a domain proportional to fluid (150, 93, 100) -> (1.5, 0.93, 1.0)
	2. Scale data using minmax scalar
	3. Split into input-output pair
	4. Split into train and validation index
	"""
	data = generateDatasetFluid(folder,splitLen=ntsteps)
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler, vorticity = data.scaling(scaler)
	splitData = data.splitDataset()
	combinedData = data.combined_data()

	mesh = RectilinearMeshGenerator(
		real_space=data.get_grid_coords(),
		reference_coords = [20,20,20],
		data=splitData
	)

	return mesh, splitData, scaler

def dataloader(radius_train, batch_size, t_extend=1):
	with open("preprocessing/sim_metadata.json", "r") as f:
		metadata = json.load(f)

	num_simulations = len(metadata)
	data_train = []

	for sim in range(num_simulations):
		sim_params = metadata[sim]
	
		num_samples = int((sim_params["end"]-sim_params["start"])/sim_params["gap"] + 1 ) - t_extend + 1 # to account that we form pairs
		print("Data ID : ", sim_params["sim_id"] , " Number of data samples : ", num_samples)

		memb_mesh, memb_scaler = dataloader_memb(sim_params["memb_file"], sim_params["start"], sim_params["end"], sim_params["gap"], t_extend)
		flow_mesh, flow_data, flow_scaler = dataloader_flow(sim_params["flow_file"], t_extend)

		edge_index = {
			"memb"	:	memb_mesh.getEdgeAttr(radius_train['radius_memb']),
			"flow"	:	flow_mesh.ball_connectivity(radius_train['radius_flow'])
		}

		print("Flow Edges : ", edge_index['flow'].shape)
		print("Memb Edges : ", edge_index['memb'].shape)

		scaler = {
			"memb"	:	memb_scaler,
			"flow"	:	flow_scaler
		}

		for sample in range(num_samples):
			for pair_num in range(1,t_extend):

				#Edge attr at t=0
				edge_attr_memb = memb_mesh.attributes(sample)
				edge_attr_flow = flow_mesh.attributes(sample)

				# Membrane data (t=0, t=pair_num)
				data_sample_memb = memb_mesh.getInputOutput(sample)
				x_memb = torch.tensor(data_sample_memb[0], dtype=torch.float32, requires_grad=True)
				y_memb = torch.tensor(data_sample_memb[1][pair_num-1], dtype=torch.float32, requires_grad=True) #-1 becoz here i separate input(0th one)

				# Flow data (t=0, t=pair_num)
				x_flow = torch.tensor(flow_data[sample, 0, :], dtype=torch.float32, requires_grad=True).view(-1, 1)
				y_flow = torch.tensor(flow_data[sample, pair_num, :], dtype=torch.float32, requires_grad=True).view(-1, 1)

				#Calculating cross attribute at t=0 and might have to think what attributes sh
				sharedData = generateSharedData(flow_graph=flow_data[sample, 0, :], eulerian_domain=flow_mesh.get_grid_coords(), memb_graph=data_sample_memb[0], radius=radius_train['radius_cross'])
				get_cross_domain_edges = sharedData.computeSharedEdgeIndex()
				get_cross_domain_edgeAttr = sharedData.computeSharedEdgeAttr(get_cross_domain_edges[('membrane', 'to', 'flow')])
				print(sample, pair_num, get_cross_domain_edgeAttr[('membrane', 'to', 'flow')].shape)

				data = HeteroData()
				
				#node features
				data['flow'].x = x_flow
				data['memb'].x = x_memb
				data['flow'].y = y_flow
				data['memb'].y = y_memb
				data['memb'].bc = data_sample_memb[2]

				data['tau'] = sim_params["dt"]*pair_num

				#edge feature
				data['flow','to','flow'].edge_index = torch.tensor(edge_index['flow'],dtype=torch.long)
				# data['flow','to','membrane'].edge_index = torch.tensor(get_cross_domain_edges[('flow', 'to', 'membrane')],dtype=torch.long)
				data['memb','to','flow'].edge_index = torch.tensor(get_cross_domain_edges[('membrane', 'to', 'flow')],dtype=torch.long)
				data['memb','to','memb'].edge_index = torch.tensor(edge_index['memb'],dtype=torch.long)

				#edge attribuets
				data['flow','to','flow'].edge_attr = torch.tensor(edge_attr_flow, dtype=torch.float32)
				# data['flow','to','membrane'].edge_attr = torch.tensor(get_cross_domain_edgeAttr[('flow', 'to', 'membrane')], dtype=torch.float32)
				data['memb','to','flow'].edge_attr = torch.tensor(get_cross_domain_edgeAttr[('membrane', 'to', 'flow')], dtype=torch.float32)
				data['memb','to','memb'].edge_attr = torch.tensor(edge_attr_memb, dtype=torch.float32)
			
				data_train.append(data)

	indices = np.arange(len(data_train))
	np.random.shuffle(indices)
	val_split = 0.3
	num_val_samples = int(len(data_train) * val_split)
	val_indices = indices[:num_val_samples]
	train_indices = indices[num_val_samples:]
	with open('train_val_indices.csv', 'w') as f:
		f.write(f'Train indices: {train_indices}\n')
		f.write(f'Val indices: {val_indices}\n')

	train_loader = DataLoader([data_train[i] for i in train_indices], batch_size=batch_size, shuffle=True)
	val_loader = DataLoader([data_train[i] for i in val_indices], batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, scaler