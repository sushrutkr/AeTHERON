import os
import sys
from pathlib import Path
import random
from timeit import default_timer
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from model_standalone.spectralGraphNetwork import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import HeteroData
import json
from pprint import pprint
from collections import defaultdict
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from dataloader.membrane_data_contruct import *
from dataloader.flow_data_contruct import *
from dataloader.shared_data_contruct import *

class obtain_global_statistics():
	def __init__(self):
		self.df_flow = pd.read_csv("./data/sim_stats_flow.csv")
		self.df_memb = pd.read_csv("./data/sim_stats_membrane.csv")

		self.flow_stats = defaultdict()
		self.memb_stats = defaultdict()

		self.compute_flow_stats()
		self.compute_memb_stats()
	
	@staticmethod
	def global_mean(shard_mean, shard_size):
		"""
		Compute the global mean across all shards, weighted by shard size.
		
		Args:
				shard_mean (pd.Series): Mean of the sum of components (or scalar mean) for each shard.
				shard_size (pd.Series): Number of node-timestep pairs in each shard (nNodes * ntsteps).
		
		Returns:
				float: Global mean.
		"""
		total_size = shard_size.sum()
		if total_size == 0:
				raise ValueError("Total shard size cannot be zero.")
		weighted_sum = (shard_mean * shard_size).sum()
		return weighted_sum / total_size
	
	@staticmethod
	def global_variance(shard_var, shard_size, shard_mean=None, global_mean=None):
		"""
		Compute the global variance across all shards.
		
		Args:
				shard_var (pd.Series): Variance for each shard.
				shard_size (pd.Series): Number of node-timestep pairs in each shard (nNodes * ntsteps).
				shard_mean (pd.Series, optional): Mean of the sum of components (or scalar mean) for each shard.
				global_mean (float, optional): Global mean (if already computed).
		
		Returns:
				float: Global variance.
		"""
		if shard_mean is None or global_mean is None:
				raise ValueError("shard_mean and global_mean must be provided for variance calculation.")
		
		total_size = shard_size.sum()
		if total_size == 0:
				raise ValueError("Total shard size cannot be zero.")
		
		# Compute the global variance: within-shard variance + between-shard variance
		within_variance = (shard_var * shard_size).sum()
		between_variance = (shard_size * (shard_mean - global_mean)**2).sum()
		return (within_variance + between_variance) / total_size
	
	@staticmethod
	def compute_global_percentile(shard_percentiles, shard_size=None):
		"""
		Compute the global 95th percentile from per-shard 95th percentiles.
		
		Args:
				shard_percentiles (pd.Series): 95th percentiles for each shard (e.g., velocity_scale).
				shard_size (pd.Series, optional): Number of node-timestep pairs in each shard (nNodes * ntsteps).
						If provided, weights the percentiles by shard size.
		
		Returns:
				float: Global 95th percentile.
		"""
		if len(shard_percentiles) == 0:
			raise ValueError("Shard percentiles cannot be empty.")
		
		if shard_size is not None:
			if len(shard_size) != len(shard_percentiles):
				raise ValueError("shard_size must have the same length as shard_percentiles.")
			total_size = shard_size.sum()
			if total_size == 0:
				raise ValueError("Total shard size cannot be zero.")
			
			# Approximate the global 95th percentile by weighting the per-shard percentiles
			# This is an approximation, as we don't have the full distribution
			sorted_indices = np.argsort(shard_percentiles.values)
			sorted_percentiles = shard_percentiles.values[sorted_indices]
			sorted_sizes = shard_size.values[sorted_indices]
			cumulative_sizes = np.cumsum(sorted_sizes)
			target_size = 0.95 * total_size
			# Find the percentile value where the cumulative size reaches 95% of the total
			idx = np.searchsorted(cumulative_sizes, target_size, side="right")
			if idx == 0:
				return sorted_percentiles[0]
			elif idx >= len(sorted_percentiles):
				return sorted_percentiles[-1]
			else:
				# Linear interpolation between the two nearest percentiles
				fraction = (target_size - cumulative_sizes[idx-1]) / (cumulative_sizes[idx] - cumulative_sizes[idx-1])
				return sorted_percentiles[idx-1] + fraction * (sorted_percentiles[idx] - sorted_percentiles[idx-1])
	
		else:
			# If shard_size is not provided, simply take the 95th percentile of the per-shard percentiles
			return np.percentile(shard_percentiles, 95)

	def compute_flow_stats(self):
		"""
		Compute global statistics for the flow dataset.
		"""
		
		shard_size = self.df_flow['nNodes'] * self.df_flow['ntsteps']
		
		self.flow_stats["velocity_scale"] = obtain_global_statistics.compute_global_percentile(self.df_flow["velocity_scale"], shard_size)
		self.flow_stats["pressure_scale"] = obtain_global_statistics.compute_global_percentile(self.df_flow["pressure_scale"], shard_size)

		#old standardisation code		
		# # Compute global means first
		# velocity_global_mean = self.global_mean(self.df_flow["velocity_mean"], shard_size)
		# pressure_global_mean = self.global_mean(self.df_flow["pressure_mean"], shard_size)
		
		# # Compute global variances using the global means
		# self.flow_stats["velocity_mean"] = velocity_global_mean
		# self.flow_stats["velocity_variance"] = self.global_variance(
		# 		self.df_flow["velocity_variance"], shard_size, 
		# 		shard_mean=self.df_flow["velocity_mean"], global_mean=velocity_global_mean
		# )
		# self.flow_stats["pressure_mean"] = pressure_global_mean
		# self.flow_stats["pressure_variance"] = self.global_variance(
		# 		self.df_flow["pressure_variance"], shard_size, 
		# 		shard_mean=self.df_flow["pressure_mean"], global_mean=pressure_global_mean
		# )

		return

	def compute_memb_stats(self):
		"""
		Compute global statistics for the membrane dataset.
		"""
		shard_size = self.df_memb['nNodes'] * self.df_memb['ntsteps']
		
		#old standardisation code
		# # Compute global means first
		# velocity_global_mean = self.global_mean(self.df_memb["velocity_mean"], shard_size)
		# force_global_mean = self.global_mean(self.df_memb["force_mean"], shard_size)
		
		# # Compute global variances using the global means
		# self.memb_stats["velocity_mean"] = velocity_global_mean
		# self.memb_stats["velocity_variance"] = self.global_variance(
		# 		self.df_memb["velocity_variance"], shard_size, 
		# 		shard_mean=self.df_memb["velocity_mean"], global_mean=velocity_global_mean
		# )

		# self.memb_stats["pressure_mean"] = velocity_global_mean
		# self.memb_stats["pressure_variance"] = self.global_variance(
		# 		self.df_memb["pressure_variance"], shard_size, 
		# 		shard_mean=self.df_memb["pressure_mean"], global_mean=velocity_global_mean
		# )

		# self.memb_stats["force_mean"] = force_global_mean
		# self.memb_stats["force_variance"] = self.global_variance(
		# 		self.df_memb["force_variance"], shard_size, 
		# 		shard_mean=self.df_memb["force_mean"], global_mean=force_global_mean
		# )

		self.memb_stats["velocity_scale"] = obtain_global_statistics.compute_global_percentile(self.df_memb["velocity_scale"], shard_size)
		self.memb_stats["pressure_scale"] = obtain_global_statistics.compute_global_percentile(self.df_memb["pressure_scale"], shard_size)
		self.memb_stats["force_scale"] = obtain_global_statistics.compute_global_percentile(self.df_memb["force_scale"], shard_size)
		self.memb_stats["pointMass_min"] = np.min(self.df_memb["pointMass_min"])
		self.memb_stats["pointMass_max"] = np.max(self.df_memb["pointMass_max"])
		self.memb_stats["max_coordinate"] = np.max(self.df_memb["max_coordinate"])
		self.memb_stats["min_coordinate"] = np.min(self.df_memb["min_coordinate"])


		
		return
			
	def get_stats(self):
		"""
		Return the computed global statistics.
		
		Returns:
				tuple: (flow_stats, memb_stats)
		"""
		return self.flow_stats, self.memb_stats

def dataloader_memb(folder, ninit, nend, ngap, stats=None, ntsteps=1):
	"""
	"""
	if stats is None:
			raise ValueError("stats must be provided for membrane data standardisation.")
	
	data = generateDatasetMembrane(ninit, nend, ngap, splitLen=ntsteps, folder=folder)
	data.scaling(stats)
	data.splitData()
	nodes, vel, pressure, forceExt, elem, pointMass, bc_nodes = data.get_output_split()
	mesh = unstructMeshGenerator(nodes=nodes, 
															vel=vel, 
															pressure=pressure, 
															forceExt=forceExt, 
															pointMass=pointMass, 
															elem=elem, 
															bc_nodes=bc_nodes)

	return mesh

def dataloader_flow(folder, stats=None, ntsteps=1):
	"""
	1. Define the bound of spatial domain. Currently assuming a domain proportional to fluid (150, 93, 100) -> (1.5, 0.93, 1.0)
	2. Scale data using minmax scalar
	3. Split into input-output pair
	4. Split into train and validation index
	"""
	if stats is None:
			raise ValueError("stats must be provided for flow data standardisation.")
	
	data = generateDatasetFluid(folder,splitLen=ntsteps)
	data.scaling(stats)
	data.splitDataset()
	splitData = data.getSplitData()

	print("splitdata size : ", splitData.shape)

	mesh = RectilinearMeshGenerator(
		real_space=data.get_grid_coords(),
		reference_coords = [20,20,20],
		data=splitData
	)

	return mesh, splitData
	
def dataGenerate(radius_train, batch_size, t_extend=1, val_split = 0.3, world_size=None, rank=None, loadData = False, cache_file="./data_train_cache.pt"):
	"""
	
	"""

	print("Computing global statistics for standardisation")
	stats = obtain_global_statistics()
	flow_stats, memb_stats = stats.get_stats()
	print(flow_stats)
	print(memb_stats)
	
	if loadData != True:
		if rank == 0:
			with open("./data/sim_metadata.json", "r") as f:
				metadata = json.load(f)

			num_simulations = len(metadata)
			data_train = []

			for sim in range(num_simulations):
				sim_params = metadata[sim]

				num_samples = int((sim_params["end"]-sim_params["start"])/sim_params["gap"] + 1 ) - t_extend # think as the # of first entry for pairs
				print("Data ID : ", sim_params["sim_id"] , ", Number of data samples : ", num_samples, ", each with pairs : ",t_extend)

				# t_extend += 1, just to make it compatible with older dataloader because we want to obtain 0th datapoint along with t_extend
				memb_mesh = dataloader_memb(sim_params["memb_file"], sim_params["start"], sim_params["end"], sim_params["gap"], memb_stats, t_extend+1)
				flow_mesh, flow_data = dataloader_flow(sim_params["flow_file"], flow_stats, t_extend+1)

				edge_index = {
					"memb"	:	memb_mesh.getEdgeAttr(radius_train['radius_memb']),
					"flow"	:	flow_mesh.ball_connectivity(radius_train['radius_flow'])
				}

				print("Flow Nodes : ", flow_data.shape)
				print("Flow Edges : ", edge_index['flow'].shape)
				print("Memb Edges : ", edge_index['memb'].shape)

				start_index = int((sim_params["start"]/sim_params["gap"]) - 1)
				print("Starting reading at : ", start_index)

				for sample in range(num_samples):
					for pair_num in range(1,t_extend+1):
						#Edge attr at t=0
						edge_attr_memb = memb_mesh.attributes(sample)
						edge_attr_flow = flow_mesh.attributes(sample)

						# Membrane data (t=0, t=pair_num)
						data_sample_memb = memb_mesh.getInputOutput(sample)
						x_memb = torch.tensor(data_sample_memb[0], dtype=torch.float32, requires_grad=True)
						y_memb = torch.tensor(data_sample_memb[1][pair_num-1], dtype=torch.float32, requires_grad=True) #-1 becoz I have already separated into input(0th one) and outputs

						# Flow data (t=0, t=pair_num)
						x_flow = torch.tensor(flow_data[start_index+sample, 0, :], dtype=torch.float32, requires_grad=True).transpose(0, 1)
						y_flow = torch.tensor(flow_data[start_index+sample, pair_num, :], dtype=torch.float32, requires_grad=True).transpose(0, 1)

						#Calculating cross attribute at t=0 and might have to think what attributes sh
						sharedData = generateSharedData(flow_graph=flow_data[start_index+sample, 0, :], eulerian_domain=flow_mesh.get_grid_coords(), memb_graph=data_sample_memb[0], radius=radius_train['radius_cross'])
						get_cross_domain_edges = sharedData.computeSharedEdgeIndex()
						get_cross_domain_edgeAttr = sharedData.computeSharedEdgeAttr(get_cross_domain_edges[('membrane', 'to', 'flow')])
						print(start_index+sample, pair_num, sim_params["dt"]*pair_num, get_cross_domain_edgeAttr[('membrane', 'to', 'flow')].shape)
						# print(x_flow[48971,:])
						# print(y_flow[48971,:])
						# print("--")
						# print(x_memb[12,:])
						# print(y_memb[12,:])

						data = HeteroData()
						

						#node features
						data['flow'].x = x_flow
						data['memb'].x = x_memb
						data['flow'].y = y_flow
						data['memb'].y = y_memb
						data['memb'].bc = data_sample_memb[2]

						data['tau'] = sim_params["dt"]*pair_num

						# inMem = 0
						# print(data['memb'].x[inMem,:], data['memb'].y[inMem,:])
						# inMem = 53223
						# print(data['flow'].x[inMem,:], data['flow'].y[inMem,:])


						# print(data['tau'])
						# pprint(get_cross_domain_edges[('membrane', 'to', 'flow')], width=200)
						# edges = get_cross_domain_edges[('membrane', 'to', 'flow')]
						# filtered_elements = edges[0, edges[1] == 53223]
						# pprint(filtered_elements.tolist())

						#edge feature
						data['flow','to','flow'].edge_index = torch.tensor(edge_index['flow'],dtype=torch.long)
						# data['flow','to','membrane'].edge_index = torch.tensor(get_cross_domain_edges[('flow', 'to', 'membrane')],dtype=torch.long)
						data['memb','to','flow'].edge_index = torch.tensor(get_cross_domain_edges[('membrane', 'to', 'flow')],dtype=torch.long)
						data['memb','to','memb'].edge_index = torch.tensor(edge_index['memb'],dtype=torch.long)

						#edge attribuets
						data['flow','to','flow'].edge_attr = torch.tensor(edge_attr_flow, dtype=torch.float16)
						# data['flow','to','membrane'].edge_attr = torch.tensor(get_cross_domain_edgeAttr[('flow', 'to', 'membrane')], dtype=torch.float32)
						data['memb','to','flow'].edge_attr = torch.tensor(get_cross_domain_edgeAttr[('membrane', 'to', 'flow')], dtype=torch.float16)
						data['memb','to','memb'].edge_attr = torch.tensor(edge_attr_memb, dtype=torch.float16)
					
						data_train.append(data)

			torch.save(data_train, cache_file)
			
		dist.barrier()
	
	else:
		print(f"[rank {rank}] loading cached data…")
		dist.barrier()
		data_train = torch.load(cache_file)

	dist.barrier()
	print(len(data_train))
	indices = np.arange(len(data_train))
	np.random.shuffle(indices)
	num_val_samples = int(len(data_train) * val_split)
	val_indices = indices[:num_val_samples]
	train_indices = indices[num_val_samples:]
	with open('./data/train_val_indices.csv', 'w') as f:
		f.write(f'Train indices: {train_indices}\n')
		f.write(f'Val indices: {val_indices}\n')
	
	train_dataset = [data_train[i] for i in train_indices]
	val_dataset = [data_train[i] for i in val_indices]

	train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
	val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

	#old single GPU training
	# train_loader = DataLoader([data_train[i] for i in train_indices], batch_size=batch_size, shuffle=False)
	# val_loader = DataLoader([data_train[i] for i in val_indices], batch_size=batch_size, shuffle=False)


	return train_loader, val_loader


def dataloader(radius_train, batch_size, t_extend=1, val_split = 0.3,loadData = False,cache_file="/mnt/Sushrut/data_train_cache.pt"):
	"""
	
	"""

	print("Computing global statistics for standardisation")
	stats = obtain_global_statistics()
	flow_stats, memb_stats = stats.get_stats()
	print(flow_stats)
	print(memb_stats)
	
	if loadData != True:
		with open("./data/sim_metadata.json", "r") as f:
			metadata = json.load(f)

		num_simulations = len(metadata)
		data_train = []

		for sim in range(num_simulations):
			sim_params = metadata[sim]

			num_samples = int((sim_params["end"]-sim_params["start"])/sim_params["gap"] + 1 ) - t_extend # think as the # of first entry for pairs
			print("Data ID : ", sim_params["sim_id"] , ", Number of data samples : ", num_samples, ", each with pairs : ",t_extend)

			# t_extend += 1, just to make it compatible with older dataloader because we want to obtain 0th datapoint along with t_extend
			memb_mesh = dataloader_memb(sim_params["memb_file"], sim_params["start"], sim_params["end"], sim_params["gap"], memb_stats, t_extend+1)
			flow_mesh, flow_data = dataloader_flow(sim_params["flow_file"], flow_stats, t_extend+1)

			edge_index = {
				"memb"	:	memb_mesh.getEdgeAttr(radius_train['radius_memb']),
				"flow"	:	flow_mesh.ball_connectivity(radius_train['radius_flow'])
			}

			print("Flow Nodes : ", flow_data.shape)
			print("Flow Edges : ", edge_index['flow'].shape)
			print("Memb Edges : ", edge_index['memb'].shape)

			start_index = int((sim_params["start"]/sim_params["gap"]) - 1)
			print("Starting reading at : ", start_index)

			for sample in range(num_samples):
				for pair_num in range(1,t_extend+1):
					#Edge attr at t=0
					edge_attr_memb = memb_mesh.attributes(sample)
					edge_attr_flow = flow_mesh.attributes(sample)

					# Membrane data (t=0, t=pair_num)
					data_sample_memb = memb_mesh.getInputOutput(sample)
					x_memb = torch.tensor(data_sample_memb[0], dtype=torch.float32, requires_grad=True)
					y_memb = torch.tensor(data_sample_memb[1][pair_num-1], dtype=torch.float32, requires_grad=True) #-1 becoz I have already separated into input(0th one) and outputs

					# Flow data (t=0, t=pair_num)
					x_flow = torch.tensor(flow_data[start_index+sample, 0, :], dtype=torch.float32, requires_grad=True).transpose(0, 1)
					y_flow = torch.tensor(flow_data[start_index+sample, pair_num, :], dtype=torch.float32, requires_grad=True).transpose(0, 1)

					#Calculating cross attribute at t=0 and might have to think what attributes sh
					sharedData = generateSharedData(flow_graph=flow_data[start_index+sample, 0, :], eulerian_domain=flow_mesh.get_grid_coords(), memb_graph=data_sample_memb[0], radius=radius_train['radius_cross'])
					get_cross_domain_edges = sharedData.computeSharedEdgeIndex()
					get_cross_domain_edgeAttr = sharedData.computeSharedEdgeAttr(get_cross_domain_edges[('membrane', 'to', 'flow')])
					print(start_index+sample, pair_num, sim_params["dt"]*pair_num, get_cross_domain_edgeAttr[('membrane', 'to', 'flow')].shape)
					# print(x_flow[48971,:])
					# print(y_flow[48971,:])
					# print("--")
					# print(x_memb[12,:])
					# print(y_memb[12,:])

					data = HeteroData()
					

					#node features
					data['flow'].x = x_flow
					data['memb'].x = x_memb
					data['flow'].y = y_flow
					data['memb'].y = y_memb
					data['memb'].bc = data_sample_memb[2]

					data['tau'] = sim_params["dt"]*pair_num

					# inMem = 0
					# print(data['memb'].x[inMem,:], data['memb'].y[inMem,:])
					# inMem = 53223
					# print(data['flow'].x[inMem,:], data['flow'].y[inMem,:])


					# print(data['tau'])
					# pprint(get_cross_domain_edges[('membrane', 'to', 'flow')], width=200)
					# edges = get_cross_domain_edges[('membrane', 'to', 'flow')]
					# filtered_elements = edges[0, edges[1] == 53223]
					# pprint(filtered_elements.tolist())

					#edge feature
					data['flow','to','flow'].edge_index = torch.tensor(edge_index['flow'],dtype=torch.long)
					# data['flow','to','membrane'].edge_index = torch.tensor(get_cross_domain_edges[('flow', 'to', 'membrane')],dtype=torch.long)
					data['memb','to','flow'].edge_index = torch.tensor(get_cross_domain_edges[('membrane', 'to', 'flow')],dtype=torch.long)
					data['memb','to','memb'].edge_index = torch.tensor(edge_index['memb'],dtype=torch.long)

					#edge attribuets
					data['flow','to','flow'].edge_attr = torch.tensor(edge_attr_flow, dtype=torch.float16)
					# data['flow','to','membrane'].edge_attr = torch.tensor(get_cross_domain_edgeAttr[('flow', 'to', 'membrane')], dtype=torch.float32)
					data['memb','to','flow'].edge_attr = torch.tensor(get_cross_domain_edgeAttr[('membrane', 'to', 'flow')], dtype=torch.float16)
					data['memb','to','memb'].edge_attr = torch.tensor(edge_attr_memb, dtype=torch.float16)
				
					data_train.append(data)

		torch.save(data_train, cache_file)
	else:
		print(f"Loading cached data from {cache_file}")
		data_train = torch.load(cache_file)

	print(len(data_train))
	indices = np.arange(len(data_train))
	np.random.shuffle(indices)
	num_val_samples = int(len(data_train) * val_split)
	val_indices = indices[:num_val_samples]
	train_indices = indices[num_val_samples:]
	with open('./data/train_val_indices.csv', 'w') as f:
		f.write(f'Train indices: {train_indices}\n')
		f.write(f'Val indices: {val_indices}\n')

	train_loader = DataLoader([data_train[i] for i in train_indices], batch_size=batch_size, shuffle=False)
	val_loader = DataLoader([data_train[i] for i in val_indices], batch_size=batch_size, shuffle=False)


	return train_loader, val_loader