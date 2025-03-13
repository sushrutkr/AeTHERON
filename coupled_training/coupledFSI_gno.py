# Standard library imports
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
from utilities import *
from model.neuralFSI import *
# from model_standalone.spectralGraphNetwork import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import HeteroData

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def dataloader_memb(folder, radius_train, batch_size, ninit, nend, ngap, ntsteps=1):
	data = generateDatasetMembrane(ninit, nend, ngap, splitLen=ntsteps, folder=folder+"1e6/")
	nodes, vel, forceExt, elem, pointMass, bc_nodes = data.get_output_split()

	nodes -= 20 #since 20 is center of rotation

	scaler = StandardScaler()

	mesh = unstructMeshGenerator(nodes=nodes, vel=vel, forceExt=forceExt, pointMass=pointMass, elem=elem, bc_nodes=bc_nodes)

	return mesh, scaler

def dataloader_flow(folder, radius_train, batch_size, ntsteps=1):
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

def dataloader(folder, radius_train, batch_size, ninit, nend, ngap, ntsteps=1):
	num_samples = int((nend-ninit)/ngap + 1 ) - 1 # -1 is to account that we form pairs
	print("Number of data samples : ", num_samples)
	indices = np.arange(num_samples)
	np.random.shuffle(indices)
	val_split = 0.3
	num_val_samples = int(num_samples * val_split)
	val_indices = indices[:num_val_samples]
	train_indices = indices[num_val_samples:]
	with open('train_val_indices.csv', 'w') as f:
		f.write(f'Train indices: {train_indices}\n')
		f.write(f'Val indices: {val_indices}\n')

	memb_mesh, memb_scaler = dataloader_memb(folder, radius_train['radius_flow'], batch_size, ninit, nend, ngap, ntsteps)
	flow_mesh, flow_data, flow_scaler = dataloader_flow(folder, radius_train['radius_memb'], batch_size, ntsteps)

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
	
	data_train = []

	for j in range(num_samples):
		edge_attr_memb = memb_mesh.attributes(j)
		edge_attr_flow = flow_mesh.attributes(j)
		data_sample = memb_mesh.getInputOutput(j)

		# Membrane data
		data_sample_memb = memb_mesh.getInputOutput(j)
		x_memb = torch.tensor(data_sample_memb[0], dtype=torch.float32, requires_grad=True)
		y_memb = torch.tensor(data_sample_memb[1][0], dtype=torch.float32, requires_grad=True)

		# Flow data
		x_flow = torch.tensor(flow_data[j, 0, :], dtype=torch.float32, requires_grad=True).view(-1, 1)
		y_flow = torch.tensor(flow_data[j, 1, :], dtype=torch.float32, requires_grad=True).view(-1, 1)

		sharedData = generateSharedData(flow_graph=flow_data[j, 0, :], eulerian_domain=flow_mesh.get_grid_coords(), memb_graph=data_sample_memb[0], radius=radius_train['radius_cross'])
		get_cross_domain_edges = sharedData.computeSharedEdgeIndex()
		get_cross_domain_edgeAttr = sharedData.computeSharedEdgeAttr(get_cross_domain_edges[('membrane', 'to', 'flow')])
		print(j, get_cross_domain_edgeAttr[('membrane', 'to', 'flow')].shape)

		data = HeteroData()
		
		#node features
		data['flow'].x = x_flow
		data['memb'].x = x_memb
		data['flow'].y = y_flow
		data['memb'].y = y_memb
		data['memb'].bc = data_sample[2]

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

	train_loader = DataLoader([data_train[i] for i in train_indices], batch_size=batch_size, shuffle=True)
	val_loader = DataLoader([data_train[i] for i in val_indices], batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, scaler

def dataloader_test(folder, radius_train, batch_size, ninit, nend, ngap, ntsteps=1):
	"""
	1. Define the bound of spatial domain. Currently assuming a domain proportional to fluid (150, 93, 100) -> (1.5, 0.93, 1.0)
	2. Scale data using minmax scalar
	3. Split into input-output pair
	4. Split into train and validation index
	"""
	data = generateDatasetFluid(folder, splitLen=ntsteps)

	# Scale data using MinMaxScaler (same as in training)
	scaler = MinMaxScaler(feature_range=(0, 1))
	scaler, vorticity = data.scaling(scaler)
	splitData = data.splitDataset()
	combinedData = data.combined_data()

	num_samples = int((nend-ninit)/ngap + 1 ) - 1
	print("Num of samples batches, timesteps per sample : ", num_samples, splitData.shape[1])
	indices = np.arange(num_samples)
	np.random.shuffle(indices)
	val_split = 0.3
	num_val_samples = int(num_samples * val_split)
	val_indices = indices[:num_val_samples]
	train_indices = indices[num_val_samples:]

	with open('train_val_indices.csv', 'w') as f:
		f.write(f'Train indices: {train_indices}\n')
		f.write(f'Val indices: {val_indices}\n')

	mesh = RectilinearMeshGenerator(
		real_space=data.get_grid_coords(),
		reference_coords = [20,20,20],
		data=splitData
	)

	edge_index = mesh.ball_connectivity(radius_train)

	data_train = []
	for j in range(num_samples):
		edge_attr = mesh.attributes(j)
		
		# print(j, torch.tensor(splitData[j,0,:]).view(-1,1).shape)
		data_train.append(Data(
			x = torch.tensor(splitData[j,0,:], dtype=torch.float32).view(-1,1),
			y = torch.tensor(splitData[j,1,:], dtype=torch.float32).view(-1,1),
			# y=torch.tensor(splitData[j,:,:].transpose(), dtype=torch.float32),
			edge_index = torch.tensor(edge_index,dtype=torch.long),
			edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
		))


	train_loader = DataLoader([data_train[i] for i in train_indices], batch_size=batch_size, shuffle=True)
	val_loader = DataLoader([data_train[i] for i in val_indices], batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, scaler

def main(checkpoint_path=None):
	set_seed(42)

	# Parameters
	params_network = {
		'memb_net': {
			'inNodeFeatures'				: 9,
			'nNodeFeatEmbedding'		: 16,
			'nEdgeFeatures'					: 12,
			'ker_width'							: 8
		},
		'flow_net': {
			'inNodeFeatures'				: 1,
			'nNodeFeatEmbedding'		: 8,
			'nEdgeFeatures'					: 8,
			'ker_width'							: 4
		},
		'attn_dim'								: 16, #found 16 to be a better value compared to 8, 24, 32,
		'nlayers'									: 2
	}
	
	params_training = {
		'epochs' 								: 2000,
		'learning_rate' 				: 0.001 ,
		'scheduler_step' 				: 500,  
		'scheduler_gamma' 			: 0.5,
		'validation_frequency' 	: 100,
		'save_frequency' 				: 100,
	}

	params_data = {
		'location' 			: '../sample_data/',
		'batch_size' 		: 6,
		'ntsteps' 			: 2,
	}

	train_radius = {
		'radius_flow' 	: 0.1,		 # keeping it >=
		'radius_memb'		: 0.04,		 #makes sense to keep <= 2X\Delta_memb
		'radius_cross'  : 0.04     #makes sense to keep <= 2X\Delta_memb
	}

	timeRange = {
		'start' : 10,
		'end'		: 2000,
		'gap'		: 10
	}
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load data
	train_loader, val_loader, scaler = dataloader(params_data['location'],
																							  train_radius, 
																								params_data['batch_size'], 
																								timeRange['start'],
																								timeRange['end'],
																								timeRange['gap'],
																								params_data['ntsteps'])

	# train_loader, val_loader, scaler = dataloader_test('../sample_data/', train_radius['radius_flow'], 
	# 																									params_data['batch_size'], 
	# 																									timeRange['start'],
	# 																									timeRange['end'],
	# 																									timeRange['gap'], params_data['ntsteps'])
	

	print("----Loaded Data----")

	# Initialize model
	# For now only performing next time step prediction
	model_instance = neuralFSI(params=params_network).to(device)
	print(model_instance)
	optimizer = torch.optim.Adam(model_instance.parameters(), 
															lr=params_training['learning_rate'])

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
																						 step_size=params_training['scheduler_step'], 
																						 gamma=params_training['scheduler_gamma'])
	criterion = torch.nn.MSELoss()

	model_instance = torch.compile(model_instance, dynamic=True)

	# Initialize training
	start_epoch = 0
	if checkpoint_path:
		checkpoint = torch.load(checkpoint_path)        
		model_instance.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])        
		start_epoch = checkpoint['epoch'] + 1
		best_val_loss = checkpoint['val_loss']
		print(f"Resuming training from epoch {start_epoch}")
	else:
		best_val_loss = float('inf')

	#training
	for epoch in range(start_epoch, params_training['epochs']):
		model_instance.train()
		train_loss = 0.0
		for batch in train_loader:
			optimizer.zero_grad(set_to_none=True)
			batch = batch.to(device)

			with torch.autocast(device_type='cuda', dtype=torch.float16): 
				out_flow, out_memb = model_instance(batch)

				loss_memb = criterion(out_memb.view(-1, 1), batch['memb'].y.view(-1, 1))
				loss_flow = criterion(out_flow.view(-1, 1), batch['flow'].y.view(-1, 1))
			loss = loss_flow + loss_memb
				
			loss.backward()
			del out_memb, out_flow, loss_flow, loss_memb
			torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)

			optimizer.step()

			train_loss += loss.item()
			torch.cuda.empty_cache()
		
		avg_train_loss = train_loss / len(train_loader)
		print(f"Epoch {epoch+1}/{params_training['epochs']}, Train Loss: {avg_train_loss:.6f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

		# Save model 
		if (epoch + 1) % params_training['save_frequency'] == 0:
			torch.save({
				'epoch': epoch,
				'model_state_dict': model_instance.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': scheduler.state_dict(),
				'train_loss': avg_train_loss,
				'val_loss': best_val_loss
			}, f'model_epoch_{epoch+1}.pth')
			print(f"Model saved at epoch {epoch+1}")

		# Validation
		if (epoch + 1) % params_training['validation_frequency'] == 0:
			model_instance.eval()
			val_loss = 0.0
			with torch.no_grad():
				for batch in val_loader:
					batch = batch.to(device)
					with torch.autocast(device_type='cuda', dtype=torch.float16):
						out_flow, out_memb = model_instance(batch)

						loss_memb = criterion(out_memb.view(-1, 1), batch['memb'].y.view(-1, 1))
						loss_flow = criterion(out_flow.view(-1, 1), batch['flow'].y.view(-1, 1))
						loss = loss_flow + loss_memb

					del out_memb, out_flow, loss_flow, loss_memb
					val_loss += loss.item()

			avg_val_loss = val_loss / len(val_loader)
			print(f"Epoch {epoch+1}/{params_training['epochs']}, Validation Loss: {avg_val_loss:.6f}")

			# Save best model
			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				torch.save(model_instance.state_dict(), 'best_model.pth')
				print(f"Best model saved with validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
	# main()
	main('model_epoch_200.pth')