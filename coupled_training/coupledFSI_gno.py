# Standard library imports
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
from model.coupledGNN import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def paired_collate_fn(data_list):
    batch_memb = Batch.from_data_list([item[0] for item in data_list])
    batch_fluid = Batch.from_data_list([item[1] for item in data_list])
    return batch_memb, batch_fluid

def dataloader_memb(folder, radius_train, batch_size,ntsteps=1):
	data = generateDatasetMembrane(ninit=1000, nend=2000, ngap=10, splitLen=ntsteps, folder=folder+"1e6/")
	nodes, vel, elem = data.get_output_split()

	nodes -= 20

	scaler = StandardScaler()

	print("Shape of nodes, vel and elem : ", nodes.shape, vel.shape, elem.shape)
	num_samples = nodes.shape[0]
	print("Num of samples batches, timesteps per sample : ", num_samples, nodes.shape[2])
	indices = np.arange(num_samples)
	np.random.shuffle(indices)
	val_split = 0.2
	num_val_samples = int(num_samples * val_split)
	val_indices = indices[:num_val_samples]
	train_indices = indices[num_val_samples:]

	mesh = unstructMeshGenerator(nodes=nodes, vel=vel, elem=elem)

	return mesh, scaler

def dataloader_flow(folder, radius_train, batch_size, ntsteps=1):
	"""
	1. Define the bound of spatial domain. Currently assuming a domain proportional to fluid (150, 93, 100) -> (1.5, 0.93, 1.0)
	2. Scale data using minmax scalar
	3. Split into input-output pair
	4. Split into train and validation index
	"""
	data = generateDatasetFluid(folder,splitLen=ntsteps)

	scaler = MinMaxScaler(feature_range=(0,1))
	scaler, vorticity = data.scaling(scaler)
	splitData = data.splitDataset()
	print(splitData.shape)
	combinedData = data.combined_data()

	num_samples = splitData.shape[0]
	print("Num of samples batches, timesteps per sample : ", num_samples, splitData.shape[1])
	indices = np.arange(num_samples)
	np.random.shuffle(indices)
	val_split = 0.2
	num_val_samples = int(num_samples * val_split)
	val_indices = indices[:num_val_samples]
	train_indices = indices[num_val_samples:]

	mesh = CartesianMeshGenerator(real_space=[[0, 1.5],[0, 0.93],[0, 1.0]],mesh_size=[combinedData.shape[1], 
																																										combinedData.shape[2],
																																										combinedData.shape[3]],data=splitData)

	return mesh, splitData, scaler

def dataloader(folder, radius_train, batch_size, ntsteps=1):
	num_samples = 100
	indices = np.arange(num_samples)
	np.random.shuffle(indices)
	val_split = 0.2
	num_val_samples = int(num_samples * val_split)
	val_indices = indices[:num_val_samples]
	train_indices = indices[num_val_samples:]

	memb_mesh, memb_scaler = dataloader_memb(folder, radius_train, batch_size, ntsteps)
	flow_mesh, flow_data, flow_scaler = dataloader_flow(folder, radius_train, batch_size, ntsteps)

	edge_index = {
		"memb"	:	memb_mesh.getEdgeAttr(radius_train),
		"flow"	:	flow_mesh.ball_connectivity(radius_train)
	}

	scaler = {
		"memb"	:	memb_scaler,
		"flow"	:	flow_scaler
	}
	
	data_train = []

	for j in range(num_samples):
		edge_attr_memb = memb_mesh.attributes(j)
		edge_attr_flow = flow_mesh.attributes(j)
		data_sample = memb_mesh.getInputOutput(j)

		data_train.append((
			Data(
				x=torch.tensor(data_sample[0], dtype=torch.float32, requires_grad=True),
				y=torch.tensor(data_sample[1], dtype=torch.float32, requires_grad=True),
				edge_index=edge_index['memb'],
				edge_attr=torch.tensor(edge_attr_memb, dtype=torch.float32, requires_grad=True),
				),

			Data(
				x=torch.tensor(flow_data[j,0,:], dtype=torch.float32, requires_grad=True).view(-1,1),
				y=torch.tensor(flow_data[j,1,:], dtype=torch.float32, requires_grad=True).view(-1,1),
				edge_index=edge_index['flow'],
				edge_attr=torch.tensor(edge_attr_flow, dtype=torch.float32, requires_grad=True)
				)))
		
	train_loader = DataLoader([data_train[i] for i in train_indices], batch_size=batch_size, shuffle=True, collate_fn=paired_collate_fn)
	val_loader = DataLoader([data_train[i] for i in val_indices], batch_size=batch_size, shuffle=False, collate_fn=paired_collate_fn)

	return train_loader, val_loader, scaler

def main(checkpoint_path=None):
	set_seed(42)

	# Parameters
	params_network = {
		'memb_net': {
			'inNodeFeatures'				: 6,
			'nNodeFeatEmbedding'		: 16,
			'nEdgeFeatures'					: 12,
			'ker_width'							: 8,
			'nlayers'								: 2
		},
		'flow_net': {
			'inNodeFeatures'				: 1,
			'nNodeFeatEmbedding'		: 8,
			'nEdgeFeatures'					: 8,
			'ker_width'							: 4,
			'nlayers'								: 2
		},
		'attn_dim'								: 16 #found 16 to be a better value compared to 8, 24, 32
	}
	
	params_training = {
		'epochs' 								: 10,
		'learning_rate' 				: 0.001 ,
		'scheduler_step' 				: 500,  
		'scheduler_gamma' 			: 0.5,
		'validation_frequency' 	: 100,
		'save_frequency' 				: 100
	}

	params_data = {
		'location' 			: '../sample_data/',
		'radius_train' 	: 0.02,
		'batch_size' 		: 1,
		'ntsteps' 			: 2,
	}

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Load data
	train_loader, val_loader, scaler = dataloader(params_data['location'],
																							  params_data['radius_train'], 
																								params_data['batch_size'], 
																								params_data['ntsteps'])
	

	print("----Loaded Data----")

	# Initialize model
	# For now only performing next time step prediction
	model_instance = CoupledGNO(params=params_network).to(device)
	print(model_instance)
	optimizer = torch.optim.Adam(model_instance.parameters(), 
															lr=params_training['learning_rate'])

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
																						 step_size=params_training['scheduler_step'], 
																						 gamma=params_training['scheduler_gamma'])
	criterion = torch.nn.MSELoss()

	model_instance = torch.compile(model_instance)

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
		for memb_batch, flow_batch in train_loader:
			optimizer.zero_grad()
			memb_batch = memb_batch.to(device)
			flow_batch = flow_batch.to(device)

			out_memb, out_flow = model_instance(memb_batch, flow_batch)

			loss_memb = criterion(out_memb.view(-1, 1), memb_batch.y.view(-1, 1))
			loss_flow = criterion(out_flow.view(-1, 1), flow_batch.y.view(-1, 1))
			loss = loss_memb + loss_flow 
			loss.backward()

			optimizer.step()
			train_loss += loss.item()
		
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
			all_predictions = []
			all_true_values = []
			with torch.no_grad():
				for batch in val_loader:
					batch = batch.to(device)
					out = model_instance(batch)
					loss = criterion(out.view(-1, 1), batch.y.view(-1, 1))
					val_loss += loss.item()
					all_predictions.append(out.cpu().numpy())
					all_true_values.append(batch.y.cpu().numpy())

			avg_val_loss = val_loss / len(val_loader)
			print(f"Epoch {epoch+1}/{params_training['epochs']}, Validation Loss: {avg_val_loss:.6f}")

			# Calculate MAE and RMSE
	# 		all_predictions = np.concatenate(all_predictions, axis=0)
	# 		all_true_values = np.concatenate(all_true_values, axis=0)

	# 		# Inverse transform predictions and true values
	# 		predictions_original_scale = all_predictions.reshape(-1,1) #scaler.inverse_transform(all_predictions.reshape(-1, 1))
	# 		true_values_original_scale = all_true_values.reshape(-1,1) #scaler.inverse_transform(all_true_values.reshape(-1, 1))

	# 		mae = mean_absolute_error(true_values_original_scale, predictions_original_scale)
	# 		rmse = np.sqrt(mean_squared_error(true_values_original_scale, predictions_original_scale))

	# 		print(f"Validation MAE: {mae:.4f}")
	# 		print(f"Validation RMSE: {rmse:.4f}")

			# Save best model
			if avg_val_loss < best_val_loss:
				best_val_loss = avg_val_loss
				torch.save(model_instance.state_dict(), 'best_model.pth')
				print(f"Best model saved with validation loss: {best_val_loss:.6f}")

	# # Final evaluation
	# model_instance.load_state_dict(torch.load('best_model.pth'))
	# model_instance.eval()
	# all_predictions = []
	# with torch.no_grad():
	# 	for batch in val_loader:
	# 		batch = batch.to(device)
	# 		out = model_instance(batch)
	# 		all_predictions.append(out.cpu().numpy())

	# all_predictions = np.concatenate(all_predictions, axis=0)
	# predictions_original_scale = scaler.inverse_transform(all_predictions.reshape(-1, 1))
	# predictions_final = all_predictions#predictions_original_scale.reshape(-1, 30, 60).transpose(0, 2, 1)

	# np.save("predictions.npy", predictions_final)
	# print("Predictions saved as 'predictions.npy'")

if __name__ == "__main__":
	main()
	# main('model_epoch_1000.pth')