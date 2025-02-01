# Standard library imports
import random
from timeit import default_timer
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utilities import *
from coupled_training.model.coupledGNN import *
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

def ijk_to_index(i, j, k, n_y, n_z):
		return i * (n_y * n_z) + j * n_z + k

def dataloader_memb(folder, radius_train, batch_size,ntsteps=1):
	data = generateDatasetMembrane(ninit=1000, nend=2000, ngap=10, splitLen=ntsteps, folder=folder)
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
	edge_index = mesh.getEdgeAttr(radius_train)

	data_train = []
	for j in range(num_samples):
			grid = mesh.build_grid(j)
			edge_attr = mesh.attributes(j)
			data_sample = mesh.getInputOutput(j)

			data_train.append(Data(
					x=data_sample[0],
					y=data_sample[1],
					edge_index=edge_index,
					edge_attr=torch.tensor(edge_attr, dtype=torch.float32)
			))


	print('train grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

	train_loader = DataLoader([data_train[i] for i in train_indices], batch_size=batch_size, shuffle=True)
	val_loader = DataLoader([data_train[i] for i in val_indices], batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, scaler

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
	grid = mesh.get_grid()
	edge_index = mesh.ball_connectivity(radius_train)

	data_train = []
	for j in range(num_samples):
		edge_attr = mesh.attributes(j)
		
		# print(j, torch.tensor(splitData[j,0,:]).view(-1,1).shape)
		data_train.append(Data(
			x = torch.tensor(splitData[j,0,:], dtype=torch.float32).view(-1,1),
			y = torch.tensor(splitData[j,1,:], dtype=torch.float32).view(-1,1),
			# y=torch.tensor(splitData[j,:,:].transpose(), dtype=torch.float32),
			edge_index = edge_index,
			edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
		))


	print('train grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

	train_loader = DataLoader([data_train[i] for i in train_indices], batch_size=batch_size, shuffle=True)
	val_loader = DataLoader([data_train[i] for i in val_indices], batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, scaler

def main(checkpoint_path=None):
	set_seed(42)

	# Parameters
	params_network = {
		'memb_net': {
			'inNodeFeatures'				: 10,
			'nNodeFeatEmbedding'		: 20,
			'nEdgeFeatures'					: 5,
			'ker_width'							: 3,
			'nlayers'								: 2
		},
		'flow_net': {
			'inNodeFeatures'				: 15,
			'nNodeFeatEmbedding'		: 25,
			'nEdgeFeatures'					: 10,
			'ker_width'							: 4,
			'nlayers'								: 2
		},
		'attn_dim'								: 4
	}
	
	params_training = {
		'epochs' 								: 10000,
		'learning_rate' 				: 0.001 ,
		'scheduler_step' 				: 500,  
		'scheduler_gamma' 			: 0.5,
		'validation_frequency' 	: 100,
		'save_frequency' 				: 100
	}

	params_data = {
		'location' 			: './sample_data/',
		'radius_train' 	: 0.02,
		'batch_size' 		: 6,
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
			optimizer.zero_grad()
			batch = batch.to(device)
			out = model_instance(batch)
			loss = criterion(out.view(-1, 1), batch.y.view(-1, 1))
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
	# main()
	main('model_epoch_1000.pth')