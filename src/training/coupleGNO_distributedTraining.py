# Standard library imports
import os
import random
from timeit import default_timer
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utilities import *
from model.spectralGraphNetwork import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


from torch.distributed.fsdp import (
		FullyShardedDataParallel as FSDP,
		MixedPrecision,
		BackwardPrefetch,
		ShardingStrategy,
		FullStateDictConfig,
		StateDictType,
)
from torch.distributed.fsdp.wrap import (
		transformer_auto_wrap_policy,
		enable_wrap,
		wrap,
)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def dataloader(folder, radius_train, batch_size, ntsteps=1):
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

	num_samples = splitData.shape[0]
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
			edge_index = edge_index,
			edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
		))


	train_loader = DataLoader([data_train[i] for i in train_indices], batch_size=batch_size, shuffle=True)
	val_loader = DataLoader([data_train[i] for i in val_indices], batch_size=batch_size, shuffle=False)

	return train_loader, val_loader, scaler

def train_fsdp(world_size: int,  rank: int, local_rank: int, checkpoint_path=None):
	
	dist.init_process_group('nccl', world_size=world_size, rank=rank)
	set_seed(42)

	# Parameters
	radius_train = 0.1
	batch_size = 6
	width = 8  # uplifting node_features+time_emb_dim to wwidth
	ker_width = 4
	edge_features = 8
	node_features = 1
	nLayers = 2
	epochs = 5000
	learning_rate = 0.001
	scheduler_step = 500
	scheduler_gamma = 0.5
	ntsteps = 2
	time_emb_dim = 4

	validation_frequency = 100
	save_frequency = 100

	# Load data
	if rank == 0:
		train_loader, val_loader, scaler = dataloader('../sample_data/', radius_train, batch_size, ntsteps)
		print("----Loaded Data----")

	dist.barrier()

	# # Initialize model
	# # Here I am just making next time step prediction as of now so setting ntsteps = 1 and sending ntsteps-1 to model initialisation
	# auto_wrap_policy = functools.partial(
	# 				size_based_auto_wrap_policy, min_num_params=100
	# 		)

	# model_instance = SpecGNO(inNodeFeatures=node_features, nNodeFeatEmbedding=width, ker_width=ker_width, nConvolutions=nLayers, nEdgeFeatures=edge_features, ntsteps=ntsteps-1,time_emb_dim=time_emb_dim).to(rank)
	# model = FSDP(
	# 				model,
	# 				auto_wrap_policy=auto_wrap_policy,
	# 				mixed_precision=MixedPrecision(
	# 						param_dtype=torch.bfloat16,
	# 						reduce_dtype=torch.bfloat16,
	# 						buffer_dtype=torch.bfloat16,
	# 				),
	# 				device_id=rank,
	# 				sharding_strategy=ShardingStrategy.FULL_SHARD
	# 		)

	# print(rank)
	# print(model_instance)

	# optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
	# criterion = torch.nn.MSELoss()

	# # Initialize training
	# start_epoch = 0
	# if checkpoint_path:
	# 	checkpoint = torch.load(checkpoint_path)
	# 	model_instance.load_state_dict(checkpoint['model_state_dict'])
	# 	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	# 	scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
	# 	start_epoch = checkpoint['epoch'] + 1
	# 	best_val_loss = checkpoint['val_loss']
	# 	print(f"Resuming training from epoch {start_epoch}")
	# else:
	# 	best_val_loss = float('inf')

	# #training
	# for epoch in range(start_epoch, epochs):
	# 	model_instance.train()
	# 	train_loss = 0.0
	# 	for batch in train_loader:
	# 		optimizer.zero_grad()
	# 		batch = batch.to(device)
	# 		out = model_instance(batch)
	# 		loss = criterion(out.view(-1, 1), batch.y.view(-1, 1))
	# 		loss.backward()
	# 		optimizer.step()
	# 		train_loss += loss.item()

	# 	dist.barrier()
	# 	if rank == 0:
	# 		avg_train_loss = train_loss / len(train_loader)
	# 		print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

	# 	# Save model
	# 	if (epoch + 1) % save_frequency == 0:
	# 		torch.save({
	# 			'epoch': epoch,
	# 			'model_state_dict': model_instance.state_dict(),
	# 			'optimizer_state_dict': optimizer.state_dict(),
	# 			'scheduler_state_dict': scheduler.state_dict(),
	# 			'train_loss': avg_train_loss,
	# 			'val_loss': best_val_loss
	# 		}, f'model_epoch_{epoch+1}.pth')
	# 		print(f"Model saved at epoch {epoch+1}")

	# 	# Validation
	# 	if (epoch + 1) % validation_frequency == 0:
	# 		if rank == 0:
	# 			model_instance.eval()
	# 			val_loss = 0.0
	# 			all_predictions = []
	# 			all_true_values = []
	# 			with torch.no_grad():
	# 				for batch in val_loader:
	# 					batch = batch.to(device)
	# 					out = model_instance(batch)
	# 					loss = criterion(out.view(-1, 1), batch.y.view(-1, 1))
	# 					val_loss += loss.item()
	# 					all_predictions.append(out.cpu().numpy())
	# 					all_true_values.append(batch.y.cpu().numpy())

	# 			avg_val_loss = val_loss / len(val_loader)
	# 			print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.6f}")

	# 		# Save best model
	# 		if avg_val_loss < best_val_loss:
	# 			best_val_loss = avg_val_loss
	# 			torch.save(model_instance.state_dict(), 'best_model.pth')
	# 			print(f"Best model saved with validation loss: {best_val_loss:.6f}")

	dist.destroy_process_group()

if __name__ == "__main__":
	checkpoint = 'model_epoch_1000.pth'
	world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
	rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
	local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
	print("GPUs available to use : ", world_size)
	train_fsdp(world_size, rank, local_rank, checkpoint)
	# main('model_epoch_1000.pth')