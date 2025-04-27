import os
import sys
import random
from timeit import default_timer
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from model.neuralFSI import *
from dataloader.dataload import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from torch.distributed.fsdp import (
# 		FullyShardedDataParallel as FSDP,
# 		MixedPrecision,
# 		BackwardPrefetch,
# 		ShardingStrategy,
# 		FullStateDictConfig,
# 		StateDictType,
# )
# from torch.distributed.fsdp.wrap import (
# 		transformer_auto_wrap_policy,
# 		enable_wrap,
# 		wrap,
# )


def reduce_tensor(tensor, world_size):
	rt = tensor.clone()
	dist.all_reduce(rt, op=dist.ReduceOp.SUM)
	rt /= world_size
	return rt

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

def main(world_size: int,  rank: int, local_rank: int, checkpoint_path=None):
	
	torch.cuda.set_device(local_rank)
	dist.init_process_group('nccl', world_size=world_size, rank=rank)
	set_seed(42)

	# # Parameters
	# params_network = {
	# 	'memb_net': {
	# 		'inNodeFeatures'				: 10, #Current keeping same as out dimension because we want to obtain latent dimension of output
	# 		'nNodeFeatEmbedding'		: 24,
	# 		'outNodeFeatures'				: 10,
	# 		'nEdgeFeatures'					: 7,
	# 		'ker_width'							: 8
	# 	},
	# 	'flow_net': {
	# 		'inNodeFeatures'				: 4,
	# 		'nNodeFeatEmbedding'		: 24,
	# 		'outNodeFeatures'				: 4,
	# 		'nEdgeFeatures'					: 14,
	# 		'ker_width'							: 4
	# 	},
	# 	'attn_dim'								: 24, #found 16 to be a better value compared to 8, 24, 32,
	# 	'nlayers'									: 4,
	# 	'time_embedding_dim'			: 8
	# }
	
	# params_training = {
	# 	'epochs' 								: 2,
	# 	'learning_rate' 				: 0.001 ,
	# 	'scheduler_step' 				: 500,  
	# 	'scheduler_gamma' 			: 0.5,
	# 	'validation_frequency' 	: 100,
	# 	'save_frequency' 				: 100,
	# }

	# params_data = {
	# 	'batch_size' 		: 3,
	# 	'ntsteps' 			: 1, #extra from current one
	# 	'val_split'			: 0.3
	# }

	# train_radius = {
	# 	'radius_flow' 	: 0.08,		 # keeping it >=
	# 	'radius_memb'		: 0.04,		 #makes sense to keep <= 2X\Delta_memb
	# 	'radius_cross'  : 0.04     #makes sense to keep <= 2X\Delta_memb
	# }
	with open("./input/config.yaml", "r") as f:
			config = json.load(f)

	params_network = config["params_network"]
	params_training = config["params_training"]
	params_data = config["params_data"]
	train_radius = config["train_radius"]

	print("Network Parameters:", params_network)
	print("Training Parameters:", params_training)
	print("Data Parameters:", params_data)
	print("Train Radius:", train_radius)

	# Load data
	train_loader, val_loader = dataGenerate(train_radius, 
																					params_data['batch_size'],
																					params_data['ntsteps'],
																					params_data['val_split'],
																					world_size,
																					rank,
																					loadData = True,
																					cache_file="/home/skumar94/scr16_rmittal3/skumar94/GNO_FSI/TrainingData/data_train_cache.pt")
	train_sampler = train_loader.sampler
	print(f"[rank {rank}] data loaded")

	dist.barrier()

	device = torch.device(f'cuda:{local_rank}')

	model_instance = neuralFSI(params=params_network).to(device)
	model_instance = DDP(model_instance, device_ids=[local_rank], find_unused_parameters=True)

	optimizer = torch.optim.AdamW(model_instance.parameters(),
															  lr=params_training['learning_rate'], 
																weight_decay=1e-4)
	# optimizer = torch.optim.Adam(model_instance.parameters(), 
	# 														lr=params_training['learning_rate'])

	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
	# 																					 step_size=params_training['scheduler_step'], 
	# 																					 gamma=params_training['scheduler_gamma'])

	scheduler = torch.optim.lr_scheduler.OneCycleLR(
			optimizer,
			max_lr=0.004,
			epochs=params_training['epochs'],
			steps_per_epoch=len(train_loader),
			pct_start=0.5,
			anneal_strategy="cos",
	)

	# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
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
		train_sampler.set_epoch(epoch)
		model_instance.train()
		train_loss = 0.0
		flow_loss_batch = 0.0
		memb_loss_batch = 0.0 

		seen_idxs = []

		for batch in train_loader:
			optimizer.zero_grad(set_to_none=True)
			batch = batch.to(device)
			# seen_idxs.extend(batch.idx.squeeze().tolist())

			with torch.autocast(device_type='cuda', dtype=torch.float16): 
				out_flow, out_memb = model_instance(batch)

				loss_memb = criterion(out_memb.view(-1, 1), batch['memb'].y.view(-1, 1))
				loss_flow = criterion(out_flow.view(-1, 1), batch['flow'].y.view(-1, 1))

			loss = params_training['flow_weight']*loss_flow + params_training['memb_weight']*loss_memb
				
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model_instance.parameters(), 1.0)

			optimizer.step()
			scheduler.step()

			train_loss += loss.item()
			flow_loss_batch += loss_flow.item()
			memb_loss_batch += loss_memb.item()

			torch.cuda.empty_cache()
			del out_memb, out_flow, loss_flow, loss_memb
		
		avg_train_loss = train_loss / len(train_loader)
		avg_flow_loss = flow_loss_batch / len(train_loader)
		avg_memb_loss = memb_loss_batch / len(train_loader)

		reduced_loss = loss
		reduced_loss_flow = avg_flow_loss
		reduced_loss_memb = avg_memb_loss

		# print(f"[Rank {rank}] saw indices: {seen_idxs}")
		
		if rank == 0:
			print(
					f"Epoch {epoch+1}/{params_training['epochs']}, Train Loss: {reduced_loss:.6f}, "
					f"Flow loss: {reduced_loss_flow:.6f}, "
					f"Memb loss: {reduced_loss_memb:.6f}, "
					f"lr: {optimizer.param_groups[0]['lr']:.6f}"
				)

			# Save model 
			if (epoch + 1) % params_training['save_frequency'] == 0:
				torch.save({
					'epoch': epoch,
					'model_state_dict': model_instance.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'scheduler_state_dict': scheduler.state_dict(),
					'train_loss': avg_train_loss,
					'val_loss': best_val_loss
				}, f'./utils/model_epoch_{epoch+1}.pth')
				print(f"Model saved at epoch {epoch+1}")

		# Validation
		if (epoch + 1) % params_training['validation_frequency'] == 0 and rank == 0:
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
			if rank == 0:
				if avg_val_loss < best_val_loss:
					best_val_loss = avg_val_loss
					torch.save(model_instance.state_dict(), './utils/best_model.pth')
					print(f"Best model saved with validation loss: {best_val_loss:.6f}")

		# scheduler.step()

	dist.barrier()
	dist.destroy_process_group()

if __name__ == "__main__":
	checkpoint = None #'model_epoch_1000.pth'
	world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
	rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
	local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))
	if rank == 0:
		print("GPUs available to use : ", world_size)
	main(world_size, rank, local_rank, checkpoint)
	# main('model_epoch_1000.pth')