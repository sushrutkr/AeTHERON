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
from coupled_training.preprocessing.utilities import *
from coupled_training.model.neuralFSI import *
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

def dataloader_inference(folder, radius_train, batch_size, ntsteps=1):
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
				y=torch.tensor(data_sample[1][0], dtype=torch.float32, requires_grad=True),
				edge_index=edge_index['memb'],
				edge_attr=torch.tensor(edge_attr_memb, dtype=torch.float32, requires_grad=True),
				),

			Data(
				x=torch.tensor(flow_data[j,0,:], dtype=torch.float32, requires_grad=True).view(-1,1),
				y=torch.tensor(flow_data[j,1,:], dtype=torch.float32, requires_grad=True).view(-1,1),
				edge_index=edge_index['flow'],
				edge_attr=torch.tensor(edge_attr_flow, dtype=torch.float32, requires_grad=True)
				)))
		
	inference_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False, collate_fn=paired_collate_fn)

	return inference_loader, scaler

def inference(checkpoint_path):
	"""
	Perform inference using the trained model.
	"""
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
		'epochs' 								: 2000,
		'learning_rate' 				: 0.001 ,
		'scheduler_step' 				: 500,  
		'scheduler_gamma' 			: 0.5,
		'validation_frequency' 	: 100,
		'save_frequency' 				: 100
	}

	params_data = {
		'location' 			: '../sample_data/',
		'radius_train' 	: 0.01,
		'batch_size' 		: 1,
		'ntsteps' 			: 2,
	}

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print("----Loaded Data----")

	# Initialize model
	# For now only performing next time step prediction
	model_instance = CoupledGNO(params=params_network).to(device)
	model_instance = torch.compile(model_instance)

	checkpoint = torch.load(checkpoint_path)
	#--- If using model_epoch_100.pth and saving all keys and if
	# just storing model_state_dict then remove ["model_state_dict"]
	model_instance.load_state_dict(checkpoint['model_state_dict'])
	

	# dataloader for inference
	inference_loader, scaler = dataloader_inference(params_data['location'],
																									params_data['radius_train'], 
																									params_data['batch_size'], 
																									params_data['ntsteps'])

	flow_predictions = []
	memb_predictions = []

	with torch.no_grad():
		for memb_batch, flow_batch in inference_loader:
			memb_batch = memb_batch.to(device)
			flow_batch = flow_batch.to(device)

			out_memb, out_flow = model_instance(memb_batch, flow_batch)
			
			#--- Processing Membrane Data
			expanded_out_memb = np.expand_dims(out_memb.cpu().numpy(), axis=0)
			memb_predictions.append(expanded_out_memb)

			#--- Processing Flow Data
			out_reshaped = out_flow.cpu().view(out_flow.shape[0], -1).numpy().transpose() #the transpose here is to make first dim as 1 for easy append.
			out_inverse_transformed = scaler["flow"].inverse_transform(out_reshaped)
			flow_predictions.append(out_inverse_transformed)

			#appending another dim to make it compatible with postprocessing
			# memb_predictions = np.expand_dims(memb_predictions, axis=0)
			# flow_predictions = np.expand_dims(flow_predictions, axis=0)


	# Concatenate predictions
	memb_predictions = np.concatenate(memb_predictions, axis=0)
	flow_predictions = np.concatenate(flow_predictions, axis=0)

	print(memb_predictions.shape,flow_predictions.shape)

	# Save predictions to a file
	np.save("./Post_Proc_Data/fluid/inference_predictions.npy", flow_predictions)
	np.save("./Post_Proc_Data/membrane/inference_predictions.npy", memb_predictions)
	

if __name__ == "__main__":
	checkpoint_path = 'model_epoch_100.pth'  # Path to the saved model checkpoint
	inference(checkpoint_path)