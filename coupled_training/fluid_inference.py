# Standard library imports
import numpy as np
import torch
from torch_geometric.data import DataLoader
from utilities import *
from coupled_training.model.coupledGNN import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dataloader_inference(folder, radius_train, batch_size, ntsteps=1):
    """
    Prepare data for inference.
    """
    data = generateDatasetFluid(folder, splitLen=ntsteps)

    # Scale data using MinMaxScaler (same as in training)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler, vorticity = data.scaling(scaler)
    splitData = data.splitDataset()
    combinedData = data.combined_data()

    num_samples = splitData.shape[0]
    print("Num of samples batches, timesteps per sample : ", num_samples, splitData.shape[1])

    mesh = CartesianMeshGenerator(
        real_space=[[0, 1.5], [0, 0.93], [0, 1.0]],
        mesh_size=[combinedData.shape[1], combinedData.shape[2], combinedData.shape[3]],
        data=splitData
    )
    grid = mesh.get_grid()
    edge_index = mesh.ball_connectivity(radius_train)

    data_inference = []
    for j in range(num_samples):
        edge_attr = mesh.attributes(j)
        data_inference.append(Data(
            x=torch.tensor(splitData[j, 0, :], dtype=torch.float32).view(-1, 1),
            y=torch.tensor(splitData[j, 1, :], dtype=torch.float32).view(-1, 1),  # Optional: Ground truth for evaluation
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32)
        ))

    inference_loader = DataLoader(data_inference, batch_size=batch_size, shuffle=False)
    return inference_loader, scaler

def inference(checkpoint_path, folder='./sample_data/', radius_train=0.02, batch_size=1, ntsteps=2):
    """
    Perform inference using the trained model.
    """
    # Model parameters (must match training)
    node_features = 1
    width = 8
    ker_width = 4
    edge_features = 8
    nLayers = 2
    time_emb_dim = 4

    # Initialize the model
    model_instance = SpecGNO(
        inNodeFeatures=node_features,
        nNodeFeatEmbedding=width,
        ker_width=ker_width,
        nConvolutions=nLayers,
        nEdgeFeatures=edge_features,
        ntsteps=ntsteps - 1,
        time_emb_dim=time_emb_dim
    ).to(device)

    # Load the trained model checkpoint
    model_instance.load_state_dict(torch.load(checkpoint_path))
    model_instance.eval()

    # Prepare the dataloader for inference
    inference_loader, scaler = dataloader_inference(folder, radius_train, batch_size, ntsteps)

    all_predictions = []
    all_true_values = []  # Optional: For evaluation purposes if ground truth is available

    with torch.no_grad():
        for batch in inference_loader:
            batch = batch.to(device)
            out = model_instance(batch)  # Perform forward pass
            out_reshaped = out.cpu().view(out.shape[0], -1).numpy()
            out_inverse_transformed = scaler.inverse_transform(out_reshaped)
            out = out_inverse_transformed.reshape(out.shape)
            all_predictions.append(out)

    # Concatenate predictions and true values (if needed)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # Save predictions to a file
    np.save("./Post_Proc_Data/fluid/inference_predictions.npy", all_predictions)
    

if __name__ == "__main__":
    checkpoint_path = 'fluid_checkpoint.pth'  # Path to the saved model checkpoint
    inference(checkpoint_path)