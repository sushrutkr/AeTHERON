# Standard library imports
import numpy as np
import torch
from torch_geometric.data import DataLoader
from utilities import *
from model.spectralGraphNetwork import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dataloader_inference(folder, radius_train, batch_size, ntsteps=1):
    # Load dataset for inference
    data = generateDatasetMembrane(ninit=3000, nend=4000, ngap=10, splitLen=ntsteps, folder=folder)
    nodes, vel, elem = data.get_output_split()

    nodes[:, 0] -= 20
    nodes[:, 1] -= 20
    nodes[:, 2] -= 20

    mesh = unstructMeshGenerator(nodes=nodes, vel=vel, elem=elem)
    edge_index = mesh.getEdgeAttr(radius_train)

    data_inference = []
    num_samples = nodes.shape[0]
    for j in range(num_samples):
        grid = mesh.build_grid(j)
        edge_attr = mesh.attributes(j)
        data_sample = mesh.getInputOutput(j)

        data_inference.append(Data(
            x=data_sample[0],
            y=data_sample[1],  # Ground truth (optional for evaluation purposes)
            edge_index=edge_index,
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32)
        ))

    inference_loader = DataLoader(data_inference, batch_size=batch_size, shuffle=False)
    return inference_loader

def inference(checkpoint_path, folder='./sample_data/1e6/', radius_train=0.03, batch_size=5, ntsteps=3):
    # Load the trained model
    node_features = 6
    width = 64
    ker_width = 8
    edge_features = 12
    nLayers = 2
    time_emb_dim = 32

    model_instance = SpecGNO(
        inNodeFeatures=node_features,
        nNodeFeatEmbedding=width,
        ker_width=ker_width,
        nConvolutions=nLayers,
        nEdgeFeatures=edge_features,
        ntsteps=ntsteps,
        time_emb_dim=time_emb_dim
    ).to(device)

    # Load the checkpoint
    model_instance.load_state_dict(torch.load(checkpoint_path))
    model_instance.eval()

    # Prepare the dataloader for inference
    inference_loader = dataloader_inference(folder, radius_train, batch_size, ntsteps)

    all_predictions = []
    all_true_values = []  # Optional: For evaluation purposes if ground truth is available

    with torch.no_grad():
        for batch in inference_loader:
            batch = batch.to(device)
            out = model_instance(batch)  # Perform forward pass
            all_predictions.append(out.cpu().numpy())
            all_true_values.append(batch.y.cpu().numpy())  # Optional

    # Concatenate predictions and true values (if needed)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # Optional: Save ground truth values for comparison (if available)
    all_true_values = np.concatenate(all_true_values, axis=0) if all_true_values else None

    # Save predictions to a file
    np.save("./Post_Proc_Data/membrane/inference_predictions.npy", all_predictions)
    
    if all_true_values is not None:
        np.save("ground_truth.npy", all_true_values)

    print("Inference completed. Predictions saved as 'inference_predictions.npy'.")
    if all_true_values is not None:
        print("Ground truth saved as 'ground_truth.npy'.")

if __name__ == "__main__":
    checkpoint_path = 'membrane_checkpoint.pth'  # Path to the saved model checkpoint
    inference(checkpoint_path)