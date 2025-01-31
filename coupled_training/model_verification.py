import torch
from torch_geometric.data import Data, DataLoader
from coupled_training.model.coupledGNN import *

def create_simple_graphs():
    # Graph 1: 5 nodes in a line
    x1 = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)  # 5 nodes with 1 feature each
    edge_index1 = torch.tensor([[0, 1, 2, 3],
                                [1, 2, 3, 4]], dtype=torch.long)  # Edges connecting nodes in a line
    edge_attr1 = torch.tensor([[10], [11], [12], [13]], dtype=torch.float)  # 4 edges with 1 feature each

    # Graph 2: 6 nodes in a line
    x2 = torch.tensor([[31], [32], [33], [34], [35]], dtype=torch.float)  # 6 nodes with 1 feature each
    edge_index2 = torch.tensor([[0, 1, 2, 3],
                                [1, 2, 3, 4]], dtype=torch.long)  # Edges connecting nodes in a line
    edge_attr2 = torch.tensor([[27], [28], [29], [30]], dtype=torch.float)  # 5 edges with 1 feature each

    data_list = [
        Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1),
        Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)
    ]

    return data_list

# Create simple graphs and data loader
data_list = create_simple_graphs()
data_loader = DataLoader(data_list, batch_size=2)

# Print the data to verify
for data in data_loader:
    print(data.x)
    print(data.edge_index)
    print(data.edge_attr)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
radius_train = 0.03
batch_size = 2
width = 9  # uplifting node_features+time_emb_dim to wwidth
ker_width = 32  
depth = 2
edge_features = 1
node_features = 1
nLayers = 2
epochs = 1
learning_rate = 0.001 
scheduler_step = 500  
scheduler_gamma = 0.5
validation_frequency = 10
ntsteps = 3
time_emb_dim = 6

model_instance = SpecGNO(inNodeFeatures=node_features, nNodeFeatEmbedding=width, ker_width=ker_width, nConvolutions=nLayers, nEdgeFeatures=edge_features, ntsteps=ntsteps,time_emb_dim=time_emb_dim).to(device)

print(model_instance)

optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

for epoch in range(0, epochs):
    model_instance.train()
    train_loss = 0.0
    for batch in data_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        out = model_instance(batch)