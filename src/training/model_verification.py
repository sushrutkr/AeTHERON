import torch
from torch_geometric.data import Data, DataLoader
from model.neuralFSI import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def create_simple_graphs():
#     # Graph 1: 5 nodes in a line
#     x1 = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float)  # 5 nodes with 1 feature each
#     edge_index1 = torch.tensor([[0, 1, 2, 3],
#                                 [1, 2, 3, 4]], dtype=torch.long)  # Edges connecting nodes in a line
#     edge_attr1 = torch.tensor([[10], [11], [12], [13]], dtype=torch.float)  # 4 edges with 1 feature each

#     # Graph 2: 6 nodes in a line
#     x2 = torch.tensor([[31], [32], [33], [34], [35]], dtype=torch.float)  # 6 nodes with 1 feature each
#     edge_index2 = torch.tensor([[0, 1, 2, 3],
#                                 [1, 2, 3, 4]], dtype=torch.long)  # Edges connecting nodes in a line
#     edge_attr2 = torch.tensor([[27], [28], [29], [30]], dtype=torch.float)  # 5 edges with 1 feature each

#     data_list = [
#         Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1),
#         Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2)
#     ]

#     return data_list

# # Create simple graphs and data loader
# data_list = create_simple_graphs()
# data_loader = DataLoader(data_list, batch_size=2)

# # Print the data to verify
# for data in data_loader:
#     print(data.x)
#     print(data.edge_index)
#     print(data.edge_attr)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Parameters
# radius_train = 0.03
# batch_size = 2
# width = 9  # uplifting node_features+time_emb_dim to wwidth
# ker_width = 32  
# depth = 2
# edge_features = 1
# node_features = 1
# nLayers = 2
# epochs = 1
# learning_rate = 0.001 
# scheduler_step = 500  
# scheduler_gamma = 0.5
# validation_frequency = 10
# ntsteps = 3
# time_emb_dim = 6

# model_instance = SpecGNO(inNodeFeatures=node_features, nNodeFeatEmbedding=width, ker_width=ker_width, nConvolutions=nLayers, nEdgeFeatures=edge_features, ntsteps=ntsteps,time_emb_dim=time_emb_dim).to(device)

# print(model_instance)

# optimizer = torch.optim.Adam(model_instance.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

# for epoch in range(0, epochs):
#     model_instance.train()
#     train_loss = 0.0
#     for batch in data_loader:
#         optimizer.zero_grad()
#         batch = batch.to(device)
#         out = model_instance(batch)

# from torch_geometric.data import HeteroData, Batch
# from torch_geometric.loader import DataLoader

# class HeteroBatch(HeteroData):
#     def __inc__(self, key, value, store):
#         if 'edge_index' in key:
#             # Get node type for src and dst
#             src_type, _, dst_type = key[0], key[1], key[2]
            
#             # Return increments for source and destination nodes
#             return torch.tensor([
#                 self[src_type].num_nodes, 
#                 self[dst_type].num_nodes
#             ])
#         return super().__inc__(key, value, store)

# # Create list of HeteroData graphs
# test_data = [
#     HeteroBatch(  # Graph 1
#         memb={'x': torch.randn(5, 8), 'y': torch.randn(5)},
#         flow={'x': torch.randn(3, 4)},
#         memb__to__flow={'edge_index': torch.tensor([[0,1],[0,0]], dtype=torch.long)}
#     ),
#     HeteroBatch(  # Graph 2
#         memb={'x': torch.randn(3, 8), 'y': torch.randn(3)},
#         flow={'x': torch.randn(2, 4)},
#         memb__to__flow={'edge_index': torch.tensor([[0],[0]], dtype=torch.long)}
#     )
# ]

# loader = DataLoader(test_data, batch_size=2, collate_fn=lambda b: Batch.from_data_list(b, cls=HeteroBatch))
# batch = next(iter(loader))

# print(batch['memb', 'to', 'flow'].edge_index)


d = 4

tau = torch.tensor([0.02]).to(device)
nBatches = 1

time_embeddings =  get_timestep_embedding(tau, d, structure="ordered")
print(time_embeddings.shape)

nMembNodes = 3
nFlowNodes = 5

time_repeat = time_embeddings.repeat_interleave(nMembNodes, dim=0)
print(time_repeat)


