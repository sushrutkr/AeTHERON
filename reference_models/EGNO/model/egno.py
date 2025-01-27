from model.basic import EGNN
from model.layer_no import TimeConv, get_timestep_embedding, TimeConv_x
import torch.nn as nn
import torch


class EGNO(EGNN):
    def __init__(self, n_layers, in_node_nf, in_edge_nf, hidden_nf, activation=nn.SiLU(), device='cpu', with_v=False,
                 flat=False, norm=False, use_time_conv=True, num_modes=2, num_timesteps=8, time_emb_dim=32):
        self.time_emb_dim = time_emb_dim
        in_node_nf = in_node_nf + self.time_emb_dim

        super(EGNO, self).__init__(n_layers, in_node_nf, in_edge_nf, hidden_nf, activation, device, with_v, flat, norm)
        self.use_time_conv = use_time_conv
        self.num_timesteps = num_timesteps
        self.device = device
        self.hidden_nf = hidden_nf
        
        print(hidden_nf,in_node_nf)

        if use_time_conv:
            self.time_conv_modules = nn.ModuleList() #nn.ModuleList is initialization to create list of layers, look down
            self.time_conv_x_modules = nn.ModuleList()
            for i in range(n_layers):
                self.time_conv_modules.append(TimeConv(hidden_nf, hidden_nf, num_modes, activation, with_nin=False))
                self.time_conv_x_modules.append(TimeConv_x(2, 2, num_modes, activation, with_nin=False))

        self.to(self.device)

    def forward(self, x, h, edge_index, edge_fea, v=None, loc_mean=None):  # [BN, H]
        #batches - 3
        #ntsteps - 4

        #x.shape: torch.Size([5271, 3]), h.shape: torch.Size([5271, 2]), edge_index.shape: torch.Size([2, 90114]), edge_fea.shape: torch.Size([90114, 2]) v.shape: torch.Size([5271, 3]) loc_mean.shape: torch.Size([5271, 3])
        #1757 X 3 = 5271, 1757 Nodes and 3 batches
        T = self.num_timesteps

        num_nodes = h.shape[0]
        num_edges = edge_index[0].shape[0]

        cumsum = torch.arange(0, T).to(self.device) * num_nodes
        #print(cumsum) #tensor([    0,  5271, 10542, 15813], device='cuda:0')
        cumsum_nodes = cumsum.repeat_interleave(num_nodes, dim=0)
        cumsum_edges = cumsum.repeat_interleave(num_edges, dim=0)
        #print(cumsum_edges) #tensor([    0,     0,     0,  ..., 15813, 15813, 15813], device='cuda:0')
        # print(cumsum_edges.shape) #torch.Size([360456]) <- 90114 repeated 4 times

        #H_T is probablu embedding dimension
        #H is size of feature vector of each node
        #BN - Maybe BatchNodes
        time_emb = get_timestep_embedding(torch.arange(T).to(x), embedding_dim=self.time_emb_dim, max_positions=10000)  # [T, H_t]
        # print("shape of time embeddings : ",time_emb.shape) #shape of time embeddings :  torch.Size([4, 8])
        h = h.unsqueeze(0).repeat(T, 1, 1)  # [T, BN, H]
        # print(h.shape) torch.Size([4, 5271, 2]) # 2 correspond to velocity magnitude and scaled z coordinate
        time_emb = time_emb.unsqueeze(1).repeat(1, num_nodes, 1)  # [T, BN, H_t], H_t is time-embedding dimension
        # print(time_emb.shape) #torch.Size([4, 5271, 8])
        h = torch.cat((h, time_emb), dim=-1)  # [T, BN, H+H_t] -> [4, 5271, 2+8=10]
        h = h.view(-1, h.shape[-1])  # [T*BN, H+H_t]

        h = self.embedding(h) #applies a NN to node features and uplift it. in GNN class
        # print(h.shape) #torch.Size([21084, 128])
        x = x.repeat(T, 1)
        loc_mean = loc_mean.repeat(T, 1)
        #print(cumsum_edges) #tensor([    0,     0,     0,  ..., 15813, 15813, 15813], device='cuda:0')

        # Batching and index shifting for it.
        edges_0 = edge_index[0].repeat(T) + cumsum_edges
        edges_1 = edge_index[1].repeat(T) + cumsum_edges 
        """
        The dataloader from torch geometric does all of it itself.
        Repeating edge_index: edge_index[0].repeat(T) and edge_index[1].repeat(T) repeat the source and target nodes T times to create a batched representation for T time steps.
        Adding cumsum_edges: cumsum_edges is used to offset the node indices for each time step. This ensures that the node indices are unique across different time steps.
        Example for usage to accomodate batches after 1st batch edge indices are translated by number of nodes
        edges_0 = tensor([    0,     1,     2,  5271,  5272,  5273, 10542, 10543, 10544, 15813, 15814, 15815], device='cuda:0')
        edges_1 = tensor([    1,     2,     0,  5272,  5273,  5271, 10543, 10544, 10542, 15814, 15815, 15813], device='cuda:0')
        """

        edge_index = [edges_0, edges_1]
        v = v.repeat(T, 1)

        edge_fea = edge_fea.repeat(T, 1)

        for i in range(self.n_layers):
            if self.use_time_conv:
                time_conv = self.time_conv_modules[i]
                h_test = h.view(T, num_nodes, self.hidden_nf)
                # print(h_test.shape) #torch.Size([4, 5271, 128]) 1757X3batches = 5271
                h = time_conv(h.view(T, num_nodes, self.hidden_nf)).view(T * num_nodes, self.hidden_nf)
                #print(h.shape) #torch.Size([21084, 128])
                """
                h is reshaped to [T, num_nodes, self.hidden_nf] and passed through the time_conv module.
                The output is reshaped back to [T * num_nodes, self.hidden_nf].
                Here the FNO is learning how the time-evolution is happening, remember in forwards pass
                there is no information of subsequent timesteps as we are just making the copy of 1st timesteps n times
                so the parameters of network are changing to predict actual node feature embedding of subsequent timesteps
                """
                x_translated = x - loc_mean
                time_conv_x = self.time_conv_x_modules[i]
                X = torch.stack((x_translated, v), dim=-1)
                temp = time_conv_x(X.view(T, num_nodes, 3, 2))
                # print(temp.shape) #torch.Size([4, 5271, 3, 2])
                """
                x_translated is the translated coordinates of the nodes.
                X is created by stacking x_translated and v (velocities) along a new dimension, resulting in a shape of [T * num_nodes, 3, 2].
                X is reshaped to [T, num_nodes, 3, 2] and passed through the time_conv_x module.
                The output temp is split into two parts: the updated coordinates x and velocities v.
                """
                x = temp[..., 0].view(T * num_nodes, 3) + loc_mean
                v = temp[..., 1].view(T * num_nodes, 3)

            x, v, h = self.layers[i](x, h, edge_index, edge_fea, v=v)
            # print(x.shape, v.shape, h.shape) #torch.Size([21084, 3]) torch.Size([21084, 3]) torch.Size([21084, 128])
        return (x, v, h) if v is not None else (x, h)

"""
The forward function in the provided code is part of a neural network model, likely a graph neural network (GNN) given the use of edge_index and edge_fea. Here's a step-by-step mathematical explanation of what the function does:

Initialization:

T is the number of timesteps.
num_nodes is the number of nodes in the graph.
num_edges is the number of edges in the graph.
Cumulative Sums:

cumsum is a tensor of shape [T] where each element is the product of the timestep index and num_nodes.
cumsum_nodes and cumsum_edges are tensors where each element is repeated num_nodes and num_edges times, respectively.

Time Embedding:

time_emb is a tensor of shape [T, H_t] where each row is the embedding of the corresponding timestep.
h is expanded to shape [T, BN, H] by repeating it T times.
time_emb is expanded to shape [T, BN, H_t] by repeating it num_nodes times.
h and time_emb are concatenated along the last dimension to form a tensor of shape [T, BN, H + H_t].
h is reshaped to [T * BN, H + H_t].

Embedding:

h is passed through an embedding layer.
x and loc_mean are repeated T times.
edges_0 and edges_1 are adjusted by adding cumsum_edges to account for the repeated edges across timesteps.
edge_index is updated to reflect the new edge indices.
v and edge_fea are repeated T times.

Time Convolution and Layer Processing:

For each layer in the network:
If use_time_conv is True, apply time convolution to h and reshape it.
Compute x_translated as the difference between x and loc_mean.
Stack x_translated and v along a new dimension.
Apply time convolution to the stacked tensor and update x and v.
Pass x, h, edge_index, edge_fea, and v through the current layer.
Output:

Return (x, v, h) if v is not None, otherwise return (x, h).

"""