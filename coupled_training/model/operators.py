import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch_scatter.composite import scatter_softmax

class intraMessagePassing(MessagePassing):
	def __init__(self,
								in_channels,
								out_channels,
								nn,
								aggr='mean',
								root_weight=True,
								bias=True,
								**kwargs):
		super(intraMessagePassing, self).__init__(aggr=aggr, **kwargs)

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.nn = nn
		self.aggr = aggr

		if root_weight:
				self.root = Parameter(torch.Tensor(in_channels, out_channels))
		else:
				self.register_parameter('root', None)

		if bias:
				self.bias = Parameter(torch.Tensor(out_channels))
		else:
				self.register_parameter('bias', None)

		self.reset_parameters()

	def reset_parameters(self):
		reset(self.nn)
		size = self.in_channels
		uniform(size, self.root)
		uniform(size, self.bias)

	def forward(self, x, edge_index, edge_attr):
		""""""
		x = x.unsqueeze(-1) if x.dim() == 1 else x
		pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
		return self.propagate(edge_index, x=x, pseudo=pseudo)

	def message(self, x_j, pseudo):
		weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
		return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

	def update(self, aggr_out, x):
		if self.root is not None:
				aggr_out = aggr_out + torch.mm(x, self.root)
		if self.bias is not None:
				aggr_out = aggr_out + self.bias
		return aggr_out
	
class crossAttnMessagePassing(MessagePassing):
	def __init__(self, memb_feat, flow_feat, attn_dim):
		super().__init__(aggr='add')
		self.W_Q = nn.Linear(flow_feat, attn_dim)
		self.W_K = nn.Linear(memb_feat, attn_dim)
		self.W_V = nn.Linear(memb_feat, attn_dim)
		self.out_proj = nn.Linear(attn_dim, flow_feat)
		self.scale = attn_dim ** -0.5

	def forward(self, x_fluid, x_memb, edge_index, edge_attr):
		return self.propagate(edge_index, x_fluid=x_fluid, x_memb=x_memb, edge_attr=edge_attr)

	def message(self, x_fluid_i, x_memb_j, edge_index, edge_attr):
		query = self.W_Q(x_fluid_i) #[nEdges, attn_dim]
		key = self.W_K(x_memb_j) #[nEdges, attn_dim]
		value = self.W_V(x_memb_j) #[nEdges, attn_dim]

		#Q^T K = (query * key).sum(dim=-1) and thn self.scale scales to not let it grow too fast
		score = (query * key).sum(dim=-1) * self.scale #torch.Size([nEdges])

		#torch.softmax summs for all nEdges and reducing information, so scatter only sums neighbours
		alpha = scatter_softmax(score, edge_index[1], dim=0) 
		
		# I still have to use edge_attr 
		return self.out_proj(value * alpha.unsqueeze(-1))

class DenseNet(torch.nn.Module):
	def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
		super(DenseNet, self).__init__()

		self.n_layers = len(layers) - 1

		assert self.n_layers >= 1

		self.layers = nn.ModuleList()

		for j in range(self.n_layers):
			self.layers.append(nn.Linear(layers[j], layers[j+1]))

			if j != self.n_layers - 1:
				if normalize:
						self.layers.append(nn.BatchNorm1d(layers[j+1]))

				self.layers.append(nonlinearity())

		if out_nonlinearity is not None:
			self.layers.append(out_nonlinearity())

	def forward(self, x):
		for _, l in enumerate(self.layers):
			x = l(x)

		return x
	
# class FlashCrossAttention(nn.Module):
#   def __init__(self, node_dim, edge_dim, hidden_dim=32):
#     super().__init__()
#     self.hidden_dim = hidden_dim
		
#     # Project G2 edge features (fluid) to query
#     self.query_proj = nn.Linear(edge_dim, hidden_dim, bias=False)
		
#     # Project G1 node features (structure) to key/value
#     self.key_proj = nn.Linear(node_dim, hidden_dim, bias=False)
#     self.value_proj = nn.Linear(node_dim, hidden_dim, bias=False)
		
#     # Output projection
#     self.out_proj = nn.Linear(hidden_dim, edge_dim, bias=False)

#   def forward(self, edge_feats, node_feats):

#     # Project inputs
#     Q = self.query_proj(edge_feats)  # [E, hidden_dim]
#     K = self.key_proj(node_feats)    # [N, hidden_dim]
#     V = self.value_proj(node_feats)  # [N, hidden_dim]

#     # FlashAttention (memory-efficient)
#     attn_out = scaled_dot_product_attention(Q, K, V)
		
#     # Output projection
#     return self.out_proj(attn_out)  # [E, edge_dim]

#     # return attn_out  # [E, hidden_dim]
