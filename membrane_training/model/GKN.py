import torch
from torch_geometric.data import Data
import torch.nn as nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform
import torch.nn.functional as F

class NNConv(MessagePassing):
  def __init__(self,
                in_channels,
                out_channels,
                nn,
                aggr='add',
                root_weight=True,
                bias=True,
                **kwargs):
    super(NNConv, self).__init__(aggr=aggr, **kwargs)

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

  def __repr__(self):
    return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                self.out_channels)

class MembraneGraphNet(MessagePassing):
  def __init__(self,
                in_channels,
                out_channels,
                phi,
                gamma,
                beta,
                aggr='add',
                root_weight=True,
                bias=True,
                **kwargs):
    super(MembraneGraphNet, self).__init__(aggr=aggr, **kwargs)

    self.in_channels = in_channels
    self.out_channels = out_channels

    self.phi = phi
    self.gamma = gamma
    self.beta = beta

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
    reset(self.phi)
    reset(self.gamma)
    # reset(self.beta)
    size = self.in_channels
    uniform(size, self.root)
    uniform(size, self.bias)

  def forward(self, x, edge_index, edge_attr):
    return self.propagate(edge_index, x=x, edge_attr=edge_attr)

  def message(self, x_i, x_j, edge_attr):
    return self.phi(torch.cat([x_i, x_j, edge_attr]),axis=-1)

  def update(self, aggr_out, x):
    return self.gamma(torch.cat([x, aggr_out], dim=-1))


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
