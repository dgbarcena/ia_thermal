# models/nnconv_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class NNConvNet(nn.Module):
    def __init__(self, input_dim, edge_dim, hidden_dim, output_dim, num_layers,
                 use_dropout=False, dropout_rate=0.1,
                 use_batchnorm=False, use_residual=False):
        super(NNConvNet, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.node_input_dim = input_dim
        self.edge_attr_dim = edge_dim

        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.bns = nn.ModuleList()

        in_dim = input_dim
        for i in range(num_layers):
            nn_edge = nn.Sequential(
                nn.Linear(edge_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, in_dim * hidden_dim)
            )
            conv = NNConv(in_dim, hidden_dim, nn_edge, aggr='mean')
            self.convs.append(conv)

            if use_batchnorm:
                self.bns.append(nn.BatchNorm1d(hidden_dim))

            in_dim = hidden_dim

        # MLP final, probar también sin la mlp final
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_layers):
            residual = x
            x = self.convs[i](x, edge_index, edge_attr)
            if self.use_batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            if self.use_residual:
                x = x + residual if residual.shape == x.shape else x
            if self.use_dropout:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)

        out = self.mlp(x)
        return out
