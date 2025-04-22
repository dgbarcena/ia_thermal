# gat_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 heads=4,
                 use_dropout=False, dropout_rate=0.2,
                 use_batchnorm=False,
                 use_residual=False):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate
        self.heads = heads

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batchnorm else None

        # Capa de entrada
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=True))
        if self.use_batchnorm:
            self.norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # Capas ocultas
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True))
            if self.use_batchnorm:
                self.norms.append(nn.BatchNorm1d(hidden_dim * heads))

        # Capa de salida (sin concatenación)
        self.convs.append(GATConv(hidden_dim * heads, output_dim, heads=1, concat=False))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            residual = x

            x = self.convs[i](x, edge_index)

            if i != self.num_layers - 1:
                if self.use_batchnorm:
                    x = self.norms[i](x)

                x = F.relu(x)

                if self.use_dropout:
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)

                if self.use_residual:
                    if residual.shape == x.shape:
                        x = x + residual

        return x
