# gcn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 use_dropout=False, dropout_rate=0.2,
                 use_batchnorm=False,
                 use_residual=False):
        super(GCN, self).__init__()

        self.num_layers = num_layers
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batchnorm else None

        # Capa de entrada
        self.convs.append(GCNConv(input_dim, hidden_dim))
        if self.use_batchnorm:
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Capas ocultas
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            if self.use_batchnorm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Capa de salida (sin BatchNorm ni Dropout ni Residual aquí)
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            residual = x

            x = self.convs[i](x, edge_index)

            if i != self.num_layers - 1:  # No aplicar en la última capa
                if self.use_batchnorm:
                    x = self.norms[i](x)

                x = F.relu(x)

                if self.use_dropout:
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)

                if self.use_residual:
                    if residual.shape == x.shape:
                        x = x + residual  # Conexión residual segura

        return x
