# nnconv_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class NNConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 edge_dim=3,
                 use_dropout=False, dropout_rate=0.2,
                 use_batchnorm=False,
                 use_residual=False):
        super(NNConvNet, self).__init__()

        self.num_layers = num_layers
        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.use_residual = use_residual
        self.dropout_rate = dropout_rate

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_batchnorm else None

        # Capa de entrada
        nn1 = nn.Sequential(nn.Linear(edge_dim, hidden_dim * input_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_dim * input_dim, hidden_dim * input_dim))
        self.convs.append(NNConv(input_dim, hidden_dim, nn1, aggr='mean'))
        if self.use_batchnorm:
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Capas ocultas
        for _ in range(num_layers - 2):
            nn_hidden = nn.Sequential(nn.Linear(edge_dim, hidden_dim * hidden_dim),
                                      nn.SiLU(),
                                      nn.Linear(hidden_dim * hidden_dim, hidden_dim * hidden_dim))
            self.convs.append(NNConv(hidden_dim, hidden_dim, nn_hidden, aggr='mean'))
            if self.use_batchnorm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        # Capa de salida
        nn_out = nn.Sequential(nn.Linear(edge_dim, hidden_dim * output_dim),
                               nn.SiLU(),
                               nn.Linear(hidden_dim * output_dim, hidden_dim * output_dim))
        self.convs.append(NNConv(hidden_dim, output_dim, nn_out, aggr='mean'))

    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_layers):
            residual = x
            x = self.convs[i](x, edge_index, edge_attr)

            if i != self.num_layers - 1:
                if self.use_batchnorm:
                    x = self.norms[i](x)
                x = F.silu(x)
                if self.use_dropout:
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)
                if self.use_residual and residual.shape == x.shape:
                    x = x + residual

        return x
