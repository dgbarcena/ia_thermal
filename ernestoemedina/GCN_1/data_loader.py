# data_loader.py
# Este módulo se encarga de cargar el archivo PCB_dataset.pth y prepararlo para su uso con PyTorch Geometric.

import torch
from torch_geometric.data import Data
import numpy as np
from Dataset_loader import PCBDataset



def load_pcb_dataset(file_path):
    """
    Carga el dataset de PCB y lo convierte en una lista de gráficos de PyTorch Geometric.
    Devuelve una lista de objetos 'Data' compatibles con PyTorch Geometric.
    """
    dataset = torch.load(file_path)
    graphs = []

    for i in range(len(dataset)): ### Notable: Optimizar evitando que se cree la matriz de adyacencia cada vez que se llame a load_...
        sample_input, sample_output, sample_scalar = dataset[i]

        # Extraer las matrices de potencias e interfaces
        potencias = sample_input[0].flatten().unsqueeze(1)  # Convertimos la matriz (num_nodos,1)
        nodal_features = potencias

        # El target son las temperaturas de cada nodo
        target = sample_output.flatten()  # (num_nodos,)

        # Número total de nodos
        num_nodos = nodal_features.size(0)

        # Crear la matriz de adyacencia
        edge_index = create_adjacency_matrix(int(np.sqrt(num_nodos)))

        # Crear un objeto Data para PyTorch Geometric
        data = Data(x=nodal_features, edge_index=edge_index, y=target)
        graphs.append(data)

    return graphs


def create_adjacency_matrix(grid_size):
    """
    Crea una matriz de adyacencia densa para un grafo en forma de grid (ej: 13x13).
    Devuelve un 'edge_index' compatible con PyTorch Geometric.
    """
    edge_index = []

    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            if i > 0:
                edge_index.append([node_id, node_id - grid_size])  # Conexión con el nodo superior
            if i < grid_size - 1:
                edge_index.append([node_id, node_id + grid_size])  # Conexión con el nodo inferior
            if j > 0:
                edge_index.append([node_id, node_id - 1])  # Conexión con el nodo izquierdo
            if j < grid_size - 1:
                edge_index.append([node_id, node_id + 1])  # Conexión con el nodo derecho

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index
