import torch
from torch_geometric.data import Data

def generate_edge_index(grid_size):
    """
    Genera la matriz de adyacencia bidireccional (edge_index) para una malla 2D cuadrada.
    Conectividad de 4 vecinos (right/left/top/bottom).
    """
    edge_index = []

    for y in range(grid_size):
        for x in range(grid_size):
            idx = y * grid_size + x

            # Derecha
            if x < grid_size - 1:
                edge_index.append([idx, idx + 1])
                edge_index.append([idx + 1, idx])

            # Abajo
            if y < grid_size - 1:
                edge_index.append([idx, idx + grid_size])
                edge_index.append([idx + grid_size, idx])

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


def build_graph_list(dataset, edge_index):
    """
    Convierte todo un dataset normalizado en una lista de grafos PyG.
    """
    graphs = []
    for idx in range(len(dataset)):
        graph = build_graph_from_sample(dataset, idx, edge_index)
        graphs.append(graph)
    return graphs