# Dataset_loader.py
# Módulo completo para cargar, estandarizar y preparar datasets para PyTorch Geometric.

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import os
import numpy as np

# Variables globales de estandarización
target_mean = None
target_std = None

# Cache para evitar recalcular edge_index si todos los gráficos son iguales
cached_edge_index = None

# ---------------------------- #
#        Dataset base         #
# ---------------------------- #

class PCBDataset(Dataset):
    def __init__(self, inputs_dataset, outputs_dataset, scalar_dataset):
        assert len(inputs_dataset) == len(outputs_dataset) == len(scalar_dataset), "Todos los sets deben tener la misma longitud"
        self.inputs_dataset = inputs_dataset
        self.outputs_dataset = outputs_dataset
        self.scalar_dataset = scalar_dataset

    def __len__(self):
        return len(self.inputs_dataset)

    def __getitem__(self, idx):
        return self.inputs_dataset[idx], self.outputs_dataset[idx], self.scalar_dataset[idx]


# ---------------------------- #
#     Carga y Guardado .pth   #
# ---------------------------- #

def prepare_dataset(file_path, device='cpu', apply_standardization=True, force_standardize=False):
    """
    Carga y prepara el dataset completo:
    - Carga desde archivo .pth
    - Convierte PCBDataset a lista de objetos Data
    - Aplica estandarización si se requiere

    Args:
        file_path (str): Ruta al archivo .pth
        device (str): 'cpu' o 'cuda'
        apply_standardization (bool): Si True, aplica estandarización a los targets
        force_standardize (bool): Si True, fuerza recalcular media y std aunque ya estén definidos

    Returns:
        graphs (list[Data]): Lista de gráficos PyTorch Geometric
        target_mean (float): Media de las temperaturas
        target_std (float): Desviación estándar de las temperaturas
    """
    graphs, target_mean, target_std = load_dataset(file_path, device=device)

    if isinstance(graphs, PCBDataset):
        graphs = load_pcb_dataset(graphs, device=device)

    if apply_standardization:
        graphs = standardize_data(graphs, device=device, force_standardize=force_standardize)

    if target_mean is None or target_std is None:
        raise ValueError("target_mean o target_std no definidos tras la carga. Revisa el dataset o aplica estandarización.")

    return graphs, target_mean, target_std


def load_dataset(file_path, device='cpu'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")

    data_dict = torch.load(file_path)

    if isinstance(data_dict, dict):
        dataset = data_dict['dataset']
        global target_mean, target_std
        target_mean = data_dict.get('target_mean', None)
        target_std = data_dict.get('target_std', None)
        print("Dataset cargado con estadísticas de estandarización.")
        return dataset, target_mean, target_std

    elif isinstance(data_dict, PCBDataset):
        print("Dataset cargado como objeto PCBDataset.")
        return data_dict, None, None

    elif isinstance(data_dict, list) and isinstance(data_dict[0], Data):
        print("Dataset cargado como lista de gráficos tipo Data.")
        for graph in data_dict:
            graph.to(device)
        return data_dict, None, None

    else:
        raise TypeError("El archivo debe ser un diccionario, una lista de objetos 'Data' o un PCBDataset.")



def save_dataset(dataset, file_path, save_stats=True):
    global target_mean, target_std
    dir_path = os.path.dirname(file_path)
    if dir_path != "":
        os.makedirs(dir_path, exist_ok=True)

    data_to_save = {'dataset': dataset}
    if save_stats and target_mean is not None and target_std is not None:
        data_to_save['target_mean'] = target_mean
        data_to_save['target_std'] = target_std

    torch.save(data_to_save, file_path)
    print(f"Dataset guardado correctamente en: {file_path}")


# ---------------------------- #
#     Estandarización global   #
# ---------------------------- #

def standardize_data(graphs, device='cpu', force_standardize=False):
    global target_mean, target_std

    if target_mean is not None and target_std is not None and not force_standardize:
        print("Dataset ya estandarizado. Usa force_standardize=True si deseas volver a hacerlo.")
        return graphs

    all_targets = torch.cat([graph.y for graph in graphs]).to(device)
    target_mean = all_targets.mean()
    target_std = all_targets.std()

    for graph in graphs:
        graph.y = (graph.y - target_mean) / target_std

    print(f"Dataset estandarizado (mean={target_mean:.2f}, std={target_std:.2f})")
    return graphs


def denormalize_graphs(graphs):
    global target_mean, target_std
    if target_mean is None or target_std is None:
        raise ValueError("No se puede desnormalizar: faltan target_mean o target_std.")
    
    for graph in graphs:
        graph.y = graph.y * target_std + target_mean
    print("Dataset desnormalizado.")
    return graphs


# ---------------------------- #
#     Conversión a PyG Data    #
# ---------------------------- #

def load_pcb_dataset(file_path, device='cpu', reuse_adjacency_matrix=True):
    global cached_edge_index

    dataset = torch.load(file_path)
    graphs = []

    # Soporta tanto PCBDataset como lista directa
    if isinstance(dataset, PCBDataset):
        dataset = [(dataset.inputs_dataset[i], dataset.outputs_dataset[i], dataset.scalar_dataset[i])
                   for i in range(len(dataset))]

    for i, (sample_input, sample_output, sample_scalar) in enumerate(dataset):
        potencias = sample_input[0].view(-1, 1)
        target = sample_output.view(-1)

        if potencias.shape[0] != target.shape[0]:
            raise ValueError(f"Input y target tienen distinta longitud en muestra {i}")

        num_nodos = potencias.size(0)
        grid_size = int(np.sqrt(num_nodos))

        if reuse_adjacency_matrix and cached_edge_index is not None:
            edge_index = cached_edge_index
        else:
            edge_index = create_adjacency_matrix(grid_size)
            if reuse_adjacency_matrix:
                cached_edge_index = edge_index

        data = Data(
            x=potencias.to(device),
            edge_index=edge_index.to(device),
            y=target.to(device)
        )
        graphs.append(data)

    return graphs


def create_adjacency_matrix(grid_size):
    edge_index = []
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            if i > 0: edge_index.append([node_id, node_id - grid_size])
            if i < grid_size - 1: edge_index.append([node_id, node_id + grid_size])
            if j > 0: edge_index.append([node_id, node_id - 1])
            if j < grid_size - 1: edge_index.append([node_id, node_id + 1])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()
