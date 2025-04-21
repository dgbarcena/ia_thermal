import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import glob
import random

def generate_edge_index(grid_size):
    """
    Genera la matriz de adyacencia (edge_index) para una malla 2D cuadrada.
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



def build_graph_from_sample(dataset, idx, edge_index):
    """
    Convierte una muestra en un grafo PyG: x ∈ [nxn, 3], y ∈ [nxn, 1]
    """
    
    input_tensor = dataset.inputs[idx]  # (3, n, n)
    x = input_tensor.reshape(3, -1).T   # [nxn, 3]

    output_tensor = dataset.outputs[idx]  # (n, n)
    y = output_tensor.reshape(-1, 1)      # [nxn, 1]

    mask_fixed_temp = get_fixed_temp_mask(x)

    return Data(x=x, y=y, edge_index=edge_index, mask_fixed_temp=mask_fixed_temp)

def build_graph_list(dataset, edge_index):
    """
    Convierte todo un dataset normalizado en una lista de grafos PyG.
    """
    graphs = []
    for idx in range(len(dataset)):
        graph = build_graph_from_sample(dataset, idx, edge_index)
        graphs.append(graph)
    return graphs

def get_fixed_temp_mask(x):
    """
    Devuelve una máscara booleana indicando qué nodos tienen temperatura fijada (input_temp ≠ 0).
    """
    # x: [N, 3] -> asumimos x[:, 0] es la temperatura
    return x[:, 0] != 0

def masked_mse_loss(pred, target, mask_fixed_temp):
    """
    Calcula el MSE solo en los nodos donde `mask_fixed_temp` es False (i.e., no es condición de contorno).
    """
    mask = ~mask_fixed_temp  # invertir la máscara
    pred_masked = pred[mask]
    target_masked = target[mask]
    return F.mse_loss(pred_masked, target_masked)


def load_latest_model(model, folder="saved_models"):
    """
    Carga el modelo más recientemente guardado en la carpeta especificada.
    """
    model_files = glob.glob(os.path.join(folder, "*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No se encontraron modelos en {folder}")

    # Ordenar por fecha de modificación
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model = model_files[0]

    # Cargar el modelo
    model.load_state_dict(torch.load(latest_model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    print(f" Modelo cargado desde: {latest_model}")

    return model

def load_model_by_name(model, filename, folder="saved_models"):
    """
    Carga un modelo especificado por su nombre de archivo desde la carpeta dada.
    
    Args:
        model: instancia del modelo a cargar.
        filename: nombre del archivo (ej. "GCN_Layers_10_HDim_64_Epochs_1000_Lr_0.001_Batch_32.pth").
        folder: carpeta donde está guardado el modelo.
    
    Returns:
        model con los pesos cargados.
    """
    model_path = os.path.join(folder, filename)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    print(f"Modelo cargado desde: {model_path}")
    return model

def get_dataloaders_optuna(batch_size, dataset_path="Datasets/PCB_Dataset_Norm.pth"):  # De cara a la optimización de hiperparámetros
    """
    Carga el dataset, genera los grafos y devuelve los dataloaders + input_dim.
    """
    # 1. Cargar el dataset normalizado
    dataset = torch.load(dataset_path)

    # 2. Crear edge_index
    grid_size = dataset.outputs.shape[-1]  # asumimos que las salidas son (n, n)
    edge_index = generate_edge_index(grid_size)

    # 3. Convertir en lista de grafos
    graphs = build_graph_list(dataset, edge_index)
    random.shuffle(graphs)
    
    # 4. Dividir dataset
    total = len(graphs)
    train_split = int(0.7 * total)
    val_split = int(0.85 * total)

    train_graphs = graphs[:train_split]
    val_graphs = graphs[train_split:val_split]
    test_graphs = graphs[val_split:]
    # 5. Crear DataLoaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    test_loader = DataLoader(test_graphs, batch_size=batch_size)

    # 6. input_dim = número de canales de entrada
    input_dim = graphs[0].x.shape[1]

    return train_loader, val_loader, test_loader, input_dim