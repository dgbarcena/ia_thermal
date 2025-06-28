# utils/dataset_loader.py
import torch
from torch_geometric.loader import DataLoader
import random
import os

def load_dataset(path, batch_size, val_split=0.1, test_split=0.1, shuffle=True, seed=42):


    # Cargar el diccionario con dataset y normalización
    data_bundle = torch.load(path)
    full_dataset = data_bundle["dataset"]
    norm_info = data_bundle.get("normalization_info", {})

    if shuffle:
        random.seed(seed)
        random.shuffle(full_dataset)

    # División
    total_size = len(full_dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * total_size)
    train_size = total_size - val_size - test_size

    train_data = full_dataset[:train_size]
    val_data = full_dataset[train_size:train_size+val_size]
    test_data = full_dataset[train_size+val_size:]

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, norm_info

def load_model_by_name(model, filename, folder="saved_models"):
    """
    Carga un modelo especificado por su nombre de archivo desde la carpeta dada.
    
    Args:
        model: instancia del modelo a cargar.
        filename: nombre del archivo (ej. "3D_GCN_Layers_10_HDim_64_Epochs_1000_Lr_0.001_Batch_32.pth").
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