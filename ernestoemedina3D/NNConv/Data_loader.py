# utils/dataset_loader.py
import torch
from torch_geometric.loader import DataLoader
import random

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
