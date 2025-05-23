import sys
import os

# Añadir la raíz del proyecto al sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
    
from tqdm import tqdm
import numpy as np
import torch
from Convolutional_NN.Dataset_Class import *
from typing import Sequence, Union, Optional
from ismaelgallo.Dataset_Class_mlp import PCBDataset_mlp


def porcentaje_error_bajo_umbral(T_true: np.ndarray, T_pred: np.ndarray, umbral: float = 5.0) -> float:
    """
    Calcula el porcentaje de valores predichos con error absoluto menor que un umbral dado.

    Args:
        T_true (np.ndarray): Array real de temperaturas de tamaño [T, 13, 13]
        T_pred (np.ndarray): Array predicho de temperaturas de tamaño [T, 13, 13]
        umbral (float): Umbral de error absoluto (en K). Por defecto, 5.0.

    Returns:
        float: Porcentaje de predicciones con error < umbral
    """
    error_abs = np.abs(T_pred - T_true)
    total_vals = T_true.size
    correctos = np.sum(error_abs < umbral)
    porcentaje = 100.0 * correctos / total_vals
    return porcentaje

def nodos_siempre_dentro_umbral(T_true: np.ndarray, T_pred: np.ndarray, umbral: float = 5.0):
    """
    Identifica los nodos (i,j) cuyo error absoluto nunca supera el umbral a lo largo del tiempo.

    Args:
        T_true (np.ndarray): Array real de temperaturas de tamaño [T, 13, 13]
        T_pred (np.ndarray): Array predicho de temperaturas de tamaño [T, 13, 13]
        umbral (float): Umbral de error absoluto (en K). Por defecto, 5.0.

    Returns:
        nodos (list): Lista de tuplas (i, j) de nodos siempre dentro del umbral
        porcentaje (float): Porcentaje de nodos que cumplen la condición
        mascara (np.ndarray): Array [13, 13] con True en nodos válidos
    """
    error_abs = np.abs(T_pred - T_true)  # [T, 13, 13]
    mascara = np.all(error_abs < umbral, axis=0)  # [13, 13]
    nodos = list(zip(*np.where(mascara)))
    porcentaje = 100.0 * np.sum(mascara) / mascara.size
    return porcentaje, nodos, mascara

def porcentaje_nodos_siempre_dentro_por_umbral(T_true: np.ndarray, T_pred: np.ndarray, umbrales: np.ndarray):
    """
    Calcula el porcentaje de nodos cuyo error nunca supera cada umbral de la lista.

    Args:
        T_true (np.ndarray): Temperaturas reales, shape [T, 13, 13]
        T_pred (np.ndarray): Temperaturas predichas, shape [T, 13, 13]
        umbrales (np.ndarray): Array de valores de umbral (float)

    Returns:
        np.ndarray: Porcentaje de nodos buenos para cada umbral
    """
    error_abs = np.abs(T_pred - T_true)  # [T, 13, 13]
    porcentajes = []

    for u in umbrales:
        mascara = np.all(error_abs < u, axis=0)  # [13, 13], True donde nunca supera el umbral
        porcentaje = 100.0 * np.sum(mascara) / mascara.size
        porcentajes.append(porcentaje)

    return np.array(porcentajes)

def extract_boundary_conditions_from_dataset(input_tensor, dataset: PCBDataset, nodes_side=13):
    """
    Extrae las condiciones de contorno originales (desnormalizadas) a partir de un input_tensor y el dataset asociado.
    input_tensor: tensor de forma [batch, sequence_length, 3, nodes_side, nodes_side]
    """
    input_0 = input_tensor[0, 0]  # [3, 13, 13]

    T_interfaces1 = input_0[0]
    Q_heaters1 = input_0[1]
    T_env1 = input_0[2]

    # Extraer los valores originales usando los métodos de desnormalización del dataset
    T_interfaces_raw = torch.tensor([
        T_interfaces1[0, 0],
        T_interfaces1[0, nodes_side - 1],
        T_interfaces1[nodes_side - 1, nodes_side - 1],
        T_interfaces1[nodes_side - 1, 0]
    ], device=input_tensor.device)
    T_interfaces_in = dataset.denormalize_T_interfaces(T_interfaces_raw)

    Q_heaters_raw = torch.tensor([
        Q_heaters1[6, 3],
        Q_heaters1[3, 6],
        Q_heaters1[9, 3],
        Q_heaters1[9, 9]
    ], device=input_tensor.device)
    Q_heaters_in = dataset.denormalize_Q_heaters(Q_heaters_raw)

    T_env_in = dataset.denormalize_T_env(T_env1[0, 0])

    return Q_heaters_in, T_interfaces_in, T_env_in


def extract_all_boundary_conditions(input_tensor, dataset: PCBDataset, nodes_side=13):
    """
    Extrae las condiciones de contorno desnormalizadas de todos los ejemplos del batch.
    Retorna tres listas: Q_heaters_all, T_interfaces_all, T_env_all.
    """
    batch_size = input_tensor.shape[0]
    Q_heaters_all = []
    T_interfaces_all = []
    T_env_all = []

    for i in range(batch_size):
        q, t_int, t_env = extract_boundary_conditions_from_dataset(input_tensor[i:i+1], dataset, nodes_side)
        Q_heaters_all.append(q)
        T_interfaces_all.append(t_int)
        T_env_all.append(t_env)

    Q_heaters_all = torch.stack(Q_heaters_all)       # [batch_size, 4]
    T_interfaces_all = torch.stack(T_interfaces_all) # [batch_size, 4]
    T_env_all = torch.stack(T_env_all)               # [batch_size]

    return Q_heaters_all, T_interfaces_all, T_env_all



def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs} - Training", leave=False)

    for batch in loop:
        # Detectar si es (x, y) o (x, t, y)
        if len(batch) == 3:
            x, t, y = batch
            x, t, y = x.to(device), t.to(device), y.to(device)
            y_pred = model(x, t)
        else:
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_pred = model(x)

        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)



def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                x, t, y = batch
                x, t, y = x.to(device), t.to(device), y.to(device)
                y_pred = model(x, t)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_pred = model(x)

            loss = criterion(y_pred, y)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def predict_temperature_mlp(
    model: torch.nn.Module,
    dataset: PCBDataset_mlp,
    Q_heaters: Sequence[float],
    T_interfaces: Sequence[float],
    T_env: float,
    time_raw_seq: Optional[Union[np.ndarray, torch.Tensor]] = None,
    n_steps: int = 1001,
    device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Inferencia completa de la evolución térmica de la PCB.

    Si `time_raw_seq` es None, se crea automáticamente el vector
    de tiempo normalizado de longitud 1001 por defecto.

    Devuelve un array (n_steps, 169) con los mapas de temperatura
    desnormalizados.
    """
    # 0) Device y modelo  
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 1) Generar o convertir la secuencia de tiempo cruda
    if time_raw_seq is None:
        time_raw = torch.arange(0, n_steps, dtype=torch.float32)  # (n_steps,)
    else:
        if isinstance(time_raw_seq, np.ndarray):
            time_raw = torch.from_numpy(time_raw_seq.astype(np.float32))
        else:
            time_raw = time_raw_seq.clone().float()
    time_raw = time_raw.to(device)

    # 2) Crear todos los inputs normalizados
    inputs = []
    for t in time_raw.cpu().numpy():
        inp = dataset.create_input_from_values(
            Q_heaters=Q_heaters,
            T_interfaces=T_interfaces,
            T_env=T_env,
            time=t
        )
        inputs.append(inp.unsqueeze(0))
    X = torch.cat(inputs, dim=0).to(device)  # (n_steps, 10)

    # 3) Inferir en batch
    with torch.no_grad():
        preds_norm = model(X)  # (n_steps, 169)

    # 4) Desnormalizar la salida
    temps_denorm = dataset.denormalize_output(preds_norm)  # (n_steps,169)
    return temps_denorm
