# train_eval.py
# Este módulo se encarga de entrenar, evaluar y predecir con la GCN.

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
from torch_geometric.loader import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Train
def train(model, loader, optimizer, device):
    global target_mean, target_std
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        out = out.view(-1)
        
        true_vals = data.y  # Ya está estandarizado

        loss = F.mse_loss(out, true_vals)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)


# Evaluate
def evaluate(model, loader, device, nodos_por_grafico=None, error_threshold=5.0, #Valores por defecto si no se definen en main
             percentage_threshold=None, plot_results=True, normalize=True):
    """
    Evalúa el modelo GCN utilizando un conjunto de datos y permite calcular el error en Kelvin o en porcentaje.
    
    Args:
        model (torch.nn.Module): Modelo GCN a evaluar.
        loader (DataLoader): DataLoader del conjunto de datos.
        device (torch.device): Dispositivo para evaluar el modelo (CPU o GPU).
        nodos_por_grafico (int): Número total de nodos por gráfico.
        error_threshold (float): Error permitido en Kelvin si se usa error absoluto.
        percentage_threshold (float): Umbral del error en porcentaje si se usa error relativo.
        plot_results (bool): Indica si se deben graficar los resultados.
        normalize (bool): Indica si se debe trabajar con datos normalizados o desnormalizados.
        
    Returns:
        mean_mse, mean_mae, mean_r2, mean_accuracy: Métricas promedio para todo el conjunto de datos.
    """
    global target_mean, target_std
    model.eval()
    all_mse, all_mae, all_r2, all_accuracy = [], [], [], []
    all_true_vals, all_pred_vals = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            out = out.view(-1)

            true_batch = data.y.cpu()
            pred_batch = out.cpu()
            
            # Si se solicita desnormalización, se aplica a ambos
            if not normalize:
                true_batch = true_batch * target_std + target_mean
                pred_batch = pred_batch * target_std + target_mean

            total_nodos = true_batch.shape[0]

            if total_nodos % nodos_por_grafico != 0:
                raise ValueError(f"El número total de nodos ({total_nodos}) no es divisible por nodos_por_grafico ({nodos_por_grafico}).")
            
            # Dividir en gráficos individuales
            true_vals_dividido = torch.split(true_batch, nodos_por_grafico)
            pred_vals_dividido = torch.split(pred_batch, nodos_por_grafico)
            
            for true_vals, pred_vals in zip(true_vals_dividido, pred_vals_dividido):
                
                # Calcular métricas en la escala correcta
                if normalize:
                    mse = mean_squared_error(true_vals, pred_vals)
                    mae = mean_absolute_error(true_vals, pred_vals)
                    r2 = r2_score(true_vals, pred_vals)
                    
                    if percentage_threshold is not None:
                        # Calcular error porcentual sobre datos normalizados
                        relative_error = torch.abs((true_vals - pred_vals) / true_vals) * 100
                        within_threshold = relative_error <= percentage_threshold
                    else:
                        within_threshold = torch.abs(true_vals - pred_vals) <= (error_threshold / target_std)
                        
                else:
                    mse = mean_squared_error(true_vals, pred_vals)
                    mae = mean_absolute_error(true_vals, pred_vals)
                    r2 = r2_score(true_vals, pred_vals)
                    
                    if percentage_threshold is not None:
                        # Calcular error porcentual sobre datos desnormalizados
                        relative_error = torch.abs((true_vals - pred_vals) / true_vals) * 100
                        within_threshold = relative_error <= percentage_threshold
                    else:
                        within_threshold = torch.abs(true_vals - pred_vals) <= error_threshold

                accuracy_within_threshold = torch.sum(within_threshold.float()).item() / len(true_vals) * 100

                all_mse.append(mse)
                all_mae.append(mae)
                all_r2.append(r2)
                all_accuracy.append(accuracy_within_threshold)
                
                all_true_vals.append(true_vals)
                all_pred_vals.append(pred_vals)

    if plot_results and len(all_true_vals) > 0:
        idx = random.randint(0, len(all_true_vals) - 1)
        plot_temperature_maps(all_true_vals[idx], all_pred_vals[idx])
        
    mean_mse = np.mean(all_mse)
    mean_mae = np.mean(all_mae)
    mean_r2 = np.mean(all_r2)
    mean_accuracy = np.mean(all_accuracy)

    return mean_mse, mean_mae, mean_r2, mean_accuracy



# Gráficas
def plot_temperature_maps(true_vals, pred_vals):
    """
    Muestra gráficas de Temperaturas Reales, Temperaturas Predichas y Error Absoluto (en Kelvin) para un gráfico dado.
    
    Parámetros:
    - true_vals: Tensor o array con las temperaturas reales (dimensión: num_nodos,).
    - pred_vals: Tensor o array con las temperaturas predichas (dimensión: num_nodos,).
    """
    true_vals = true_vals.numpy()
    pred_vals = pred_vals.numpy()

    total_nodos = true_vals.shape[0]
    nodos_lado = int(np.sqrt(total_nodos))
    if nodos_lado ** 2 != total_nodos:
        raise ValueError("El número de nodos no forma cuadrado perfecto.")

    true_vals = true_vals.reshape(nodos_lado, nodos_lado)
    pred_vals = pred_vals.reshape(nodos_lado, nodos_lado)
    error_abs = np.abs(true_vals - pred_vals)

    # Fijar escala común de colores para los mapas de temperatura
    vmin_temp = min(true_vals.min(), pred_vals.min())
    vmax_temp = max(true_vals.max(), pred_vals.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im1 = axes[0].imshow(true_vals, cmap='jet', vmin=vmin_temp, vmax=vmax_temp)
    axes[0].set_title("Temperaturas Reales (K)")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(pred_vals, cmap='jet', vmin=vmin_temp, vmax=vmax_temp)
    axes[1].set_title("Temperaturas Predichas (K)")
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(error_abs, cmap='jet')
    axes[2].set_title("Error Absoluto (K)")
    fig.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.show()



def predict(model, loader, device, nodos_por_grafico=None, normalize=True):
    """
    Genera predicciones con un modelo GCN utilizando datos sin target.
    
    Args:
        model (torch.nn.Module): El modelo GCN entrenado.
        loader (DataLoader): DataLoader con las condiciones de contorno.
        device (torch.device): Dispositivo para evaluar el modelo (CPU o GPU).
        nodos_por_grafico (int): Número total de nodos por cada gráfico (debe ser conocido de antemano).
        normalize (bool): Indica si las predicciones deben ser devueltas en su forma normalizada o desnormalizada.
        
    Returns:
        list: Lista de predicciones individuales por gráfico.
    """
    global target_mean, target_std
    model.eval()
    all_pred_vals = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index)
            out = out.view(-1)
            
            if not normalize:
                # Desnormalizar la predicción si se solicita
                out = out * target_std + target_mean

            # Dividir las predicciones en gráficos individuales
            total_nodos = out.shape[0]

            if nodos_por_grafico is None:
                raise ValueError("Debe especificarse el argumento 'nodos_por_grafico' para dividir correctamente las predicciones.")

            if total_nodos % nodos_por_grafico != 0:
                raise ValueError(f"El número total de nodos ({total_nodos}) no es divisible por nodos_por_grafico ({nodos_por_grafico}).")

            pred_vals_dividido = torch.split(out.cpu(), nodos_por_grafico)
            
            for pred_vals in pred_vals_dividido:
                all_pred_vals.append(pred_vals)
    
    return all_pred_vals  # Devolver las predicciones individuales por gráfico


# Variables globales para guardar media y desviación estándar
target_mean = None
target_std = None

def standardize_data(graphs):
    """
    Estandariza automáticamente las temperaturas target en un conjunto de gráficos
    y guarda la media y desviación estándar globalmente.
    """
    global target_mean, target_std

    all_targets = torch.cat([graph.y for graph in graphs])
    target_mean = all_targets.mean()
    target_std = all_targets.std()
    
    for graph in graphs:
        graph.y = (graph.y - target_mean) / target_std  # Estandarización de la temperatura target
    
    return graphs
