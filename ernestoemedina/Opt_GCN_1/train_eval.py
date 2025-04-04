import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import Dataset_loader as ds  # Acceso en tiempo real a target_mean y target_std


# Entrenamiento
def train(model, loader, optimizer, device, normalize=True):
    if normalize and (ds.target_mean is None or ds.target_std is None):
        raise ValueError("target_mean y target_std no están definidos. ¿Olvidaste llamar a standardize_data()?")
    
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index).view(-1)
        true_vals = data.y.view(-1)

        if normalize and ds.target_mean is not None and ds.target_std is not None:
            true_vals = (true_vals - ds.target_mean.to(device)) / ds.target_std.to(device)

        loss = F.mse_loss(out, true_vals)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, nodos_por_grafico=None, error_threshold=5.0,
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
        normalize (bool): Si True, los datos se asumen normalizados; si False, se desnormalizan para evaluación.
        
    Returns:
        mean_mse, mean_mae, mean_r2, mean_accuracy: Métricas promedio para todo el conjunto de datos.
    """

    if ds.target_mean is None or ds.target_std is None:
        raise ValueError("target_mean y target_std no están definidos. Asegúrate de llamar a prepare_dataset() con apply_standardization=True.")

    model.eval()
    all_mse, all_mae, all_r2, all_accuracy = [], [], [], []
    all_true_vals, all_pred_vals = [], []

    # Definir media y desviación estándar (usadas también si normalize=False)
    mean = ds.target_mean.to(device)
    std = ds.target_std.to(device)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.x, data.edge_index).view(-1)
            true_vals = data.y.view(-1)

            # Desnormalizar si corresponde
            if not normalize:
                true_vals = true_vals * std + mean
                pred = pred * std + mean

            pred_vals = pred  # Alias para consistencia de código

            total_nodos = true_vals.shape[0]
            if nodos_por_grafico is None or total_nodos % nodos_por_grafico != 0:
                raise ValueError(f"El número total de nodos ({total_nodos}) no es divisible por nodos_por_grafico ({nodos_por_grafico}).")

            true_split = torch.split(true_vals, nodos_por_grafico)
            pred_split = torch.split(pred_vals, nodos_por_grafico)

            for true_graph, pred_graph in zip(true_split, pred_split):
                eps = 1e-8
                mse = F.mse_loss(pred_graph, true_graph).item()
                mae = F.l1_loss(pred_graph, true_graph).item()

                ss_res = torch.sum((true_graph - pred_graph) ** 2)
                ss_tot = torch.sum((true_graph - true_graph.mean()) ** 2) + eps
                r2 = 1 - ss_res / ss_tot

                if percentage_threshold is not None:
                    relative_error = torch.abs((true_graph - pred_graph) / (true_graph + eps)) * 100
                    within = relative_error <= percentage_threshold
                else:
                    threshold = error_threshold / std.item() if normalize else error_threshold
                    within = torch.abs(true_graph - pred_graph) <= threshold

                acc = torch.sum(within).item() / len(true_graph) * 100

                all_mse.append(mse)
                all_mae.append(mae)
                all_r2.append(r2.item())
                all_accuracy.append(acc)

                all_true_vals.append(true_graph.cpu())
                all_pred_vals.append(pred_graph.cpu())

    if plot_results and len(all_true_vals) > 0:
        idx = random.randint(0, len(all_true_vals) - 1)
        plot_temperature_maps(all_true_vals[idx], all_pred_vals[idx])

    return (
        float(torch.tensor(all_mse).mean()),
        float(torch.tensor(all_mae).mean()),
        float(torch.tensor(all_r2).mean()),
        float(torch.tensor(all_accuracy).mean())
    )


# Gráficas
def plot_temperature_maps(true_vals, pred_vals):
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



# Predicción sin evaluación
def predict(model, loader, device, nodos_por_grafico=None, normalize=True):
    """
    Realiza predicciones con el modelo GCN sobre un conjunto de datos.

    Args:
        model (torch.nn.Module): El modelo entrenado.
        loader (DataLoader): DataLoader con datos de entrada (sin targets obligatorios).
        device (torch.device): Dispositivo de evaluación ('cpu' o 'cuda').
        nodos_por_grafico (int): Número de nodos por gráfico individual.
        normalize (bool): Si False, desnormaliza las salidas a escala real (K).

    Returns:
        List[Tensor]: Lista de predicciones (una por gráfico).
    """
    model.eval()
    all_pred_vals = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index).view(-1)

            if not normalize:
                out = out * ds.target_std.to(device) + ds.target_mean.to(device)

            if nodos_por_grafico is None:
                raise ValueError("nodos_por_grafico debe definirse.")
            
            if out.shape[0] % nodos_por_grafico != 0:
                raise ValueError(f"El número total de nodos ({out.shape[0]}) no es divisible por nodos_por_grafico ({nodos_por_grafico}).")        

            pred_vals_div = torch.split(out, nodos_por_grafico)
            all_pred_vals.extend([pred.cpu() for pred in pred_vals_div])  # Convertimos a CPU

    return all_pred_vals
