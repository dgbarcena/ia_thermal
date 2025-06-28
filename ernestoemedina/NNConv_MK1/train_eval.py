import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean


def train(model, loader, optimizer, device, norm_info, use_physics=False, lambda_physics=0.003,
          use_boundary_loss=True, lambda_boundary=0.01, use_heater_loss=True, lambda_heater=0.01):

    model.train()
    total_loss = torch.tensor(0.0, device=device)
    criterion = torch.nn.MSELoss()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr).view(-1)
        true_vals = batch.y.view(-1)

        mask = ~batch.mask_fixed_temp.view(-1)
        #loss_data = criterion(out[mask], true_vals[mask])  # Este valora solo los nodos no fijos, para ver qué tal generaliza el modelo
        loss_data = criterion(out, true_vals) # Este valora todos los nodos, no solo los no fijos, para ver el global


        loss_physics = compute_physics_loss(
            pred_T_norm=out,
            edge_index=batch.edge_index,
            batch=batch.batch,
            Q_heaters_norm=batch.x[:, 2],
            T_env_norm=batch.x[:, 1],
            norm_info=norm_info
        ) if use_physics else torch.tensor(0.0, device=device)

        loss_boundary = compute_boundary_loss(
            pred_T_norm=out,
            true_T_norm=batch.y.view(-1),
            mask_fixed=batch.mask_fixed_temp.view(-1)
        ) if use_boundary_loss else torch.tensor(0.0, device=device)

        loss_heater = compute_heater_loss(
            pred_T_norm=out,
            true_T_norm=batch.y.view(-1),
            Q_heaters_norm=batch.x[:, 2]
        ) if use_heater_loss else torch.tensor(0.0, device=device)

        loss = loss_data + lambda_physics * loss_physics + lambda_boundary * loss_boundary + lambda_heater * loss_heater
        loss.backward()
        optimizer.step()

        total_loss += loss.detach() * batch.num_graphs

    return total_loss.item() / len(loader.dataset)

def evaluate(model, loader, device, norm_info, error_threshold,
             use_physics=False, lambda_physics=0.003,
             use_boundary_loss=True, lambda_boundary=0.01,
             use_heater_loss=True, lambda_heater=0.01,
             percentage_threshold=None, plot_results=False):

    model.eval()

    all_mse, all_mae, all_r2, all_accuracy = [], [], [], []
    all_true_vals, all_pred_vals = [], []
    all_physics_loss, all_boundary_loss, all_heater_loss = [], [], []
    max_error_overall = 0.0
    sum_max_errors_per_graph = 0.0

    max_temp_output = norm_info["max_T_outputs"].item()

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr).view(-1)
            true_vals = data.y.view(-1)
            pred_vals = out.clone()

            mask_fixed = data.mask_fixed_temp.view(-1)

            loss_physics = compute_physics_loss(
                pred_T_norm=pred_vals,
                edge_index=data.edge_index,
                batch=data.batch,
                Q_heaters_norm=data.x[:, 2],
                T_env_norm=data.x[:, 1],
                norm_info=norm_info
            ).detach() if use_physics else torch.tensor(0.0, device=device)

            loss_boundary = compute_boundary_loss(
                pred_T_norm=pred_vals,
                true_T_norm=true_vals,
                mask_fixed=mask_fixed
            ).detach() if use_boundary_loss else torch.tensor(0.0, device=device)

            loss_heater = compute_heater_loss(
                pred_T_norm=pred_vals,
                true_T_norm=true_vals,
                Q_heaters_norm=data.x[:, 2]
            ).detach() if use_heater_loss else torch.tensor(0.0, device=device)

            all_physics_loss.append(loss_physics)
            all_boundary_loss.append(loss_boundary)
            all_heater_loss.append(loss_heater)

            nodos_por_grafico = data.num_nodes
            total_nodos = true_vals.shape[0]

            if total_nodos % nodos_por_grafico != 0:
                raise ValueError(f"El número total de nodos ({total_nodos}) no es divisible por nodos_por_grafico ({nodos_por_grafico}).")

            true_split = torch.split(true_vals, nodos_por_grafico)
            pred_split = torch.split(pred_vals, nodos_por_grafico)

            for true_graph, pred_graph in zip(true_split, pred_split):
                mask = ~data.mask_fixed_temp.view(-1)[:len(true_graph)]

                true_graph_masked = true_graph[mask]
                pred_graph_masked = pred_graph[mask]
                
                # --- Cálculo de errores máximos ---
                error_abs = torch.abs(pred_graph - true_graph)
                max_error_this_graph = error_abs.max().item()
                max_error_overall = max(max_error_overall, max_error_this_graph)
                sum_max_errors_per_graph += max_error_this_graph


                eps = 1e-8
                #mse = F.mse_loss(pred_graph_masked, true_graph_masked).detach() #Este valora solo los nodos no fijos, para ver qué tal generaliza el modelo
                mse = F.mse_loss(pred_graph, true_graph).detach() #Este valora todos los nodos
                mae = F.l1_loss(pred_graph_masked, true_graph_masked).detach()

                ss_res = torch.sum((true_graph_masked - pred_graph_masked) ** 2)
                ss_tot = torch.sum((true_graph_masked - true_graph_masked.mean()) ** 2) + eps
                r2 = (1 - ss_res / ss_tot).detach()

                if percentage_threshold is not None:
                    relative_error = torch.abs((true_graph_masked - pred_graph_masked) / (true_graph_masked + eps)) * 100
                    within = relative_error <= percentage_threshold
                else:
                    error_threshold_norm = error_threshold / max_temp_output
                    within = torch.abs(true_graph_masked - pred_graph_masked) <= error_threshold_norm

                acc = (torch.sum(within).float() / len(true_graph_masked) * 100).detach()

                all_mse.append(mse)
                all_mae.append(mae)
                all_r2.append(r2)
                all_accuracy.append(acc)

                all_true_vals.append(true_graph.cpu())
                all_pred_vals.append(pred_graph.cpu())

        if plot_results:
            true_vals_batch = true_vals.view(-1, 1)
            pred_vals_batch = pred_vals.view(-1, 1)
            batch = data.batch

            true_dense, mask = to_dense_batch(true_vals_batch, batch)
            pred_dense, _ = to_dense_batch(pred_vals_batch, batch)

            for i in range(true_dense.size(0)):
                true_graph = true_dense[i][mask[i]].squeeze().cpu()
                pred_graph = pred_dense[i][mask[i]].squeeze().cpu()

                n = true_graph.shape[0]
                lado = int(np.sqrt(n))
                if lado * lado == n:
                    plot_temperature_maps(true_graph * max_temp_output, pred_graph * max_temp_output)
                    break
            else:
                print("No se encontró ninguna muestra con número de nodos cuadrado perfecto para graficar.")

    mse_mean = torch.stack(all_mse).mean().item()
    mae_mean = torch.stack(all_mae).mean().item()
    r2_mean = torch.stack(all_r2).mean().item()
    acc_mean = torch.stack(all_accuracy).mean().item()
    rmse_mean = np.sqrt(mse_mean)

    max_error_overall_K = max_error_overall * max_temp_output
    mean_max_error_per_graph_K = (sum_max_errors_per_graph / len(all_mse)) * max_temp_output


    physics_loss_mean = torch.stack(all_physics_loss).mean().item() if use_physics else 0.0
    boundary_loss_mean = torch.stack(all_boundary_loss).mean().item() if use_boundary_loss else 0.0
    heater_loss_mean = torch.stack(all_heater_loss).mean().item() if use_heater_loss else 0.0

    val_total_loss = (
        mse_mean
        + lambda_physics * physics_loss_mean
        + lambda_boundary * boundary_loss_mean
        + lambda_heater * heater_loss_mean
    )

    return (
        mse_mean,
        mae_mean,
        r2_mean,
        acc_mean,
        physics_loss_mean,
        boundary_loss_mean,
        heater_loss_mean,
        rmse_mean,
        val_total_loss,
        max_error_overall_K,
        mean_max_error_per_graph_K,
    )



def predict(model, loader, device, norm_info):
    model.eval()
    all_pred_vals = []

    max_temp_output = norm_info["max_T_outputs"].item()
    max_temp_interfaces = norm_info["max_T_interfaces"].item()

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr).view(-1).detach()

            mask_fixed = data.mask_fixed_temp.view(-1)
            T_interfaces_input = data.x[:, 0]  # normalizado con max_T_interfaces
            T_interfaces_rescaled = T_interfaces_input * (max_temp_interfaces / max_temp_output)

            out[mask_fixed] = T_interfaces_rescaled[mask_fixed]

            nodos_por_grafico = data.num_nodes
            if out.shape[0] % nodos_por_grafico != 0:
                raise ValueError(f"El número total de nodos ({out.shape[0]}) no es divisible por nodos_por_grafico ({nodos_por_grafico}).")

            pred_vals_div = torch.split(out, nodos_por_grafico)
            all_pred_vals.extend([pred.cpu() for pred in pred_vals_div])

    return all_pred_vals


def plot_temperature_maps(true_vals, pred_vals):
    """
    Muestra:
    - Una figura con mapas de temperaturas reales y predichas, lado a lado.
    - Una figura separada con el mapa de error absoluto.
    """
    
    
    plt.style.use('default')
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["text.usetex"] = False  # Usa MathText mientras MiKTeX no esté listo
    plt.rcParams["axes.titlesize"] = 17
    
    # Asegurar numpy arrays
    true_vals = true_vals.numpy() if hasattr(true_vals, 'numpy') else np.array(true_vals)
    pred_vals = pred_vals.numpy() if hasattr(pred_vals, 'numpy') else np.array(pred_vals)
    
    total_nodos = true_vals.shape[0]
    nodos_lado = int(np.sqrt(total_nodos))
    if nodos_lado ** 2 != total_nodos:
        raise ValueError("El número de nodos no forma un cuadrado perfecto.")
    
    true_vals = true_vals.reshape(nodos_lado, nodos_lado)
    pred_vals = pred_vals.reshape(nodos_lado, nodos_lado)
    error_abs = np.abs(true_vals - pred_vals)
    
    # Escala común para temperaturas
    vmin_temp = min(true_vals.min(), pred_vals.min())
    vmax_temp = max(true_vals.max(), pred_vals.max())

    # === FIGURA 1: True y Predicted ===
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))  # Tamaño más ancho solo para esta figura

    im1 = axes1[0].imshow(true_vals, cmap='jet', vmin=vmin_temp, vmax=vmax_temp)
    axes1[0].set_title("Real Temperatures (K)")
    fig1.colorbar(im1, ax=axes1[0])

    im2 = axes1[1].imshow(pred_vals, cmap='jet', vmin=vmin_temp, vmax=vmax_temp)
    axes1[1].set_title("Predicted Temperatures (K)")
    fig1.colorbar(im2, ax=axes1[1])

    fig1.suptitle("Real vs Predicted Temperature Maps")
    fig1.tight_layout()
    plt.show()

    # === FIGURA 2: Error Absoluto ===
    fig2, ax2 = plt.subplots(figsize=(6, 5))  # Figura separada para el error

    im3 = ax2.imshow(error_abs, cmap='jet')
    ax2.set_title("Absolute Error (K)")
    fig2.colorbar(im3, ax=ax2)

    fig2.suptitle("Absolute Error Map")
    fig2.tight_layout()
    plt.show()
    
def compute_physics_loss(pred_T_norm, edge_index, batch, Q_heaters_norm, T_env_norm, norm_info):
    """
    Calcula el error físico de un grafo con conducción y radiación al entorno.
    
    pred_T_norm: temperaturas predichas normalizadas
    edge_index: conectividad del grafo [2, num_edges]
    batch: asignación de nodos a grafos
    Q_heaters_norm: potencias disipadas normalizadas
    T_env_norm: temperatura del entorno (una por nodo) normalizada
    norm_info: diccionario con valores máximos de normalización
    GLx, GLy: conductancias en x e y
    GR: coeficiente de radiación nodal
    """
    sigma = 5.67e-8  # Constante de Stefan-Boltzmann
    L = 0.1  # m
    thickness = 0.001  # m
    board_k = 15  # W/(K*m)
    ir_emissivity = 0.8
    n = 13  # nodos por lado

    dx = L / (n - 1)
    dy = L / (n - 1)
    GLx = thickness * board_k * dy / dx
    GLy = thickness * board_k * dx / dy
    GR = 2 * dx * dy * ir_emissivity
    
    # Desnormalización
    max_temp_output = norm_info["max_T_outputs"].item()
    max_Q_heaters = norm_info["max_Q_heaters"].item()
    max_T_env = norm_info["max_T_env"].item()

    T = pred_T_norm * max_temp_output
    Q_heaters = Q_heaters_norm * max_Q_heaters
    T_env = T_env_norm * max_T_env

    # Flujo por conducción homogénea (GLx, GLy)
    i, j = edge_index
    Tij = T[i] - T[j]

    # Detección de direcciones: horizontal o vertical
    dif = torch.abs(i - j)
    dx = dif == 1
    dy = ~dx

    Qij = torch.zeros_like(Tij)
    Qij[dx] = GLx * Tij[dx]
    Qij[dy] = GLy * Tij[dy]

    # Flujo neto por nodo (conducción)
    nodal_residuals = torch.zeros_like(T)
    nodal_residuals.index_add_(0, i, -Qij)
    nodal_residuals.index_add_(0, j, Qij)

    # Flujo por radiación: GR * σ * (T^4 - T_env^4)
    Q_rad = GR * sigma * (T**4 - T_env**4)
    nodal_residuals -= Q_rad

    # Añadir heaters si existen
    if Q_heaters is not None:
        nodal_residuals -= Q_heaters.view(-1)

    # Agrupación por grafo
    physics_loss = scatter_mean((nodal_residuals ** 2), batch, dim=0).mean()

    return physics_loss

def compute_boundary_loss(pred_T_norm, true_T_norm, mask_fixed):
    """
    Calcula el error en los nodos de condición de contorno (interfaces fijas).
    
    pred_T_norm: tensor [nodos] con temperaturas predichas normalizadas
    true_T_norm: tensor [nodos] con temperaturas target normalizadas
    mask_fixed: máscara booleana [nodos] con True donde hay condición de contorno
    """
    pred_fixed = pred_T_norm[mask_fixed]
    true_fixed = true_T_norm[mask_fixed]

    boundary_loss = F.mse_loss(pred_fixed, true_fixed)
    return boundary_loss


def compute_heater_loss(pred_T_norm, true_T_norm, Q_heaters_norm):
    """
    Calcula el error en los nodos donde se disipa potencia (heaters).
    
    pred_T_norm: tensor [nodos] con temperaturas predichas normalizadas
    true_T_norm: tensor [nodos] con temperaturas target normalizadas
    Q_heaters_norm: tensor [nodos] con potencias disipadas normalizadas
    """
    # Identificar nodos de heater (potencia disipada > 0)
    heater_mask = Q_heaters_norm > 0

    pred_heaters = pred_T_norm[heater_mask]
    true_heaters = true_T_norm[heater_mask]

    heater_loss = F.mse_loss(pred_heaters, true_heaters)
    return heater_loss

class EarlyStopping:
    def __init__(self, patience=100):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
