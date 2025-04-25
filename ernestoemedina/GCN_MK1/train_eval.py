import torch
import torch.nn.functional as F
import random
import numpy as np
import os
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_batch



def train(model, loader, optimizer, device, use_physics=False, lambda_physics=0.3):
    model.train()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()
    # Cargar info de normalización para la pérdida física
    norm_info = torch.load(os.path.join("Datasets", "normalization_info.pth"))

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index).view(-1)
        true_vals = batch.y.view(-1)

        # Enmascarar condiciones de contorno (temperatura fija)
        mask = ~batch.mask_fixed_temp.view(-1)
        loss_data = criterion(out[mask], true_vals[mask])

        # Penalización física opcional
        if use_physics:
            loss_physics = compute_physics_loss(
                pred_T_norm=out,
                edge_index=batch.edge_index,
                batch=batch.batch,
                Q_heaters_norm=batch.x[:, 2],
                T_env_norm=batch.x[:, 1],
                norm_info=norm_info
            )
            loss = loss_data + lambda_physics * loss_physics
        else:
            loss = loss_data

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)



def evaluate(model, loader, device, error_threshold, use_physics=False, percentage_threshold=None, plot_results=False):
    model.eval()

    all_mse, all_mae, all_r2, all_accuracy = [], [], [], []
    all_true_vals, all_pred_vals = [], []
    all_physics_loss = []

    # Leer info de normalización (usada tanto para plotting como física)
    norm_info = torch.load(os.path.join("Datasets", "normalization_info.pth"))
    max_temp_output = norm_info["max_T_outputs"].item()
    max_temp_interfaces = norm_info["max_T_interfaces"].item()

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index).view(-1)
            true_vals = data.y.view(-1)
            pred_vals = out.clone()

            # Sustituir condiciones de contorno por valor real
            mask_fixed = data.mask_fixed_temp.view(-1)
            T_interfaces_input = data.x[:, 0]
            T_interfaces_rescaled = T_interfaces_input * (max_temp_interfaces / max_temp_output)
            pred_vals[mask_fixed] = T_interfaces_rescaled[mask_fixed]
            true_vals[mask_fixed] = T_interfaces_rescaled[mask_fixed]

            # Solo mostrar la pérdida física si está activada
            if use_physics:
                loss_physics = compute_physics_loss(
                    pred_T_norm=pred_vals,
                    edge_index=data.edge_index,
                    batch=data.batch,
                    Q_heaters_norm=data.x[:, 2],
                    T_env_norm=data.x[:, 1],
                    norm_info=norm_info
                )
                all_physics_loss.append(loss_physics.item())

            total_nodos = true_vals.shape[0]
            nodos_por_grafico = data.num_nodes

            if total_nodos % nodos_por_grafico != 0:
                raise ValueError(f"El número total de nodos ({total_nodos}) no es divisible por nodos_por_grafico ({nodos_por_grafico}).")

            true_split = torch.split(true_vals, nodos_por_grafico)
            pred_split = torch.split(pred_vals, nodos_por_grafico)

            for true_graph, pred_graph in zip(true_split, pred_split):
                mask = ~data.mask_fixed_temp.view(-1)[:len(true_graph)]

                true_graph_masked = true_graph[mask]
                pred_graph_masked = pred_graph[mask]

                eps = 1e-8
                mse = F.mse_loss(pred_graph_masked, true_graph_masked).item()
                mae = F.l1_loss(pred_graph_masked, true_graph_masked).item()

                ss_res = torch.sum((true_graph_masked - pred_graph_masked) ** 2)
                ss_tot = torch.sum((true_graph_masked - true_graph_masked.mean()) ** 2) + eps
                r2 = 1 - ss_res / ss_tot

                if percentage_threshold is not None:
                    relative_error = torch.abs((true_graph_masked - pred_graph_masked) / (true_graph_masked + eps)) * 100
                    within = relative_error <= percentage_threshold
                else:
                    error_threshold_norm = error_threshold / max_temp_output
                    within = torch.abs(true_graph_masked - pred_graph_masked) <= error_threshold_norm

                acc = torch.sum(within).item() / len(true_graph_masked) * 100

                all_mse.append(mse)
                all_mae.append(mae)
                all_r2.append(r2.item())
                all_accuracy.append(acc)

                all_true_vals.append(true_graph.cpu())
                all_pred_vals.append(pred_graph.cpu())

        # Mostrar un mapa si se pide
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

    physics_loss_mean = float(torch.tensor(all_physics_loss).mean()) if use_physics else None

    return (
        float(torch.tensor(all_mse).mean()),
        float(torch.tensor(all_mae).mean()),
        float(torch.tensor(all_r2).mean()),
        float(torch.tensor(all_accuracy).mean()),
        physics_loss_mean
    )



def predict(model, loader, device):
    model.eval()
    all_pred_vals = []

    # Cargar info de normalización
    norm_info = torch.load(os.path.join("Datasets", "normalization_info.pth"))
    max_temp_output = norm_info["max_T_outputs"].item()
    max_temp_interfaces = norm_info["max_T_interfaces"].item()

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index).view(-1)

            # Recuperar nodos de temperatura fija
            mask_fixed = data.mask_fixed_temp.view(-1)
            T_interfaces_input = data.x[:, 0]  # ya está normalizado con max_T_interfaces
            T_interfaces_rescaled = T_interfaces_input * (max_temp_interfaces / max_temp_output)

            # Sobrescribir los nodos fijos en la predicción
            out[mask_fixed] = T_interfaces_rescaled[mask_fixed]

            nodos_por_grafico = data.num_nodes
            if out.shape[0] % nodos_por_grafico != 0:
                raise ValueError(f"El número total de nodos ({out.shape[0]}) no es divisible por nodos_por_grafico ({nodos_por_grafico}).")

            pred_vals_div = torch.split(out, nodos_por_grafico)
            all_pred_vals.extend([pred.cpu() for pred in pred_vals_div])

    return all_pred_vals



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
    from torch_geometric.utils import to_dense_batch
    residuals_dense, mask = to_dense_batch(nodal_residuals.view(-1, 1), batch)
    physics_loss = torch.mean((residuals_dense[mask]) ** 2)

    return physics_loss

