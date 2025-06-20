import torch
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects


def train(model, train_loader, optimizer, device, norm_info,
          use_physics=False, lambda_physics=0.003, 
          use_heater_loss=True, lambda_heater=0.01):
    model.train()  # Poner el modelo en modo entrenamiento
    criterion = torch.nn.MSELoss()  # Usamos MSE 
    total_loss = 0.0
    
    for data in train_loader:
        # Cargar datos
        inputs = data.x.to(device)
        targets = data.y.to(device)
        edge_index = data.edge_index.to(device)
        edge_attr_physics = data.edge_attr.to(device)
        Qi = data.x[:, 0].to(device)
        
        # Predicción del modelo
        optimizer.zero_grad()
        predictions = model(inputs, edge_index) 
        
        # Pérdida estándar (MSE)
        loss_data = criterion(predictions, targets)

        # Pérdida física (si se aplica)
        loss_physics = compute_physics_loss(predictions, edge_index, edge_attr_physics, Qi, norm_info) if use_physics else torch.tensor(0.0, device=device)
        
        # Pérdida de los disipadores de potencia (si se aplica)
        loss_heater = compute_heater_loss(predictions, targets, Qi) if use_heater_loss else torch.tensor(0.0, device=device)
        
        # Backpropagation
        loss = loss_data + lambda_physics * loss_physics + lambda_heater * loss_heater
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()

    return total_loss.item() / len(train_loader)  # Devolver el promedio de la pérdida de la época


def evaluate(model, data_loader, device, norm_info, lambda_physics=0.003, use_physics=False, 
             lambda_heater=0.01, use_heater_loss=False, 
             error_threshold_K=3.0, plot_results=False):
    """
    Evaluar el modelo en el conjunto de validación o test.

    :param model: El modelo entrenado.
    :param data_loader: El DataLoader con los datos de evaluación.
    :param device: El dispositivo en el que se está ejecutando (CPU o CUDA).
    :param norm_info: Información de normalización.
    :param lambda_physics: Peso de la pérdida física.
    :param use_physics: Si se debe usar la pérdida física.
    :param lambda_heater: Peso de la pérdida de disipadores.
    :param use_heater_loss: Si se debe usar la pérdida de disipadores.
    :param error_threshold_K: Umbral para el error en Kelvin.
    :param percentage_threshold: Umbral de porcentaje.
    :param plot_results: Si True, genera gráficos de las temperaturas y el error por nodo.
    
    :return: Un tuple con las métricas de evaluación: MSE, MAE, R2, accuracy, etc.
    """
    model.eval()  # Establecer el modelo en modo de evaluación
    error_threshold_K_norm = error_threshold_K / norm_info["max_T_real"]  # Normalizar el umbral de error
    
    mse_total = 0.0
    mae_total = 0.0
    r2_total = 0.0
    accuracy_total = 0.0
    physics_loss_total = 0.0
    heater_loss_total = 0.0
    rmse_total = 0.0
    total_loss = 0.0

    with torch.no_grad():  # No necesitamos calcular gradientes para la evaluación
        for data in data_loader:
            inputs = data.x.to(device)
            targets = data.y.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr_physics = data.edge_attr.to(device) if hasattr(data, 'edge_attr') else None
            Qi = data.x[:, 0].to(device)
            
            # Predicción del modelo
            predictions = model(inputs, edge_index)
            # Añade temporalmente esto en evaluate()
            #print("Ejemplo de T_real:", targets[0].item(), "T_pred:", predictions[0].item())

            # Cálculo de las pérdidas estándar (MSE, RMSE, MAE, R2)
            mse = torch.mean((predictions - targets) ** 2)
            mae = torch.mean(torch.abs(predictions - targets))
            r2 = 1 - torch.sum((targets - predictions) ** 2) / torch.sum((targets - torch.mean(targets)) ** 2)
            rmse = torch.sqrt(mse)
            
            # Cálculo del porcentaje de nodos con error dentro del umbral
            error = torch.abs(targets - predictions)
            accuracy = torch.mean((error < error_threshold_K_norm).float()) * 100  # En porcentaje
            
            # Cálculo de la pérdida física (si se aplica)
            loss_physics = compute_physics_loss(predictions, edge_index, edge_attr_physics, Qi, norm_info) if use_physics else torch.tensor(0.0, device=device)
            
            # Cálculo de la pérdida de los heaters (si se aplica)
            loss_heater = compute_heater_loss(predictions, targets, Qi) if use_heater_loss else torch.tensor(0.0, device=device)
            
            # Cálculo de la pérdida total (MSE + física + disipadores)
            loss_data = mse  # Se asume que ya se está calculando la MSE
            loss = loss_data + lambda_physics * loss_physics + lambda_heater * loss_heater
            
            # Acumular pérdidas y métricas
            mse_total += mse.item()
            mae_total += mae.item()
            r2_total += r2.item()
            accuracy_total += accuracy.item()
            physics_loss_total += loss_physics.item()
            heater_loss_total += loss_heater.item()
            rmse_total += rmse.item()
            total_loss += loss.item()
            
        # Graficar los resultados si se solicita
        if plot_results:
            plot_graph_idx = 0
            data = data_loader.dataset[plot_graph_idx].to(device)  # Obtener el primer grafo del DataLoader
            inputs = data.x.to(device)
            targets = data.y.to(device)
            edge_index = data.edge_index.to(device)
            
            predictions = model(inputs, edge_index)
            T_predicted = predictions.view(-1)

            plot_temperatures_and_error(plot_graph_idx, data.cpu(), norm_info, T_predicted.cpu())
    
    # Calcular el promedio de las métricas
    num_graphs = len(data_loader)
    mse_mean = mse_total / num_graphs
    mae_mean = mae_total / num_graphs
    r2_mean = r2_total / num_graphs
    acc_mean = accuracy_total / num_graphs
    physics_loss_mean = physics_loss_total / num_graphs
    heater_loss_mean = heater_loss_total / num_graphs
    rmse_mean = rmse_total / num_graphs
    val_total_loss = total_loss / num_graphs

    return (
        mse_mean,
        mae_mean,
        r2_mean,
        acc_mean,
        physics_loss_mean,
        heater_loss_mean,
        rmse_mean,
        val_total_loss,
    )


def predict(model, data_loader, device, norm_info, plot_results=False):
    """
    Realiza predicciones usando el modelo para un conjunto de datos dado.

    :param model: El modelo entrenado.
    :param data_loader: El DataLoader con los datos de evaluación.
    :param device: El dispositivo en el que se está ejecutando (CPU o CUDA).
    :param norm_info: Información de normalización.
    :param plot_results: Si True, genera gráficos de las temperaturas y el error por nodo.
    
    :return: Un tensor con las predicciones del modelo para el conjunto de datos.
    """
    model.eval()  # Establecer el modelo en modo de evaluación
    
    all_predictions = []  # Para almacenar las predicciones de todos los lotes
    all_targets = []      # Para almacenar los valores reales (targets) si necesitas compararlos

    with torch.no_grad():  # No necesitamos calcular gradientes para la evaluación
        for graph_idx, data in enumerate(data_loader):
            inputs = data.x.to(device)
            targets = data.y.to(device)
            edge_index = data.edge_index.to(device)
            
            # Predicción del modelo
            predictions = model(inputs, edge_index)  
            
            all_predictions.append(predictions.cpu())  # Almacenar las predicciones
            all_targets.append(targets.cpu())          # Almacenar los targets si necesitas compararlos

        # Graficar los resultados si se solicita
        if plot_results:
            plot_temperatures_and_error(graph_idx, data.cpu(), norm_info, predictions.cpu().numpy())
    
    # Concatenar todas las predicciones y targets para devolverlos
    all_predictions = torch.cat(all_predictions, dim=0)  # Concatenar todas las predicciones en un solo tensor
    all_targets = torch.cat(all_targets, dim=0)          # Concatenar todos los targets en un solo tensor

    return all_predictions, all_targets

def plot_temperatures_and_error(graph_idx, data, norm_info, T_predicted):
    """
    Visualiza el grafo coloreado por T_real, T_predicted y error absoluto.

    Parameters:
    - graph_idx: Índice del grafo.
    - data: Objeto torch_geometric.data.Data.
    - norm_info: Diccionario con valores máximos de normalización.
    - T_predicted: Temperaturas predichas normalizadas (tensor 1D).
    """

    # === Configuración estilo LaTeX + Times New Roman ===
    plt.style.use('default')
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 11,
        "text.usetex": True  # ponlo en True solo si tienes LaTeX instalado
    })

    print(f"Grafo {graph_idx}: {data}")

    # === Desnormalización ===
    T_real = data.y[:, 0] * norm_info["max_T_real"]
    T_predicted = T_predicted.reshape(-1) * norm_info["max_T_real"]
    error = torch.abs(T_real - T_predicted)

    # === Validaciones ===
    external_ids = data.external_ids
    T_real = T_real.view(-1)
    error = error.view(-1)

    assert T_predicted.shape[0] == external_ids.shape[0], "Mismatch entre nodos y predicciones"
    assert T_real.shape[0] == external_ids.shape[0], "Mismatch entre nodos y T_real"

    # === Crear grafo con nodos y atributos ===
    G = nx.Graph()
    Showexternal_ids = True  # Cambiar a False si no se quieren usar IDs externos
    
    if Showexternal_ids:
        for idx, node_id in enumerate(external_ids): #Descomentar para usar IDs externos
            G.add_node(
                int(node_id),
                T_real=T_real[idx].item(),
                T_predicted=T_predicted[idx].item(),
                error=error[idx].item()
            )
    else:
        for idx in range(len(external_ids)):
            G.add_node(
                idx,  # numeración interna
                T_real=T_real[idx].item(),
                T_predicted=T_predicted[idx].item(),
                error=error[idx].item()
            )
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()

    edge_dict = {}
    for j in range(edge_index.shape[1]):
        u_idx, v_idx = edge_index[0, j], edge_index[1, j]
        if Showexternal_ids:
            u_real, v_real = int(external_ids[u_idx]), int(external_ids[v_idx]) #Descomentar para usar IDs externos
            key = tuple(sorted((u_real, v_real)))
        else:
            key = tuple(sorted((u_idx, v_idx)))  # usa directamente índices internos

        gl, gr = edge_attr[j, 0], edge_attr[j, 1]
        if key in edge_dict:
            edge_dict[key]["GL"] = max(edge_dict[key]["GL"], gl)
            edge_dict[key]["GR"] = max(edge_dict[key]["GR"], gr)
        else:
            edge_dict[key] = {"GL": gl, "GR": gr}

    for (u, v), attr in edge_dict.items():
        G.add_edge(u, v, **attr)

    pos = nx.spring_layout(G, seed=42)

    def plot_graph(attr_name, title):
        plt.figure(figsize=(12, 9))
        colors = [G.nodes[n][attr_name] for n in G.nodes]
        norm = mcolors.Normalize(vmin=min(colors), vmax=max(colors))
        cmap = plt.get_cmap("jet")
        nodes = nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=cmap, node_size=600)
        nx.draw_networkx_edges(G, pos, alpha=0.4)

        for n in G.nodes:
            val = G.nodes[n][attr_name]
            rgba = cmap(norm(val))
            brightness = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            label_color = "white" if brightness < 0.5 else "black"
            txt = plt.text(*pos[n], str(n), fontsize=20, ha='center', va='center', color=label_color,
                           fontweight='bold')
            txt.set_path_effects([
                path_effects.Stroke(linewidth=1.5, foreground="black" if label_color == "white" else "white"),
                path_effects.Normal()
            ])

        cbar = plt.colorbar(nodes)

        if plt.rcParams["text.usetex"]:
            cbar.set_label(rf"\textbf{{{title}}} [K]", fontsize=20)
            plt.title(rf"\textbf{{Graph {graph_idx} coloured as {title}}}")
        else:
            cbar.set_label(f"{title} [K]", fontsize=20)
            plt.title(f"Grafo {graph_idx} coloreado por {title}")

        plt.axis("off")
        plt.tight_layout()

    # === Visualizaciones ===
    plot_graph("T_real", "Real Temperature")
    plot_graph("T_predicted", "Predicted Temperature")
    plot_graph("error", "Absolute Error")
    plt.show()




def compute_physics_loss(predictions, edge_index, edge_attr_physics, Qi, norm_info):
    """
    Calcula la pérdida física como error cuadrático medio entre la suma de flujos 
    entrantes (conductivo + radiativo) y la potencia Qi inyectada, por nodo.
    """
    # Desnormalizar
    max_temp = norm_info['max_T_real']
    max_power = norm_info['max_Qi']
    predictions = predictions.view(-1) * max_temp
    Qi = Qi.view(-1) * max_power

    # Obtener GL y GR
    GL = edge_attr_physics[:, 0]
    GR = edge_attr_physics[:, 1]
    sigma = 5.67e-8  # Constante de Stefan-Boltzmann en W/(m^2 K^4)

    # Nodos fuente y destino de cada arista
    src, dst = edge_index

    # Temperaturas en nodos fuente y destino
    T_src = predictions[src]
    T_dst = predictions[dst]

    # Flujos por arista
    q_cond = GL * (T_dst - T_src)
    q_rad = GR * sigma * (T_dst**4 - T_src**4)
    q_total = q_cond + q_rad  # Tamaño: [num_edges]

    # Sumar flujos entrantes a cada nodo
    residual = torch.zeros_like(predictions)
    residual = residual.index_add(0, src, q_total)

    # MSE por nodo: diferencia entre flujo total entrante y Qi inyectado
    physics_loss = torch.mean((residual - Qi) ** 2)
    
    return physics_loss



def compute_heater_loss(predictions, targets, Qi):
    """
    Calcula la pérdida asociada al calentador utilizando el MSE para los nodos
    que tienen Qi diferente de cero.
    
    :param predictions: Entradas del modelo (potencias y temperaturas normalizadas)
    :param targets: Salidas reales del modelo (temperaturas)
    :param Qi: Potencia en cada nodo (vectores de las potencias de entrada)
    :return: Pérdida MSE para los nodos con Qi distinto de cero.
    """

    # Filtrar los nodos donde Qi != 0
    mask = Qi != 0
    predicted_temps = predictions[mask]  # Temperaturas predichas para nodos con Qi != 0
    actual_temps = targets[mask]  # Temperaturas reales para nodos con Qi != 0

    # Calcular MSE entre temperaturas predichas y reales para esos nodos
    loss_heaters = torch.mean((predicted_temps - actual_temps) ** 2)
    
    return loss_heaters

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
                

