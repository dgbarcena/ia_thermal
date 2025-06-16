import os, sys, json, time, platform
import numpy as np
import torch
from torch import amp
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Sequence, Union, Optional

from ismaelgallo.Dataset_Class_convlstm import PCBDataset_convlstm

# # Añadir la raíz del proyecto al sys.path
# ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)
    
# # from Convolutional_NN.Dataset_Class import *
# from ismaelgallo.Dataset_Class_convlstm import PCBDataset_convlstm


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

def extract_boundary_conditions_from_dataset(input_tensor, dataset: PCBDataset_convlstm, nodes_side=13):
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


def extract_all_boundary_conditions(input_tensor, dataset: PCBDataset_convlstm, nodes_side=13):
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

def rollout_convlstm_6ch_onetoone(
    model: torch.nn.Module,
    dataset: PCBDataset_convlstm,
    Q_heaters: np.ndarray, 
    T_interfaces: np.ndarray,
    T_env: np.ndarray,
    T_init: np.ndarray = np.full((13, 13), 298.0),
    # input_tensor: torch.Tensor,
    n_steps: int = 1001,
    device: Optional[torch.device] = None,
    autorregresive: bool = True,
    denormalize: bool = True,
) -> np.ndarray:
    """
    Realiza una inferencia completa de la evolución térmica de la PCB.
    Devuelve un array (n_steps, 13, 13) con los mapas de temperatura desnormalizados.
    """
    
    T_init_tensor = torch.tensor(T_init, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [13, 13]
    
    # Device y modelo  
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Crear el input de arranque de la red
    input_tensor0 = dataset.create_input_from_values(Q_heaters, T_interfaces, T_env, T_init_tensor, sequence_length=n_steps, autorregress=True)
    
    # se inicializa todo igual y luego se updatea
    input_seq = input_tensor0.repeat(1, n_steps, 1, 1, 1)  # [1, n_steps, 6, 13, 13]
    output_norm = torch.zeros((1, n_steps, 1, 13, 13), dtype=torch.float32).to(device)  # [1, n_steps, 1, 13, 13]

    with torch.no_grad():
        for t in range(n_steps):
            output_norm[:, t, :, :, :] = model(input_seq[:, t, :, :, :])
            if t + 1 < n_steps:
                input_seq[:, t + 1, 3, :, :] = output_norm[:, t, :, :, :]

    # Desnormalizar la salida
    output_norm = output_norm.squeeze(2).squeeze(0)  # [n_steps, 13, 13]
    temps_denorm = dataset.denormalize_output(output_norm)  # (n_steps,13, 13)
    temps_denorm = temps_denorm.cpu().numpy()  # Convertir a numpy
    
    return temps_denorm


#%%
def seleccionar_dispositivo(use_cuda: bool = True, verbose: bool = True) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if verbose:
        print("📟 Dispositivo seleccionado:", device)
    return device

#%%
def get_system_specs(device: torch.device) -> dict:
    specs = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "device": str(device)
    }
    if torch.cuda.is_available():
        specs["gpu_name"] = torch.cuda.get_device_name(0)
        specs["gpu_memory_total_GB"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        specs["cuda_version"] = torch.version.cuda
    return specs

#%%
def configurar_graficos(verbose=False):
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.grid': True,
        'grid.alpha': 0.75,
        'grid.linestyle': '--',
        'lines.linewidth': 2,
    })
    sns.set_context('paper')
    sns.set_style('whitegrid')
    sns.set_palette('colorblind')  # Paleta con buen contraste y amigable
    if verbose:
        print("🎨 Estilo gráfico configurado con fondo blanco y paleta 'colorblind'.")
        
    
#%%
# def entrenar_modelo_cuda(
#         model, train_loader, val_loader,
#         criterion, optimizer, scheduler,
#         model_path, json_path,
#         epochs, early_stop_patience,
#         p0=1.0, p_min=0.0, decay_epochs=50,
#         device=None, start_epoch=0,
#         train_loss=None, val_loss=None,
#         best_val_loss=float('inf'),
#         elapsed_previous=0.0,
#         start_datetime=None,
#         system_specs=None,
#         amp_enabled=True,   # mixtura de precisión
#         clip_grad=None,
#         use_compile=True):  # activar torch.compile

#     # Preparación
#     device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model  = model.to(device)

#     # torch.compile con mode reduce-overhead (CUDAGraphs ya desactivado via env var)
#     if use_compile and torch.__version__ >= '2' and device.type == 'cuda':
#         props = torch.cuda.get_device_properties(device)
#         if props.major >= 7:
#             model = torch.compile(model, mode="reduce-overhead")
#             print("Modelo compilado (CUDAGraphs desactivado).")
#         else:
#             print(f"torch.compile desactivado: CC {props.major}.{props.minor} < 7.0")

#     torch.backends.cudnn.benchmark = True

#     # AMP: GradScaler y autocast (API torch.amp)
#     scaler = amp.GradScaler(enabled=amp_enabled)
#     autocast_ctx = (lambda: amp.autocast(device_type='cuda',
#                                          dtype=torch.float16,
#                                          enabled=True)) if amp_enabled else nullcontext

#     # Inicializar listas y tiempos
#     train_loss = [] if train_loss is None else train_loss
#     val_loss   = [] if val_loss   is None else val_loss
#     start_datetime = start_datetime or time.strftime("%Y-%m-%d %H:%M:%S")
#     start_time_training = time.time() - elapsed_previous*60
#     epochs_without_improvement = 0

#     # Bucle de epochs
#     for epoch in range(start_epoch, epochs):
#         model.train()
#         total_loss = 0.0
#         p_gt = max(p_min, p0 - epoch*(p0-p_min)/decay_epochs)

#         for x_batch, y_batch in tqdm(train_loader,
#                                      desc=f"Epoch {epoch+1}/{epochs}",
#                                      leave=False):
#             x_batch = x_batch.to(device, non_blocking=True)
#             y_batch = y_batch.to(device, non_blocking=True)
#             _, T, _, _, _ = x_batch.shape

#             bc_seq, t_prev = x_batch[:, :, :5], x_batch[:, 0, 5:6]
#             hidden, loss_accum = None, 0.0

#             optimizer.zero_grad(set_to_none=True)
#             with autocast_ctx():
#                 for t in range(T):
#                     if t == 0:
#                         t_in = t_prev
#                     else:
#                         mask = (torch.rand_like(t_prev) < p_gt).float()
#                         t_in = mask * y_batch[:, t-1] + (1-mask) * t_prev.detach()

#                     x_t = torch.cat([bc_seq[:, t], t_in], dim=1).unsqueeze(1)
#                     pred_seq, hidden = model(x_t, hidden)
#                     t_prev = pred_seq[:, 0]
#                     loss_accum += criterion(t_prev, y_batch[:, t])

#                 loss_batch = loss_accum / T

#             scaler.scale(loss_batch).backward()
#             if clip_grad:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
#             scaler.step(optimizer)
#             scaler.update()
#             total_loss += loss_batch.item()

#         train_loss.append(total_loss/len(train_loader))

#         # Validación
#         epoch_val_loss = _validacion(model, val_loader, criterion, device)
#         val_loss.append(epoch_val_loss)

#         # Scheduler & Early stopping
#         scheduler.step(epoch_val_loss)
#         lr = optimizer.param_groups[0]['lr']
#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             torch.save(model.state_dict(), model_path)
#             epochs_without_improvement = 0
#         else:
#             epochs_without_improvement += 1
#             if epochs_without_improvement >= early_stop_patience:
#                 print(f"⚠️ Early stopping en epoch {epoch+1}")
#                 break

#         # Guardar parametros en JSON
#         _guardar_json(json_path, {
#             'start_datetime': start_datetime,
#             'training_duration_minutes': (time.time()-start_time_training)/60,
#             'system_specs': system_specs,
#             'hidden_dims': list(model.convlstm.hidden_dim),
#             'kernel_size': model.convlstm.kernel_size[0],
#             'batch_size': train_loader.batch_size,
#             'lr': lr,
#             'scheduler': {
#                 'type':'ReduceLROnPlateau',
#                 'factor':scheduler.factor,
#                 'patience':scheduler.patience,
#                 'final_lr':lr
#             },
#             'early_stop_patience':early_stop_patience,
#             'epochs_trained':epoch+1,
#             'best_val_loss':best_val_loss,
#             'train_loss':train_loss,
#             'val_loss':val_loss
#         })

#     print(f"Entrenamiento finalizado en {(time.time()-start_time_training)/60:.2f} min")
#     return train_loss, val_loss, best_val_loss

# # Funciones auxiliares
# def _validacion(model, val_loader, criterion, device):
#     model.eval()
#     total = 0.0
#     with torch.no_grad():
#         for x_val, y_val in val_loader:
#             x_val = x_val.to(device, non_blocking=True)
#             y_val = y_val.to(device, non_blocking=True)
#             _, T, _, _, _ = x_val.shape
#             bc_seq, t_prev = x_val[:, :, :5], x_val[:, 0, 5:6]
#             hidden, loss_accum = None, 0.0
#             for t in range(T):
#                 x_t = torch.cat([bc_seq[:, t], t_prev], dim=1).unsqueeze(1)
#                 pred_seq, hidden = model(x_t, hidden)
#                 t_prev = pred_seq[:, 0]
#                 loss_accum += criterion(t_prev, y_val[:, t])
#             total += (loss_accum/T).item()
#     return total/len(val_loader)

# def _guardar_json(ruta, params):
#     with open(ruta, 'w') as f:
#         json.dump(params, f, indent=4)



# # Auxiliares

# def _validacion(model, val_loader, criterion, device):
#     model.eval()
#     total = 0.0
#     with torch.no_grad():
#         for x_val, y_val in val_loader:
#             x_val = x_val.to(device, non_blocking=True)
#             y_val = y_val.to(device, non_blocking=True)
#             _, T, _, _, _ = x_val.shape

#             bc_seq, t_prev = x_val[:, :, :5], x_val[:, 0, 5:6]
#             hidden, loss_accum = None, 0.0

#             for t in range(T):
#                 x_t = torch.cat([bc_seq[:, t], t_prev], dim=1).unsqueeze(1)
#                 pred_seq, hidden = model(x_t, hidden)
#                 t_prev = pred_seq[:, 0]
#                 loss_accum += criterion(t_prev, y_val[:, t])

#             total += (loss_accum / T).item()
#     return total / len(val_loader)

# def _guardar_json(ruta, params):
#     with open(ruta, 'w') as f:
#         json.dump(params, f, indent=4)

# #%%

# # --- 2. Función de rollout autoregresivo puro ---
# @torch.no_grad()
# def rollout(model, bc_seq, T0):
#     """
#     Predicción autoregresiva completa.

#     Args:
#         model   : modelo PyTorch
#         bc_seq  : tensor (B, T, 5, H, W) con canales de contorno ya en device
#         T0      : tensor (B, 1, H, W) condición inicial ya en device

#     Returns:
#         preds   : tensor (B, T, 1, H, W) con predicciones autoregresivas
#     """
#     B, T, _, H, W = bc_seq.shape
#     preds = []
#     t_prev = T0
#     hidden = None

#     for t in range(T):
#         x_t = torch.cat([bc_seq[:, t], t_prev], dim=1).unsqueeze(1)  # (B,1,6,H,W)
#         pred_seq, hidden = model(x_t, hidden)
#         t_prev = pred_seq[:, 0]
#         preds.append(t_prev)

#     return torch.stack(preds, dim=1)

# #%%
# def predict_from_conditions(Q_heaters: np.ndarray,
#                             T_interfaces: np.ndarray,
#                             T_env: float,
#                             T_seq_initial: np.ndarray,
#                             sequence_length: int,
#                             model: torch.nn.Module,
#                             dataset: PCBDataset_convlstm,
#                             device: torch.device = None) -> np.ndarray:
#     """
#     Genera una predicción completa de long. sequence_length a partir de:
#       - Q_heaters:        (4,)             np.ndarray
#       - T_interfaces:     (4,)             np.ndarray
#       - T_env:            scalar           float
#       - T_seq_initial:    (13,13)          np.ndarray (el mapa inicial)
#       - sequence_length:  número de pasos a predecir
#       - model:            tu PCB_ConvLSTM cargado y en .eval()
#       - dataset:          instancia de PCBDataset_convlstm con create_input_from_values
#       - device:           opcional, torch.device

#     Devuelve:
#       np.ndarray de forma (sequence_length, 13, 13) con la serie desnormalizada.
#     """

#     model.eval()
#     if device is None:
#         device = next(model.parameters()).device

#     # 1) Primer input (1,1,6,13,13)
#     input0 = dataset.create_input_from_values(
#         Q_heaters, T_interfaces, T_env,
#         T_seq=np.expand_dims(T_seq_initial, 0),
#         sequence_length=sequence_length,
#         autorregress=True
#     ).to(device)

#     # 2) Prepara contornos y t_prev
#     # bc_static: (1, seq_len, 5, 13,13)
#     bc_static = input0[:, :1, :5, :, :].repeat(1, sequence_length, 1, 1, 1)
#     # t_prev: (1,1,1,13,13)
#     t_prev = input0[:, :1, 5:6, :, :]

#     # 3) Roll-out autoregresivo
#     preds_norm = []
#     hidden = None
#     with torch.no_grad():
#         for t in range(sequence_length):
#             # concatenar por canal (dim=2)
#             x_t = torch.cat([bc_static[:, t:t+1], t_prev], dim=2)  # → (1,1,6,13,13)
#             pred_t, hidden = model.forward_step(x_t, hidden)       # → (1,1,1,13,13)
#             t_prev = pred_t                                        # mantener shape
#             preds_norm.append(t_prev)

#     preds_norm = torch.cat(preds_norm, dim=1)  # (1, T, 1, 13,13)

#     # 4) Desnormalizar y to numpy
#     mean = dataset.T_outputs_mean.to(device)
#     std  = dataset.T_outputs_std.to(device)
#     preds_denorm = preds_norm * std + mean      # (1, T, 1, H, W)
#     preds_denorm = preds_denorm.squeeze(0).squeeze(1)  # (T, 13,13)
#     return preds_denorm.cpu().numpy()


def predict_from_conditions(Q_heaters: np.ndarray,
                            T_interfaces: np.ndarray,
                            T_env: float,
                            sequence_length: int = 1001,
                            T_seq_initial: np.ndarray = None,
                            model=None,
                            dataset=None,
                            device=None) -> np.ndarray:
    """
    Genera una predicción completa de long. sequence_length a partir de:
      - Q_heaters:        (4,)             np.ndarray
      - T_interfaces:     (4,)             np.ndarray
      - T_env:            scalar           float
      - T_seq_initial:    (13,13)          np.ndarray (el mapa inicial)
      - sequence_length:  número de pasos a predecir
      - model:            tu PCB_ConvLSTM cargado y en .eval()
      - dataset:          instancia de PCBDataset_convlstm con create_input_from_values
      - device:           opcional, torch.device

    Devuelve:
      np.ndarray de forma (sequence_length, 13, 13) con la serie desnormalizada.
    """

    model.eval()
    if device is None:
        device = next(model.parameters()).device
        
    if T_seq_initial is None:
        T_seq_initial = np.full((13, 13), 298.0)  # Mapa por defecto a 298 K

    # 1) Primer input (1,1,6,13,13)
    input0 = dataset.create_input_from_values(
        Q_heaters, T_interfaces, T_env,
        T_seq=np.expand_dims(T_seq_initial, 0),
        sequence_length=sequence_length,
        autorregress=True
    ).to(device)

    # 2) Prepara contornos y t_prev
    # bc_static: (1, seq_len, 5, 13,13)
    bc_static = input0[:, :1, :5, :, :].repeat(1, sequence_length, 1, 1, 1)
    # t_prev: (1,1,1,13,13)
    t_prev = input0[:, :1, 5:6, :, :]

    # 3) Roll-out autoregresivo
    preds_norm = []
    hidden = None
    with torch.no_grad():
        for t in range(sequence_length):
            # concatenar por canal (dim=2)
            x_t = torch.cat([bc_static[:, t:t+1], t_prev], dim=2)  # → (1,1,6,13,13)
            pred_t, hidden = model.forward_step(x_t, hidden)       # → (1,1,1,13,13)
            t_prev = pred_t                                        # mantener shape
            preds_norm.append(t_prev)

    preds_norm = torch.cat(preds_norm, dim=1)  # (1, T, 1, 13,13)

    # 4) Desnormalizar y to numpy
    mean = dataset.T_outputs_mean.to(device)
    std  = dataset.T_outputs_std.to(device)
    preds_denorm = preds_norm * std + mean      # (1, T, 1, H, W)
    preds_denorm = preds_denorm.squeeze(0).squeeze(1)  # (T, 13,13)
    return preds_denorm.cpu().numpy()


#%%
def downsample_solver_output(T_solver, step_interval):
    """
    Recorta la salida del solver tomando muestras cada 'step_interval' pasos.
    
    Args:
        T_solver: numpy array (seq_len, H, W) - salida directa del solver
        step_interval: int, intervalo de pasos (ej: 10 para tomar pasos 0, 10, 20, ...)
    
    Returns:
        T_solver_downsampled: numpy array recortado
    """
    # Generar índices: 0, step_interval, 2*step_interval, ...
    max_steps = T_solver.shape[0]  # seq_len
    indices = list(range(0, max_steps, step_interval))
    
    # print(f"Solver output original: {max_steps} pasos")
    # print(f"Solver output recortado: {len(indices)} pasos (cada {step_interval} pasos)")
    # print(f"Índices seleccionados: {indices[:10]}{'...' if len(indices) > 10 else ''}")
    
    # Recortar usando los índices
    T_solver_downsampled = T_solver[indices, ...]
    
    return T_solver_downsampled


#%%
def simular_casos_consecutivos(Q_casos, T_interfaces_casos, T_env_casos, time_casos, T_init_global=298.0, dt_solver=1, dt_output=5, solver='transient', display=False):
    """
    Simula múltiples casos consecutivos de PCB usando arrays agrupados, donde cada caso 
    usa como condición inicial el estado final del caso anterior.
    
    Args:
        Q_casos: array (n_casos, 4) con potencias de calentadores
        T_interfaces_casos: array (n_casos, 4) con temperaturas de interfaces
        T_env_casos: array (n_casos,) con temperaturas ambiente
        time_casos: array (n_casos,) con duraciones de cada caso
        T_init_global: temperatura inicial global
        dt_solver: paso de tiempo del solver (usualmente 1s)
        dt_output: paso de tiempo de salida deseado (para hacer downsampling)
        solver: tipo de solver
        display: mostrar información de progreso
    
    Returns:
        dict con resultados combinados y metadatos
    """
    
    # Validación de entrada
    n_casos = len(time_casos)
    if Q_casos.shape[0] != n_casos:
        raise ValueError(f"Q_casos debe tener {n_casos} filas, tiene {Q_casos.shape[0]}")
    if T_interfaces_casos.shape[0] != n_casos:
        raise ValueError(f"T_interfaces_casos debe tener {n_casos} filas, tiene {T_interfaces_casos.shape[0]}")
    if T_env_casos.shape[0] != n_casos:
        raise ValueError(f"T_env_casos debe tener {n_casos} elementos, tiene {T_env_casos.shape[0]}")
    
    T_casos = []
    T_init_actual = T_init_global
    
    if display:
        print(f"🚀 Iniciando simulación de {n_casos} casos consecutivos")
        print(f"🌡️ Temperatura inicial global: {T_init_global} K")
        print(f"⏱️ dt_solver: {dt_solver}s, dt_output: {dt_output}s")
    
    for i in range(n_casos):
        # Extraer parámetros del caso actual
        Q_heaters_i = Q_casos[i]
        T_interfaces_i = T_interfaces_casos[i]
        T_env_i = T_env_casos[i]
        time_i = time_casos[i]
        
        if display:
            print(f"\n📊 Caso {i+1}/{n_casos}:")
            print(f"   ⏱️ Duración: {time_i} s")
            print(f"   🔥 Q_heaters: {Q_heaters_i}")
            print(f"   🌡️ T_interfaces: {T_interfaces_i}")
            print(f"   🌍 T_env: {T_env_i}")
        
        # Ejecutar simulación del caso actual con dt_solver
        T_caso, *_ = PCB_case_2(
            solver=solver,
            display=False,
            time=time_i,
            dt=dt_solver,
            Q_heaters=Q_heaters_i,
            T_interfaces=T_interfaces_i,
            Tenv=T_env_i,
            T_init=T_init_actual
        )
        
        # Reshape y downsample si es necesario
        T_caso_reshaped = T_caso.reshape(T_caso.shape[0], 13, 13)
        
        # Aplicar downsampling si dt_output > dt_solver
        if dt_output > dt_solver:
            T_caso_downsampled = downsample_solver_output(T_caso_reshaped, dt_output)
        else:
            T_caso_downsampled = T_caso_reshaped
        
        T_casos.append(T_caso_downsampled)
        
        # La temperatura inicial del siguiente caso es la final del actual (sin downsample)
        if i < n_casos - 1:
            T_init_actual = T_caso[-1, :]  # Último estado temporal del solver original
    
    # Concatenar todos los casos, eliminando duplicados en las uniones
    T_combined = T_casos[0]
    indices_cambio = []
    tiempos_cambio = []
    tiempo_acumulado = 0
    
    for i in range(1, len(T_casos)):
        # El índice de cambio es la longitud actual de T_combined
        indices_cambio.append(len(T_combined))
        
        # Tiempo acumulado hasta este cambio
        tiempo_acumulado += time_casos[i-1]
        tiempos_cambio.append(tiempo_acumulado)
        
        # Excluir el primer punto del caso actual (que corresponde al último del anterior)
        T_combined = np.concatenate([T_combined, T_casos[i][1:]], axis=0)
    
    # Crear array de tiempo completo
    tiempo_total = int(np.sum(time_casos))
    tiempo_array = np.arange(0, len(T_combined)) * dt_output
    
    resultado = {
        'T_combined': T_combined,
        'T_casos': T_casos,
        'tiempo_array': tiempo_array,
        'tiempos_casos': time_casos.tolist(),
        'tiempo_total': tiempo_total,
        'indices_cambio': indices_cambio,
        'tiempos_cambio': tiempos_cambio,
        'dt_solver': dt_solver,
        'dt_output': dt_output,
        'Q_casos': Q_casos,
        'T_interfaces_casos': T_interfaces_casos,
        'T_env_casos': T_env_casos,
        'time_casos': time_casos
    }
    
    if display:
        print(f"\n✅ Simulación completada:")
        print(f"   📏 Tiempo total: {tiempo_total} s")
        print(f"   📊 Puntos temporales: {len(T_combined)}")
        print(f"   🔄 Cambios en índices: {resultado['indices_cambio']}")
        print(f"   🔄 Cambios en tiempos: {resultado['tiempos_cambio']} s")
    
    return resultado


#%%
def plot_casos_consecutivos(resultado, nodos=[(6, 6)], save_as_pdf=False, filename='casos_consecutivos_arrays', show_details=False):
    """
    Grafica la evolución de temperatura para múltiples nodos específicos mostrando los diferentes casos.
    Versión que permite múltiples nodos en el mismo gráfico.
    
    Args:
        resultado: dict con resultados de simular_casos_consecutivos
        nodos: lista de tuplas (i, j) con las coordenadas de los nodos a graficar
        save_as_pdf: bool, guardar como PDF
        filename: nombre del archivo
        show_details: bool, mostrar información detallada (por defecto False)
    """
    
    T_combined = resultado['T_combined']
    indices_cambio = resultado['indices_cambio']
    tiempos_cambio = resultado['tiempos_cambio']
    tiempos_casos = resultado['tiempos_casos']
    tiempo_array = resultado['tiempo_array']
    
    # Reshape si es necesario
    if len(T_combined.shape) == 2:  # (tiempo, nodos_linearizados)
        T_combined = T_combined.reshape(T_combined.shape[0], 13, 13)
    
    # Crear colores para cada caso
    n_casos = len(tiempos_casos)
    colores_casos = plt.cm.tab10(np.linspace(0, 1, n_casos))
    
    # Crear estilos de línea para cada nodo
    estilos_linea = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5))]
    if len(nodos) > len(estilos_linea):
        # Si hay más nodos que estilos, repetir estilos
        estilos_linea = estilos_linea * (len(nodos) // len(estilos_linea) + 1)
    
    plt.figure(figsize=(14, 8))
    
    # Calcular índices de inicio y fin para cada caso
    indices_inicio = [0] + indices_cambio.copy()
    indices_fin = indices_cambio.copy() + [len(tiempo_array)]
    
    # Para cada nodo
    for nodo_idx, (i, j) in enumerate(nodos):
        temp_nodo = T_combined[:, i, j]
        estilo_linea = estilos_linea[nodo_idx]
        
        # Plotear cada segmento con color diferente por caso
        for caso_idx, (duracion, color) in enumerate(zip(tiempos_casos, colores_casos)):
            inicio = indices_inicio[caso_idx]
            fin = indices_fin[caso_idx]
            
            # Para continuidad visual, incluir el primer punto del siguiente si existe
            if caso_idx < len(tiempos_casos) - 1:
                fin_plot = fin + 1 if fin < len(tiempo_array) else fin
            else:
                fin_plot = fin
            
            # Etiqueta solo para el primer nodo de cada caso (evitar duplicados en leyenda)
            if nodo_idx == 0:
                label_caso = f'Case {caso_idx+1} ({duracion}s)'  # 🔄 Cambiado a inglés
            else:
                label_caso = None
                
            plt.plot(tiempo_array[inicio:fin_plot], temp_nodo[inicio:fin_plot], 
                    color=color, linewidth=2, linestyle=estilo_linea,
                    label=label_caso)
    
    # Marcar puntos de cambio con líneas verticales
    for tiempo_cambio in tiempos_cambio:
        plt.axvline(x=tiempo_cambio, color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.text(tiempo_cambio, plt.ylim()[1]*0.95, f'Change',  # 🔄 Cambiado a inglés
                rotation=90, ha='right', va='top', fontsize=8, color='red')
    
    # Añadir leyenda personalizada para los nodos
    from matplotlib.lines import Line2D
    
    # Leyenda para casos (ya está en el plot)
    legend_casos = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Cases')  # 🔄 Cambiado a inglés
    
    # Leyenda adicional para nodos
    if len(nodos) > 1:
        legend_elements_nodos = []
        for nodo_idx, (i, j) in enumerate(nodos):
            estilo_linea = estilos_linea[nodo_idx]
            legend_elements_nodos.append(
                Line2D([0], [0], color='black', linestyle=estilo_linea, linewidth=2,
                       label=f'Node ({i},{j})')  # 🔄 Cambiado a inglés
            )
        
        legend_nodos = plt.legend(handles=legend_elements_nodos, 
                                 bbox_to_anchor=(1.05, 0.7), loc='upper left', 
                                 title='Nodes')  # 🔄 Cambiado a inglés
        plt.gca().add_artist(legend_casos)  # Mantener ambas leyendas
    
    plt.xlabel('Time [s]')  # 🔄 Cambiado a inglés
    plt.ylabel('Temperature [K]')  # 🔄 Cambiado a inglés
    
    # Título adaptativo (solo para visualización en pantalla)
    if len(nodos) == 1:
        titulo = f'Temperature Evolution - Node {nodos[0]} - Consecutive Cases'  # 🔄 Cambiado a inglés
    else:
        titulo = f'Temperature Evolution - {len(nodos)} Nodes - Consecutive Cases'  # 🔄 Cambiado a inglés
    
    plt.title(titulo)  # 🆕 Título siempre visible en pantalla
    
    plt.grid(True, alpha=0.3)
    
    # 🔧 AJUSTAR RANGO DE ABSCISAS (eje X)
    plt.xlim(tiempo_array[0], tiempo_array[-1])
    
    plt.tight_layout()
    
    if save_as_pdf:
        # 🆕 Remover título antes de guardar PDF
        plt.title('')  # Eliminar título para el PDF
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf', bbox_inches='tight')
        # 🆕 Restaurar título después de guardar
        plt.title(titulo)  # Restaurar título para visualización
    
    plt.show()
    
    # ✅ INFORMACIÓN DETALLADA CONDICIONAL (también en inglés)
    if show_details:
        # Estadísticas para todos los nodos
        print(f"📊 Selected nodes: {nodos}")  # 🔄 Cambiado a inglés
        print(f"⏱️ Time resolution: {resultado['dt_output']} s")  # 🔄 Cambiado a inglés
        
        # Información detallada por nodo
        for nodo_idx, (i, j) in enumerate(nodos):
            temp_nodo = T_combined[:, i, j]
            print(f"\n🌡️ NODE ({i},{j}):")  # 🔄 Cambiado a inglés
            print(f"   Initial temperature: {temp_nodo[0]:.2f} K")  # 🔄 Cambiado a inglés
            print(f"   Final temperature: {temp_nodo[-1]:.2f} K")  # 🔄 Cambiado a inglés
            print(f"   Total increment: {temp_nodo[-1] - temp_nodo[0]:.2f} K")  # 🔄 Cambiado a inglés
        
        # Información detallada de casos (solo una vez)
        print(f"\n📋 CASE SUMMARY:")  # 🔄 Cambiado a inglés
        Q_casos = resultado['Q_casos']
        T_interfaces_casos = resultado['T_interfaces_casos']
        T_env_casos = resultado['T_env_casos']
        
        for idx in range(len(tiempos_casos)):
            inicio = indices_inicio[idx]
            fin = indices_fin[idx] - 1
            
            print(f"  Case {idx+1}: Q={Q_casos[idx]}, Ti={T_interfaces_casos[idx]}, Tenv={T_env_casos[idx]}")  # 🔄 Cambiado a inglés
            print(f"           Time: {tiempo_array[inicio]:.1f}s → {tiempo_array[fin-1]:.1f}s")  # 🔄 Cambiado a inglés
            
            # Mostrar cambio de temperatura para cada nodo en este caso
            for nodo_idx, (i, j) in enumerate(nodos):
                temp_nodo = T_combined[:, i, j]
                temp_inicial = temp_nodo[inicio]
                temp_final = temp_nodo[fin] if fin < len(temp_nodo) else temp_nodo[-1]
                delta_T = temp_final - temp_inicial
                print(f"           Node ({i},{j}): {temp_inicial:.2f}K → {temp_final:.2f}K (ΔT={delta_T:.2f}K)")  # 🔄 Cambiado a inglés
    
    return tiempo_array, {f"nodo_{i}_{j}": T_combined[:, i, j] for i, j in nodos}


#%%
def predict_casos_consecutivos(Q_casos, T_interfaces_casos, T_env_casos, time_casos, 
                              model, dataset, T_init_global=298.0, dt_output=5, device=None, display=False):
    """
    Predice múltiples casos consecutivos usando ConvLSTM, donde cada caso 
    usa como condición inicial el estado final del caso anterior.
    
    Args:
        Q_casos: array (n_casos, 4) con potencias de calentadores
        T_interfaces_casos: array (n_casos, 4) con temperaturas de interfaces
        T_env_casos: array (n_casos,) con temperaturas ambiente
        time_casos: array (n_casos,) con duraciones de cada caso
        model: modelo ConvLSTM entrenado
        dataset: dataset para normalización/desnormalización
        T_init_global: temperatura inicial global
        dt_output: paso de tiempo de salida
        device: dispositivo de cómputo
        display: mostrar información de progreso
    
    Returns:
        dict con resultados combinados y metadatos
    """
    
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    # Validación de entrada
    n_casos = len(time_casos)
    if Q_casos.shape[0] != n_casos:
        raise ValueError(f"Q_casos debe tener {n_casos} filas, tiene {Q_casos.shape[0]}")
    if T_interfaces_casos.shape[0] != n_casos:
        raise ValueError(f"T_interfaces_casos debe tener {n_casos} filas, tiene {T_interfaces_casos.shape[0]}")
    if T_env_casos.shape[0] != n_casos:
        raise ValueError(f"T_env_casos debe tener {n_casos} elementos, tiene {T_env_casos.shape[0]}")
    
    T_casos = []
    T_map_actual = np.full((13, 13), T_init_global)  # Mapa inicial
    
    if display:
        print(f"🚀 Iniciando predicción de {n_casos} casos consecutivos")
        print(f"🌡️ Temperatura inicial global: {T_init_global} K")
        print(f"⏱️ dt_output: {dt_output}s")
    
    for i in range(n_casos):
        # Extraer parámetros del caso actual
        Q_heaters_i = Q_casos[i]
        T_interfaces_i = T_interfaces_casos[i]
        T_env_i = T_env_casos[i]
        time_i = time_casos[i]
        sequence_length_i = int(time_i // dt_output + 1)
        
        if display:
            print(f"\n📊 Caso {i+1}/{n_casos}:")
            print(f"   ⏱️ Duración: {time_i} s ({sequence_length_i} pasos)")
            print(f"   🔥 Q_heaters: {Q_heaters_i}")
            print(f"   🌡️ T_interfaces: {T_interfaces_i}")
            print(f"   🌍 T_env: {T_env_i}")
        
        # Predecir el caso actual usando predict_from_conditions
        T_pred_caso = predict_from_conditions(
            Q_heaters=Q_heaters_i,
            T_interfaces=T_interfaces_i,
            T_env=T_env_i,
            sequence_length=sequence_length_i,
            T_seq_initial=T_map_actual,  # Usar el estado final del caso anterior
            model=model,
            dataset=dataset,
            device=device
        )
        
        T_casos.append(T_pred_caso)
        
        # La temperatura inicial del siguiente caso es la final del actual
        if i < n_casos - 1:
            T_map_actual = T_pred_caso[-1, :, :]  # Último estado temporal
    
    # Concatenar todos los casos, eliminando duplicados en las uniones
    T_combined = T_casos[0]
    indices_cambio = []
    tiempos_cambio = []
    tiempo_acumulado = 0
    
    for i in range(1, len(T_casos)):
        # El índice de cambio es la longitud actual de T_combined
        indices_cambio.append(len(T_combined))
        
        # Tiempo acumulado hasta este cambio
        tiempo_acumulado += time_casos[i-1]
        tiempos_cambio.append(tiempo_acumulado)
        
        # Excluir el primer punto del caso actual (que corresponde al último del anterior)
        T_combined = np.concatenate([T_combined, T_casos[i][1:]], axis=0)
    
    # Crear array de tiempo completo
    tiempo_total = int(np.sum(time_casos))
    tiempo_array = np.arange(0, len(T_combined)) * dt_output
    
    resultado = {
        'T_combined': T_combined,
        'T_casos': T_casos,
        'tiempo_array': tiempo_array,
        'tiempos_casos': time_casos.tolist(),
        'tiempo_total': tiempo_total,
        'indices_cambio': indices_cambio,
        'tiempos_cambio': tiempos_cambio,
        'dt_output': dt_output,
        'Q_casos': Q_casos,
        'T_interfaces_casos': T_interfaces_casos,
        'T_env_casos': T_env_casos,
        'time_casos': time_casos
    }
    
    if display:
        print(f"\n✅ Predicción completada:")
        print(f"   📏 Tiempo total: {tiempo_total} s")
        print(f"   📊 Puntos temporales: {len(T_combined)}")
        print(f"   🔄 Cambios en índices: {resultado['indices_cambio']}")
        print(f"   🔄 Cambios en tiempos: {resultado['tiempos_cambio']} s")
    
    return resultado


def generate_unique_cases(n_data):
    """
    Genera casos únicos evitando duplicados para asegurar diversidad en el análisis.
    
    Args:
        n_data: Número de casos únicos a generar
        
    Returns:
        Q_list: Array de potencias de heaters [W] - shape (n_data, 4)
        T_int_list: Array de temperaturas de interfaces [K] - shape (n_data, 4)  
        T_env_list: Array de temperaturas ambiente [K] - shape (n_data,)
    """
    seen = set()
    Q_list, T_int_list, T_env_list = [], [], []
    
    while len(Q_list) < n_data:
        # Generar condiciones aleatorias
        Q = tuple(np.random.uniform(0.5, 1.5, 4).round(6))        # Potencias [0.5-1.5W]
        T_int = tuple(np.random.uniform(270, 320, 4).round(2))    # Interfaces [270-320K]
        T_env = round(float(np.random.uniform(270, 320)), 2)      # Ambiente [270-320K]
        
        # Crear clave única para evitar duplicados
        key = Q + T_int + (T_env,)
        
        if key not in seen:
            seen.add(key)
            Q_list.append(Q)
            T_int_list.append(T_int)
            T_env_list.append(T_env)
    
    return np.array(Q_list), np.array(T_int_list), np.array(T_env_list)