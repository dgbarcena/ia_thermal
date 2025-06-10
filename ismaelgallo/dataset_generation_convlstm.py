import numpy as np
import os
import sys
import time
import torch

#%%
def downsample_sequences(input_seq, output_seq, step_interval):
    """
    Recorta las secuencias tomando muestras cada 'step_interval' pasos.
    
    Args:
        input_seq: tensor (n_data, seq_len, channels, H, W)
        output_seq: tensor (n_data, seq_len, H, W) 
        step_interval: int, intervalo de pasos (ej: 10 para tomar pasos 0, 10, 20, ...)
    
    Returns:
        input_seq_downsampled: tensor recortado
        output_seq_downsampled: tensor recortado
    """
    # Generar índices: 0, step_interval, 2*step_interval, ...
    max_steps = input_seq.shape[1]  # seq_len
    indices = list(range(0, max_steps, step_interval))
    
    print(f"  Secuencia original: {max_steps} pasos")
    print(f"  Secuencia recortada: {len(indices)} pasos (cada {step_interval} pasos)")
    print(f"  Índices seleccionados: {indices[:10]}{'...' if len(indices) > 10 else ''}")
    
    # Recortar usando los índices
    input_seq_downsampled = input_seq[:, indices, ...]
    output_seq_downsampled = output_seq[:, indices, ...]
    
    return input_seq_downsampled, output_seq_downsampled

#%%

base_path = os.path.dirname(__file__)

# Añadir 'scripts'
script_path = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.append(os.path.abspath(script_path))

# path directorie for saving datasets
path = os.path.join(base_path, 'datasets')
if not os.path.exists(path):
    os.makedirs(path)

from PCB_solver_tr import PCB_case_2
from Dataset_Class_convlstm import PCBDataset_convlstm

# Parámetros de configuración
n_train = 2500
n_validation = 500
n_test = 50
n_data = n_train + n_test + n_validation

# Define los índices para cada split
idx_train = slice(n_test, n_test + n_train)
idx_val = slice(n_test + n_train, n_test + n_train + n_validation)
idx_test = slice(0, n_test)

nodes_side = 13
time_sim = 650
dt = 1
T_init = 298.0

# =============== CONFIGURACIÓN DE ITERACIONES ===============
# step_intervals = list(range(1, 11))  # Lista de step_intervals a procesar
step_intervals = [1, 2, 5, 10, 20, 50, 100]
return_bc_options = [True, False]  # Lista de opciones para return_bc (NO TOCAR)

print(f"🚀 Generando datasets para:")
print(f"   Step intervals: {step_intervals}")
print(f"   Return BC options: {return_bc_options}")
print(f"   Total combinaciones: {len(step_intervals) * len(return_bc_options)}")
print("=" * 60)

# =============== GENERACIÓN DE DATOS (UNA SOLA VEZ) ===============
np.random.seed(0)

def generate_unique_cases(n_data):
    seen = set()
    Q_list, T_int_list, T_env_list = [], [], []
    while len(Q_list) < n_data:
        Q = tuple(np.random.uniform(0.5, 1.5, 4).round(6))
        T_int = tuple(np.random.uniform(270, 320, 4).round(2))
        T_env = round(float(np.random.uniform(270, 320)), 2)
        key = Q + T_int + (T_env,)
        if key not in seen:
            seen.add(key)
            Q_list.append(Q)
            T_int_list.append(T_int)
            T_env_list.append(T_env)
    return np.array(Q_list), np.array(T_int_list), np.array(T_env_list)

Q_random, T_interfaces_random, T_env_random = generate_unique_cases(n_data)

print("📊 Generando datos del solver (una sola vez)...")
time_start = time.time()

input_seq = []
output_seq = []

for i in range(n_data):
    if i % 100 == 0:
        print(f"  Generando elemento {i} | tiempo: {time.time()-time_start:.2f}s")
    
    T, _, _, _ = PCB_case_2(
        solver='transient', display=False, time=time_sim, dt=dt, T_init=T_init,
        Q_heaters=Q_random[i], T_interfaces=T_interfaces_random[i], Tenv=T_env_random[i]
    )
    T = T.reshape(-1, 13, 13)
    output_seq.append(T)

    seq_len = T.shape[0]
    input_case = []
    for t in range(seq_len):
        # Construir mapas para cada canal
        T_map = np.zeros((13, 13), dtype=np.float32)
        Q_map = np.zeros((13, 13), dtype=np.float32)
        T_env_map = np.full((13, 13), T_env_random[i], dtype=np.float32)

        # Posicionar valores de interfaces
        T_map[0, 0] = T_interfaces_random[i, 0]
        T_map[0, -1] = T_interfaces_random[i, 1]
        T_map[-1, -1] = T_interfaces_random[i, 2]
        T_map[-1, 0] = T_interfaces_random[i, 3]

        # Posicionar valores de heaters
        Q_map[6, 3] = Q_random[i, 0]
        Q_map[3, 6] = Q_random[i, 1]
        Q_map[9, 3] = Q_random[i, 2]
        Q_map[9, 9] = Q_random[i, 3]

        # Canal 6: output anterior o condición inicial
        if t == 0:
            T_init_map = np.full((13, 13), T_init, dtype=np.float32)
        else:
            T_init_map = output_seq[-1][t-1]
        
        input_t = np.stack([T_map, Q_map, T_env_map, T_init_map], axis=0)
        input_case.append(input_t)
    input_seq.append(input_case)

time_generation = time.time() - time_start
print(f"✅ Datos generados en {time_generation:.2f}s")

# Convertir a tensores
input_seq_original = torch.tensor(np.array(input_seq, dtype=np.float32))
output_seq_original = torch.tensor(np.array(output_seq, dtype=np.float32))

# Calcular estadísticas (una sola vez)
T_interfaces_mean = T_interfaces_random.mean()
T_interfaces_std = T_interfaces_random.std()
Q_heaters_mean = Q_random.mean()
Q_heaters_std = Q_random.std()
T_env_mean = T_env_random.mean()
T_env_std = T_env_random.std()

print("=" * 60)

# =============== BUCLES DE PROCESAMIENTO ===============
total_combinations = len(step_intervals) * len(return_bc_options)
current_combination = 0

for step_interval in step_intervals:
    step_start_time = time.time()  # Tiempo inicio para este step_interval
    print(f"\n🔄 Procesando step_interval = {step_interval}")
    
    # Aplicar downsample
    input_seq, output_seq = downsample_sequences(input_seq_original, output_seq_original, step_interval)
    
    # Calcular estadísticas específicas para este downsampling
    output_mean = output_seq.mean()
    output_std = output_seq.std()
    
    for return_bc in return_bc_options:
        current_combination += 1
        bc_str = "phy" if return_bc else "std"
        
        print(f"\n  📦 [{current_combination}/{total_combinations}] Creando datasets con return_bc={return_bc} ({bc_str})")
        
        # Crear dataset completo
        dataset = PCBDataset_convlstm(
            T_interfaces=input_seq[:, :, 0, ...],
            Q_heaters=input_seq[:, :, 1, ...],
            T_env=input_seq[:, :, 2, ...],
            T_outputs=output_seq,
            T_interfaces_mean=T_interfaces_mean,
            T_interfaces_std=T_interfaces_std,
            Q_heaters_mean=Q_heaters_mean,
            Q_heaters_std=Q_heaters_std,
            T_env_mean=T_env_mean,
            T_env_std=T_env_std,
            T_outputs_mean=output_mean,
            T_outputs_std=output_std,
            return_bc=return_bc
        )
        
        # Crear datasets por splits
        dataset_train = PCBDataset_convlstm(
            T_interfaces=input_seq[idx_train, :, 0, ...],
            Q_heaters=input_seq[idx_train, :, 1, ...],
            T_env=input_seq[idx_train, :, 2, ...],
            T_outputs=output_seq[idx_train],
            T_interfaces_mean=T_interfaces_mean,
            T_interfaces_std=T_interfaces_std,
            Q_heaters_mean=Q_heaters_mean,
            Q_heaters_std=Q_heaters_std,
            T_env_mean=T_env_mean,
            T_env_std=T_env_std,
            T_outputs_mean=output_mean,
            T_outputs_std=output_std,
            return_bc=return_bc
        )
        
        dataset_val = PCBDataset_convlstm(
            T_interfaces=input_seq[idx_val, :, 0, ...],
            Q_heaters=input_seq[idx_val, :, 1, ...],
            T_env=input_seq[idx_val, :, 2, ...],
            T_outputs=output_seq[idx_val],
            T_interfaces_mean=T_interfaces_mean,
            T_interfaces_std=T_interfaces_std,
            Q_heaters_mean=Q_heaters_mean,
            Q_heaters_std=Q_heaters_std,
            T_env_mean=T_env_mean,
            T_env_std=T_env_std,
            T_outputs_mean=output_mean,
            T_outputs_std=output_std,
            return_bc=return_bc
        )
        
        dataset_test = PCBDataset_convlstm(
            T_interfaces=input_seq[idx_test, :, 0, ...],
            Q_heaters=input_seq[idx_test, :, 1, ...],
            T_env=input_seq[idx_test, :, 2, ...],
            T_outputs=output_seq[idx_test],
            T_interfaces_mean=T_interfaces_mean,
            T_interfaces_std=T_interfaces_std,
            Q_heaters_mean=Q_heaters_mean,
            Q_heaters_std=Q_heaters_std,
            T_env_mean=T_env_mean,
            T_env_std=T_env_std,
            T_outputs_mean=output_mean,
            T_outputs_std=output_std,
            return_bc=return_bc
        )
        
        # Guardar datasets
        print(f"    💾 Guardando archivos...")

        if return_bc:
            # Con física - usar sufijo 'phy'
            torch.save(dataset_train, os.path.join(path, f'PCB_convlstm_dt{step_interval}_phy_6ch_dataset_train.pth'))
            torch.save(dataset_test, os.path.join(path, f'PCB_convlstm_dt{step_interval}_phy_6ch_dataset_test.pth'))
            torch.save(dataset_val, os.path.join(path, f'PCB_convlstm_dt{step_interval}_phy_6ch_dataset_val.pth'))
            torch.save(dataset, os.path.join(path, f'PCB_convlstm_dt{step_interval}_phy_6ch_dataset.pth'))
        else:
            # Sin física - sin sufijo (como antes)
            torch.save(dataset_train, os.path.join(path, f'PCB_convlstm_dt{step_interval}_6ch_dataset_train.pth'))
            torch.save(dataset_test, os.path.join(path, f'PCB_convlstm_dt{step_interval}_6ch_dataset_test.pth'))
            torch.save(dataset_val, os.path.join(path, f'PCB_convlstm_dt{step_interval}_6ch_dataset_val.pth'))
            torch.save(dataset, os.path.join(path, f'PCB_convlstm_dt{step_interval}_6ch_dataset.pth'))
                
        # Limpiar memoria
        del dataset, dataset_train, dataset_val, dataset_test
        
        print(f"    ✅ Datasets guardados para dt{step_interval}_{bc_str}")
        
     # Print del tiempo transcurrido para este step_interval
    step_elapsed_time = time.time() - step_start_time
    total_elapsed_time = time.time() - time_start
    print(f"\n⏱️  Completado dt{step_interval} en {step_elapsed_time:.2f}s | Tiempo total transcurrido: {total_elapsed_time:.2f}s")

# =============== LIMPIEZA FINAL ===============
print("\n🧹 Limpiando memoria...")
del input_seq_original, output_seq_original, input_seq, output_seq
torch.cuda.empty_cache() if torch.cuda.is_available() else None

import gc
gc.collect()

total_time = time.time() - time_start
print(f"\n🎉 ¡Completado! Tiempo total: {total_time:.2f}s")
print(f"📊 Se generaron {total_combinations} sets de datasets")