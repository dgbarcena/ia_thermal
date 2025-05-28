import numpy as np
import os
import sys
import time
import torch

def append_and_save(dataset_new, filename):
    """
    Añade datos nuevos a un dataset .pth existente, comprobando que la longitud temporal coincide.
    Si el archivo no existe, lo crea.
    Al finalizar, imprime la longitud final del dataset.
    """
    full_path = os.path.join(path, filename)
    if os.path.exists(full_path):
        dataset_old = torch.load(full_path, weights_only=False)
        # Comprobar que la longitud temporal coincide
        if dataset_old.inputs.shape[1] != dataset_new.inputs.shape[1]:
            raise ValueError(
                f"La longitud temporal (shape[1]) no coincide: "
                f"existente={dataset_old.inputs.shape[1]}, nuevo={dataset_new.inputs.shape[1]}"
            )
        # Concatenar datos
        dataset_old.inputs = torch.cat([dataset_old.inputs, dataset_new.inputs], dim=0)
        dataset_old.outputs = torch.cat([dataset_old.outputs, dataset_new.outputs], dim=0)
        dataset_to_save = dataset_old
    else:
        dataset_to_save = dataset_new
    torch.save(dataset_to_save, full_path)
    print(f"✅ Guardado '{filename}'. Longitud final del dataset: {len(dataset_to_save)} muestras.")

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

solver = 'transient' # steady or transient

n_train = 2500
n_validation = 500
n_test = 50
n_data = n_train+n_test+n_validation  

# Define los índices para cada split
idx_train = slice(n_test, n_test + n_train)
idx_val = slice(n_test + n_train, n_test + n_train + n_validation)
idx_test = slice(0, n_test)

nodes_side = 13
time_sim = 100
dt = 1
T_init = 298.0

input_seq = []
output_seq = []

np.random.seed(0)

Q_random = np.random.uniform(0.5, 1.0, (n_data, 4))
T_interfaces_random = np.random.uniform(280, 310, (n_data, 4))
T_env_random = np.random.uniform(280, 310, n_data)

time_start = time.time()
    
input_seq = []
output_seq = []

for i in range(n_data):
    if i % 100 == 0:
        print("Generating element number: ", i, " | time: ", time.time()-time_start)
    T, _, _, _ = PCB_case_2(
        solver=solver, display=False, time=time_sim, dt=dt, T_init=T_init,
        Q_heaters=Q_random[i], T_interfaces=T_interfaces_random[i], Tenv=T_env_random[i]
    )  # T.shape = (T, 13, 13)
    T = T.reshape(-1, 13, 13)  # (seq_len, 13, 13)
    output_seq.append(T)  # (seq_len, 13, 13)

    seq_len = T.shape[0]
    input_case = []
    for t in range(seq_len):
        # Construye los mapas para cada canal
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
        # Stack canales
        input_t = np.stack([T_map, Q_map, T_env_map, T_init_map], axis=0)  # (4, 13, 13)
        input_case.append(input_t)
    input_seq.append(input_case)  # (seq_len, 4, 13, 13)
    
time_end = time.time()
time_generation_data = time_end-time_start
print("Time to generate the data: ",time_generation_data)

input_seq = np.array(input_seq, dtype=np.float32)   # (n_data, seq_len, 4, 13, 13)
output_seq = np.array(output_seq, dtype=np.float32) # (n_data, seq_len, 13, 13)

# Luego sigue igual:
input_seq = torch.tensor(input_seq, dtype=torch.float32)
output_seq = torch.tensor(output_seq, dtype=torch.float32)

# transform the lists into numpy arrays
input_seq = np.array(input_seq)   # (n_data, seq_len, 9)
output_seq = np.array(output_seq) # (n_data, seq_len, 13, 13)

if solver == 'transient':
    output_seq = output_seq.reshape(output_seq.shape[0], output_seq.shape[1], nodes_side,nodes_side) # reshaping the data grid-shape
elif solver == 'steady':
    output_seq = output_seq.reshape(output_seq.shape[0], nodes_side, nodes_side) # reshaping the data grid-shape
else:
    raise ValueError("Solver must be 'transient' or 'steady'")

input_seq = torch.tensor(input_seq, dtype=torch.float32)
output_seq = torch.tensor(output_seq, dtype=torch.float32)

# calculate averages and standard deviations
T_interfaces_mean = T_interfaces_random.mean() 
T_interfaces_std = T_interfaces_random.std()
Q_heaters_mean = Q_random.mean()
Q_heaters_std = Q_random.std()
T_env_mean = T_env_random.mean()
T_env_std = T_env_random.std()
output_mean = output_seq.mean() 
output_std = output_seq.std()


dataset = PCBDataset_convlstm(
    T_interfaces=input_seq[:, :, 0, ...],      # (n_data, seq_len, 13, 13)
    Q_heaters=input_seq[:, :, 1, ...],         # (n_data, seq_len, 13, 13)
    T_env=input_seq[:, :, 2, ...],             # (n_data, seq_len, 13, 13)
    T_outputs=output_seq,                      # (n_data, seq_len, 13, 13)
    T_interfaces_mean=T_interfaces_mean,
    T_interfaces_std=T_interfaces_std,
    Q_heaters_mean=Q_heaters_mean,
    Q_heaters_std=Q_heaters_std,
    T_env_mean=T_env_mean,
    T_env_std=T_env_std,
    T_outputs_mean=output_mean,
    T_outputs_std=output_std, 
    return_bc=True
)

# Dataset de entrenamiento
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
    return_bc=True
)

# Dataset de validación
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
    return_bc=True
)

# Dataset de test
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
    return_bc=True
)

# path directorie for saving datasets
path = os.path.join(base_path,'datasets')
if not os.path.exists(path):
    os.makedirs(path)
    
append_and_save(dataset_train, 'PCB_convlstm_phy_6ch_transient_dataset_train.pth')
append_and_save(dataset_val, 'PCB_convlstm_phy_6ch_transient_dataset_val.pth')
append_and_save(dataset_test, 'PCB_convlstm_phy_6ch_transient_dataset_test.pth')
append_and_save(dataset, 'PCB_convlstm_phy_6ch_transient_dataset.pth')

time_end = time.time()
print("Total time to generate and save the dataset: ", time_end - time_start)


# # %%
# print("input_seq shape:", input_seq.shape)     # Esperado: (n_data, seq_len, 9)
# print("output_seq shape:", output_seq.shape)   # Esperado: (n_data, seq_len, 13, 13)

# x, y = dataset[0]
# print("x shape:", x.shape)  # Esperado: (seq_len, 4, 13, 13)
# print("y shape:", y.shape)  # Esperado: (seq_len, 13, 13)

# from torch.utils.data import DataLoader

# loader = DataLoader(dataset, batch_size=2)
# for x_batch, y_batch in loader:
#     print(x_batch.shape)  # (batch, seq_len, 4, 13, 13)
#     print(y_batch.shape)  # (batch, seq_len, 13, 13)
#     break