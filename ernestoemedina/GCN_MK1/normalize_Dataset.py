import torch
import os
from Dataset_Class import PCBDataset

# Ruta del archivo original
dir_path = os.path.dirname(os.path.realpath(__file__))
original_path = os.path.join(dir_path, "Datasets", "PCB_dataset.pth")
output_path = os.path.join(dir_path, "Datasets", "PCB_Dataset_Norm.pth")
info_path = os.path.join(dir_path, "Datasets", "normalization_info.pth")

# Cargar dataset original
dataset = torch.load(original_path)

# Listas para los nuevos tensores normalizados
T_interfaces_norm = []
Q_heaters_norm = []
T_env_norm = []
T_outputs_real = []

# Desestandarizar todo el dataset
for i in range(len(dataset)):
    T_interfaces_real = dataset.denormalize_T_interfaces(dataset.T_interfaces[i])
    Q_heaters_real = dataset.denormalize_Q_heaters(dataset.Q_heaters[i])
    T_env_real = dataset.denormalize_T_env(dataset.T_env[i])
    T_output_real = dataset.denormalize_output(dataset.outputs[i])

    T_interfaces_norm.append(T_interfaces_real)
    Q_heaters_norm.append(Q_heaters_real)
    T_env_norm.append(T_env_real)
    T_outputs_real.append(T_output_real)

# Convertir a tensores
T_interfaces_real = torch.stack(T_interfaces_norm)
Q_heaters_real = torch.stack(Q_heaters_norm)
T_env_real = torch.stack(T_env_norm)
T_outputs_real = torch.stack(T_outputs_real)

# Calcular máximos por tipo
max_T_interfaces = torch.max(T_interfaces_real)
max_Q_heaters = torch.max(Q_heaters_real)
max_T_env = torch.max(T_env_real)
max_T_outputs = torch.max(T_outputs_real)

# Normalizar
T_interfaces_normalized = T_interfaces_real / max_T_interfaces
Q_heaters_normalized = Q_heaters_real / max_Q_heaters
T_env_normalized = T_env_real / max_T_env
T_outputs_normalized = T_outputs_real / max_T_outputs

# Crear dataset normalizado (con std=1 ficticio solo para evitar errores si se llama a .denormalize_output)
dataset_normalizado = PCBDataset(
    T_interfaces_normalized,
    Q_heaters_normalized,
    T_env_normalized,
    T_outputs_normalized,
    torch.tensor(0.0), torch.tensor(1.0),
    torch.tensor(0.0), torch.tensor(1.0),
    torch.tensor(0.0), torch.tensor(1.0),
    torch.tensor(0.0), torch.tensor(1.0)  # output: mean=0, std=1 (no se usa más)
)

# Guardar el dataset normalizado
torch.save(dataset_normalizado, output_path)

# Guardar los valores máximos
normalization_info = {
    "max_T_interfaces": max_T_interfaces,
    "max_Q_heaters": max_Q_heaters,
    "max_T_env": max_T_env,
    "max_T_outputs": max_T_outputs
}
torch.save(normalization_info, info_path)

# Mensajes de éxito
print(" Dataset normalizado guardado como:", output_path)
print(f"Max T_interfaces: {max_T_interfaces.item():.4f}")
print(f"Max Q_heaters: {max_Q_heaters.item():.4f}")
print(f"Max T_env: {max_T_env.item():.4f}")
print(f"Max T_outputs: {max_T_outputs.item():.4f}")
