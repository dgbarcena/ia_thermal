import torch
from torch.utils.data import Dataset, TensorDataset
import os

#%%

def load_dataset_convlstm(base_path='.', folder='datasets', dataset_type=None, solver='transient'):
    """
    Carga un dataset .pth desde una carpeta, por defecto el dataset base completo.
    
    Parámetros:
    - base_path: ruta base (por defecto, carpeta actual)
    - folder: subcarpeta donde están los archivos (por defecto, 'datasets')
    - dataset_type: 'train', 'test', 'val' o None (por defecto carga el dataset base completo)
    - solver: 'transient' o 'steady' (por defecto 'transient')
    """
    if dataset_type is None:
        filename = f'PCB_convlstm_6ch_{solver}_dataset.pth'
    else:
        valid_types = ['train', 'test', 'val']
        if dataset_type not in valid_types:
            raise ValueError(f"Tipo de dataset inválido. Usa uno de: {valid_types} o None para el dataset base.")
        filename = f"PCB_convlstm_6ch_{solver}_dataset_{dataset_type}.pth"

    full_path = os.path.join(base_path, folder, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {full_path}")

    print(f"✅ Cargando ConvLSTM {solver} dataset {'base' if dataset_type is None else dataset_type} desde: {full_path}")
    return torch.load(full_path)


#%%
# Dataset class for the PCB data
# This class is used to create a dataset for the PCB data. It takes the input and output data, normalizes it, and returns it in a format that can be used by the model.
class PCBDataset_convlstm(Dataset):
    def __init__(self,T_interfaces:torch.tensor,Q_heaters:torch.tensor,T_env:torch.tensor,T_outputs:torch.tensor,
                 T_interfaces_mean:torch.tensor,T_interfaces_std:torch.tensor,Q_heaters_mean:torch.tensor,
                 Q_heaters_std:torch.tensor,T_env_mean:torch.tensor,T_env_std:torch.tensor,T_outputs_mean:torch.tensor,
                 T_outputs_std:torch.tensor,
                 return_bc:bool = False):
        
        """
        Espera tensores de entrada con shape:
        - T_interfaces: (N, T, 13, 13)
        - Q_heaters: (N, T, 13, 13)
        - T_env: (N, T, 13, 13)
        - T_outputs: (N, T, 13, 13)
        """
        
        self.return_bc = return_bc
        
        mask_interfaces = (T_interfaces != 0).float()
        # print(f"Mask interfaces: {mask_interfaces}")  # (N, T, 13, 13)
        mask_heaters = (Q_heaters != 0).float()
        # print(f"Mask heaters: {mask_heaters}")  # (N, T, 13, 13)
        
        self.mask_interfaces = mask_interfaces
        self.mask_heaters = mask_heaters
        
        self.T_interfaces_mean = T_interfaces_mean
        self.T_interfaces_std = T_interfaces_std
        self.Q_heaters_mean = Q_heaters_mean
        self.Q_heaters_std = Q_heaters_std
        self.T_outputs_mean = T_outputs_mean
        self.T_outputs_std = T_outputs_std
        self.T_env_mean = T_env_mean
        self.T_env_std = T_env_std

        # self.inputs = torch.empty([T_interfaces.shape[0],4,13,13])
        # self.T_interfaces = (T_interfaces-T_interfaces_mean)/T_interfaces_std
        # self.Q_heaters = (Q_heaters-Q_heaters_mean)/Q_heaters_std
        # self.T_env= (T_env-T_env_mean)/T_env_std
        # self.outputs = (T_outputs-T_outputs_mean)/T_outputs_std
        # self.inputs[:,0,:,:] = self.T_interfaces
        # self.inputs[:,1,:,:] = self.Q_heaters
        # self.inputs[:,2,:,:] = self.T_env
        
        # T_init = 298.0 # default value
        # T_init_norm = (torch.full_like(self.outputs[0], T_init) - T_outputs_mean) / T_outputs_std
        # first_step = T_init_norm.unsqueeze(0)  # (1, 13, 13)
        # self.inputs[:,3,:,:] = torch.cat([first_step, self.outputs[:-1]], dim=0)
        
        # print(f"Input shape: {T_interfaces.shape}")  # (N, T, 13, 13)
        N, T, _, _ = T_interfaces.shape
        self.inputs = []
        
        T_current = torch.zeros_like(T_outputs)
        T_current[:, 0] = 298.0  # Condición inicial para t=0 (puedes normalizar si lo necesitas)
        T_current[:, 1:] = T_outputs[:, :-1]  # Para t>0, el output del paso anterior
        
        
        # Normalizar
        self.T_interfaces = (T_interfaces - T_interfaces_mean) / T_interfaces_std
        self.Q_heaters = (Q_heaters - Q_heaters_mean) / Q_heaters_std
        self.T_env = (T_env - T_env_mean) / T_env_std
        self.outputs = (T_outputs - T_outputs_mean) / T_outputs_std
        self.T_current = (T_current - T_outputs_mean) / T_outputs_std
        
        self.T_interfaces = self.T_interfaces * mask_interfaces # aplicar máscara
        self.Q_heaters = self.Q_heaters * mask_heaters # aplicar máscara
        
        for n in range(N):
            seq_inputs = []
            for t in range(T):
                # Usar directamente los mapas normalizados de cada instante
                T_map = self.T_interfaces[n, t]   # (13, 13)
                Q_map = self.Q_heaters[n, t]      # (13, 13)
                T_env_map = self.T_env[n, t]      # (13, 13)
                T_current_map = self.T_current[n, t]  # (13, 13)
                mask_interfaces_map = self.mask_interfaces[n, t]  # (13, 13)
                mask_heaters_map = self.mask_heaters[n, t]  # (13, 13)
        
                # Stack canales
                input_t = torch.stack([T_map, Q_map, T_env_map, mask_interfaces_map, mask_heaters_map, T_current_map], dim=0)  # (6, 13, 13)
                seq_inputs.append(input_t)
            seq_inputs = torch.stack(seq_inputs, dim=0)  # (T, 6, 13, 13)
            self.inputs.append(seq_inputs)
        self.inputs = torch.stack(self.inputs, dim=0)  # (N, T, 6, 13, 13)
        # print(f"Input shape: {self.inputs.shape}")  # (N, T, 4, 13, 13)
        # self.outputs ya tiene shape (N, T, 13, 13)

    def denormalize_T_interfaces(self,x):
        tensor_device = x.device
        mean = self.T_interfaces_mean.to(tensor_device)
        std = self.T_interfaces_std.to(tensor_device)
        return x*std+mean
    
    def denormalize_input(self, x):
        """
        Desnormaliza el input completo (6 canales) usando las funciones individuales.
        Entrada: x de forma (..., 4, 13, 13)
        Salida: tensor desnormalizado con los 6 canales
        """
        device = x.device
        x_denorm = torch.empty_like(x)

        x_denorm[..., 0, :, :] = self.denormalize_T_interfaces(x[..., 0, :, :].to(device))
        x_denorm[..., 1, :, :] = self.denormalize_Q_heaters(x[..., 1, :, :].to(device))
        x_denorm[..., 2, :, :] = self.denormalize_T_env(x[..., 2, :, :].to(device))
        x_denorm[..., 3, :, :] = x[..., 3, :, :]  # Mask interfaces (no desnormalizar)
        x_denorm[..., 4, :, :] = x[..., 4, :, :]  # Mask heaters (no desnormalizar)
        x_denorm[..., 5, :, :] = self.denormalize_output(x[..., 5, :, :].to(device))

        return x_denorm
    
    def create_input_from_values(self, Q_heaters, T_interfaces, T_env, T_seq, sequence_length=1001, autorregress=False):
        """
        Crea un input normalizado de forma (1, T, 6, 13, 13) a partir de:
        - Q_heaters: np.array de shape (4,)
        - T_interfaces: np.array de shape (4,)
        - T_env: float o escalar
        - T_seq: np.array de shape (T, 13, 13)
        - autorregress: bool, si True, se calcula solo el primer paso de la secuencia
        - sequence_length: int, longitud de la secuencia (por defecto 1001)

        Devuelve: tensor (1, T, 6, 13, 13)
        """
        nodes_side = 13

        # Convertir a tensores
        Q_heaters = torch.tensor(Q_heaters, dtype=torch.float32)
        T_interfaces = torch.tensor(T_interfaces, dtype=torch.float32)
        T_env = torch.tensor(T_env, dtype=torch.float32)
        T_seq = torch.tensor(T_seq, dtype=torch.float32)
        
        # Normalizar
        Q_norm = (Q_heaters - self.Q_heaters_mean) / self.Q_heaters_std
        T_int_norm = (T_interfaces - self.T_interfaces_mean) / self.T_interfaces_std
        T_env_norm = (T_env - self.T_env_mean) / self.T_env_std
        T_seq_norm = (T_seq - self.T_outputs_mean) / self.T_outputs_std

        # Crear mapas (6, 13, 13)
        Q_map = torch.zeros((13, 13))
        T_map = torch.zeros((13, 13))
        T_env_map = torch.full((13, 13), T_env_norm)

        # Posicionar los valores
        T_map[0, 0] = T_int_norm[0]
        T_map[0, -1] = T_int_norm[1]
        T_map[-1, -1] = T_int_norm[2]
        T_map[-1, 0] = T_int_norm[3]
        mask_interfaces_map = (T_map != 0).float() # máscara de interfaces

        Q_map[6, 3] = Q_norm[0]
        Q_map[3, 6] = Q_norm[1]
        Q_map[9, 3] = Q_norm[2]
        Q_map[9, 9] = Q_norm[3]
        mask_heaters_map = (Q_map != 0).float() # máscara de heaters

        # Stack y replicar en el tiempo
        if autorregress:
            print('Calculando input tensor para predicción, solo el primer paso de la secuencia')
            input_bc = torch.stack([T_map, Q_map, T_env_map, mask_interfaces_map, mask_heaters_map], dim=0)
            input_tensor = torch.zeros((1, 1, 6, nodes_side, nodes_side), dtype=torch.float32) # se añade una dimensión de tiempo
            input_tensor[:, :, 0:5, :, :] = input_bc[0:5, :, :]
            input_tensor[:, :, 5, :, :] = T_seq_norm[0, :, :]
            print(f"Input tensor shape: {input_tensor.shape}")
        else:
            print('Calculando input tensor para entrenamiento, toda la secuencia con datos del solver')
            input_bc = torch.stack([T_map, Q_map, T_env_map, mask_interfaces_map, mask_heaters_map], dim=0)  # (, 13, 13)
            input_tensor = torch.zeros((1, sequence_length, 6, nodes_side, nodes_side), dtype=torch.float32)
            input_tensor[:, :, 0:5, :, :] = input_bc[0:5, :, :] 
            input_tensor[:, :, 5, :, :] = T_seq_norm  # (T, 13, 13)
            print(f"Input tensor shape: {input_tensor.shape}")  # (1, T, 4, 13, 13)

        return input_tensor.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def denormalize_T_env(self,x):
        tensor_device = x.device
        mean = self.T_env_mean.to(tensor_device)
        std = self.T_env_std.to(tensor_device)
        return x*std+mean
    
    def denormalize_Q_heaters(self,x):
        tensor_device = x.device
        mean = self.Q_heaters_mean.to(tensor_device)
        std = self.Q_heaters_std.to(tensor_device)
        return x*std+mean

    def denormalize_output(self,x):
        tensor_device = x.device
        mean = self.T_outputs_mean.to(tensor_device)
        std = self.T_outputs_std.to(tensor_device)
        return x*std+mean

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]
        if output_data.ndim == 2 and output_data.shape[-1] == 169:
            output_data = output_data.view(-1, 13, 13)

        if self.return_bc:
            t_int = self.denormalize_T_interfaces(self.T_interfaces[idx])
            q_heat = self.denormalize_Q_heaters(self.Q_heaters[idx])
            t_env = self.denormalize_T_env(self.T_env[idx])
            return input_data, output_data, q_heat, t_int, t_env

        return input_data, output_data
    
    def to_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.inputs = self.inputs.to(device)
        self.outputs = self.outputs.to(device)
        
    def extract_bc_values(self, q_map, t_int_map, t_env_map):
        # Heaters: extraer los 4 valores de entrada
        q = torch.tensor([
            q_map[6, 3],
            q_map[3, 6],
            q_map[9, 3],
            q_map[9, 9]
        ])

        # Interfaces: 4 esquinas
        t_int = torch.tensor([
            t_int_map[0, 0],
            t_int_map[0, -1],
            t_int_map[-1, -1],
            t_int_map[-1, 0]
        ])

        # T_env: es constante
        t_env = torch.tensor([t_env_map[0, 0]])

        return q, t_int, t_env

        
    
    
#%%    

# -----------------------------------------------------------------------------
# TrimmedDataset: Wrapper para recorte eficiente de muestras y tiempo
# -----------------------------------------------------------------------------
class TrimmedDataset_convlstm(Dataset):
    def __init__(self, base_dataset: Dataset, max_samples: int = None,
                 time_steps_input: int = None, time_steps_output: int = None,
                 solver: str = 'transient'):
        """
        Dataset recortado para datos estacionarios o transitorios.

        Args:
            base_dataset (Dataset): dataset base
            max_samples (int): número máximo de muestras a devolver
            time_steps_input (int): número de pasos temporales de entrada (solo si solver='transient')
            time_steps_output (int): número de pasos temporales de salida (solo si solver='transient')
            solver (str): 'steady' o 'transient'
        """
        self.base_dataset = base_dataset
        self.max_samples = max_samples or len(base_dataset)
        self.time_steps_input = time_steps_input
        self.time_steps_output = time_steps_output
        assert solver in ['steady', 'transient'], "solver must be 'steady' or 'transient'"
        self.solver = solver

    def __len__(self):
        return min(self.max_samples, len(self.base_dataset))

    def __getitem__(self, idx):
        if hasattr(self.base_dataset, 'return_bc') and self.base_dataset.return_bc:
            data = self.base_dataset[idx]
        else:
            original_return_bc = getattr(self.base_dataset, 'return_bc', False)
            self.base_dataset.return_bc = False
            data = self.base_dataset[idx]
            self.base_dataset.return_bc = original_return_bc
    
        if isinstance(data, tuple) and len(data) > 2:
            input_data, output_data, *bcs = data
        else:
            input_data, output_data = data
            bcs = []
    
        if self.solver == 'transient':
            # Recortar la dimensión temporal (primer eje) tanto en input como en output
            if self.time_steps_output is not None and output_data.shape[0] > self.time_steps_output:
                input_data = input_data[:self.time_steps_output]
                output_data = output_data[:self.time_steps_output]
            
        # Asegurar que el output tenga una dimensión de canal
        if output_data.ndim == 2:
            output_data = output_data.unsqueeze(0).unsqueeze(1)
        elif output_data.ndim == 3:
            output_data = output_data.unsqueeze(1)
    
        return (input_data, output_data, *bcs) if bcs else (input_data, output_data)
    

# -----------------------------------------------------------------------------
# load_trimmed_dataset: Carga y recorta dataset desde .pth
# -----------------------------------------------------------------------------
def load_trimmed_dataset_convlstm(base_path='.', folder='datasets', dataset_type=None,
                         max_samples=None, time_steps_input=None, time_steps_output=None,
                         to_device=False, solver='transient'):
    """
    Carga un dataset base y lo encapsula en un TrimmedDataset, compatible con casos transitorios y estacionarios.

    Args:
        base_path (str): ruta base donde está la carpeta del dataset
        folder (str): subcarpeta donde están los archivos .pth
        dataset_type (str): tipo de dataset ('train', 'test', 'val') o None
        max_samples (int): número máximo de muestras a cargar
        time_steps_input (int): número de pasos temporales de entrada (si es transitorio)
        time_steps_output (int): número de pasos temporales de salida (si es transitorio)
        to_device (bool): si se debe mover a CUDA (si está disponible)
        solver (str): 'transient' o 'steady'

    Returns:
        TrimmedDataset
    """
    if dataset_type is None:
        filename = f'PCB_convlstm_6ch_{solver}_dataset.pth'
    else:
        valid_types = ['train', 'test', 'val']
        if dataset_type not in valid_types:
            raise ValueError(f"Tipo de dataset inválido. Usa uno de: {valid_types} o None.")
        filename = f"PCB_convlstm_6ch_{solver}_dataset_{dataset_type}.pth"

    full_path = os.path.join(base_path, folder, filename)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"❌ No se encontró el archivo: {full_path}")

    print(f"✅ Cargando dataset {'base' if dataset_type is None else dataset_type} desde: {full_path}")
    base_dataset = torch.load(full_path)

    if to_device and hasattr(base_dataset, 'to_device'):
        base_dataset.to_device()
        print("📦 Dataset movido a:", "CUDA" if torch.cuda.is_available() else "CPU")

    return TrimmedDataset_convlstm(base_dataset,
                          max_samples=max_samples,
                          time_steps_input=time_steps_input,
                          time_steps_output=time_steps_output,
                          solver=solver)


# -----------------------------------------------------------------------------
# prepare_data_for_convlstm: ajusta input/output para entrenamiento ConvLSTM
# -----------------------------------------------------------------------------
def prepare_data_for_convlstm(dataset, device='cuda', with_bc=False):
    """
    Prepara input, output y (opcionalmente) condiciones de contorno para entrenar un modelo ConvLSTM.

    Parámetros:
    - dataset: instancia tipo TrimmedDataset
    - device: dispositivo donde enviar los tensores (default: 'cuda')
    - with_bc: si True, también devuelve bc_all concatenando [Q, T_int, T_env]

    Returns:
    - x: (N, T, C, H, W)
    - y: (N, T, 1, H, W)
    - (opcional) bc_all: (N, 9)
    """
    x_list, y_list, bc_list = [], [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        x, y = sample[:2]
        x_list.append(x)
        y_list.append(y)

        if with_bc and len(sample) == 5:
            q, t_int, t_env = sample[2:]

            # Asegurar que t_env es escalar
            if t_env.ndim == 2:
                t_env = t_env[0, 0]

            bc = torch.cat([q, t_int, t_env.unsqueeze(0)], dim=0)  # (9,)
            bc_list.append(bc)

    x = torch.stack(x_list)  # (N, C, H, W)
    y = torch.stack(y_list)  # (N, T, H, W) o (N, H, W)

    # Ajustar dimensiones
    if y.ndim == 4:
        y = y.unsqueeze(2)  # (N, T, 1, H, W)
    elif y.ndim != 5:
        raise ValueError(f"[ERROR] Forma inesperada para y: {y.shape}")

    T = y.shape[1]
    x = x.unsqueeze(1).repeat(1, T, 1, 1, 1)  # (N, T, C, H, W)

    x = x.to(device)
    y = y.to(device)

    if with_bc:
        if not bc_list:
            raise RuntimeError("⚠️ `with_bc=True`, pero el dataset no contiene condiciones de contorno.")
        bc_all = torch.stack(bc_list).to(device)  # (N, 9)
        # print(f"[DEBUG] x: {x.shape}, y: {y.shape}, bc_all: {bc_all.shape}")
        return x, y, bc_all

    # print(f"[DEBUG] x: {x.shape}, y: {y.shape}")
    return x, y


# -----------------------------------------------------------------------------
# Función auxiliar para preparar datos (con repetición temporal)
# -----------------------------------------------------------------------------
def prepare_data_with_bc(dataset, device):
    inputs, outputs, q_list, t_int_list, t_env_list = [], [], [], [], []

    for i in range(len(dataset)):
        x, y, q_raw, t_int_raw, t_env_raw = dataset[i]

        # Recoger x e y
        inputs.append(x)
        outputs.append(y)

        # Reducir mapas espaciales a vectores (4,) y (1,)
        # Para Q y T_interfaces: extraer los 4 valores no-cero
        q_vals = q_raw[q_raw != 0].flatten()
        t_int_vals = t_int_raw[t_int_raw != 0].flatten()

        # Para T_env: usar el valor medio o directamente uno (es constante)
        t_env_val = t_env_raw[0, 0] if t_env_raw.ndim == 2 else t_env_raw

        q_list.append(q_vals)
        t_int_list.append(t_int_vals)
        t_env_list.append(torch.tensor([t_env_val], dtype=torch.float32))

    # Convertir a tensores
    x = torch.stack(inputs)  # (B, C, H, W)
    y = torch.stack(outputs)  # (B, T, 1, H, W) o (B, T, H, W)
    q = torch.stack(q_list)  # (B, 4)
    t_int = torch.stack(t_int_list)  # (B, 4)
    t_env = torch.stack(t_env_list)  # (B, 1)

    # Asegurar canal en y
    if y.ndim == 4:
        y = y.unsqueeze(2)  # (B, T, 1, H, W)

    # Repetir input en tiempo
    T = y.shape[1]
    x = x.unsqueeze(1).repeat(1, T, 1, 1, 1)  # (B, T, C, H, W)

    # Concatenar condiciones
    bc_all = torch.cat([q, t_int, t_env], dim=1)  # (B, 9)

    return TensorDataset(x.to(device), y.to(device), bc_all.to(device))


# class FlattenedTemporalDataset(Dataset):
#     def __init__(self, base_dataset):
#         # Accede al dataset base si es un wrapper
#         if hasattr(base_dataset, 'base_dataset'):
#             self.base_dataset = base_dataset.base_dataset
#         else:
#             self.base_dataset = base_dataset

#         self.inputs = self.base_dataset.inputs  # (N, T, C, H, W)
#         self.outputs = self.base_dataset.outputs  # (N, T, H, W)
#         N, T = self.inputs.shape[:2]
#         self.N = N
#         self.T = T

#     def __len__(self):
#         return self.N * self.T

#     def __getitem__(self, idx):
#         n = idx // self.T
#         t = idx % self.T
#         x = self.inputs[n, t]    # (C, H, W)
#         y = self.outputs[n, t]   # (H, W)
#         return x, y
    
    
def get_base_with_inputs_outputs(ds):
    # Busca recursivamente el dataset que tiene los atributos .inputs y .outputs
    while not (hasattr(ds, "inputs") and hasattr(ds, "outputs")):
        if hasattr(ds, "base_dataset"):
            ds = ds.base_dataset
        elif hasattr(ds, "base"):
            ds = ds.base
        else:
            raise AttributeError("No se encontró un dataset con .inputs y .outputs")
    return ds

class PCBDatasetSliding(Dataset):
    """
    Envuelve un PCBDataset_convlstm y lo presenta como
    ventanas deslizantes Many→One.
    """
    def __init__(self, base_ds, window_size=5, stride=1):
        # Buscar el dataset real con .inputs y .outputs
        self.base = get_base_with_inputs_outputs(base_ds)
        self.K = window_size
        self.stride = stride

        N, T, _, _, _ = self.base.inputs.shape  # (N,T,C,H,W)
        self.samples = [
            (n, t)
            for n in range(N)
            for t in range(self.K, T, stride)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        n, t = self.samples[idx]
        K = self.K
        X = self.base.inputs[n, t-K:t]           # (K, 6, 13, 13)
        y = self.base.outputs[n, t].unsqueeze(0) # (1, 13, 13)
        return X, y