import torch
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from Dataset_Class import PCBDataset
import os

# === Ruta del dataset .pth ===
dataset_path = os.path.join("Datasets", "PCB_dataset.pth")
dataset = torch.load(dataset_path)


# === Acumular todas las features por nodo (Qi, T, T_env) ===
all_features = []

for item in dataset:
    input_tensor = item[0]  # Tensor de shape [3, 13, 13]
    Qi = input_tensor[0].reshape(-1, 1).numpy()
    T = input_tensor[1].reshape(-1, 1).numpy()
    T_env = input_tensor[2].reshape(-1, 1).numpy()
    
    # Cada fila es un nodo
    features = np.hstack([Qi, T, T_env])
    all_features.append(features)

# === Unir todos los nodos de todos los grafos ===
X = np.vstack(all_features)
df = pd.DataFrame(X, columns=['Qi', 'T', 'T_env'])

# === Añadir constante y calcular VIF ===
X_const = add_constant(df)
vif_df = pd.DataFrame()
vif_df['Variable'] = X_const.columns
vif_df['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

print(vif_df)
