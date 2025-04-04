# model_saver.py

import torch
import numpy as np

def save_model(model, num_layers, hidden_dim, epochs, learning_rate, test_mse, test_accuracy):
    """
    Guarda un modelo GCN con un nombre que incluye su estructura, hiperparámetros y rendimiento.
    
    Args:
        model (torch.nn.Module): El modelo GCN entrenado.
        num_layers (int): Número de capas en el modelo GCN.
        hidden_dim (int): Dimensión de las capas ocultas.
        epochs (int): Número total de épocas de entrenamiento.
        learning_rate (float): Tasa de aprendizaje utilizada durante el entrenamiento.
        test_mse (float): Error medio cuadrático (MSE) en el conjunto de prueba.
        test_accuracy (float): Porcentaje de aciertos dentro del umbral de error.
    """
    error_kelvin = round(np.sqrt(test_mse), 4)  # Convertimos el MSE a Kelvin tomando la raíz cuadrada
    accuracy = round(test_accuracy, 2)  # Redondeamos el porcentaje de aciertos a 2 decimales
    
    # Crear un nombre de archivo seguro para cualquier sistema operativo
    file_name = (f"GCN_Layers-{num_layers}_HDim-{hidden_dim}_Epochs-{epochs}_Lr-{learning_rate}"
                 f"_ErrorK-{error_kelvin}_Acc-{accuracy}.pth").replace(":", "-").replace("%", "")
    
    # Guardar el modelo
    torch.save(model.state_dict(), file_name)
    print(f"Modelo guardado correctamente como: {file_name}")
