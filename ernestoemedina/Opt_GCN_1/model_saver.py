import torch
import os

def save_model(model, num_layers, hidden_dim, epochs, learning_rate, 
               test_mse, test_accuracy, best_model_state=None, 
               save_best_model_to_file=True, device='cpu'):
    """
    Guarda un modelo GCN entrenado, incluyendo su configuración y rendimiento en el nombre del archivo.
    
    Args:
        model (torch.nn.Module): El modelo GCN entrenado.
        num_layers (int): Número de capas del modelo.
        hidden_dim (int): Dimensión de capas ocultas.
        epochs (int): Número de épocas de entrenamiento.
        learning_rate (float): Tasa de aprendizaje utilizada.
        test_mse (float): Error cuadrático medio en el conjunto de test.
        test_accuracy (float): Precisión dentro del umbral en test (%).
        best_model_state (dict): Estado del mejor modelo (opcional).
        save_best_model_to_file (bool): Si True, guarda el archivo .pth.
        device (str): 'cpu' o 'cuda', define cómo se guarda el modelo.
    """
    # Crear nombre de archivo seguro e informativo
    error_kelvin = round(test_mse ** 0.5, 4)
    accuracy = round(test_accuracy, 2)

    file_name = (
        f"GCN_Layers-{num_layers}_HDim-{hidden_dim}_Epochs-{epochs}"
        f"_Lr-{learning_rate}_ErrorK-{error_kelvin}_Acc-{accuracy}.pth"
    ).replace(":", "-").replace("%", "")

    # Carpeta de salida
    os.makedirs("models", exist_ok=True)
    file_path = os.path.join("models", file_name)

    # Determinar qué estado guardar
    model_state = best_model_state if best_model_state is not None else model.state_dict()

    # Convertir a CPU si es necesario
    if device == 'cuda':
        torch.save(model_state, file_path)
    else:
        model_state_cpu = {k: v.cpu() for k, v in model_state.items()}
        torch.save(model_state_cpu, file_path)

    if save_best_model_to_file:
        print(f"Modelo guardado como: {file_path}")
    else:
        print("ℹModelo actualizado en memoria pero no se guardó como archivo.")
