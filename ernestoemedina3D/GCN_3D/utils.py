import os
import torch
from ernestoemedina.GCN.GCN_model import GCN  # o ajusta según tu estructura

def load_model_by_name(model, model_path):
    """Carga los pesos del modelo desde un archivo .pth"""
    print(f"Intentando cargar modelo desde:\n{os.path.abspath(model_path)}")
    checkpoint = torch.load(model_path, map_location=model.device if hasattr(model, 'device') else 'cpu')
    model.load_state_dict(checkpoint)
    return model

def load_trained_model(config, device, model_filename):
    """
    Carga un modelo NNConvNet ya entrenado con los parámetros de `config`.

    Parameters:
    - config: diccionario con parámetros del modelo.
    - device: torch.device ('cuda' o 'cpu').
    - model_filename: nombre del archivo .pth guardado.

    Returns:
    - modelo cargado y listo para usar.
    """
    model = GCN(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        num_layers=config["num_layers"],
        use_dropout=config["use_dropout"],
        dropout_rate=config["dropout_rate"],
        use_batchnorm=config["use_batchnorm"],
        use_residual=config["use_residual"]
    ).to(device)

    model_path = os.path.join("saved_models", model_filename)  # ajusta la carpeta si es necesario
    model = load_model_by_name(model, model_path)
    return model
