import torch
from torch_geometric.data import Data
import glob

def generate_edge_index(grid_size):
    """
    Genera la matriz de adyacencia (edge_index) para una malla 2D cuadrada.
    Conectividad de 4 vecinos (right/left/top/bottom).
    """
    edge_index = []

    for y in range(grid_size):
        for x in range(grid_size):
            idx = y * grid_size + x

            # Derecha
            if x < grid_size - 1:
                edge_index.append([idx, idx + 1])
                edge_index.append([idx + 1, idx])

            # Abajo
            if y < grid_size - 1:
                edge_index.append([idx, idx + grid_size])
                edge_index.append([idx + grid_size, idx])

    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()



def build_graph_from_sample(dataset, idx, edge_index):
    """
    Convierte una muestra en un grafo PyG: x ∈ [nxn, 3], y ∈ [nxn, 1]
    """

    input_tensor = dataset.inputs[idx]  # (3, n, n)
    x = input_tensor.reshape(3, -1).T  # [nxn, 3]

    output_tensor = dataset.outputs[idx]  # (n, n)
    y = output_tensor.reshape(-1, 1)      # [nxn, 1]

    return Data(x=x, y=y, edge_index=edge_index)

def build_graph_list(dataset, edge_index):
    """
    Convierte todo un dataset normalizado en una lista de grafos PyG.
    """
    graphs = []
    for idx in range(len(dataset)):
        graph = build_graph_from_sample(dataset, idx, edge_index)
        graphs.append(graph)
    return graphs


def load_latest_model(model, folder="saved_models"):
    """
    Carga el modelo más recientemente guardado en la carpeta especificada.
    """
    model_files = glob.glob(os.path.join(folder, "*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No se encontraron modelos en {folder}")

    # Ordenar por fecha de modificación
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model = model_files[0]

    # Cargar el modelo
    model.load_state_dict(torch.load(latest_model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    print(f" Modelo cargado desde: {latest_model}")
    return model