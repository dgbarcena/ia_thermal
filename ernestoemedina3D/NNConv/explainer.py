# PREGUNTAR DAVID:CAMBIAR TB PARA QUE ACEPTE GLAGRA
import torch
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig

def run_full_gnnexplainer(model, dataset, graph_idx, target_node_idx, norm_info, device):
    model.eval()
    data = dataset[graph_idx].to(device)
    print(f"data.x.shape = {data.x.shape}")
    assert data.edge_index.max().item() < data.num_nodes

    print(f"data.edge_index.max() = {data.edge_index.max().item()}")
    print(f"data.num_nodes = {data.num_nodes}")

    ### 1. GLOBAL (sobre todo el grafo)
    explainer_global = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        edge_mask_type='object',
        node_mask_type='attributes',
        model_config=ModelConfig(
            mode='regression',
            task_level='node',
            return_type='raw'
        )
    )
    explanation_global = explainer_global(data.x, data.edge_index, edge_attr=data.edge_attr)
    node_mask_global = explanation_global.node_mask
    edge_mask_global = explanation_global.edge_mask

    plot_node_importance(graph_idx, data, node_mask_global, norm_info, title="Global Node Importance")
    plot_edge_importance(graph_idx, data, edge_mask_global, norm_info, title="Global Edge Importance")
    assert 0 <= target_node_idx < data.num_nodes, f"Índice fuera de rango: {target_node_idx}"

    ### 2. LOCAL (para un nodo específico)
    explainer_local = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='phenomenon',
        edge_mask_type='object',
        node_mask_type='attributes',
        model_config=ModelConfig(
            mode='regression',
            task_level='node',
            return_type='raw'
        )
    )
    with torch.no_grad():
        all_preds = model(data.x, data.edge_index, data.edge_attr)
        prediction = all_preds.detach().to(device).float()
    explanation_local = explainer_local(
        data.x,
        data.edge_index,
        edge_attr=data.edge_attr,
        index=target_node_idx,
        target=prediction
    )
    node_mask_local = explanation_local.node_mask
    edge_mask_local = explanation_local.edge_mask

    plot_node_importance(graph_idx, data, node_mask_local, norm_info, title=f"Local Node Importance (T_{target_node_idx})")
    plot_edge_importance(graph_idx, data, edge_mask_local, norm_info, title=f"Local Edge Importance (T_{target_node_idx})")


def plot_node_importance(graph_idx, data, node_mask, norm_info, title="Node Importance"):
    G = nx.Graph()

    node_importance = torch.norm(node_mask, p=2, dim=1)

    for i, ext_id in enumerate(data.external_ids.cpu()):
        G.add_node(i, importance=node_importance[i].item(), T_real=data.y[i].item() * norm_info["max_T_real"])

    edge_index = data.edge_index.cpu().numpy()
    for j in range(edge_index.shape[1]):
        u, v = edge_index[:, j]
        G.add_edge(u, v)

    # Este bloque va al final, cuando ya están los nodos creados
    pos = nx.spring_layout(G, seed=42)

    _plot_graph(G, pos, attr="importance", title=title, graph_idx=graph_idx)



def plot_edge_importance(graph_idx, data, edge_mask, norm_info, title="Edge Importance"):
    G = nx.Graph()
    

    for i in range(len(data.external_ids)):
        G.add_node(i, T_real=data.y[i].item() * norm_info["max_T_real"])

    edge_index = data.edge_index.cpu().numpy()
    for j in range(edge_index.shape[1]):
        u, v = edge_index[:, j]
        importance = edge_mask[j].item()
        G.add_edge(u, v, importance=importance)
        
    pos = nx.spring_layout(G, seed=42)

    _plot_graph(G, pos, attr="importance", title=title, graph_idx=graph_idx, is_edge=True)


def _plot_graph(G, pos, attr="importance", title="Importance", graph_idx=0, is_edge=False, highlight_node=None):
    plt.figure(figsize=(10, 8))

    # === Valores de color y normalización ===
    if is_edge:
        values = [G[u][v][attr] for u, v in G.edges()]
    else:
        values = [G.nodes[n][attr] for n in G.nodes()]
    
    vmin, vmax = min(values), max(values)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    # === Dibujar nodos/aristas y colorbar ===
    if is_edge:
        nx.draw_networkx_edges(G, pos, edge_color=values, edge_cmap=cmap,
                               edge_vmin=vmin, edge_vmax=vmax, width=2)
        nx.draw_networkx_nodes(G, pos, node_color="darkgray", node_size=500)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
    else:
        nodes = nx.draw_networkx_nodes(G, pos, node_color=values, cmap=cmap, node_size=500)
        cbar = plt.colorbar(nodes, ax=plt.gca())

    cbar.set_label(f"{title}", fontsize=14)

    # === Etiquetas y layout ===
    nx.draw_networkx_labels(G, pos, labels={n: f"{n}" for n in G.nodes()}, font_size=14, font_color='White')
    plt.title(f"Graph {graph_idx}: {title}", fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
