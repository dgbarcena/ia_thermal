import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.animation import PillowWriter
import os
from utils import *

#%%

def plot_sample(output,target):

    plt.style.use('default')

    plt.rcParams["figure.figsize"] = (6,4)

    plt.rcParams["font.family"] = "Times New Roman"

    plt.rcParams["font.size"] = 12

    plt.rcParams["text.usetex"] = False

    plt.rcParams["axes.titlesize"] = 11

    # Convertir los tensores a numpy y asegurarse de que están en CPU
    try:
        output_np = output.squeeze().cpu().detach().numpy()
        target_np = target.squeeze().cpu().detach().numpy()
    except:
        output_np=output
        target_np=target
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Asegurarse de que las celdas de la grilla sean lo suficientemente grandes para el texto
    fig.tight_layout(pad=3.0)
    
    # Output de la red
    axs[1].imshow(output_np, cmap='viridis', interpolation='nearest')
    axs[1].title.set_text('Output')
    for i in range(output_np.shape[0]):
        for j in range(output_np.shape[1]):
            text = axs[1].text(j, i, f'{output_np[i, j]:.0f}',
                            ha="center", va="center", color="w", fontsize=6)

    # Target
    axs[0].imshow(target_np, cmap='viridis', interpolation='nearest')
    axs[0].title.set_text('Target')
    for i in range(target_np.shape[0]):
        for j in range(target_np.shape[1]):
            text = axs[0].text(j, i, f'{target_np[i, j]:.0f}',
                            ha="center", va="center", color="w", fontsize=6)

    # Calcular la diferencia
    diferencia_np =np.abs( output_np - target_np)
    
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    
    # Asegurarse de que las celdas de la grilla sean lo suficientemente grandes para el texto
    fig2.tight_layout(pad=3.0)
    
    # Diferencia entre el output de la red y el target
    cax2 = ax2.imshow(diferencia_np, cmap='viridis', interpolation='nearest')
    ax2.title.set_text('Absolute Error')

    fig.colorbar(cax2)
    plt.show()


#%%
def visualizar_valores_vectoreslatentes(output, target):

    plt.style.use('default')

    plt.rcParams["figure.figsize"] = (6,4)

    plt.rcParams["font.family"] = "Times New Roman"

    plt.rcParams["font.size"] = 12

    plt.rcParams["text.usetex"] = False

    plt.rcParams["axes.titlesize"] = 11
    
    # Convert tensors to numpy and make sure they're on CPU
    if type(output) == torch.Tensor:
        output_np = output.cpu().detach().numpy()
    else:
        output_np = output
    if type(target) == torch.Tensor:
        target_np = target.cpu().detach().numpy()
    else: 
        target_np = target
    
    # Create a plot with 2 subplots: one for the output and one for the target
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot the output vector
    axs[1].plot(list(range(1,10)),output_np.flatten(), marker='.', linestyle='--', color='darkcyan')
    axs[1].title.set_text('Output')
    axs[1].grid(False)
    
    # Plot the target vector
    axs[0].plot(list(range(1,10)),target_np.flatten(), marker='.', linestyle='-', color='crimson')
    axs[0].title.set_text('Target')
    axs[0].grid(False)

    # Show the plot
    plt.show()

    # Create a plot with 2 subplots: one for the output and one for the target
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    
    # Plot the output vector
    axs.plot(list(range(1,10)),output_np.flatten(), marker='.', linestyle='--', color='darkcyan')
    axs.plot(list(range(1,10)),target_np.flatten(), marker='.', linestyle='--', color='crimson')
    axs.grid(False)

    
    # Plot the target vector
    plt.show()
    
#%%

def plot_loss_curves(train_loss, val_loss, save_as_pdf=False, filename='loss_curves'):
    """
    Grafica las curvas de pérdida de entrenamiento y validación.
    Ajusta el eje x solo a las épocas representadas y permite guardar como PDF.
    """
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, max(len(train_loss), len(val_loss)) + 1)
    plt.plot(epochs[:len(train_loss)], train_loss, label='Pérdida Entrenamiento', color='tab:blue')
    plt.plot(epochs[:len(val_loss)], val_loss, label='Pérdida Validación', color='tab:orange')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.title('Curvas de pérdida durante el entrenamiento')
    plt.legend()
    plt.grid(True)
    plt.xlim(epochs[0], epochs[max(len(train_loss), len(val_loss)) - 1])
    plt.tight_layout()
    if save_as_pdf:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf')
    plt.show()
    

#%%
def plot_error_map(y_pred, y_true, i=0, t=0):
    """
    Muestra el mapa de temperaturas reales, predichas y el error (por pixel) en un instante concreto.
    Parámetros:
        y_pred: tensor con shape (B, T, 1, H, W)
        y_true: tensor con shape (B, T, 1, H, W)
        i: índice de la muestra
        t: timestep dentro de la secuencia
    """
    real = y_true[i, t].squeeze().detach().cpu().numpy()
    pred = y_pred[i, t].squeeze().detach().cpu().numpy()
    error = pred - real
    abs_error = abs(error)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    
    im0 = axs[0].imshow(real, cmap='hot')
    axs[0].set_title("Temperatura real")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(pred, cmap='hot')
    axs[1].set_title("Temperatura predicha")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(abs_error, cmap='viridis')
    axs[2].set_title("Absolute error")  
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()
    
#%%
def plot_se_map(y_pred, y_true, time=0, dt=1, show_pred=True, return_mse=False):
    """
    Muestra el mapa de temperaturas reales, predichas y el Squared Error (por pixel) en un instante concreto.
    
    Parámetros:
        y_pred: array con shape (T, H, W)
        y_true: tensor con shape (T, H, W)
        time: instante de tiempo real (en segundos)
        dt: intervalo de tiempo entre pasos
        show_pred: si es True, muestra el mapa de temperaturas predichas también
        return_mse: si es True, devuelve el valor del MSE
    """
    t = time // dt

    real = y_true[t, :, :]
    pred = y_pred[t, :, :]
    sq_diff = (pred - real) ** 2
    mse = np.mean(sq_diff)

    if show_pred:
        # Rango común de temperatura
        vmin = min(real.min(), pred.min())
        vmax = max(real.max(), pred.max())

        fig, axs = plt.subplots(1, 3, figsize=(14, 4))
        
        im0 = axs[0].imshow(real, cmap='hot', vmin=vmin, vmax=vmax)
        axs[0].set_title("Real temperature [K]")
        plt.colorbar(im0, ax=axs[0])

        im1 = axs[1].imshow(pred, cmap='hot', vmin=vmin, vmax=vmax)
        axs[1].set_title("Predicted temperature [K]")
        plt.colorbar(im1, ax=axs[1])

        im2 = axs[2].imshow(sq_diff, cmap='viridis')
        axs[2].set_title("Squared error [K²]")
        plt.colorbar(im2, ax=axs[2])
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        
        im = ax.imshow(sq_diff, cmap='viridis')
        ax.set_title("Squared error [K²]")
        plt.colorbar(im, ax=ax)

    fig.suptitle(f'Temperature map at t = {time:.2f} s', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print(f"MSE: {mse:.8f} K^2")

    if return_mse:
        return mse
        
#%% 
def plot_nodes_evolution(y_pred, y_true, nodes_idx, dt=1, together=True, save_as_pdf=False, filename='nodes_evolution'):
    """
    Shows the temporal evolution of real and predicted temperatures in a series of nodes.
    
    Parameters:
        y_pred: array with shape (T, H, W)
        y_true: array with shape (T, H, W)
        nodes_idx: list of node indices to show [(idx1, idy1), (idx2, idy2), ...]
        dt: time interval between each time step
        together: if True, shows all evolutions in a single plot
        save_as_pdf: if True, saves the figure as PDF in the 'figures' folder
        filename: base filename (without extension)
    """
    time = np.arange(y_pred.shape[0]) * dt

    if together:
        # =============== FIGURE CONFIGURATION WITH WHITE BACKGROUND ===============
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        for i, node_idx in enumerate(nodes_idx):
            color = plt.cm.tab10(i % 10)
            label = f'Node ({node_idx[0]}, {node_idx[1]})'

            y_true_node = y_true[:, node_idx[0], node_idx[1]]
            y_pred_node = y_pred[:, node_idx[0], node_idx[1]]

            ax.plot(time, y_true_node, label=f'{label} - Ground Truth', color=color, linewidth=2)
            ax.plot(time, y_pred_node, 'x', label=f'{label} - Prediction', color=color, markersize=4)
        
        ax.set_xlabel('Time [s]', color='black')
        ax.set_ylabel('Temperature [K]', color='black')
        
        # Título solo para visualización (se eliminará al guardar PDF)
        title_handle = ax.set_title('Time evolution of temperature in selected nodes', color='black')
        
        # Limitar eje x a los pasos representados
        ax.set_xlim(time[0], time[-1])
        
        # Configurar cuadrícula
        ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
        
        ax.legend()
        ax.tick_params(colors='black')
        plt.tight_layout()

        if save_as_pdf:
            # Eliminar título antes de guardar
            title_handle.set_visible(False)
            os.makedirs('figures', exist_ok=True)
            plt.savefig(f'figures/{filename}.pdf', format='pdf', facecolor='white')
            # Restaurar título para visualización
            title_handle.set_visible(True)
        plt.show()

    else:
        # =============== MULTIPLE SUBPLOTS CONFIGURATION ===============
        fig, axs = plt.subplots(len(nodes_idx), 1, figsize=(12, 3 * len(nodes_idx)), sharex=True)
        fig.patch.set_facecolor('white')
        
        if len(nodes_idx) == 1:
            axs = [axs]

        for i, node_idx in enumerate(nodes_idx):
            axs[i].set_facecolor('white')
            axs[i].plot(time, y_true[:, node_idx[0], node_idx[1]], label='Ground truth', color='blue', linewidth=2)
            axs[i].plot(time, y_pred[:, node_idx[0], node_idx[1]], 'x', label='Prediction', color='orange', markersize=4)
            axs[i].set_title(f"Node ({node_idx[0]}, {node_idx[1]})", color='black')

            axs[i].set_ylabel('Temperature [K]', color='black')
            
            # Limitar eje x a los pasos representados
            axs[i].set_xlim(time[0], time[-1])
            
            # Configurar cuadrícula
            axs[i].grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
            axs[i].tick_params(colors='black')

            if i == len(nodes_idx) - 1:
                axs[i].set_xlabel('Time [s]', color='black')
            if i == 0:
                axs[i].legend(loc='upper right')

        # Título principal para visualización
        main_title = fig.suptitle('Time evolution of temperature in selected nodes', fontsize=16, color='black')
        fig.align_ylabels()
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_as_pdf:
            # Eliminar título principal antes de guardar
            main_title.set_visible(False)
            os.makedirs('figures', exist_ok=True)
            fig.savefig(f'figures/{filename}.pdf', format='pdf', facecolor='white')
            # Restaurar título para visualización
            main_title.set_visible(True)
        plt.show()
        
    
#%%
def plot_nodes_evolution_3way(y_true, y_pred_nofis, y_pred_fis, nodes_idx, dt=1, together=True, save_as_pdf=False, filename='nodes_evolution_3way'):
    """
    Muestra la evolución temporal de las temperaturas reales y predichas (modelo sin física y con física) en una serie de nodos.

    Parámetros:
        y_true: array (T, H, W) - ground truth
        y_pred_nofis: array (T, H, W) - predicción modelo sin física
        y_pred_fis: array (T, H, W) - predicción modelo con física
        nodes_idx: lista de índices de los nodos a mostrar [(idx1, idy1), ...]
        dt: intervalo de tiempo entre cada paso de tiempo
        together: si es True, muestra todas las evoluciones en un solo gráfico
        save_as_pdf: si es True, guarda la figura como PDF en la carpeta 'figures'
        filename: nombre base del archivo (sin extensión)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    time = np.arange(y_true.shape[0]) * dt

    if together:
        plt.figure(figsize=(12, 6))
        for i, node_idx in enumerate(nodes_idx):
            color = plt.cm.tab10(i % 10)
            label = f'Node ({node_idx[0]}, {node_idx[1]})'

            y_true_node = y_true[:, node_idx[0], node_idx[1]]
            y_pred_nofis_node = y_pred_nofis[:, node_idx[0], node_idx[1]]
            y_pred_fis_node = y_pred_fis[:, node_idx[0], node_idx[1]]

            plt.plot(time, y_true_node, label=f'{label} - Ground Truth', color=color, linewidth=2)
            plt.plot(time, y_pred_nofis_node, 'x--', label=f'{label} - No Physics', color=color, alpha=0.7)
            plt.plot(time, y_pred_fis_node, 'o:', label=f'{label} - Physics', color=color, alpha=0.7)

        plt.xlabel('Time [s]')
        plt.ylabel('Temperature [K]')
        plt.title('Time evolution of temperature in selected nodes')
        plt.xlim(time[0], time[-1])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_as_pdf:
            os.makedirs('figures', exist_ok=True)
            plt.savefig(f'figures/{filename}.pdf', format='pdf')
        plt.show()

    else:
        fig, axs = plt.subplots(len(nodes_idx), 1, figsize=(12, 3 * len(nodes_idx)), sharex=True)
        if len(nodes_idx) == 1:
            axs = [axs]

        for i, node_idx in enumerate(nodes_idx):
            color = plt.cm.tab10(i % 10)
            axs[i].plot(time, y_true[:, node_idx[0], node_idx[1]], label='Ground truth', color=color, linewidth=2)
            axs[i].plot(time, y_pred_nofis[:, node_idx[0], node_idx[1]], 'x--', label='No Physics', color=color, alpha=0.7)
            axs[i].plot(time, y_pred_fis[:, node_idx[0], node_idx[1]], 'o:', label='Physics', color=color, alpha=0.7)
            axs[i].set_title(f"Node ({node_idx[0]}, {node_idx[1]})")
            axs[i].set_ylabel('Temperature [K]')
            axs[i].set_xlim(time[0], time[-1])
            if i == len(nodes_idx) - 1:
                axs[i].set_xlabel('Time [s]')
            if i == 0:
                axs[i].legend(loc='upper right')

        fig.suptitle('Time evolution of temperature in selected nodes', fontsize=16)
        fig.align_ylabels()
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_as_pdf:
            os.makedirs('figures', exist_ok=True)
            fig.savefig(f'figures/{filename}.pdf', format='pdf')
        plt.show()
        
        
#%%

def compare_error_maps_2d(err1_map, err2_map, 
                          titles=("Error Map 1", "Error Map 2"),
                          return_mse=False,
                          save_as_pdf=False,
                          filename='compare_error_maps'):
    """
    Compara visualmente dos mapas de error 2D (por ejemplo, error cuadrático por píxel).

    Parámetros:
        err1_map: array (H, W) – primer mapa de error
        err2_map: array (H, W) – segundo mapa de error
        titles: tupla de strings – títulos personalizados para los mapas
        return_mse: bool – si es True, devuelve el MSE global de cada mapa
        save_as_pdf: bool – si es True, guarda la figura como PDF en 'figures'
        filename: string – nombre base del archivo (sin extensión)

    Devuelve:
        mse1, mse2: MSE global para cada mapa (si return_mse == True)
    """
    mse1 = np.mean(err1_map**2)
    mse2 = np.mean(err2_map**2)

    vmax = max(err1_map.max(), err2_map.max())

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    im1 = axs[0].imshow(err1_map, cmap='viridis', vmin=0, vmax=vmax)
    axs[0].set_title(f"{titles[0]}\nMSE = {mse1:.6f} K²")
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(err2_map, cmap='viridis', vmin=0, vmax=vmax)
    axs[1].set_title(f"{titles[1]}\nMSE = {mse2:.6f} K²")
    plt.colorbar(im2, ax=axs[1])

    plt.tight_layout()

    if save_as_pdf:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf')

    plt.show()

    if return_mse:
        return mse1, mse2
    

#%%

def plot_prediction_and_error(y_pred, y_true, t=0, cmap='hot', save_as_pdf=False, filename='prediction_and_error'):
    """
    Representa la predicción de un modelo junto con el error absoluto en cada punto.
    
    Parámetros:
        y_pred: array o tensor con shape (T, H, W) – predicciones del modelo
        y_true: array o tensor con shape (T, H, W) – valores reales del solver
        t: int – índice del tiempo que se desea visualizar
        cmap: str – esquema de colores para los mapas de temperatura
        save_as_pdf: bool – si es True, guarda la figura como PDF en 'figures'
        filename: str – nombre base del archivo (sin extensión)
    """
    # Asegurarse de que los datos estén en formato NumPy
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()

    # Extraer los datos en el tiempo t
    pred = y_pred[t, :, :]
    real = y_true[t, :, :]
    abs_error = np.abs(pred - real)

    # Rango común de temperatura para predicción y valores reales
    vmin = min(real.min(), pred.min())
    vmax = max(real.max(), pred.max())

    # Crear la figura con dos subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Mapa de predicción
    im0 = axs[0].imshow(pred, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_title("Predicted Temperature [K]")
    plt.colorbar(im0, ax=axs[0])

    # Mapa de error absoluto
    im1 = axs[1].imshow(abs_error, cmap='viridis')
    axs[1].set_title("Absolute Error [K]")
    plt.colorbar(im1, ax=axs[1])

    # Ajustar diseño
    plt.tight_layout()

    # Guardar como PDF si es necesario
    if save_as_pdf:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf')

    # Mostrar la figura
    plt.show()
    
#%%


def generar_gif_pcb_comparacion(model_preds, solver_data, dt=1, nombre_archivo="comparison_evolution",
                                 guardar_en_figures=False, duracion_total=10.0):
    """
    Genera un GIF o animación comparando las predicciones del modelo y los datos del solver.
    Fondo blanco, texto de tiempo grande y barra de progreso con fondo.

    Args:
        model_preds (np.ndarray): Array (T, H, W) de predicciones del modelo.
        solver_data (np.ndarray): Array (T, H, W) de datos del solver (ground truth).
        dt (float): Paso temporal entre frames (en segundos).
        nombre_archivo (str or None): Nombre base del archivo (sin extensión). Si None, no se guarda.
        guardar_en_figures (bool): Si True, guarda el gif en la carpeta 'figures/' junto al notebook.
        duracion_total (float): Duración total del gif en segundos.

    Returns:
        ani (matplotlib.animation.FuncAnimation): Objeto de animación para uso en Jupyter.
    """

    assert model_preds.shape == solver_data.shape, "Las formas de model_preds y solver_data deben coincidir."
    total_frames = model_preds.shape[0]

    # Calcular fps e intervalo para mantener la duración total deseada
    fps = total_frames / duracion_total
    interval_ms = int(1000 / fps)

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Fondo blanco
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    ax2.set_facecolor('white')

    vmin = min(model_preds.min(), solver_data.min())
    vmax = max(model_preds.max(), solver_data.max())
    im1 = ax1.imshow(model_preds[0], vmin=vmin, vmax=vmax, cmap='jet')
    im2 = ax2.imshow(solver_data[0], vmin=vmin, vmax=vmax, cmap='jet')

    ax1.set_title("Modelo", fontsize=14, color='black')
    ax2.set_title("Solver", fontsize=14, color='black')
    for ax in [ax1, ax2]:
        ax.axis('off')

    # Barra de color común
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(im1, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black', fontsize=10)


    # Barra de progreso con fondo
    progress_ax = fig.add_axes([0.25, 0.01, 0.5, 0.02])
    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis('off')
    background_bar = Rectangle((0, 0), 1, 1, color='lightgray')
    progress_bar = Rectangle((0, 0), 0, 1, color='blue')
    progress_ax.add_patch(background_bar)
    progress_ax.add_patch(progress_bar)

    # Texto del tiempo fuera de la barra
    tiempo_num = fig.text(0.5, 0.045, "0.00 s", ha='center', va='bottom', fontsize=18, color='black')

    # Función de actualización
    def update(frame):
        im1.set_data(model_preds[frame])
        im2.set_data(solver_data[frame])
        tiempo_actual = frame * dt
        progress_bar.set_width((frame + 1) / total_frames)
        tiempo_num.set_text(f"{tiempo_actual:.2f} s")
        return im1, im2, progress_bar, tiempo_num

    # Crear animación (¡sin blit!)
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)

    if nombre_archivo and guardar_en_figures:
        base_path = os.getcwd()
        carpeta = os.path.join(base_path, "figures")
        os.makedirs(carpeta, exist_ok=True)
        ruta_salida = os.path.join(carpeta, f"{nombre_archivo}.gif")
        print(f"Guardando gif en: {ruta_salida}")
        ani.save(ruta_salida, writer='pillow', fps=fps, savefig_kwargs={'facecolor': 'white'})
        print("Gif guardado.")
    plt.close()
    
    return ani

#%%
def generar_gif_pcb_comparacion_3way(model_preds_nofis, model_preds_fis, solver_data, dt=1, nombre_archivo="comparison_evolution_3way",
                                     guardar_en_figures=False, duracion_total=10.0):
    """
    Genera un GIF comparando las predicciones de dos modelos (sin física y con física) y los datos del solver.
    Muestra los tres mapas lado a lado, con barra de progreso y tiempo.

    Args:
        model_preds_nofis (np.ndarray): Array (T, H, W) de predicciones del modelo sin física.
        model_preds_fis (np.ndarray): Array (T, H, W) de predicciones del modelo con física.
        solver_data (np.ndarray): Array (T, H, W) de datos del solver (ground truth).
        dt (float): Paso temporal entre frames (en segundos).
        nombre_archivo (str or None): Nombre base del archivo (sin extensión). Si None, no se guarda.
        guardar_en_figures (bool): Si True, guarda el gif en la carpeta 'figures/' junto al notebook.
        duracion_total (float): Duración total del gif en segundos.

    Returns:
        ani (matplotlib.animation.FuncAnimation): Objeto de animación para uso en Jupyter.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    import os

    assert model_preds_nofis.shape == model_preds_fis.shape == solver_data.shape, "Las formas deben coincidir."
    total_frames = model_preds_nofis.shape[0]

    # Calcular fps e intervalo para mantener la duración total deseada
    fps = total_frames / duracion_total
    interval_ms = int(1000 / fps)

    # Crear figura
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')

    vmin = min(model_preds_nofis.min(), model_preds_fis.min(), solver_data.min())
    vmax = max(model_preds_nofis.max(), model_preds_fis.max(), solver_data.max())
    im0 = axs[0].imshow(model_preds_nofis[0], vmin=vmin, vmax=vmax, cmap='jet')
    im1 = axs[1].imshow(model_preds_fis[0], vmin=vmin, vmax=vmax, cmap='jet')
    im2 = axs[2].imshow(solver_data[0], vmin=vmin, vmax=vmax, cmap='jet')

    axs[0].set_title("Modelo sin física", fontsize=14, color='black')
    axs[1].set_title("Modelo con física", fontsize=14, color='black')
    axs[2].set_title("Solver", fontsize=14, color='black')
    for ax in axs:
        ax.axis('off')

    # Barra de color común
    cbar_ax = fig.add_axes([0.94, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(im0, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black', fontsize=10)

    # Barra de progreso con fondo
    progress_ax = fig.add_axes([0.25, 0.01, 0.5, 0.02])
    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis('off')
    background_bar = Rectangle((0, 0), 1, 1, color='lightgray')
    progress_bar = Rectangle((0, 0), 0, 1, color='blue')
    progress_ax.add_patch(background_bar)
    progress_ax.add_patch(progress_bar)

    # Texto del tiempo fuera de la barra
    tiempo_num = fig.text(0.5, 0.045, "0.00 s", ha='center', va='bottom', fontsize=18, color='black')

    # Función de actualización
    def update(frame):
        im0.set_data(model_preds_nofis[frame])
        im1.set_data(model_preds_fis[frame])
        im2.set_data(solver_data[frame])
        tiempo_actual = frame * dt
        progress_bar.set_width((frame + 1) / total_frames)
        tiempo_num.set_text(f"{tiempo_actual:.2f} s")
        return im0, im1, im2, progress_bar, tiempo_num

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)

    if nombre_archivo and guardar_en_figures:
        base_path = os.getcwd()
        carpeta = os.path.join(base_path, "figures")
        os.makedirs(carpeta, exist_ok=True)
        ruta_salida = os.path.join(carpeta, f"{nombre_archivo}.gif")
        print(f"Guardando gif en: {ruta_salida}")
        ani.save(ruta_salida, writer='pillow', fps=fps, savefig_kwargs={'facecolor': 'white'})
        print("Gif guardado.")
    plt.close()

    return ani


#%%


def generar_gif_error_evolucion(model_preds, solver_data, dt=1, nombre_archivo="error_evolution",
                                 guardar_en_figures=False, duracion_total=10.0):
    """
    Genera un GIF o animación mostrando la evolución del error absoluto entre las predicciones del modelo y los datos del solver.
    Fondo blanco, texto de tiempo grande y barra de progreso con fondo.

    Args:
        model_preds (np.ndarray): Array (T, H, W) de predicciones del modelo.
        solver_data (np.ndarray): Array (T, H, W) de datos del solver (ground truth).
        dt (float): Paso temporal entre frames (en segundos).
        nombre_archivo (str or None): Nombre base del archivo (sin extensión). Si None, no se guarda.
        guardar_en_figures (bool): Si True, guarda el gif en la carpeta 'figures/' junto al notebook.
        duracion_total (float): Duración total del gif en segundos.

    Returns:
        ani (matplotlib.animation.FuncAnimation): Objeto de animación para uso en Jupyter.
    """
    assert model_preds.shape == solver_data.shape, "Las formas de model_preds y solver_data deben coincidir."
    total_frames = model_preds.shape[0]

    error_abs = np.abs(model_preds - solver_data)

    # Calcular fps e intervalo para mantener la duración total deseada
    fps = total_frames / duracion_total
    interval_ms = int(1000 / fps)

    # Crear figura
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    vmin = 0.0
    vmax = error_abs.max()
    im = ax.imshow(error_abs[0], vmin=vmin, vmax=vmax, cmap='hot')
    ax.set_title("Error absoluto", fontsize=14, color='black')
    ax.axis('off')

    # Barra de color
    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black', fontsize=10)

    # Barra de progreso con fondo
    progress_ax = fig.add_axes([0.25, 0.01, 0.5, 0.02])
    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis('off')
    background_bar = Rectangle((0, 0), 1, 1, color='lightgray')
    progress_bar = Rectangle((0, 0), 0, 1, color='blue')
    progress_ax.add_patch(background_bar)
    progress_ax.add_patch(progress_bar)

    # Texto del tiempo
    tiempo_num = fig.text(0.5, 0.045, "0.00 s", ha='center', va='bottom', fontsize=18, color='black')

    # Función de actualización
    def update(frame):
        im.set_data(error_abs[frame])
        tiempo_actual = frame * dt
        progress_bar.set_width((frame + 1) / total_frames)
        tiempo_num.set_text(f"{tiempo_actual:.2f} s")
        return im, progress_bar, tiempo_num

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)

    if nombre_archivo and guardar_en_figures:
        base_path = os.getcwd()
        carpeta = os.path.join(base_path, "figures") if guardar_en_figures else base_path
        os.makedirs(carpeta, exist_ok=True)
        ruta_salida = os.path.join(carpeta, f"{nombre_archivo}.gif")
        ani.save(ruta_salida, writer='pillow', fps=fps, savefig_kwargs={'facecolor': 'white'})
        plt.close()
    else:
        plt.close()

    return ani


#%%
def generar_gif_error_evolucion_2way(model_preds_nofis, model_preds_fis, solver_data, dt=1, nombre_archivo="error_evolution_2way",
                                     guardar_en_figures=False, duracion_total=10.0):
    """
    Genera un GIF mostrando la evolución del error absoluto de dos modelos respecto al solver.
    Muestra ambos mapas de error lado a lado, con barra de progreso y tiempo.

    Args:
        model_preds_nofis (np.ndarray): Array (T, H, W) de predicciones del modelo sin física.
        model_preds_fis (np.ndarray): Array (T, H, W) de predicciones del modelo con física.
        solver_data (np.ndarray): Array (T, H, W) de datos del solver (ground truth).
        dt (float): Paso temporal entre frames (en segundos).
        nombre_archivo (str or None): Nombre base del archivo (sin extensión). Si None, no se guarda.
        guardar_en_figures (bool): Si True, guarda el gif en la carpeta 'figures/' junto al notebook.
        duracion_total (float): Duración total del gif en segundos.

    Returns:
        ani (matplotlib.animation.FuncAnimation): Objeto de animación para uso en Jupyter.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle
    import os

    assert model_preds_nofis.shape == model_preds_fis.shape == solver_data.shape, "Las formas deben coincidir."
    total_frames = model_preds_nofis.shape[0]

    error_abs_nofis = np.abs(model_preds_nofis - solver_data)
    error_abs_fis = np.abs(model_preds_fis - solver_data)

    # Calcular fps e intervalo para mantener la duración total deseada
    fps = total_frames / duracion_total
    interval_ms = int(1000 / fps)

    # Crear figura
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')

    vmax = max(error_abs_nofis.max(), error_abs_fis.max())
    im0 = axs[0].imshow(error_abs_nofis[0], vmin=0.0, vmax=vmax, cmap='hot')
    im1 = axs[1].imshow(error_abs_fis[0], vmin=0.0, vmax=vmax, cmap='hot')

    axs[0].set_title("Error absoluto - Sin física", fontsize=14, color='black')
    axs[1].set_title("Error absoluto - Con física", fontsize=14, color='black')
    for ax in axs:
        ax.axis('off')

    # Barra de color común
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(im0, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black', fontsize=10)

    # Barra de progreso con fondo
    progress_ax = fig.add_axes([0.25, 0.01, 0.5, 0.02])
    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis('off')
    background_bar = Rectangle((0, 0), 1, 1, color='lightgray')
    progress_bar = Rectangle((0, 0), 0, 1, color='blue')
    progress_ax.add_patch(background_bar)
    progress_ax.add_patch(progress_bar)

    # Texto del tiempo fuera de la barra
    tiempo_num = fig.text(0.5, 0.045, "0.00 s", ha='center', va='bottom', fontsize=18, color='black')

    # Función de actualización
    def update(frame):
        im0.set_data(error_abs_nofis[frame])
        im1.set_data(error_abs_fis[frame])
        tiempo_actual = frame * dt
        progress_bar.set_width((frame + 1) / total_frames)
        tiempo_num.set_text(f"{tiempo_actual:.2f} s")
        return im0, im1, progress_bar, tiempo_num

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)

    if nombre_archivo and guardar_en_figures:
        base_path = os.getcwd()
        carpeta = os.path.join(base_path, "figures")
        os.makedirs(carpeta, exist_ok=True)
        ruta_salida = os.path.join(carpeta, f"{nombre_archivo}.gif")
        print(f"Guardando gif en: {ruta_salida}")
        ani.save(ruta_salida, writer='pillow', fps=fps, savefig_kwargs={'facecolor': 'white'})
        print("Gif guardado.")
    plt.close()

    return ani

#%%
def generar_gif_temperatura(temperaturas, dt=1.0, nombre_archivo="temperatura_evolucion",
                        guardar_en_figures=False, duracion_total=10.0):
    """
    Genera un GIF mostrando la evolución de la temperatura en el tiempo.

    Args:
        temperaturas (np.ndarray): Array (T, H, W) con la evolución temporal de la temperatura.
        dt (float): Paso temporal entre frames (en segundos).
        nombre_archivo (str or None): Nombre base del archivo (sin extensión). Si None, no se guarda.
        guardar_en_figures (bool): Si True, guarda el gif en la carpeta 'figures/' junto al notebook.
        duracion_total (float): Duración total del gif en segundos.

    Returns:
        ani (matplotlib.animation.FuncAnimation): Objeto de animación para uso en Jupyter.
    """
    assert temperaturas.ndim == 3, "temperaturas debe tener forma (T, H, W)"
    total_frames = temperaturas.shape[0]

    # Calcular fps e intervalo para lograr la duración deseada
    fps = total_frames / duracion_total
    interval_ms = 1000 / fps # Cambio importante aquí

    # Crear figura
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    vmin = temperaturas.min()
    vmax = temperaturas.max()
    im = ax.imshow(temperaturas[0], vmin=vmin, vmax=vmax, cmap='hot')
    ax.set_title("Temperatura", fontsize=14, color='black')
    ax.axis('off')

    # Barra de color
    cbar_ax = fig.add_axes([0.88, 0.2, 0.03, 0.6])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black', fontsize=10)

    # Barra de progreso
    progress_ax = fig.add_axes([0.25, 0.01, 0.5, 0.02])
    progress_ax.set_xlim(0, 1)
    progress_ax.set_ylim(0, 1)
    progress_ax.axis('off')
    background_bar = Rectangle((0, 0), 1, 1, color='lightgray')
    progress_bar = Rectangle((0, 0), 0, 1, color='red')
    progress_ax.add_patch(background_bar)
    progress_ax.add_patch(progress_bar)

    # Texto del tiempo
    tiempo_num = fig.text(0.5, 0.045, "0.00 s", ha='center', va='bottom', fontsize=18, color='black')

    # Función de actualización
    def update(frame):
        im.set_data(temperaturas[frame])
        tiempo_actual = frame * dt
        progress_bar.set_width((frame + 1) / total_frames)
        tiempo_num.set_text(f"{tiempo_actual:.2f} s")
        return im, progress_bar, tiempo_num

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False)

    if nombre_archivo:
        base_path = os.getcwd()
        carpeta = os.path.join(base_path, "figures") if guardar_en_figures else base_path
        os.makedirs(carpeta, exist_ok=True)
        ruta_salida = os.path.join(carpeta, f"{nombre_archivo}.gif")
        writer = PillowWriter(fps=fps)
        ani.save(ruta_salida, writer=writer, savefig_kwargs={'facecolor': 'white'})
        plt.close()
    else:
        plt.close()

    return ani


#%%
def plot_mae_per_pixel(y_true, y_pred, dataset=None, save_as_pdf=False, filename='mae_per_pixel'):
    """
    Calcula y grafica el error absoluto medio (MAE) por píxel acumulado en el tiempo para una predicción.
    Si se proporciona un dataset, se desnormaliza el error.
    Compatible con modelos que usan (x, y) o (x, t, y).
    Ahora acepta arrays de numpy o tensores torch.
    """
    # Convertir a numpy si es tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # MAE por píxel acumulado en el tiempo
    error_map = np.mean(np.abs(y_true - y_pred), axis=0)  # (H, W)

    # Desnormalizar si corresponde
    if dataset is not None:
        std = dataset.T_outputs_std.cpu().numpy() if isinstance(dataset.T_outputs_std, torch.Tensor) else dataset.T_outputs_std
        error_map = error_map * std

    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    im = ax.imshow(error_map, cmap='hot')
    ax.set_title("MAE acumulado [K]" if dataset is not None else "MAE acumulado", color='black')
    ax.axis('off')

    # Barra de color al margen derecho
    cbar_ax = fig.add_axes([0.88, 0.2, 0.04, 0.6])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('MAE [K]' if dataset is not None else 'MAE', fontsize=12, color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_as_pdf:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf', facecolor='white')
    plt.show()
    

    
#%%
def plot_mae_per_pixel_2way(y_true, y_pred_nofis, y_pred_fis, dataset=None, 
                            titles=("MAE sin física", "MAE con física"),
                            save_as_pdf=False, filename='mae_per_pixel_2way'):
    """
    Calcula y grafica el MAE por píxel acumulado en el tiempo para dos predicciones.
    Si se proporciona un dataset, se desnormaliza el error.
    Muestra ambos mapas lado a lado con una barra de color común al margen derecho.

    Parámetros:
        y_true: array o tensor (T, H, W) – ground truth
        y_pred_nofis: array o tensor (T, H, W) – predicción modelo sin física
        y_pred_fis: array o tensor (T, H, W) – predicción modelo con física
        dataset: objeto dataset para desnormalizar (opcional)
        titles: tupla de strings – títulos personalizados para los mapas
        save_as_pdf: bool – si es True, guarda la figura como PDF en 'figures'
        filename: string – nombre base del archivo (sin extensión)
    """

    # Convertir a numpy si es tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_nofis, torch.Tensor):
        y_pred_nofis = y_pred_nofis.detach().cpu().numpy()
    if isinstance(y_pred_fis, torch.Tensor):
        y_pred_fis = y_pred_fis.detach().cpu().numpy()

    # MAE por píxel acumulado en el tiempo
    error_map_nofis = np.mean(np.abs(y_true - y_pred_nofis), axis=0)  # (H, W)
    error_map_fis = np.mean(np.abs(y_true - y_pred_fis), axis=0)      # (H, W)

    # Desnormalizar si corresponde
    if dataset is not None:
        std = dataset.T_outputs_std.cpu().numpy() if isinstance(dataset.T_outputs_std, torch.Tensor) else dataset.T_outputs_std
        error_map_nofis = error_map_nofis * std
        error_map_fis = error_map_fis * std

    vmax = max(error_map_nofis.max(), error_map_fis.max())
    vmin = 0

    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')

    im0 = axs[0].imshow(error_map_nofis, cmap='hot', vmin=vmin, vmax=vmax)
    axs[0].set_title(titles[0], color='black')
    axs[0].axis('off')

    im1 = axs[1].imshow(error_map_fis, cmap='hot', vmin=vmin, vmax=vmax)
    axs[1].set_title(titles[1], color='black')
    axs[1].axis('off')

    # Barra de color común al margen derecho (solo una para ambos)
    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(im1, cax=cbar_ax)
    cbar.set_label('MAE [K]' if dataset is not None else 'MAE', fontsize=12, color='black')
    cbar.ax.yaxis.set_tick_params(color='black')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='black')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save_as_pdf:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf', facecolor='white')
    plt.show()
    
    
#%%
def plot_mae_per_frame(y_true, y_pred, dataset=None, dt = 1, save_as_pdf=False, filename='mae_per_frame', yscale='log'):
    """
    Calcula y grafica el error absoluto medio (MAE) por paso temporal en escala original 
    para una muestra del dataloader. Compatible con modelos que usan (x, y) o (x, t, y).
    Ahora acepta arrays de numpy o tensores torch.
    Parámetros:
        y_true: array o tensor (T, H, W) – ground truth
        y_pred: array o tensor (T, H, W) – predicción del modelo
        dt: float – paso temporal entre frames (en segundos)
        dataset: objeto dataset para desnormalizar (opcional)
        save_as_pdf: bool – si es True, guarda la figura como PDF en 'figures'
        filename: string – nombre base del archivo (sin extensión)
        yscale: str – escala del eje y ('linear' o 'log')
    """
    # Convertir a numpy si es tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Desnormalizar si corresponde
    if dataset is not None:
        std = dataset.T_outputs_std.cpu().numpy() if isinstance(dataset.T_outputs_std, torch.Tensor) else dataset.T_outputs_std
        y_true = y_true * std
        y_pred = y_pred * std

    # MAE por paso temporal
    mae_per_t = np.mean(np.abs(y_true - y_pred), axis=(1, 2))  # (T,)
    # Convertir a escala de tiempoç
    steps = np.arange(len(mae_per_t)) * dt  # (T,)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, mae_per_t, marker='o')
    plt.title("Error absoluto medio por paso temporal (desnormalizado)")
    plt.xlabel("Paso temporal t")
    plt.ylabel("MAE [K]" if dataset is not None else "MAE")
    if yscale == 'log':
        plt.yscale('log')
    elif yscale == 'linear':
        plt.yscale('linear')
    else:
        raise ValueError("y_scale debe ser 'log' o 'linear'")
    plt.xlim(0, steps[-1])
    plt.grid(True)
    plt.tight_layout()
    if save_as_pdf:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf')
    plt.show()


#%%
def plot_mae_per_frame_2way(y_true, y_pred_nofis, y_pred_fis, dataset=None, dt= 1, 
                            labels=("Sin física", "Con física"),
                            save_as_pdf=False, filename='mae_per_frame_2way', yscale='log'):
    """
    Calcula y grafica el MAE por paso temporal para dos predicciones.
    Si se proporciona un dataset, se desnormaliza el error.
    Muestra ambas curvas en el mismo gráfico.

    Parámetros:
        y_true: array o tensor (T, H, W) – ground truth
        y_pred_nofis: array o tensor (T, H, W) – predicción modelo sin física
        y_pred_fis: array o tensor (T, H, W) – predicción modelo con física
        dataset: objeto dataset para desnormalizar (opcional)
        dt: float – paso temporal entre frames (en segundos)
        labels: tupla de strings – etiquetas para las curvas
        save_as_pdf: bool – si es True, guarda la figura como PDF en 'figures'
        filename: string – nombre base del archivo (sin extensión)
        yscale: str – escala del eje y ('linear' o 'log')
    """

    # Convertir a numpy si es tensor
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred_nofis, torch.Tensor):
        y_pred_nofis = y_pred_nofis.detach().cpu().numpy()
    if isinstance(y_pred_fis, torch.Tensor):
        y_pred_fis = y_pred_fis.detach().cpu().numpy()

    # Desnormalizar si corresponde
    if dataset is not None:
        std = dataset.T_outputs_std.cpu().numpy() if isinstance(dataset.T_outputs_std, torch.Tensor) else dataset.T_outputs_std
        y_true = y_true * std
        y_pred_nofis = y_pred_nofis * std
        y_pred_fis = y_pred_fis * std

    # MAE por paso temporal
    mae_nofis = np.mean(np.abs(y_true - y_pred_nofis), axis=(1, 2))  # (T,)
    mae_fis = np.mean(np.abs(y_true - y_pred_fis), axis=(1, 2))      # (T,)
    # Convertir a escala de tiempo
    steps = np.arange(len(mae_nofis)) * dt  # (T,)

    plt.figure(figsize=(8, 4))
    plt.plot(steps, mae_nofis, marker='o', label=labels[0])
    plt.plot(steps, mae_fis, marker='s', label=labels[1])
    plt.title("Error absoluto medio por paso temporal (desnormalizado)")
    plt.xlabel("Paso temporal t")
    plt.ylabel("MAE [K]" if dataset is not None else "MAE")
    plt.xlim(steps[0], steps[-1])  # Limitar eje x a los valores representados
    if yscale == 'log':
        plt.yscale('log')
    elif yscale == 'linear':
        plt.yscale('linear')
    else:
        raise ValueError("y_scale debe ser 'log' o 'linear'")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_as_pdf:
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf')
    plt.show()
    
    
#%%
def plot_accuracy_curve_by_threshold(T_true, T_pred, umbrales=None, save_as_pdf=False, filename='accuracy_curve_by_threshold'):
    """
    Plots the accuracy curve showing the percentage of nodes that remain within 
    the error threshold throughout the entire sequence.
    
    Parameters:
        T_true: array with shape (T, H, W) - ground truth temperatures
        T_pred: array with shape (T, H, W) - predicted temperatures  
        umbrales: array of error thresholds to evaluate. If None, uses np.linspace(0, 25, 26)
        save_as_pdf: if True, saves the figure as PDF in the 'figures' folder
        filename: base filename (without extension)
    """
    # Default thresholds if not provided
    if umbrales is None:
        umbrales = np.linspace(0, 25, 26)
    
    # Calculate percentages for each threshold
    porcentajes = porcentaje_nodos_siempre_dentro_por_umbral(T_true, T_pred, umbrales)
    
    # =============== FIGURE CONFIGURATION WITH WHITE BACKGROUND ===============
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Plot the accuracy curve
    ax.plot(umbrales, porcentajes, label="Node percentage", linewidth=2, color='#1f77b4')
    
    # Configure axes
    ax.set_xlabel('Error threshold [K]', color='black')
    ax.set_ylabel('Percentage of nodes with error < threshold\nthroughout entire sequence [%]', color='black')
    
    # Title only for visualization (will be removed when saving PDF)
    title_handle = ax.set_title('Cumulative accuracy curve by error threshold', color='black')
    
    # Set axis limits
    ax.set_xlim(0, umbrales[-1])
    ax.set_ylim(0, 100)
    
    # Configure grid
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    # Configure legend and ticks
    ax.legend()
    ax.tick_params(colors='black')
    
    plt.tight_layout()
    
    if save_as_pdf:
        # Remove title before saving
        title_handle.set_visible(False)
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf', facecolor='white')
        # Restore title for visualization
        title_handle.set_visible(True)
    
    plt.show()


def plot_accuracy_comparison_by_threshold(T_true, T_pred_list, model_names, umbrales=None, 
                                        save_as_pdf=False, filename='accuracy_comparison_by_threshold'):
    """
    Plots comparison of accuracy curves for multiple models showing the percentage of nodes 
    that remain within the error threshold throughout the entire sequence.
    
    Parameters:
        T_true: array with shape (T, H, W) - ground truth temperatures
        T_pred_list: list of arrays with shape (T, H, W) - predicted temperatures for each model
        model_names: list of strings with model names for legend
        umbrales: array of error thresholds to evaluate. If None, uses np.linspace(0, 25, 26)
        save_as_pdf: if True, saves the figure as PDF in the 'figures' folder
        filename: base filename (without extension)
    """
    # Default thresholds if not provided
    if umbrales is None:
        umbrales = np.linspace(0, 25, 26)
    
    # Validate inputs
    if len(T_pred_list) != len(model_names):
        raise ValueError("Number of prediction arrays must match number of model names")
    
    # =============== FIGURE CONFIGURATION WITH WHITE BACKGROUND ===============
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Define colors for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Plot accuracy curve for each model
    for i, (T_pred, model_name) in enumerate(zip(T_pred_list, model_names)):
        porcentajes = porcentaje_nodos_siempre_dentro_por_umbral(T_true, T_pred, umbrales)
        color = colors[i % len(colors)]
        ax.plot(umbrales, porcentajes, label=model_name, linewidth=2, color=color)
    
    # Configure axes
    ax.set_xlabel('Error threshold [K]', color='black')
    ax.set_ylabel('Percentage of nodes with error < threshold\nthroughout entire sequence [%]', color='black')
    
    # Title only for visualization (will be removed when saving PDF)
    title_handle = ax.set_title('Model comparison: Cumulative accuracy curve by error threshold', color='black')
    
    # Set axis limits
    ax.set_xlim(0, umbrales[-1])
    ax.set_ylim(0, 100)
    
    # Configure grid
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5)
    
    # Configure legend and ticks
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(colors='black')
    
    plt.tight_layout()
    
    if save_as_pdf:
        # Remove title before saving
        title_handle.set_visible(False)
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf', bbox_inches='tight', facecolor='white')
        # Restore title for visualization
        title_handle.set_visible(True)
    
    plt.show()


def plot_accuracy_metrics_summary(T_true, T_pred_list, model_names, umbrales=[1, 3, 5, 10], 
                                save_as_pdf=False, filename='accuracy_metrics_summary'):
    """
    Creates a summary table/bar chart showing accuracy metrics for specific thresholds.
    
    Parameters:
        T_true: array with shape (T, H, W) - ground truth temperatures
        T_pred_list: list of arrays with shape (T, H, W) - predicted temperatures for each model
        model_names: list of strings with model names
        umbrales: list of specific thresholds to evaluate
        save_as_pdf: if True, saves the figure as PDF in the 'figures' folder
        filename: base filename (without extension)
    """
    # Calculate metrics for each model and threshold
    results = {}
    for model_name, T_pred in zip(model_names, T_pred_list):
        results[model_name] = []
        for umbral in umbrales:
            porcentaje, _, _ = nodos_siempre_dentro_umbral(T_true, T_pred, umbral=umbral)
            results[model_name].append(porcentaje)
    
    # =============== FIGURE CONFIGURATION WITH WHITE BACKGROUND ===============
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Prepare data for bar chart
    x = np.arange(len(umbrales))
    width = 0.8 / len(model_names)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Create bars for each model
    for i, (model_name, percentages) in enumerate(results.items()):
        offset = (i - len(model_names)/2 + 0.5) * width
        color = colors[i % len(colors)]
        bars = ax.bar(x + offset, percentages, width, label=model_name, color=color, alpha=0.8)
        
        # Add value labels on top of bars
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{percentage:.1f}%', ha='center', va='bottom', fontsize=9, color='black')
    
    # Configure axes
    ax.set_xlabel('Error threshold [K]', color='black')
    ax.set_ylabel('Percentage of good nodes [%]', color='black')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{u}K' for u in umbrales])
    
    # Title only for visualization (will be removed when saving PDF)
    title_handle = ax.set_title('Model comparison: Accuracy metrics for specific thresholds', color='black')
    
    # Set axis limits
    ax.set_ylim(0, 105)
    
    # Configure grid
    ax.grid(True, alpha=0.3, color='gray', linestyle='-', linewidth=0.5, axis='y')
    
    # Configure legend and ticks
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(colors='black')
    
    plt.tight_layout()
    
    if save_as_pdf:
        # Remove title before saving
        title_handle.set_visible(False)
        os.makedirs('figures', exist_ok=True)
        plt.savefig(f'figures/{filename}.pdf', format='pdf', bbox_inches='tight', facecolor='white')
        # Restore title for visualization
        title_handle.set_visible(True)
    
    plt.show()
    
    # Print summary table
    print("\n📊 ACCURACY METRICS SUMMARY")
    print("="*60)
    print(f"{'Model':<20}", end="")
    for umbral in umbrales:
        print(f"{'Error < ' + str(umbral) + 'K':<12}", end="")
    print()
    print("-"*60)
    
    for model_name, percentages in results.items():
        print(f"{model_name:<20}", end="")
        for percentage in percentages:
            print(f"{percentage:>10.1f}%", end="  ")
        print()