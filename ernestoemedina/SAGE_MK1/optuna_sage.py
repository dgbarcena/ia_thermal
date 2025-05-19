import optuna
import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sage_model import GraphSAGE
from train_eval import train, evaluate
from dataset_utils import get_dataloaders_optuna

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    dir_path = "/content/drive/MyDrive/ErnestoData"
else:
    dir_path = os.getcwd()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders(batch_size, dir_path):
    return get_dataloaders_optuna(batch_size, dir_path)

def objective(trial):
    # --- Hiperparámetros a optimizar ---
    num_layers = trial.suggest_int("num_layers", 3, 10)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    


    # --- Cargar datos ---
    train_loader, val_loader, test_loader, input_dim, norm_info = get_dataloaders(batch_size=batch_size, dir_path=dir_path)

    # --- Crear modelo ---
    model = GraphSAGE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=num_layers,
        use_dropout=True,
        dropout_rate=dropout_rate,
        use_batchnorm=True,
        use_residual=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Scheduler con ReduceLROnPlateau ---
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    # --- Entrenamiento + Evaluación en validación ---
    for epoch in range(50):
        train(
            model,
            train_loader,
            optimizer,
            device,
            norm_info=norm_info,
            use_physics=True,
            lambda_physics=0.003,
            use_boundary_loss=True,
            lambda_boundary=1,
            use_heater_loss=True,
            lambda_heater=10
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            norm_info=norm_info,
            error_threshold=1.0,
            use_physics=True,
            use_boundary_loss=True,
            use_heater_loss=True,
            lambda_physics=0.003,
            lambda_boundary=1,
            lambda_heater=10
        )

        val_total_loss = val_metrics[-1]
        scheduler.step(val_total_loss)

    return val_total_loss  # Métrica a minimizar


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="gcn_study_1",
        storage="sqlite:///gcn_optuna_1.db",  # Persistente para dashboard
        load_if_exists=True
    )

    study.optimize(objective, n_trials=100)

    print("Mejor conjunto de hiperparámetros encontrados:")
    for key, val in study.best_trial.params.items():
        print(f"{key}: {val}")
