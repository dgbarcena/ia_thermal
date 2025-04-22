import optuna
import torch
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
from gat_model import GAT
from train_eval import train, evaluate
from dataset_utils import get_dataloaders_optuna

def get_dataloaders(batch_size):
    return get_dataloaders_optuna(batch_size)

def objective(trial):
    # --- Hiperparámetros a optimizar ---
    num_layers = trial.suggest_int("num_layers", 10, 30)
    hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.2)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    heads = trial.suggest_categorical("heads", [4, 8])


    # --- Cargar datos ---
    train_loader, val_loader, test_loader, input_dim = get_dataloaders(batch_size=batch_size)

    # --- Crear modelo ---
    model = GAT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        num_layers=num_layers,
        heads=heads,
        use_dropout=True,
        dropout_rate=dropout_rate,
        use_batchnorm=False,
        use_residual=False
    ).to("cuda")

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
        train(model, train_loader, optimizer, "cuda")
        val_loss, _, _, _ = evaluate(model, val_loader, "cuda", error_threshold=1.0)
        scheduler.step(val_loss)

    return val_loss  # Métrica a minimizar

if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="gcn_study_2",
        storage="sqlite:///gcn_optuna_2.db",  # Persistente para dashboard
        load_if_exists=True
    )

    study.optimize(objective, n_trials=100)

    print("Mejor conjunto de hiperparámetros encontrados:")
    for key, val in study.best_trial.params.items():
        print(f"{key}: {val}")
