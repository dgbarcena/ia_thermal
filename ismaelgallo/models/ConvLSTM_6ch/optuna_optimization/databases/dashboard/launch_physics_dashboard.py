#!/usr/bin/env python3
"""
Script para lanzar el dashboard de Optuna - Optimización de Pesos de Física.
Ejecutar con: python launch_physics_dashboard.py
"""

import optuna

def main():
    # Ruta a la base de datos
    db_path = r"c:\Users\ismael.gallo\Desktop\ia_thermal\ismaelgallo\models\ConvLSTM_6ch\optuna_optimization\databases\physics_weights_optimization_20250616_193544.db"
    storage_url = f"sqlite:///{db_path}"

    print("🚀 Iniciando dashboard de Optuna para Pesos de Física...")
    print(f"📁 Base de datos: {db_path}")
    print("🌐 Dashboard disponible en: http://localhost:8080")
    print("💡 Presiona Ctrl+C para detener")

    try:
        # Lanzar dashboard
        optuna.dashboard.run_server(storage=storage_url, host="localhost", port=8080)
    except KeyboardInterrupt:
        print("\n👋 Dashboard detenido")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Instala optuna-dashboard: pip install optuna-dashboard")

if __name__ == "__main__":
    main()
