from pathlib import Path
import mlflow

# Dossier principal
MLFLOW_DIR = Path("/kaggle/working/mlflow")

# Dossiers de stockage
ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"

MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Base de données SQLite
DB_PATH = MLFLOW_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH}"

# Configuration MLflow
mlflow.set_tracking_uri(TRACKING_URI)

# Créer ou récupérer l'expérience
mlflow.set_experiment("brief_5_experiment")

print("Dossier MLflow :", MLFLOW_DIR)
print("Base SQLite :", DB_PATH)
print("Artifacts :", ARTIFACTS_DIR)