from pathlib import Path
import mlflow

# Détecter la racine du projet
PROJECT_ROOT = Path.cwd().resolve()

# Si le notebook est exécuté depuis le dossier notebook/
if PROJECT_ROOT.name == "notebook":
    PROJECT_ROOT = PROJECT_ROOT.parent

# Dossier principal
MLFLOW_DIR = PROJECT_ROOT / "mlflow"

# Stockage des artifacts
ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"

# Base de données
DB_PATH = MLFLOW_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{DB_PATH}"

# Créer les dossiers
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Configuration MLflow
mlflow.set_tracking_uri(TRACKING_URI)

# Créer ou récupérer l'expérience
EXPERIMENT_NAME = "neuro_vision_experiment"

# Créer ou récupérer l'expérience
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
EXPERIMENT_ARTIFACT_URI = ARTIFACTS_DIR.as_uri()

if experiment is None:
    experiment_id = mlflow.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=ARTIFACTS_DIR.resolve().as_uri(),
    )
    experiment = mlflow.get_experiment(experiment_id)

mlflow.set_experiment(EXPERIMENT_NAME)

# Vérification
print("Project root :", PROJECT_ROOT)
print("Tracking URI :", mlflow.get_tracking_uri())
print("Database :", DB_PATH.resolve())
print("Artifacts :", experiment.artifact_location)
print("Experiment ID :", experiment.experiment_id)