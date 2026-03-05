
from pathlib import Path
import torch

# ------------------- 1. Architecture (Modèle) -------------------
# Note : YOLOv8 utilise généralement des fichiers .yaml pour définir l'architecture.
# On définit ici les paramètres pour l'appel de fonction .train()
model_config = dict(
    model='./models/yolov8m.pt',      # Modèle de base (Nano) - à adapter (s, m, l, x)
    type='YOLOv8m',           # Type générique
)

# ------------------- 2. Données (Data) -------------------
data_root = './data'
input_size = 640  # Taille d'entrée pour YOLOv8 (peut être 320, 640, 1280, etc.)

# Configuration du dataset (format YAML attendu par YOLOv8)
# On peut pointer vers un fichier .yaml existant
data_args = dict(
    data='./data/data.yaml', # Chemin vers votre fichier de description de données
    imgsz=input_size,
    batch=2,                # Équivalent à samples_per_gpu
    workers=4                # Équivalent à workers_per_gpu
)

# ------------------- 3. Optimisation & Hyperparamètres -------------------
# Traduction de votre AdamW lr 1e-4 et de la stratégie de decay
training_params = dict(
    epochs=30,
    optimizer='AdamW',
    lr0=1e-4,                # Initial Learning Rate
    lrf=0.1,                 # Final Learning Rate (lr0 * lrf)
    momentum=0.9,            # Proche de votre config originale
    weight_decay=0.01,       # Standard pour AdamW
    warmup_epochs=0,         # Désactivé comme dans votre exemple commenté
    close_mosaic=10,         # Désactive l'augmentation mosaïque les 10 dernières époques
    seed=42
)

# ------------------- 4. Évaluation & Runtime -------------------
# YOLOv8 gère les hooks de logging (Tensorboard) automatiquement
runtime_config = dict(
    val=True,
    save=True,
    device= 'cuda:0' if torch.cuda.is_available() else 'cpu',             # Automatique (CPU ou GPU)
    project='./workdirs',      # Votre work_dir
    name='yolov8_experiment',    # Nom de l'expérience
    exist_ok=True,
    visualize=True           # Équivalent au VisualizerHook
)