"""
config for YOLOv11m-pose adapted with b2e30 and adamw lr1e-4
"""

from pathlib import Path
import torch

# ------------------- 1. Architecture (Modèle) -------------------
# On passe sur le modèle v11m-pose (Medium)
model_config = dict(
    model='./models/yolo11m-pose.pt',  # Utilisation du dernier modèle v11
    type='YOLO11',            # Type mis à jour pour la version 11
)

# ------------------- 2. Données (Data) -------------------
data_root = './data'
input_size = 640  # Taille standard recommandée pour v11m

# Configuration du dataset
data_args = dict(
    data='./data/data.yaml', # Doit contenir les infos kpt_shape pour le pose estimation
    imgsz=input_size,
    batch=2,                 # Gardé à 2 selon ton setup original
    workers=4
)

# ------------------- 3. Optimisation & Hyperparamètres -------------------
training_params = dict(
    epochs=30,
    optimizer='AdamW',
    lr0=1e-4,                # Ton Learning Rate cible
    lrf=0.1,                 # Final Learning Rate (1e-5 en fin d'entraînement)
    momentum=0.9,
    weight_decay=0.01,
    warmup_epochs=0,         # Toujours désactivé
    close_mosaic=10,         # Important : aide à affiner la précision des points clés
    seed=42,
    # Spécificités Pose
    kobj=1.0,                # Poids de la perte d'objet pour les points clés
    pose=12.0,               # Poids de la perte de position des points clés
)

# ------------------- 4. Évaluation & Runtime -------------------
runtime_config = dict(
    val=True,
    save=True,
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    project='workdirs',
    name='yolo11_pose_experiment',
    exist_ok=True,
    visualize=True           # Génère des prédictions visuelles durant la validation
)