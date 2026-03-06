import cv2
import os
import glob
import csv
import json
import subprocess
import shutil # Ajouté pour le nettoyage
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION GLOBALE ---
BASE_DIR = './data/test/video3'
INPUT_DIR = "./data/test/video3"
TEMP_RESIZED_DIR = "./data/test/video3/resized_videos" # Doit être différent de INPUT_DIR
JSON_OUTPUT_DIR = "./data/test/video3/output_jsons"
WEIGHTS = "./weights/v2/tracknetv2_b2e30_adamw_lr1e-4/best_model.pth"
ARCH = "v2"
THRESHOLD = 0.5

def step1_resize_and_rename(input_path, output_path):
    """Nettoie la destination et redimensionne."""
    
    # --- ACTION CORRECTIVE : Vider le dossier de destination s'il existe ---
    if os.path.exists(output_path):
        print(f"🧹 Nettoyage du dossier temporaire : {output_path}")
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    video_extensions = ['*.mp4', '*.mov', '*.MP4', '*.MOV']
    files = []
    # On récupère la liste des fichiers AVANT de commencer à en créer de nouveaux
    for ext in video_extensions:
        files.extend(glob.glob(os.path.join(input_path, ext)))
    
    # Filtrer pour éviter de s'auto-inclure si les dossiers sont imbriqués
    files = [f for f in files if output_path not in f]
    files = sorted(list(set(files))) # Supprime les doublons éventuels

    print(f"--- 🛠️ ÉTAPE 1 : Resize & Renommage ({len(files)} vidéos sources détectées) ---")
    
    for i, v_path in enumerate(files, 1):
        new_name = f"video{i}.mp4"
        final_v_path = os.path.join(output_path, new_name)
        
        cap = cv2.VideoCapture(v_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Si déjà 1280x720, on copie juste avec le nouveau nom
        if (width, height) == (1280, 720):
            print(f"📦 {new_name} : Déjà 720p, copie simple...")
            cap.release()
            shutil.copy2(v_path, final_v_path)
            continue

        print(f"📐 Resizing {new_name} (depuis {width}x{height})...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_v_path, fourcc, fps, (1280, 720))
        
        for _ in tqdm(range(total_frames), desc=f"Processing {new_name}", leave=False):
            ret, frame = cap.read()
            if not ret: break
            out.write(cv2.resize(frame, (1280, 720)))
        
        cap.release()
        out.release()

def step2_run_tracknet():
    """Lance l'inférence via le script track.py"""
    print(f"\n--- 🎾 ÉTAPE 2 : Inférence TrackNet (Architecture: {ARCH}) ---")
    subprocess.run([
        "python", "track.py", 
        TEMP_RESIZED_DIR, 
        WEIGHTS, 
        "--arch", ARCH, 
        "--threshold", str(THRESHOLD),
        "--device", "cuda:0"
    ])

def step3_csv_to_json(inference_results_dir, final_json_dir):
    """Convertit les CSV résultants en JSON 1280x720"""
    print(f"\n--- 📝 ÉTAPE 3 : Conversion CSV vers JSON ---")
    if not os.path.exists(final_json_dir):
        os.makedirs(final_json_dir)

    # On cherche les CSV dans le dossier créé par TrackNet (v2_thresh_0.5)
    search_pattern = os.path.join(inference_results_dir, "**", "*_data.csv")
    csv_files = glob.glob(search_pattern, recursive=True)
    print(f"{len(csv_files)} fichiers CSV trouvés pour conversion.")

    for csv_path in csv_files:
        json_data = []
        # Nom de sortie basé sur le dossier parent (ex: video1_data.json)
        file_name = os.path.basename(csv_path).replace('.csv', '.json')
        target_path = os.path.join(final_json_dir, file_name)

        with open(csv_path, encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('detected') == '1':
                    json_data.append({
                        "image_id": int(row['frame_number']),
                        "Ball_x": int(float(row['x']) * 1280 / 512),
                        "Ball_y": int(float(row['y']) * 720 / 288),
                        "Ball_type": 0
                    })
        
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
        print(f"✨ JSON généré : {file_name}")

# --- EXÉCUTION DU PIPELINE ---
if __name__ == "__main__":
    # # 1. Préparation des vidéos
    step1_resize_and_rename(INPUT_DIR, TEMP_RESIZED_DIR)
    
    # # 2. Prédiction (TrackNet)
    step2_run_tracknet()
    
    # 3. Génération des fichiers JSON finaux
    # Note: TrackNet crée un dossier nommé selon l'architecture et le seuil
    inference_folder = os.path.join(TEMP_RESIZED_DIR, f"{ARCH}_thresh_{THRESHOLD}")
    step3_csv_to_json(inference_folder, JSON_OUTPUT_DIR)
    
    print("\n🚀 PIPELINE TERMINÉ AVEC SUCCÈS !")