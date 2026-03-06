import glob
import json
import cv2
import os
import re
import csv

# --- Fonctions de gestion des dossiers ---

def get_next_game_index(directory):
    if not os.path.exists(directory):
        return 1
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    game_nums = [int(re.search(r'game(\d+)', f).group(1)) for f in folders if re.search(r'game(\d+)', f)]
    return max(game_nums) + 1 if game_nums else 1

def preprocess_data(data):
    new_data = []
    # Assurer le tri par image_id
    data = sorted(data, key=lambda x: x['image_id'])
    
    for i in range(len(data) - 1):
        current_pt = data[i]
        next_pt = data[i+1]
        curr_id = current_pt['image_id']
        next_id = next_pt['image_id']
        
        new_data.append({
            "file name": curr_id,
            "visibility": 1,
            "x-coordinate": current_pt['Ball_x'],
            "y-coordinate": current_pt['Ball_y']
        })
        
        diff = next_id - curr_id
        if 1 < diff < 10:
            for j in range(1, diff):
                new_data.append({
                    "file name": curr_id + j,
                    "visibility": 0,
                    "x-coordinate": 0,
                    "y-coordinate": 0
                })
    
    last_pt = data[-1]
    new_data.append({
        "file name": last_pt['image_id'],
        "visibility": 1,
        "x-coordinate": last_pt['Ball_x'],
        "y-coordinate": last_pt['Ball_y']
    })
    return new_data

def extract_clips(data, video_source, current_game_path, min_frames, max_frames=100):
    """Découpe en segments de 60-100 images et extrait les frames."""
    clips = []
    current_clip = []
    
    for i in range(len(data)):
        current_clip.append(data[i])
        
        # Conditions de fermeture du clip :
        # 1. C'est le dernier élément
        # 2. OU il y a une discontinuité dans les frames
        # 3. OU on a atteint la taille maximale (100)
        is_last = (i == len(data) - 1)
        has_gap = (not is_last and data[i+1]['file name'] != data[i]['file name'] + 1)
        is_full = (len(current_clip) >= max_frames)

        if is_last or has_gap or is_full:
            if len(current_clip) >= min_frames:
                clips.append(current_clip)
            current_clip = []

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir {video_source}")
        return

    for clip_idx, clip_data in enumerate(clips, start=1):
        clip_name = f"Clip{clip_idx}"
        clip_folder = os.path.join(current_game_path, clip_name)
        os.makedirs(clip_folder, exist_ok=True)
        
        final_rows = []
        print(f"Extraction de {clip_name} ({len(clip_data)} frames)...")
        
        for local_id, frame_info in enumerate(clip_data):
            original_frame_id = frame_info['file name']
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_id)
            ret, frame = cap.read()
            
            if ret:
                img_name = f"{local_id:04d}.jpg"
                cv2.imwrite(os.path.join(clip_folder, img_name), frame)
                
                row = {
                    "file name": img_name,
                    "visibility": frame_info['visibility'],
                    "x-coordinate": frame_info['x-coordinate'],
                    "y-coordinate": frame_info['y-coordinate']
                }
                final_rows.append(row)

        # Sauvegarde en CSV (Label.csv)
        csv_path = os.path.join(clip_folder, "Label.csv")
        column_names = ["file name", "visibility", "x-coordinate", "y-coordinate"]
        with open(csv_path, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=column_names)
            writer.writeheader()
            writer.writerows(final_rows)

    cap.release()

# --- Main ---

if __name__ == "__main__":
    SOURCE_DIR = "./data/videos/resized_videos/"
    BENCHMARK_DIR = "./data/benchmark"
    MIN_FRAMES = 60
    MAX_FRAMES = 100

    json_files = glob.glob(os.path.join(SOURCE_DIR, "*_selected_points.json"))
    print(f"--- {len(json_files)} fichiers JSON trouvés ---")

    for json_input in sorted(json_files):
        video_base_name = os.path.basename(json_input).replace("_selected_points.json", "")
        video_input = os.path.join(SOURCE_DIR, f"{video_base_name}.mp4")

        if not os.path.exists(video_input):
            continue

        next_game_num = get_next_game_index(BENCHMARK_DIR)
        game_path = os.path.join(BENCHMARK_DIR, f"game{next_game_num}")
        os.makedirs(game_path, exist_ok=True)
        
        print(f"\nTraitement de {video_base_name} -> {game_path}...")

        with open(json_input, "r") as f:
            raw_data = json.load(f)
        
        if raw_data:
            clean_data = preprocess_data(raw_data)
            # On passe MIN et MAX frames ici
            extract_clips(clean_data, video_input, game_path, MIN_FRAMES, MAX_FRAMES)
        
    print(f"\nTerminé !")