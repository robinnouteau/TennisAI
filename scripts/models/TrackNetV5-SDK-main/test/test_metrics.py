import json
import pandas as pd
import numpy as np
from scipy.spatial import distance
from pathlib import Path

def compute_metrics(csv_path, gt_dict, min_dist=5):
    """
    Calcule les métriques pour un CSV donné par rapport à un dictionnaire de vérité terrain.
    """
    # --- CONFIGURATION (Scaling 512x288 -> 1280x720) ---
    scale_x, scale_y = 1280/512, 720/288
    
    # 1. Chargement et nettoyage du CSV
    df_pred = pd.read_csv(csv_path)
    df_pred['frame_number'] = pd.to_numeric(df_pred['frame_number'], errors='coerce')
    df_pred = df_pred.dropna(subset=['frame_number']) 
    df_pred['frame_number'] = df_pred['frame_number'].astype(int)

    tp, fp1, fp2, fn, tn = 0, 0, 0, 0, 0
    
    # 2. Comparaison frame par frame
    for _, row in df_pred.iterrows():
        f_idx = int(row['frame_number'])
        pred_detected = int(row['detected'])
        
        x_pred_scaled = float(row['x']) * scale_x
        y_pred_scaled = float(row['y']) * scale_y
        
        is_in_gt = f_idx in gt_dict
        
        if pred_detected == 1:
            if is_in_gt:
                x_gt, y_gt = gt_dict[f_idx]
                dist = distance.euclidean((x_pred_scaled, y_pred_scaled), (x_gt, y_gt))
                if dist <= min_dist:
                    tp += 1
                else:
                    fp1 += 1 # Erreur de localisation
            else:
                fp2 += 1 # Bruit (Fausse détection)
        else:
            if is_in_gt:
                fn += 1 # Oubli
            else:
                tn += 1 # Vrai négatif

    # 3. Calculs finaux
    eps = 1e-15
    fp = fp1 + fp2
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / (total + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return {
        'Total': total,
        'TP': tp,
        'FP1': fp1,
        'FP2': fp2,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Accuracy': round(accuracy, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1-Score': round(f1, 4)
    }

def main():
    # --- CONFIGURATION DES CHEMINS ---
    base_dir = Path('./data/test')
    video_id = 'video2'
    json_path = './data/test/video2/video2_selected_points.json'
    
    # Liste des seuils à tester
    thresholds = ['0.5']
    
    # 1. Charger le JSON une seule fois pour tout le test
    with open(json_path, 'r') as f:
        gt_data = json.load(f)
    gt_dict = {item['image_id']: (item['Ball_x'], item['Ball_y']) for item in gt_data}

    summary_results = []

    print(f"🚀 Analyse comparative pour {video_id}...")
    print("-" * 50)

    # 2. BOUCLE PRINCIPALE
    for thresh in thresholds:
        csv_path = base_dir /video_id/ f"v2" / video_id / f"{video_id}_data.csv"
        
        if csv_path.exists():
            res = compute_metrics(csv_path, gt_dict)
            res['Thresh'] = thresh
            summary_results.append(res)
            print(f"✅ Seuil {thresh} traité : F1 = {res['F1-Score']}")
        else:
            print(f"❌ Chemin introuvable : {csv_path}")

    # 3. AFFICHAGE FINAL
    if summary_results:
        final_df = pd.DataFrame(summary_results)
        # On réordonne pour que Thresh soit en premier
        print("\n📊 --- TABLEAU RÉCAPITULATIF ---")
        print(final_df.to_string(index=False))
        
        # Trouver le meilleur seuil
        best_f1 = final_df.loc[final_df['F1-Score'].idxmax()]
        print(f"\n✨ Meilleur compromis : Seuil {best_f1['Thresh']} (F1: {best_f1['F1-Score']})")

if __name__ == "__main__":
    main()