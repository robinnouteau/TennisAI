import subprocess

# Liste des seuils que tu veux tester
thresholds = [0.5]
input_dir = "./data/test/video2"  # Dossier contenant les vidéos à tester
weights = "./weights/v2/tracknetv5/best_model.pth"
architecture = "v5"

for t in thresholds:
    print(f"--- 🧪 Test en cours avec Threshold = {t} ---")
    
    # On appelle ton script original comme une commande système
    subprocess.run([
        "python", "track.py", 
        input_dir, 
        weights, 
        "--arch", architecture, 
        "--threshold", str(t)
    ])