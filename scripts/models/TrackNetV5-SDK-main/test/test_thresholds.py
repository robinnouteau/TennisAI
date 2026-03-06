import subprocess

# Liste des seuils que tu veux tester
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
input_dir = "ton_dossier_video"
weights = "tes_poids.pth"
architecture = "v2"

for t in thresholds:
    print(f"--- 🧪 Test en cours avec Threshold = {t} ---")
    
    # On appelle ton script original comme une commande système
    subprocess.run([
        "python", "ton_script_original.py", 
        input_dir, 
        weights, 
        "--arch", architecture, 
        "--threshold", str(t)
    ])