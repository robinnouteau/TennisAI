import importlib
import sys

import cv2
from ultralytics import YOLO
import time
import argparse
from pathlib import Path
from model_factory.model import ModelFactory
def load_config_from_path(config_path: str):
    """
    Dynamically loads a Python file as a module from a given file system path.
    This is useful for loading configuration files without hardcoding imports.
    """
    config_path = Path(config_path)
    
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    
    cfg_module = importlib.util.module_from_spec(spec)
    
    if config_path.stem in sys.modules:
        del sys.modules[config_path.stem]
    
    spec.loader.exec_module(cfg_module)
    return cfg_module

def display_custom_metrics(yolo_results, title="Results"):
    """
    Extracts and displays the specific keys requested using real model output.
    """
    print(f"\n--- 📊 {title} ---")
    
    # Extraction des métriques de base d'Ultralytics
    # YOLO range cela dans results.results_dict ou via des attributs
    try:
        # Note: Certaines valeurs (TP, FP, FN) demandent un calcul basé sur la matrice de confusion
        # ou les résultats de validation détaillés.
        results_dict = yolo_results.results_dict
        
        # Mapping des clés demandées vers les valeurs réelles du modèle
        metrics = {
            'Total': len(yolo_results.results_dict), # Nombre de classes ou d'images
            'TP': getattr(yolo_results, 'tp', "N/A"), # True Positives
            'FP1': "N/A", # Métrique spécifique (souvent liée au tracking)
            'FP2': "N/A", # Métrique spécifique
            'FP': getattr(yolo_results, 'fp', "N/A"),
            'TN': "N/A", 
            'FN': getattr(yolo_results, 'fn', "N/A"),
            'Accuracy': results_dict.get('metrics/accuracy_score', 0.0),
            'Precision': results_dict.get('metrics/precision(B)', 0.0),
            'Recall': results_dict.get('metrics/recall(B)', 0.0),
            'F1-Score': results_dict.get('metrics/f1(B)', 0.0)
        }

        for key, value in metrics.items():
            if isinstance(value, float):
                print(f" {key:10}: {value:.4f}")
            else:
                print(f" {key:10}: {value}")
                
    except Exception as e:
        print(f"Could not extract all keys automatically: {e}")
        # Affiche au moins ce que YOLO propose par défaut si l'extraction spécifique échoue
        print(yolo_results)

    print("-" * 30)

def run_experiment(config_path: Path):
    """
    Runs a full training and evaluation cycle for a given configuration file.
    This is the core function that orchestrates the entire experiment based on the provided config.
    """
    print(f"\n🧪 Running Experiment: {config_path.name}")
    
    cfg = load_config_from_path(str(config_path))
    
    factory = ModelFactory(cfg)
    
    print("\n🛠️  Starting Training...")

    factory.train()
    
    print("\n🔍 Starting Validation...")
    val_metrics = factory.evaluate()
    display_custom_metrics(val_metrics,"Validation Metrics")

    print("\n🧪 Starting Final Testing...")

    test_metrics = factory.test()
    display_custom_metrics(test_metrics,"Test Metrics (Final)")

    print(f"\n📦 Exporting model to engine format...")
    factory.export(format='pth') 


def main():
    print("🚀 Starting player detection using YOLO")
    config_dir = Path('./configs')
    if not config_dir.is_dir():
        print(f"⚠️ Config directory not found at: {config_dir}")
        return
    
    config_paths = sorted([p for p in config_dir.glob('*.py') if p.name != '__init__.py'])
    if not config_paths:
        print("❌ No configuration files (.py) found in ./configs")
        return
    
    print("\n--- 📜 Available Configurations ---")
    for i, path in enumerate(config_paths, 1):
        print(f"  {i:>2}: {path.name}")
    print("---------------------------\n")

    while True:
        selection_str = input("👉 Choose configuration indices to run (e.g., '1 2 3'): ")
        selected_indices = selection_str.split()
        
        experiments_to_run = []
        valid_input = True
        
        if not selected_indices:
            print("⚠️ Input is empty.")
            continue
            
        try:
            for s_idx in selected_indices:
                idx = int(s_idx)
                if 1 <= idx <= len(config_paths):
                    selected_path = config_paths[idx - 1]
                    experiments_to_run.append(selected_path)
                else:
                    print(f"❌ Index {idx} out of range.")
                    valid_input = False
                    break
            if valid_input: break
        except ValueError:
            print("❌ Invalid input. Please enter numbers.")
            if valid_input:
                print("\nYou have chosen to run the following models in order:")
                for i, (_, name) in enumerate(experiments_to_run, 1):
                    print(f"  {i:>2}: {name}")
                break
        # 3. Exécution des expériences
    for config_path in experiments_to_run:
        run_experiment(config_path)

    print("\n🎉 All experiments completed successfully!")



if __name__ == "__main__":
    main()
   