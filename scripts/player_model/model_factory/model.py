import os
import torch
from ultralytics import YOLO
from pathlib import Path

class ModelFactory:
    def __init__(self, config):
        self.cfg = config
        self.model_name = self.cfg.model_config['model']
        self.model = YOLO(self.model_name)
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
    
        self.base_dir = Path(__file__).parent.parent.absolute()
        print(f"📁 Base directory for outputs: {self.base_dir}")
        self.project_path = self.base_dir / self.cfg.runtime_config.get('project', 'workdirs')

    def train(self):
        train_args = {**self.cfg.data_args, **self.cfg.training_params, **self.cfg.runtime_config}
        
        train_args['project'] = str(self.project_path)
        
        self.model.train(**train_args)
    
    def evaluate(self):
        data_path = self.cfg.data_args.get('data')
        exp_name = self.cfg.runtime_config.get('name', 'eval')

        # On utilise le chemin absolu ici aussi
        metrics = self.model.val(
            data=data_path,
            device=self.device,
            project=str(self.project_path),
            name=exp_name,
            plots=True
        )
        return metrics

    def test(self):
        data_path = self.cfg.data_args.get('data')
        exp_name = f"{self.cfg.runtime_config.get('name', 'test')}_final"

        metrics = self.model.val(
            data=data_path,
            device=self.device,
            project=str(self.project_path),
            name=exp_name,
            plots=True,
            split='test'
        )
        return metrics
    
    def export(self, format='pth'):
        exp_name = f"{self.cfg.runtime_config.get('name', 'exp')}_export"
        
        path = self.model.export(
            format=format, 
            device=self.device, 
            project=str(self.project_path), 
            name=exp_name
        )
        print(f"✅ Modèle exporté ici : {path}")
