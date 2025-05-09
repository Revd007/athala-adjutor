import torch
import sys
import os
from pathlib import Path

# --- Add project root to sys.path --- #
# Cari path root (dua level di atas file ini)
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    # Fallback jika struktur tidak seperti yang diharapkan (misal, dijalankan dari root)
    PROJECT_ROOT = Path('.').resolve() 

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f'Added {PROJECT_ROOT} to sys.path')
# --- End sys.path modification --- #

from src.utils.dataset_manager import DatasetManager
from src.ai.train_multi import train_component
from logger import logger

class ContinualLearner:
    def __init__(self, component, data_dir='./data', model_path='./data/models'):
        self.component = component
        self.dataset_manager = DatasetManager()
        self.model_path = f'{model_path}/{component}_model.pt'

    def update_model(self):
        """Update model with new data."""
        try:
            new_data = self.dataset_manager.detect_new_datasets()
            if not new_data:
                logger.info(f"No new data for {self.component}")
                return

            dataset = self.dataset_manager.load_processed_data(self.component)
            model = torch.load(self.model_path) if os.path.exists(self.model_path) else None
            params = {"lr": 1e-4, "batch_size": 32, "epochs": 10}  # From tune_multi.py
            data_path = f'{self.dataset_manager.DATA_DIR}/processed/{self.component}_train.parquet'
            metrics = train_component(self.component, data_path)
            if metrics is not None and 'model' in metrics:
                torch.save(metrics['model'], self.model_path)
                logger.info(f"Model updated for {self.component}: {metrics}")
            else:
                logger.warning(f"No model returned from train_component for {self.component}")
        except Exception as e:
            logger.error(f"Continual learning error for {self.component}: {e}")
            raise

if __name__ == "__main__":
    learner = ContinualLearner("dialog")
    learner.update_model()