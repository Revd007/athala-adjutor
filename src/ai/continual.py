import torch
from src.utils.dataset_manager import DatasetManager
from src.ai.train_multi import train_model
from logger import logger

class ContinualLearner:
    def __init__(self, component, data_dir="./data", model_path="./data/models"):
        self.component = component
        self.dataset_manager = DatasetManager(data_dir)
        self.model_path = f"{model_path}/{component}_model.pt"

    def update_model(self):
        """Update model with new data."""
        try:
            new_data = self.dataset_manager.detect_new_datasets()
            if not new_data:
                logger.info(f"No new data for {self.component}")
                return

            dataset = self.dataset_manager.load_processed_data(self.component)
            model = torch.load(self.model_path) if os.path.exists(self.model_path) else None
            params = {"lr": 1e-4, "batch_size": 32, "epochs": 5}  # From tune_multi.py
            metrics = train_model(self.component, dataset, params, model)
            torch.save(metrics["model"], self.model_path)
            logger.info(f"Model updated for {self.component}: {metrics}")
        except Exception as e:
            logger.error(f"Continual learning error for {self.component}: {e}")
            raise

if __name__ == "__main__":
    learner = ContinualLearner("dialog")
    learner.update_model()