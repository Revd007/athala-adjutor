import optuna
import torch
from src.utils.dataset_manager import DatasetManager
from src.ai.train_multi import train_model
from logger import logger

class MultiTuner:
    def __init__(self, component, data_dir="./data"):
        self.component = component
        self.dataset_manager = DatasetManager(data_dir)

    def objective(self, trial):
        """Objective function for Optuna."""
        try:
            params = {
                "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                "epochs": trial.suggest_int("epochs", 5, 20)
            }
            dataset = self.dataset_manager.load_processed_data(self.component)
            metrics = train_model(self.component, dataset, params)
            return metrics["validation_loss"]
        except Exception as e:
            logger.error(f"Tuning error for {self.component}: {e}")
            raise

    def tune(self, n_trials=10):
        """Run hyperparameter tuning."""
        try:
            study = optuna.create_study(direction="minimize")
            study.optimize(self.objective, n_trials=n_trials)
            logger.info(f"Best params for {self.component}: {study.best_params}")
            return study.best_params
        except Exception as e:
            logger.error(f"Tune multi error: {e}")
            raise

if __name__ == "__main__":
    tuner = MultiTuner("trading")
    best_params = tuner.tune()
    print(best_params)