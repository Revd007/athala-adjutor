import numpy as np
from src.utils.dataset_manager import DatasetManager
from src.ai.train_multi import train_model
from logger import logger

class ActiveLearner:
    def __init__(self, component, data_dir="./data"):
        self.component = component
        self.dataset_manager = DatasetManager(data_dir)

    def select_informative_samples(self, model, unlabeled_data, n_samples=100):
        """Select most informative samples using uncertainty sampling."""
        try:
            predictions = model.predict_proba(unlabeled_data)
            uncertainties = 1 - np.max(predictions, axis=1)
            indices = np.argsort(uncertainties)[-n_samples:]
            return indices
        except Exception as e:
            logger.error(f"Active learning selection error: {e}")
            raise

    def active_learning(self, n_iterations=3, n_samples=100):
        """Run active learning loop."""
        try:
            dataset = self.dataset_manager.load_processed_data(self.component)
            unlabeled_data = dataset["unlabeled"]
            model = torch.load(f"data/models/{self.component}_model.pt") if os.path.exists(f"data/models/{self.component}_model.pt") else None

            for i in range(n_iterations):
                indices = self.select_informative_samples(model, unlabeled_data, n_samples)
                selected_data = unlabeled_data[indices]
                labeled_data = self.dataset_manager.label_data(selected_data)  # Simulate labeling
                dataset["labeled"] = np.concatenate([dataset["labeled"], labeled_data])
                unlabeled_data = np.delete(unlabeled_data, indices, axis=0)

                params = {"lr": 1e-4, "batch_size": 32, "epochs": 5}
                metrics = train_model(self.component, dataset, params, model)
                model = metrics["model"]
                logger.info(f"Active learning iteration {i+1} for {self.component}: {metrics}")

            torch.save(model, f"data/models/{self.component}_model.pt")
        except Exception as e:
            logger.error(f"Active learning error for {self.component}: {e}")
            raise

if __name__ == "__main__":
    learner = ActiveLearner("captcha")
    learner.active_learning()