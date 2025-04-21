import torch
import torch.nn as nn
from src.dataset.manager import DatasetManager
from src.ai.model import TransformerModel
from logger import logger

def evaluate_model(component, dataset, model_path):
    """Evaluate a model for a specific component."""
    try:
        model = TransformerModel()
        model.load_state_dict(torch.load(model_path))
        model = model.cuda() if torch.cuda.is_available() else model
        model.eval()

        data = dataset
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        with torch.no_grad():
            for batch in data:
                batch = torch.tensor(batch, dtype=torch.long)
                batch = batch.cuda() if torch.cuda.is_available() else batch
                output = model(batch)
                loss = criterion(output.view(-1, output.size(-1)), batch.view(-1))
                total_loss += loss.item()

        logger.info(f"Evaluation loss for {component}: {total_loss/len(data)}")
        return {"evaluation_loss": total_loss/len(data)}
    except Exception as e:
        logger.error(f"Evaluation error for {component}: {e}")
        raise

if __name__ == "__main__":
    manager = DatasetManager()
    data = manager.load_processed_data("dialog")
    evaluate_model("dialog", data, "data/models/dialog_model.pt")