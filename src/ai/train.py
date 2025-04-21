import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.dataset.manager import DatasetManager
from src.ai.model import TransformerModel
from logger import logger

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Example: Assume data is tokenized
        return torch.tensor(self.data.iloc[idx]["input_ids"], dtype=torch.long)

def train_model(component, dataset, params, model=None):
    """Train a model for a specific component."""
    try:
        # Load dataset
        train_data = dataset  # Assume preprocessed
        train_dataset = CustomDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)

        # Initialize model
        if model is None:
            model = TransformerModel()  # Customize per component
        model = model.cuda() if torch.cuda.is_available() else model
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        for epoch in range(params["epochs"]):
            total_loss = 0
            for batch in train_loader:
                batch = batch.cuda() if torch.cuda.is_available() else batch
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output.view(-1, output.size(-1)), batch.view(-1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{params['epochs']}, Loss: {total_loss/len(train_loader)}")

        # Save model
        model_path = f"data/models/{component}_model.pt"
        torch.save(model.state_dict(), model_path)
        logger.info(f"Model saved: {model_path}")
        return {"model": model, "validation_loss": total_loss/len(train_loader)}
    except Exception as e:
        logger.error(f"Training error for {component}: {e}")
        raise

if __name__ == "__main__":
    manager = DatasetManager()
    data = manager.load_processed_data("dialog")
    params = {"lr": 1e-4, "batch_size": 32, "epochs": 5}
    train_model("dialog", data, params)