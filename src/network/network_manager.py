from sklearn.ensemble import RandomForestClassifier
from src.utils.database_manager import DatabaseManager
from logger import logger
from config import DATA_DIR
import pandas as pd
import os

class NetworkManager:
    def __init__(self, model_path=f"{DATA_DIR}/models/network_model.pt"):
        self.model = RandomForestClassifier()
        if os.path.exists(model_path):
            self.model = torch.load(model_path)
        self.db = DatabaseManager()
        logger.info("NetworkManager initialized with pre-trained Random Forest")

    def monitor_network(self):
        try:
            df = self.db.fetch_crawled_data("network")
            if df.empty:
                return "No network data"
            features = df["text"].apply(len).values.reshape(-1, 1)  # Dummy feature
            predictions = self.model.predict(features)
            result = f"Network status: {sum(predictions)} active connections"
            logger.info(result)
            return result
        except Exception as e:
            logger.error(f"Error monitoring network: {str(e)}")
            return "Error monitoring network"