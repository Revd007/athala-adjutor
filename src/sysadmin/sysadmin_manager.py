from sklearn.ensemble import IsolationForest
from src.utils.database_manager import DatabaseManager
from logger import logger
from config import DATA_DIR
import pandas as pd
import os

class SysadminManager:
    def __init__(self, model_path=f"{DATA_DIR}/models/threat_intel_model.pt"):
        self.model = IsolationForest(contamination=0.1)
        if os.path.exists(model_path):
            self.model = torch.load(model_path)
        self.db = DatabaseManager()
        logger.info("SysadminManager initialized with pre-trained Isolation Forest")

    def analyze_traffic(self):
        try:
            df = self.db.fetch_crawled_data("threat_intel")
            if df.empty:
                return []
            features = df["text"].apply(len).values.reshape(-1, 1)  # Dummy feature
            anomalies = self.model.predict(features)
            threats = df[anomalies == -1]["text"].tolist()
            logger.info(f"Detected {len(threats)} threats")
            return threats
        except Exception as e:
            logger.error(f"Error analyzing traffic: {str(e)}")
            return []