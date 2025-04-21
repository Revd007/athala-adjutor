import torch
import torch.nn as nn
from src.utils.database_manager import DatabaseManager
from logger import logger
from config import DATA_DIR
import pandas as pd
import os

class LSTMTCNGRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=64):
        super(LSTMTCNGRU, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.tcn = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.tcn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x

class MarketAnalyzer:
    def __init__(self, model_path=f"{DATA_DIR}/models/trading_model.pt"):
        self.model = LSTMTCNGRU()
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.db = DatabaseManager()
        logger.info("MarketAnalyzer initialized with pre-trained LSTM-TCN-GRU")

    def analyze(self, data):
        try:
            df = pd.DataFrame(data)
            inputs = torch.tensor(df[["open", "high", "low", "close", "volume"]].values, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(inputs).item()
            logger.info(f"Generated trading prediction: {prediction}")
            return {"prediction": prediction}
        except Exception as e:
            logger.error(f"Error analyzing market data: {str(e)}")
            return {"prediction": 0}