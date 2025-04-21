import torch
import torch.nn as nn
from logger import logger

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        logger.info("LSTM initialized")

    def forward(self, x):
        try:
            out, _ = self.lstm(x)
            out = self.fc(out[:, -1, :])
            return out
        except Exception as e:
            logger.error(f"Error in LSTM forward: {str(e)}")
            return None