import torch
import pandas as pd
from src.ai.tcn import TCN
from logger import logger

class MarketAnalyzer:
    def __init__(self, model_path="./data/models/trading_model.pt"):
        self.model = TCN(input_size=5, output_size=1, num_channels=[16, 32, 64])
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def analyze(self, data):
        """Analyze trading data and predict next price."""
        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("Empty data")

            # Preprocess: OHLCV to tensor
            features = df[["open", "high", "low", "close", "volume"]].values
            features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, features]

            # Predict
            with torch.no_grad():
                prediction = self.model(features).item()

            signals = ["buy" if prediction > df["close"].iloc[-1] else "sell"]
            result = {
                "prediction": prediction,
                "signals": signals,
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            logger.info(f"Trading analysis: {result}")
            return result
        except Exception as e:
            logger.error(f"Market analyzer error: {e}")
            raise

if __name__ == "__main__":
    analyzer = MarketAnalyzer()
    sample_data = [
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000},
        {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 1200}
    ]
    print(analyzer.analyze(sample_data))