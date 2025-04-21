import pandas as pd
from src.finance.market_analyzer import MarketAnalyzer
from logger import logger

class TradingDashboard:
    def __init__(self):
        self.analyzer = MarketAnalyzer()

    def generate_dashboard_data(self, data):
        """Generate data for trading dashboard."""
        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise ValueError("Empty data")

            # Analyze with MarketAnalyzer
            result = self.analyzer.analyze(data)

            # Prepare dashboard data
            dashboard_data = {
                "ohlcv": df[["open", "high", "low", "close", "volume"]].tail(10).to_dict(orient="records"),
                "prediction": result.get("prediction", []),
                "signals": result.get("signals", []),
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            logger.info("Trading dashboard data generated")
            return dashboard_data
        except Exception as e:
            logger.error(f"Trading dashboard error: {e}")
            raise

if __name__ == "__main__":
    dashboard = TradingDashboard()
    sample_data = [
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 1000, "time": "2025-04-20"},
        {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 1200, "time": "2025-04-21"}
    ]
    print(dashboard.generate_dashboard_data(sample_data))