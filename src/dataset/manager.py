import os
import pandas as pd
import pyarrow.parquet as pq
from logger import logger

class DatasetManager:
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")

    def load_raw_data(self, component):
        """Load raw data for a component."""
        try:
            raw_path = os.path.join(self.raw_dir, f"{component}_dataset.csv")
            if os.path.exists(raw_path):
                data = pd.read_csv(raw_path)
                logger.info(f"Loaded raw data for {component}: {raw_path}")
                return data
            else:
                logger.warning(f"No raw data found for {component}")
                return None
        except Exception as e:
            logger.error(f"Error loading raw data for {component}: {e}")
            raise

    def load_processed_data(self, component):
        """Load processed data for a component."""
        try:
            processed_path = os.path.join(self.processed_dir, f"{component}_train.parquet")
            if os.path.exists(processed_path):
                data = pq.read_table(processed_path).to_pandas()
                logger.info(f"Loaded processed data for {component}: {processed_path}")
                return data
            else:
                logger.warning(f"No processed data found for {component}")
                return None
        except Exception as e:
            logger.error(f"Error loading processed data for {component}: {e}")
            raise

    def detect_new_datasets(self):
        """Detect new datasets in raw directory."""
        try:
            datasets = [f for f in os.listdir(self.raw_dir) if f.endswith('.csv') or f.endswith('.parquet')]
            logger.info(f"Detected datasets: {datasets}")
            return datasets
        except Exception as e:
            logger.error(f"Error detecting new datasets: {e}")
            raise

    def preprocess_data(self, component):
        """Preprocess raw data and save to processed directory."""
        try:
            data = self.load_raw_data(component)
            if data is None:
                return

            # Example preprocessing (customize per component)
            data = data.dropna()
            processed_path = os.path.join(self.processed_dir, f"{component}_train.parquet")
            data.to_parquet(processed_path)
            logger.info(f"Preprocessed data saved for {component}: {processed_path}")
        except Exception as e:
            logger.error(f"Error preprocessing data for {component}: {e}")
            raise

if __name__ == "__main__":
    manager = DatasetManager()
    manager.preprocess_data("dialog")