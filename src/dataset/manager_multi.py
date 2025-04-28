import os
import pandas as pd
import pyarrow.parquet as pq
from logger import logger  # Assuming you have a logger module
from datasets import load_dataset
import kaggle
import zipfile
from pathlib import Path  # Use pathlib for cleaner path manipulation
import shutil
import re  # For cleaning Wikipedia XML data
import time
import argparse # <-- Tambahkan import argparse

# Additional imports for data generation and scraping
import requests
from bs4 import BeautifulSoup  # For web scraping
import random
from tqdm import tqdm  # For progress bars
import uuid  # For generating unique IDs

# Additional imports for trade data
import yfinance as yf
# You may need to pip install these as well if you don't have them:
#   pip install beautifulsoup4 requests tqdm yfinance

# --- Import MetaTrader5 --- #
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    logger.warning("MetaTrader5 library not found. Trading data will rely solely on yfinance.")
    logger.warning("Install it using: pip install MetaTrader5")
    MT5_AVAILABLE = False
# --- End Import --- #

# --- Define Project Root and Data Dirs ---
# Assuming PROJECT_ROOT is defined correctly as E:/athala-adjutor
# Or derive it dynamically
try:
    # Navigate three levels up from the current file (__file__) to reach the project root
    PROJECT_ROOT = Path(__file__).resolve().parents[2] 
except IndexError:
    # Fallback if the script is run from a different structure (e.g., project root itself)
    PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HF_CACHE_DIR = PROCESSED_DIR / ".cache" # Define desired cache path within processed dir
TEMP_DIR = HF_CACHE_DIR / "temp" # Define a temporary directory within the cache

logger.info(f"Project Root detected/set to: {PROJECT_ROOT}")
logger.info(f"Data Directory set to: {DATA_DIR}")
logger.info(f"Processed Directory set to: {PROCESSED_DIR}")
logger.info(f"Desired Hugging Face Cache Directory: {HF_CACHE_DIR}")
logger.info(f"Temporary Directory for this script: {TEMP_DIR}")

# --- !!! SET Environment Variables !!! ---
# Ensure the cache and temp directories exist before setting the environment variables
try:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True) # Create the temp directory

    hf_home_path_str = str(HF_CACHE_DIR.resolve()) # Get absolute path as string
    temp_path_str = str(TEMP_DIR.resolve())

    os.environ['HF_HOME'] = hf_home_path_str
    os.environ['TRANSFORMERS_CACHE'] = hf_home_path_str
    os.environ['HF_DATASETS_CACHE'] = hf_home_path_str # Explicitly set datasets cache
    os.environ['TEMP'] = temp_path_str # Set TEMP for the script's process
    os.environ['TMP'] = temp_path_str # Set TMP for the script's process

    logger.info(f"Set HF_HOME environment variable to: {hf_home_path_str}")
    logger.info(f"Set TRANSFORMERS_CACHE environment variable to: {hf_home_path_str}")
    logger.info(f"Set HF_DATASETS_CACHE environment variable to: {hf_home_path_str}")
    logger.info(f"Set TEMP environment variable to: {temp_path_str}")
    logger.info(f"Set TMP environment variable to: {temp_path_str}")

except Exception as e:
    logger.error(f"Failed to create cache/temp directory or set environment variables: {e}", exc_info=True)
    # Decide if you want to exit or continue without guaranteed caching location
    # For now, log the error and continue, but downloads might fail or go to default location.
# --- End Environment Variable Setting ---


# --- Now import libraries that use the cache ---
# Import Hugging Face datasets library *after* setting HF_HOME
try:
    from datasets import load_dataset
except ImportError:
    logger.error("Failed to import 'datasets' library. Please install it: pip install datasets")
    # Exit if datasets library is crucial and failed to import
    import sys
    sys.exit("Exiting due to missing 'datasets' library.")
# --- End Deferred Import ---

class DatasetManager:
    def __init__(self, data_dir="./data", limit=None): # <-- Tambahkan limit=None
        self.data_dir = Path(data_dir)  # Store as Path object
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.limit = limit # <-- Simpan limit
        if self.limit is not None:
            logger.info(f"Dataset processing limit set to: {self.limit} rows per component")

        # Create directories if they don't exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self, component):
        """Load raw data for a component."""
        try:
            raw_path = self.raw_dir / f"{component}_dataset.csv"
            if raw_path.exists():
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
            processed_path = self.processed_dir / f"{component}_train.parquet"
            if processed_path.exists():
                data = pq.read_table(str(processed_path)).to_pandas()  # Convert Path to string
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
            processed_path = self.processed_dir / f"{component}_train.parquet"
            data.to_parquet(str(processed_path))  # Convert Path to string
            logger.info(f"Preprocessed data saved for {component}: {processed_path}")
        except Exception as e:
            logger.error(f"Error preprocessing data for {component}: {e}")
            raise

    def download_hf_dataset(self, dataset_name, split="train", cache_dir=None):
        """Downloads a dataset from Hugging Face Hub."""
        logger.info(f"Downloading Hugging Face dataset: {dataset_name} (split: {split})")
        try:
            if cache_dir is None:
                # We rely on HF_HOME etc. set via environment variables
                # cache_dir = self.processed_dir / ".cache"
                pass # No need to redefine cache_dir here if env vars are set
            # cache_dir.mkdir(parents=True, exist_ok=True)
            
            # absolute_cache_dir_str = str(cache_dir.resolve())
            # logger.debug(f"Using cache directory (absolute string): {absolute_cache_dir_str}")
            
            # Log environment variable just before use for confirmation
            logger.debug(f"HF_HOME environment variable is currently: {os.environ.get('HF_HOME')}")
            logger.debug(f"HF_DATASETS_CACHE environment variable is currently: {os.environ.get('HF_DATASETS_CACHE')}")
            logger.debug(f"TRANSFORMERS_CACHE environment variable is currently: {os.environ.get('TRANSFORMERS_CACHE')}")
            logger.debug(f"TEMP environment variable is currently: {os.environ.get('TEMP')}")
            logger.debug(f"TMP environment variable is currently: {os.environ.get('TMP')}")

            headers = {
                'User-Agent': 'AthalaAdjutor/1.0 (contact@example.com)'
            }
            # --- Tambahkan trust_remote_code jika datasetnya adalah codeparrot --- #
            # Determine if trust_remote_code is needed
            trust_remote = False
            if "codeparrot/github-code" in dataset_name or dataset_name == "daily_dialog":
                trust_remote = True
            # trust_remote = True if "codeparrot/github-code" in dataset_name else False # Old logic
            
            if trust_remote:
                logger.warning(f"Allowing remote code execution for dataset: {dataset_name}")
            # Remove explicit cache_dir argument, rely on HF_HOME environment variable
            # dataset = load_dataset(dataset_name, split=split, cache_dir=absolute_cache_dir_str, trust_remote_code=trust_remote)
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote)
            # --- Akhir Penambahan --- #
            logger.info(f"Successfully loaded dataset: {dataset_name}")
            return dataset
        except Exception as e:
            # Let's check the specific error for daily_dialog again
            if dataset_name == "daily_dialog" and "trust_remote_code=True" in str(e):
                 logger.error(f"Failed to load {dataset_name}. It requires `trust_remote_code=True`. Attempting reload with the flag.")
                 try:
                      # Retry specifically for daily_dialog with the flag
                      dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
                      logger.info(f"Successfully reloaded dataset with trust_remote_code=True: {dataset_name}")
                      return dataset
                 except Exception as retry_e:
                     logger.error(f"Retry failed for {dataset_name}: {retry_e}")
                     # Fall through to generic error logging
            
            logger.error(f"Failed to download/load Hugging Face dataset {dataset_name}: {e}")
            # Log the cache directory being used if error occurs
            logger.error(f"HF_HOME environment variable was set to: {os.environ.get('HF_HOME')}")
            logger.error(f"HF_DATASETS_CACHE environment variable was set to: {os.environ.get('HF_DATASETS_CACHE')}")
            logger.error(f"TRANSFORMERS_CACHE environment variable was set to: {os.environ.get('TRANSFORMERS_CACHE')}")
            logger.error(f"TEMP environment variable was set to: {os.environ.get('TEMP')}")
            logger.error(f"TMP environment variable was set to: {os.environ.get('TMP')}")
            return None

    def download_kaggle_dataset(self, dataset_slug, file_name=None, unzip=True):
        """Downloads a dataset from Kaggle."""
        logger.info(f"Downloading Kaggle dataset: {dataset_slug}")
        target_dir = self.processed_dir / "kaggle_downloads" / dataset_slug.replace('/', '_')
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            kaggle.api.authenticate() # Ensure authentication
            kaggle.api.dataset_download_files(dataset_slug, path=str(target_dir), quiet=False, unzip=False) # Download zip first
            
            downloaded_files = list(target_dir.glob('*.zip'))
            if not downloaded_files:
                logger.error(f"No zip file found after attempting download for {dataset_slug}")
                return None

            zip_path = downloaded_files[0] # Assume one zip file
            logger.info(f"Downloaded {zip_path.name} from Kaggle dataset: {dataset_slug}")

            if unzip:
                logger.info(f"Unzipping {zip_path.name}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_dir)
                logger.info(f"Unzipped files to {target_dir}")
                # Optionally remove the zip file after extraction
                # os.remove(zip_path) 
            
            # Return the directory containing the (potentially unzipped) files
            return target_dir
        except Exception as e:
            logger.error(f"Failed to download/unzip Kaggle dataset {dataset_slug}: {e}")
            # Attempt cleanup of potentially partial download
            shutil.rmtree(str(target_dir), ignore_errors=True)
            return None

    def prepare_dialog_data(self, dataset_name="daily_dialog", output_filename="dialog_train.parquet"):
        """Prepares dialog training data."""
        try:
            # Panggil dataset dengan benar
            dataset = self.download_hf_dataset(dataset_name, split="train")
            if dataset is None:
              return

            # Terapkan limit jika ada
            if self.limit is not None:
                limit_count = min(len(dataset), self.limit)
                logger.info(f"Applying limit: Selecting first {limit_count} rows for dialog data.")
                # Use slicing for Hugging Face datasets before converting to pandas
                # dataset = dataset.select(range(limit_count)) # This should work
                # Alternative if select fails for some reason:
                indices = list(range(limit_count))
                dataset = dataset.select(indices)

            df = dataset.to_pandas()

            # Proses data
            if 'dialog' in df.columns:
                df['text'] = df['dialog'].apply(lambda x: "\n".join(x))
            elif 'text' in df.columns:
                df['text'] = df['text']
            else:
                logger.error("Kolom teks tidak ditemukan")
                return

            df[['text']].to_parquet(str(self.processed_dir / output_filename))  # Convert Path to string

        except Exception as e:
            logger.error(f"Gagal memproses dialog: {e}")

    def prepare_coding_data(self, dataset_name="codeparrot/github-code", output_filename="coding_train.parquet"):
        """Prepares coding training data."""
        try:
            dataset = self.download_hf_dataset(dataset_name, split="train")

            if dataset is None:
              return

            # Filter Python dulu sebelum limit (opsional, bisa dibalik jika perlu sampel semua bahasa)
            dataset = dataset.filter(lambda example: example['language'] == 'Python')
            logger.info(f"Filtered for Python code. Rows remaining: {len(dataset)}")

            # Terapkan limit jika ada
            if self.limit is not None:
                limit_count = min(len(dataset), self.limit)
                logger.info(f"Applying limit: Selecting first {limit_count} rows for coding data.")
                dataset = dataset.select(range(limit_count))

            df = dataset.to_pandas()
            df.rename(columns={'code': 'content'}, inplace=True)
            df[['content']].to_parquet(str(self.processed_dir / output_filename))  # Convert Path to string


        except Exception as e:
            logger.error(f"Gagal memproses kode: {e}")

    def prepare_math_data(self, dataset_name="gsm8k", output_filename="math_train.parquet"):
        """Prepares math training data."""
        try:

            dataset = self.download_hf_dataset(dataset_name, split="train", cache_dir = self.processed_dir / ".cache")
            if dataset is None:
              return

            # Terapkan limit jika ada
            if self.limit is not None:
                limit_count = min(len(dataset), self.limit)
                logger.info(f"Applying limit: Selecting first {limit_count} rows for math data.")
                dataset = dataset.select(range(limit_count))

            df = dataset.to_pandas()
            df.rename(columns={'question': 'problem', 'answer': 'solution'}, inplace=True)
            df[['problem', 'solution']].to_parquet(str(self.processed_dir / output_filename))  # Convert Path to string


        except Exception as e:
            logger.error(f"Gagal memproses matematika: {e}")

    def prepare_network_data(self, dataset_slug="unsw-nb15", output_filename="network_train.parquet"):
        """Prepares network data using Kaggle API."""
        try:
            data_dir = self.download_kaggle_dataset(dataset_slug)
            if data_dir is None:
                return  # Exit if the dataset wasn't downloaded

            # Load data from all CSV files within the directory
            all_data = []
            for file_path in data_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(file_path)
                    all_data.append(df)
                    logger.info(f"Successfully loaded: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

            if not all_data:
                logger.error("No data loaded from CSV files.")
                return

            # Concatenate all DataFrames into a single DataFrame
            combined_data = pd.concat(all_data, ignore_index=True)

            # Terapkan limit jika ada
            if self.limit is not None:
                limit_count = min(len(combined_data), self.limit)
                logger.info(f"Applying limit: Selecting first {limit_count} rows for network data.")
                combined_data = combined_data.head(limit_count)

            # Save the combined data to a Parquet file
            parquet_path = self.processed_dir / output_filename
            combined_data.to_parquet(str(parquet_path))  # Convert Path to string
            logger.info(f"Combined network data saved to: {parquet_path}")

        except Exception as e:
            logger.error(f"Failed to process network data: {e}")

    def prepare_trading_data(self, output_filename="trading_train.parquet"):
         """Prepares trading data using yfinance and MetaTrader5 (if available)."""
         try:
             logger.info("Preparing trading data using yfinance and MetaTrader5 (if available)...")
             from datetime import datetime, timedelta

             # --- Pisahkan Ticker untuk yfinance dan MT5 --- #
             yfinance_tickers = [
                 # Crypto (cocok untuk yfinance)
                 "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "LINK-USD",
                 # Major US Stocks (Tech)
                 "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AMD", "INTC", "CSCO", "ORCL", "ADBE", "CRM", "QCOM", "TXN", "AVGO",
                 # Major US Stocks (Finance)
                 "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "AXP", "V", "MA", "PYPL", "SQ", "COIN",
                 # Major US Stocks (Healthcare)
                 "JNJ", "PFE", "UNH", "MRK", "LLY", "ABBV", "TMO", "MDT", "DHR", "ABT", "BMY", "GILD",
                 # Major US Stocks (Consumer Goods - Cyclical & Defensive)
                 "PG", "KO", "PEP", "WMT", "COST", "HD", "NKE", "MCD", "SBUX", "TGT", "LOW", "DIS",
                 # Major US Stocks (Energy)
                 "XOM", "CVX", "SLB", "COP", "EOG", "MPC", "PSX",
                 # Major US Stocks (Industrials)
                 "GE", "BA", "CAT", "HON", "UPS", "FDX", "RTX", "LMT", "DE",
                 # Major US Stocks (Utilities)
                 "NEE", "DUK", "SO", "AEP", "EXC",
                 # Major US Stocks (Real Estate)
                 "AMT", "PLD", "EQIX", "CCI", "SPG",
                 # Major Indices
                 "^GSPC", "^DJI", "^IXIC", "^RUT", # US
                 "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI", # International
                 # Major ETFs
                 "SPY", "IVV", "VOO", "QQQ", "DIA", "IWM", "EFA", "IEFA", "VEA", "EEM", "VWO",
                 "GLD", "IAU", # Gold ETFs (biarkan di yfinance sebagai alternatif)
                 "USO", "BNO", # Oil ETFs
                 "AGG", "BND", # Bond ETFs
                 "ARKK", "ARKG", # Innovation ETFs
                 # Futures (jika tersedia di yfinance)
                 "CL=F", "GC=F", "SI=F", "HG=F", "NG=F", "ZC=F", "ZS=F", "ZW=F",
             ] # Total ticker yfinance

             mt5_symbols = [
                 # Forex (Gunakan format MT5)
                 "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD",
                 "EURGBP", "EURJPY", "GBPJPY",
                 # Komoditas (Gunakan format MT5, contoh)
                 "XAUUSD", # Gold
                 "XAGUSD", # Silver
                 # "WTI", "BRENT" # Nama bisa bervariasi tergantung broker
             ]
             # --- Akhir Pemisahan Ticker --- #

             start_date_dt = datetime.strptime("2020-01-01", '%Y-%m-%d')
             end_date_dt = datetime.now()
             end_date_str = end_date_dt.strftime('%Y-%m-%d')

             all_data = [] # List untuk menampung semua DataFrame

             # --- Ambil data dari MetaTrader 5 --- #
             mt5_data_frames = []
             if MT5_AVAILABLE:
                 mt5_initialized = False
                 logger.info("Attempting to initialize MetaTrader 5...")
                 try:
                     if not mt5.initialize():
                         logger.error(f"MetaTrader 5 initialize() failed, error code = {mt5.last_error()}")
                         logger.warning("Ensure MT5 terminal is running and logged in.")
                     else:
                         mt5_initialized = True
                         logger.info("MetaTrader 5 initialized successfully.")
                         if mt5.terminal_info() is None:
                             logger.error("Could not connect to MetaTrader 5 terminal.")
                             mt5_initialized = False # Set false if connection fails
                         else:
                             logger.info("Connected to MetaTrader 5 terminal.")
                             logger.info(f"Requesting MT5 data from {start_date_dt} to {end_date_dt} for symbols: {mt5_symbols}")
                             for symbol in mt5_symbols:
                                 try:
                                     rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start_date_dt, end_date_dt)
                                     if rates is None or len(rates) == 0:
                                         logger.warning(f"No data received from MT5 for {symbol}: {mt5.last_error()}")
                                         continue

                                     df_mt5 = pd.DataFrame(rates)
                                     df_mt5['Date'] = pd.to_datetime(df_mt5['time'], unit='s')
                                     df_mt5.set_index('Date', inplace=True)
                                     df_mt5.rename(columns={
                                         'open': 'Open',
                                         'high': 'High',
                                         'low': 'Low',
                                         'close': 'Close',
                                         'tick_volume': 'Volume'
                                     }, inplace=True)
                                     df_mt5 = df_mt5[['Open', 'High', 'Low', 'Close', 'Volume']]
                                     df_mt5['Symbol'] = symbol
                                     df_mt5['SMA_20'] = df_mt5['Close'].rolling(window=20).mean()
                                     df_mt5['EMA_12'] = df_mt5['Close'].ewm(span=12, adjust=False).mean()
                                     df_mt5['Volatility'] = df_mt5['High'] - df_mt5['Low']
                                     mt5_data_frames.append(df_mt5)
                                     logger.info(f"Successfully retrieved {len(df_mt5)} records from MT5 for {symbol}")
                                 except Exception as e_mt5:
                                     logger.error(f"Error processing MT5 symbol {symbol}: {e_mt5}")
                 finally:
                     if mt5_initialized:
                         mt5.shutdown()
                         logger.info("MetaTrader 5 connection shut down.")
             all_data.extend(mt5_data_frames)
             # --- Akhir Blok MT5 --- #

             # --- Ambil data dari yfinance --- #
             logger.info(f"{len(mt5_data_frames)} dataframes collected from MT5.")
             logger.info(f"Downloading yfinance data from {start_date_dt.strftime('%Y-%m-%d')} to {end_date_str} for {len(yfinance_tickers)} tickers.")
             yfinance_data_frames = []
             for ticker in yfinance_tickers:
                 try:
                     df_yf = yf.download(ticker, start=start_date_dt.strftime('%Y-%m-%d'), end=end_date_str)
                     if not df_yf.empty:
                         df_yf['Symbol'] = ticker
                         df_yf['SMA_20'] = df_yf['Close'].rolling(window=20).mean()
                         df_yf['EMA_12'] = df_yf['Close'].ewm(span=12, adjust=False).mean()
                         df_yf['Volatility'] = df_yf['High'] - df_yf['Low']
                         yfinance_data_frames.append(df_yf)
                         logger.info(f"Successfully downloaded data using yfinance for {ticker}")
                     else:
                         logger.warning(f"No data found using yfinance for {ticker} between {start_date_dt.strftime('%Y-%m-%d')} and {end_date_str}")
                 except Exception as e:
                     logger.error(f"Error downloading ticker {ticker} using yfinance: {e}")
             all_data.extend(yfinance_data_frames)
             # --- Akhir Blok yfinance --- #

             logger.info(f"{len(yfinance_data_frames)} dataframes collected from yfinance.")
             logger.info(f"Total dataframes collected from all sources: {len(all_data)}")

             if not all_data:
                 logger.error("No data was downloaded from any source.")
                 return

             # Combine all data
             combined_data = pd.concat(all_data, axis=0)
             logger.info(f"Shape after concatenation: {combined_data.shape}")
             combined_data = combined_data.dropna() # Drop rows with NaN from indicators
             logger.info(f"Shape after dropping NaNs: {combined_data.shape}")

             # Final check if data remains after cleaning
             if combined_data.empty:
                 logger.error("No data remaining after cleaning (dropping NaNs). Cannot save trading data.")
                 return

             # Terapkan limit jika ada
             if self.limit is not None and len(combined_data) > self.limit:
                 limit_count = self.limit
                 logger.info(f"Applying limit: Selecting first {limit_count} rows for combined trading data.")
                 combined_data = combined_data.head(limit_count)
             elif self.limit is not None:
                 logger.info(f"Limit ({self.limit}) is greater than or equal to the number of rows ({len(combined_data)}), no limiting applied for trading data.")

             # Save to parquet
             parquet_path = self.processed_dir / output_filename
             combined_data.to_parquet(str(parquet_path))
             logger.info(f"Combined trading data saved to: {parquet_path}")

         except Exception as e:
             logger.error(f"Error in prepare_trading_data: {e}", exc_info=True)

    def prepare_captcha_data(self, dataset_slug="google-street-view-house-numbers", output_filename="captcha_train.parquet"):
        """Prepares CAPTCHA data from Kaggle."""
        try:
            data_dir = self.download_kaggle_dataset(dataset_slug)
            if data_dir is None:
                return

            # Find the train CSV file (the other one is extra)
            train_file_path = next((f for f in data_dir.glob("*.csv") if 'train' in f.name.lower()), None)

            if not train_file_path:
                logger.error("No 'train' CSV file found in the downloaded dataset.")
                return

            # Load data from the CSV
            try:
                df = pd.read_csv(train_file_path)
                logger.info(f"Successfully loaded: {train_file_path}")
            except Exception as e:
                logger.warning(f"Failed to load {train_file_path}: {e}")
                return

            parquet_path = self.processed_dir / output_filename
            # Terapkan limit jika ada
            if self.limit is not None:
                limit_count = min(len(df), self.limit)
                logger.info(f"Applying limit: Selecting first {limit_count} rows for captcha data.")
                df = df.head(limit_count)

            df.to_parquet(str(parquet_path))
            logger.info(f"CAPTCHA data saved to: {parquet_path}")

        except Exception as e:
            logger.error(f"Failed to process CAPTCHA data: {e}")

    def prepare_threat_intel_data(self, output_filename="threat_intel_train.parquet"):
        """Placeholder for preparing threat intelligence data."""
        logger.warning("Threat intelligence data preparation is a placeholder and not implemented.")
        # Replace this with actual threat intelligence data preparation logic.
        # May involve external APIs or local datasets.
        # Example (using a hypothetical API):
        # try:
        #     # Replace with your API key and endpoint
        #     api_key = "YOUR_THREAT_INTEL_API_KEY"
        #     endpoint = "https://api.example.com/threats"
        #     response = requests.get(endpoint, headers={"Authorization": f"Bearer {api_key}"})
        #     response.raise_for_status()
        #     data = response.json()
        #     df = pd.DataFrame(data) # Assuming the API returns JSON
        #     df.to_parquet(str(self.processed_dir / output_filename))
        # except Exception as e:
        #     logger.error(f"Error fetching threat intel data: {e}")
        pass

    def prepare_rag_data(self, output_filename="rag_train.parquet"):
        """Placeholder for preparing RAG data."""
        logger.warning("RAG data preparation is a placeholder and not implemented.")
        # This might involve combining text from other sources
        # Example:
        # 1. Load text data from various sources (e.g., text files, web pages)
        # 2. Chunk the text into smaller segments
        # 3. Embed the text chunks using a model like SentenceTransformers
        # 4. Store the embeddings in a vector database (e.g., FAISS)

        pass

    # --- New methods for advanced data sources ---

    def prepare_reddit_conversations(self, subreddit="CasualConversation", output_filename="reddit_conversations.parquet", start_year=2015, end_year=2023):
        """Extract Reddit conversations from Pushshift API."""
        try:
            logger.info(f"Extracting Reddit conversations from r/{subreddit} ({start_year}-{end_year})")
            all_posts = []
            for year in tqdm(range(start_year, end_year + 1), desc="Processing Years"):
                for month in range(1, 13):
                    #Pushshift API
                    url = f"https://api.pushshift.io/reddit/search/submission/?subreddit={subreddit}&after={year}-{str(month).zfill(2)}-01&before={year}-{str(month + 1).zfill(2)}-01&size=500"
                    try:
                        response = requests.get(url)
                        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                        data = response.json()['data']
                        for post in data:
                            all_posts.append({
                                "title": post.get('title', ''),  # Title of the post
                                "selftext": post.get('selftext', ''), # Main content of the post
                                "author": post.get('author', ''), # Author of the post
                                "created_utc": post.get('created_utc', ''), # Time created
                                "url": post.get('url', ''), # URL of the post
                                "subreddit": post.get('subreddit', '') # Subreddit name
                            })
                        time.sleep(0.1) #sleep to avoid rate limit
                    except requests.exceptions.RequestException as e:
                        logger.warning(f"Error during Pushshift request: {e}")

            df = pd.DataFrame(all_posts)
            df['text'] = df['title'] + '\n' + df['selftext']  # Combine title and text

            # Terapkan limit jika ada
            if self.limit is not None:
                limit_count = min(len(df), self.limit)
                logger.info(f"Applying limit: Selecting first {limit_count} rows for Reddit data.")
                df = df.head(limit_count)

            df[['text']].to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"Reddit conversations saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error extracting Reddit conversations: {e}")

    def prepare_wikipedia_data(self, output_filename="wikipedia_train.parquet"):
        """Fetches Wikipedia articles via API."""
        try:
            logger.info("Fetching Wikipedia articles via API.")

            wikipedia_articles = []
            # Daftar judul bisa diperbanyak jika perlu sampel lebih besar
            example_titles = [
                # ... (Daftar judul yang sudah ada) ...
                "Artificial intelligence", "Machine learning", "Computer science",
                "Internet", "Digital technology", "Robotics",
                "World War II", "Ancient Egypt", "Roman Empire",
                "Industrial Revolution", "Renaissance", "French Revolution",
                "American Civil War", "Cold War", "Ancient Greece",
                "Physics", "Chemistry", "Biology",
                "Astronomy", "Evolution", "Quantum mechanics",
                "Classical music", "Renaissance art", "Modern art",
                "Literature", "Theatre", "Film history",
                "Western philosophy", "Eastern philosophy", "Ethics",
                "Psychology", "Sociology", "Economics",
                "Political science", "Anthropology", "Archaeology",
                "Buddhism", "Christianity", "Islam",
                "Hinduism", "Judaism", "World religions",
                "Physical geography", "Human geography", "Climate",
                "Mathematics", "Geometry", "Algebra",
                "Medicine", "Human anatomy", "Public health",
                "Architecture", "Engineering", "Agriculture",
                "Education", "Sports", "Food science"
                # Tambahkan judul lain di sini jika perlu
            ]
            logger.info(f"Fetching {len(example_titles)} sample articles from Wikipedia API...")
            for title in tqdm(example_titles, desc="Fetching Wikipedia Articles"):
                try:
                    # Gunakan User-Agent yang baik
                    headers = {'User-Agent': 'AthalaAdjutor/1.0 (Data Collection Script; contact@example.com)'}
                    response = requests.get(f"https://en.wikipedia.org/w/api.php?action=query&format=json&titles={title}&prop=extracts&exintro&explaintext", headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    page_id = next(iter(data['query']['pages']))
                    page = data['query']['pages'][page_id]
                    # Pastikan ada teks ekstrak
                    if 'extract' in page and page['extract']:
                        wikipedia_articles.append({"title": page.get('title', title), "text": page['extract']})
                    else:
                        logger.warning(f"No extract found for Wikipedia article '{title}'. Page data: {page}")
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error fetching Wikipedia article '{title}': {e}")
                time.sleep(0.1) # Jeda sopan agar tidak membebani API

            if not wikipedia_articles:
                 logger.error("No Wikipedia articles could be fetched. Skipping.")
                 return

            wikipedia_df = pd.DataFrame(wikipedia_articles)
            wikipedia_df['source'] = "Wikipedia"

            # --- Proses hanya DataFrame Wikipedia --- #
            # Terapkan limit jika ada
            if self.limit is not None:
                limit_count = min(len(wikipedia_df), self.limit)
                logger.info(f"Applying limit: Selecting first {limit_count} rows for Wikipedia data.")
                wikipedia_df = wikipedia_df.head(limit_count)

            wikipedia_df.to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"Wikipedia data saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error preparing Wikipedia data: {e}", exc_info=True)

    def prepare_stack_overflow(self, archive_url="https://archive.org/download/stackexchange/stackoverflow.com-Posts.7z", output_filename="stackoverflow.parquet"):
        """Extract Stack Overflow data from Internet Archive. This is a stub.
        Due to the size and complexity of the Stack Overflow archive, this requires more advanced processing.
        Consider using tools like 7-Zip to extract the XML files, then using an XML parser to extract the relevant data.
        This is left as a placeholder due to the complexity.
        """
        logger.warning("Stack Overflow data extraction is a placeholder and not implemented.")
        # This is a significantly more complex task.  Consider using:
        # 1.  `7z` command-line tool to extract the XML files.
        # 2.  `xml.etree.ElementTree` or `lxml` to parse the XML.
        # 3.  Iterate through the XML, extracting `Question` and `Answer` posts.
        # 4.  Clean the HTML content (Stack Overflow posts contain HTML).
        pass

    def prepare_github_codeparrot_plus(self, output_filename="github_codeparrot_plus.parquet", min_stars=100, start_year=2020, end_year=2025):
      """
      This method is a placeholder for a more complex process.
      It requires more advanced techniques for filtering GitHub repositories, parsing code,
      and handling large datasets.
      """
      logger.warning("GitHub CodeParrot+ data extraction is a placeholder and not implemented. Requires tree-sitter and GitHub API.")
      # This method would require significant work involving:
      # 1. Using the GitHub API to search for repositories with > min_stars and active commits in the specified years.
      # 2. Cloning the repositories.
      # 3. Using `tree-sitter` to parse the code and extract Abstract Syntax Trees (ASTs).
      # 4. Converting the ASTs into a suitable format for training.

    def prepare_lean4_theorem_proving(self, output_filename="lean4_theorems.parquet"):
        """Placeholder for extracting Lean4 theorem proving data. Requires knowledge of Lean4 format."""
        logger.warning("Lean4 theorem extraction is a placeholder and not implemented. Requires Lean4 parsing.")
        # Requires understanding the Lean4 language and file format.

    def prepare_imo_shortlists(self, output_filename="imo_problems.parquet", start_year=1959, end_year=2024):
        """
        Scrapes IMO problem shortlists from the official website.
        """
        try:
            logger.info(f"Scraping IMO problem shortlists ({start_year}-{end_year})")
            all_problems = []
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            for year in tqdm(range(start_year, end_year + 1), desc="Processing Years"):
                url = f"https://www.imo-official.org/problems.aspx?yr={year}"
                try:
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    soup = BeautifulSoup(response.content, 'html.parser')

                    # Adjust the selector based on the actual website structure. Inspect the website!
                    # --- USER VERIFICATION NEEDED for Selectors --- #
                    # Example: Inspect the page for year 2023 and find the correct table/row/cell tags and classes.
                    problem_tables_selector = 'table.problems' # Example: Find tables with class 'problems' (NEEDS VERIFICATION)
                    row_selector = 'tr'                   # Example: Find all table rows (NEEDS VERIFICATION)
                    cell_index_problem = 0             # Example: Problem text is in the first cell (index 0) (NEEDS VERIFICATION)
                    # cell_index_solution = 1            # Example: Solution text is in the second cell (index 1) (NEEDS VERIFICATION)
                    # --- END USER VERIFICATION NEEDED --- #

                    problem_tables = soup.select(problem_tables_selector)
                    logger.debug(f"Year {year}: Found {len(problem_tables)} elements matching selector '{problem_tables_selector}'")

                    for table_index, table in enumerate(problem_tables):
                        rows = table.select(row_selector)
                        logger.debug(f"  Table {table_index}: Found {len(rows)} elements matching selector '{row_selector}'")
                        rows_processed = 0
                        for row in table.find_all('tr'):
                            cells = row.find_all('td')
                            # Ensure we have enough cells based on expected indices
                            if len(cells) > cell_index_problem: # Check if the problem cell index exists
                                problem = cells[0].text.strip()
                                #answer = cells[1].text.strip() # uncomment if solution is present and needs to be extracted
                                all_problems.append({"year": year, "problem": problem})  # , "solution": answer})
                                rows_processed += 1
                            #else: # Optional: Log rows that don't have enough cells
                            #    logger.debug(f"    Row skipped: Found {len(cells)} cells, expected at least {max(cell_index_problem, cell_index_solution) + 1}")
                        logger.debug(f"  Table {table_index}: Successfully processed {rows_processed} rows.")

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error fetching IMO problems for {year}: {e}")

            logger.info(f"Total problems scraped from all years: {len(all_problems)}")

            if not all_problems:
                logger.error("No problems were scraped from the IMO website. Cannot save IMO data.")
                return

            df = pd.DataFrame(all_problems)
            # Terapkan limit jika ada
            if self.limit is not None:
                limit_count = min(len(df), self.limit)
                logger.info(f"Applying limit: Selecting first {limit_count} rows for IMO data.")
                df = df.head(limit_count)

            df.to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"IMO problems saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error extracting IMO problems: {e}")

    def prepare_cic_darknet2020(self, output_filename="cic_darknet2020.parquet"):
      """Placeholder for extracting CIC-Darknet2020 data.  Requires dataset download and parsing."""
      logger.warning("CIC-Darknet2020 extraction is a placeholder and not implemented. Requires dataset and parsing.")
      # This requires downloading the 2.5TB dataset from the University of New Brunswick.
      # It then requires parsing the network traffic data (likely pcap files).
      # Consider using tools like `tcpdump` or `Wireshark` to analyze the traffic.

    def prepare_malware_dna_db(self, output_filename="malware_dna.parquet"):
      """Placeholder for extracting Malware DNA data. Requires VirusShare/Hybrid Analysis access and API usage."""
      logger.warning("Malware DNA extraction is a placeholder and not implemented. Requires API access and malware analysis.")
      # This requires access to VirusShare or Hybrid Analysis and their APIs.
      # It then requires malware analysis to extract behavioral metadata.

    # --- Synthetic Data Generation Methods ---
    def generate_gan_captcha(self, num_captchas=1000000, output_filename="synthetic_captcha.parquet"):
      """Placeholder for GAN-based CAPTCHA generation. Requires GAN model."""
      logger.warning("GAN-based CAPTCHA generation is a placeholder and not implemented. Requires GAN model (StyleGAN3).")
      # This requires training a StyleGAN3 model on CAPTCHA images.
      # It then uses the trained model to generate synthetic CAPTCHAs.
      # The generated CAPTCHAs should have perturbations and noise.

    def simulate_darkweb_marketplace(self, num_transactions=10000, output_filename="darkweb_transactions.parquet"):
      """Placeholder for dark web marketplace simulation. Requires RL framework."""
      logger.warning("Dark web marketplace simulation is a placeholder and not implemented. Requires RL agents (Mesa framework).")
      # This requires building a simulation of a dark web marketplace.
      # It involves creating RL agents for sellers and buyers.
      # The simulation generates realistic transactions.

    def generate_bin_attack_data(self, num_records=100000, output_filename="bin_attack_data.parquet"):
        """Generates synthetic credit card data for BIN attacks."""
        try:
            # Sesuaikan jumlah record dengan limit jika ada
            if self.limit is not None:
                num_records = min(num_records, self.limit)
                logger.info(f"Generating limited number of BIN attack records: {num_records}")
            else:
                 logger.info(f"Generating {num_records} synthetic credit card records for BIN attacks.")

            # Sample BIN issuers (replace with a more comprehensive list)
            bin_issuers = ["411111", "522222", "377777"]

            data = []
            for _ in tqdm(range(num_records), desc="Generating Records"):
                bin_issuer = random.choice(bin_issuers)
                account_number = ''.join(random.choices('0123456789', k=16 - len(bin_issuer) - 1))  # Remaining digits (minus check digit)
                card_number = bin_issuer + account_number

                # Luhn Algorithm (Check Digit)
                sum_odd = 0
                sum_even = 0
                reverse_card_number = card_number[::-1]  # reversed string
                for i in range(len(reverse_card_number)):
                    digit = int(reverse_card_number[i])
                    if (i + 1) % 2 != 0:  # Odd position (1-based indexing)
                        sum_odd += digit
                    else:
                        doubled_digit = digit * 2
                        sum_even += doubled_digit - 9 if doubled_digit > 9 else doubled_digit  # Subtract 9 if doubled is > 9

                check_digit = (10 - ((sum_odd + sum_even) % 10)) % 10

                full_card_number = card_number + str(check_digit)
                expiry_month = str(random.randint(1, 12)).zfill(2)
                expiry_year = str(random.randint(2025, 2030))
                cvv = str(random.randint(100, 999))

                data.append({
                    "card_number": full_card_number,
                    "expiry_month": expiry_month,
                    "expiry_year": expiry_year,
                    "cvv": cvv
                })

            df = pd.DataFrame(data)
            df.to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"Generated BIN attack data saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error generating BIN attack data: {e}")

    def generate_deepfake_financial_profiles(self, num_profiles=100, output_filename="deepfake_profiles.parquet"):
        """Menggunakan model lokal untuk generate data finansial"""
        try:
            # Sesuaikan jumlah profil dengan limit jika ada
            if self.limit is not None:
                num_profiles = min(num_profiles, self.limit)
                logger.info(f"Generating limited number of financial profiles: {num_profiles}")
            else:
                 logger.info(f"Generating {num_profiles} synthetic financial profiles using local models")

            # Ganti dengan model lokal (contoh: GPT-Neo)
            from transformers import pipeline
            generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')
            
            # Template prompt
            prompt = """Generate a financial profile with:
            - Name: {name}
            - Address: {address}
            - Bank Balance: ${balance}
            - Credit Score: {score}
            """
            
            data = []
            for _ in range(num_profiles):
                # Generate data acak
                name = f"User_{uuid.uuid4().hex[:8]}"
                address = f"{random.randint(1,999)} Fake Street"
                balance = random.randint(1000, 1000000)
                score = random.randint(300, 850)
                
                # Generate text
                generated = generator(
                    prompt.format(name=name, address=address, balance=balance, score=score),
                    max_length=200,
                    do_sample=True,
                    temperature=0.7
                )
                
                data.append({
                    "name": name,
                    "address": address,
                    "balance": balance,
                    "score": score,
                    "document": generated[0]['generated_text']
                })
                
            df = pd.DataFrame(data)
            df.to_parquet(str(self.processed_dir / output_filename))
            logger.info(f"Generated financial profiles saved to {output_filename}")

        except Exception as e:
            logger.error(f"Error generating financial profiles: {e}")

    # --- NEW FEATURE DATASETS (PLACEHOLDERS - IMPLEMENT ACCORDING TO NEEDS) ---
    def prepare_new_dialog_data(self, output_filename="new_dialog_data.parquet"):
        """Placeholder for a new dialog dataset source."""
        logger.warning("New Dialog data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Scrape dialogs from a website
        # 2. Load dialog data from a local file
        # 3. Use an API to get dialog data
        pass  # Replace with actual data loading and processing

    def prepare_new_coding_data(self, output_filename="new_coding_data.parquet"):
        """Placeholder for a new coding dataset source."""
        logger.warning("New Coding data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Scrape code snippets from a coding forum
        # 2. Load code from a GitHub repository
        # 3. Use an API to get code examples
        pass  # Replace with actual data loading and processing

    def prepare_new_math_data(self, output_filename="new_math_data.parquet"):
        """Placeholder for a new math dataset source."""
        logger.warning("New Math data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Scrape math problems from a website
        # 2. Load math problems from a textbook
        # 3. Generate math problems using a symbolic math library
        pass  # Replace with actual data loading and processing

    def prepare_new_network_data(self, output_filename="new_network_data.parquet"):
        """Placeholder for a new network dataset source."""
        logger.warning("New Network data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Capture network traffic using tcpdump
        # 2. Load network traffic data from a pcap file
        # 3. Generate synthetic network traffic data
        pass  # Replace with actual data loading and processing

    def prepare_new_trading_data(self, output_filename="new_trading_data.parquet"):
        """Placeholder for a new trading dataset source."""
        logger.warning("New Trading data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Use a different trading API (e.g., Alpaca, IEX Cloud)
        # 2. Load trading data from a local CSV file
        # 3. Generate synthetic trading data
        pass  # Replace with actual data loading and processing

    def prepare_new_captcha_data(self, output_filename="new_captcha_data.parquet"):
        """Placeholder for a new CAPTCHA dataset source."""
        logger.warning("New CAPTCHA data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Scrape CAPTCHAs from a website
        # 2. Generate CAPTCHAs using a CAPTCHA library
        # 3. Load CAPTCHAs from a local file
        pass  # Replace with actual data loading and processing

    def prepare_new_threat_intel_data(self, output_filename="new_threat_intel_data.parquet"):
        """Placeholder for a new threat intelligence dataset source."""
        logger.warning("New Threat Intel data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Use a different threat intelligence API
        # 2. Load threat intelligence data from a local file
        # 3. Scrape threat intelligence data from a website
        pass  # Replace with actual data loading and processing

    def prepare_new_rag_data(self, output_filename="new_rag_data.parquet"):
        """Placeholder for a new RAG dataset source."""
        logger.warning("New RAG data preparation is a placeholder. Implement data extraction/processing here.")
        # Example:
        # 1. Load data from a new knowledge base
        # 2. Scrape data from a website
        # 3. Create a new RAG pipeline
        pass  # Replace with actual data loading and processing

    def prepare_all_training_data(self):
        """Downloads and prepares all necessary training datasets."""
        logger.info("Starting preparation of all training datasets...")
        self.prepare_dialog_data()
        # --- Aktifkan kembali prepare_coding_data --- #
        logger.warning("Preparing coding data (codeparrot/github-code). This dataset is very large and may take a long time and significant disk space (cache). Use --limit argument.")
        #self.prepare_coding_data() 
        # --- Akhir Aktivasi --- #
        self.prepare_math_data()
        self.prepare_network_data()
        self.prepare_trading_data() # Sekarang menggunakan MT5 + yfinance
        self.prepare_captcha_data()
        self.prepare_threat_intel_data()
        self.prepare_rag_data()

        # New methods:
        self.prepare_reddit_conversations()
        self.prepare_wikipedia_data()
        self.prepare_stack_overflow()
        self.prepare_github_codeparrot_plus()
        self.prepare_lean4_theorem_proving()
        self.prepare_imo_shortlists()
        #self.prepare_cic_darknet2020()
        self.prepare_malware_dna_db()

        logger.info("Finished preparation of training datasets.")


if __name__ == "__main__":
    # <-- Tambahkan ArgParse -->
    parser = argparse.ArgumentParser(description="Download and prepare various training datasets.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of rows to process and save for each dataset component.")
    args = parser.parse_args()
    # <-- Akhir Tambahan ArgParse -->

    # Initialize and run the DatasetManager with the limit from args
    manager = DatasetManager(data_dir=str(DATA_DIR), limit=args.limit) # <-- Berikan limit ke konstruktor
    manager.prepare_all_training_data()