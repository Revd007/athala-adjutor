import pandas as pd
import faiss
import numpy as np
import os
import glob
import torch
import random
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from src.utils.web_crawler import WebCrawler
from src.utils.database_manager import DatabaseManager
from src.ai.train_multi import train_component
from logger import logger
from config import DATA_DIR, KAGGLE_CREDENTIALS

class DatasetManager:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.crawler = WebCrawler()
        self.db = DatabaseManager()
        self.index = faiss.IndexFlatL2(384)
        self.components = ["dialog", "coding", "math", "trading", "captcha", "threat_intel", "network", "rag"]
        self.kaggle_api = KaggleApi()
        self.kaggle_api.authenticate()
        logger.info("DatasetManager initialized with PostgreSQL and Kaggle/Hugging Face support")

    def download_kaggle_dataset(self, dataset, component):
        try:
            os.makedirs(f"{DATA_DIR}/raw/kaggle", exist_ok=True)
            self.kaggle_api.dataset_download_files(dataset, path=f"{DATA_DIR}/raw/kaggle", unzip=True)
            files = glob.glob(f"{DATA_DIR}/raw/kaggle/*.csv") + glob.glob(f"{DATA_DIR}/raw/kaggle/*.parquet")
            for file in files:
                if file.endswith(".csv"):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_parquet(file)
                df.to_parquet(f"{DATA_DIR}/processed/{component}_kaggle.parquet")
            logger.info(f"Downloaded Kaggle dataset: {dataset} for {component}")
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {str(e)}")

    def download_huggingface_dataset(self, dataset, component):
        try:
            os.makedirs(f"{DATA_DIR}/raw/huggingface", exist_ok=True)
            hf_dataset = load_dataset(dataset)
            df = pd.DataFrame(hf_dataset["train"])
            df.to_parquet(f"{DATA_DIR}/raw/huggingface/{component}_dataset.parquet")
            df.to_parquet(f"{DATA_DIR}/processed/{component}_huggingface.parquet")
            logger.info(f"Downloaded Hugging Face dataset: {dataset} for {component}")
        except Exception as e:
            logger.error(f"Error downloading Hugging Face dataset: {str(e)}")

    def initialize_datasets(self):
        try:
            datasets = {
                "dialog": [
                    {"source": "kaggle", "dataset": "daily_dialog_dataset"},
                    {"source": "huggingface", "dataset": "daily_dialog"}
                ],
                "coding": [
                    {"source": "kaggle", "dataset": "python_code_dataset"},
                    {"source": "huggingface", "dataset": "codeparrot"}
                ],
                "math": [
                    {"source": "kaggle", "dataset": "math_problem_dataset"},
                    {"source": "huggingface", "dataset": "math_qa"}
                ],
                "trading": [
                    {"source": "kaggle", "dataset": "bitcoin_ohlcv_dataset"}
                ],
                "captcha": [
                    {"source": "kaggle", "dataset": "captcha_image_dataset"},
                    {"source": "huggingface", "dataset": "captcha_text_dataset"}
                ],
                "threat_intel": [
                    {"source": "kaggle", "dataset": "cve_dataset"}
                ],
                "network": [
                    {"source": "kaggle", "dataset": "network_traffic_dataset"}
                ],
                "rag": [
                    {"source": "huggingface", "dataset": "wiki_dpr"}
                ]
            }
            for component, sources in datasets.items():
                for source in sources:
                    if source["source"] == "kaggle":
                        self.download_kaggle_dataset(source["dataset"], component)
                    else:
                        self.download_huggingface_dataset(source["dataset"], component)
            self.crawler.crawl_for_components()
            logger.info("Initialized datasets for all components")
        except Exception as e:
            logger.error(f"Error initializing datasets: {str(e)}")

    def crawl_and_store(self, queries=None, deep_web_urls=None, dark_web_urls=None):
        try:
            os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
            if queries:
                for query, component in queries:
                    self.crawler.crawl_google(query, component)
            if deep_web_urls:
                for url, component in deep_web_urls:
                    self.crawler.crawl_deep_web(url, component)
            if dark_web_urls:
                for url, component in dark_web_urls:
                    self.crawler.crawl_dark_web(url, component)
            logger.info("Crawling completed")
            self.index_crawled_data()
        except Exception as e:
            logger.error(f"Error crawling and storing: {str(e)}")

    def index_crawled_data(self):
        try:
            for component in self.components:
                df = self.db.fetch_crawled_data(component)
                if df.empty:
                    continue
                texts = df["text"].tolist()
                embeddings = []
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                    with torch.no_grad():
                        embedding = self.model(**inputs).last_hidden_state.mean(dim=1).numpy()
                    embeddings.append(embedding)
                    self.db.store_rag_metadata(len(embeddings), text, embedding)
                embeddings = np.vstack([e for e in embeddings if e is not None])
                self.index.add(embeddings)
                faiss.write_index(self.index, f"{DATA_DIR}/processed/rag_index/index.faiss")
                pd.DataFrame({"text": texts}).to_parquet(f"{DATA_DIR}/processed/rag_index/documents.parquet")
                logger.info(f"Indexed {len(texts)} crawled documents for {component}")
        except Exception as e:
            logger.error(f"Error indexing crawled data: {str(e)}")

    def deduplicate_data(self):
        try:
            texts = []
            for component in self.components:
                df = self.db.fetch_crawled_data(component)
                texts.extend(df["text"].tolist())
            embeddings = []
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    embedding = self.model(**inputs).last_hidden_state.mean(dim=1).numpy()
                embeddings.append(embedding)
            embeddings = np.vstack([e for e in embeddings if e is not None])
            self.index.add(embeddings)
            D, I = self.index.search(embeddings, k=2)
            duplicates = [i for i, d in enumerate(D[:, 1]) if d < 0.1]
            unique_texts = [texts[i] for i in range(len(texts)) if i not in duplicates]
            unique_df = pd.DataFrame({"text": unique_texts})
            unique_df.to_parquet(f"{DATA_DIR}/processed/unique_crawled_data.parquet")
            logger.info(f"Deduplicated {len(texts) - len(unique_texts)} duplicates")
        except Exception as e:
            logger.error(f"Error deduplicating data: {str(e)}")

    def categorize_data(self):
        try:
            df = pd.read_parquet(f"{DATA_DIR}/processed/unique_crawled_data.parquet")
            categorized = {cat: [] for cat in self.components}
            for text in df["text"]:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    embedding = self.model(**inputs).last_hidden_state.mean(dim=1).numpy()
                category = random.choice(self.components)  # Ganti dengan model klasifikasi
                categorized[category].append(text)
            for category, texts in categorized.items():
                cat_df = pd.DataFrame({"text": texts})
                cat_df.to_parquet(f"{DATA_DIR}/processed/{category}_data.parquet")
            logger.info("Data categorized")
        except Exception as e:
            logger.error(f"Error categorizing data: {str(e)}")

    def detect_new_datasets(self):
        try:
            new_files = glob.glob(f"{DATA_DIR}/raw/new_dataset/*")
            if not new_files:
                logger.info("No new datasets detected")
                return None
            logger.info(f"Detected {len(new_files)} new datasets: {new_files}")
            for file in new_files:
                if file.endswith(".csv"):
                    df = pd.read_csv(file)
                elif file.endswith(".parquet"):
                    df = pd.read_parquet(file)
                else:
                    continue
                columns = df.columns.tolist()
                logger.info(f"Dataset {file} has columns: {columns}")
                category = self.infer_category(df)
                df.to_parquet(f"{DATA_DIR}/processed/{category}_new.parquet")
                logger.info(f"Stored new dataset as {category}_new.parquet")
            return new_files
        except Exception as e:
            logger.error(f"Error detecting new datasets: {str(e)}")
            return None

    def infer_category(self, df):
        try:
            columns = [col.lower() for col in df.columns]
            if any(col in columns for col in ["open", "high", "low", "close", "volume"]):
                return "trading"
            elif any(col in columns for col in ["image", "label"]):
                return "captcha"
            elif any(col in columns for col in ["ip", "domain", "cve"]):
                return "threat_intel"
            elif any(col in columns for col in ["code", "snippet"]):
                return "coding"
            elif any(col in columns for col in ["equation", "solution"]):
                return "math"
            elif any(col in columns for col in ["text", "context"]):
                return "rag"
            elif any(col in columns for col in ["packet", "traffic"]):
                return "network"
            else:
                return "dialog"
        except Exception as e:
            logger.error(f"Error inferring category: {str(e)}")
            return "dialog"

    def preprocess(self, component="all"):
        try:
            if component == "all":
                components = self.components
            else:
                components = [component]
            for comp in components:
                files = glob.glob(f"{DATA_DIR}/processed/{comp}*.parquet")
                df_crawled = self.db.fetch_crawled_data(comp)
                if not files and df_crawled.empty:
                    continue
                dfs = [pd.read_parquet(f) for f in files]
                if not df_crawled.empty:
                    dfs.append(df_crawled[["text"]])
                df = pd.concat(dfs)
                df = df.dropna().drop_duplicates()
                df.to_parquet(f"{DATA_DIR}/processed/{comp}_train.parquet")
                logger.info(f"Preprocessed {comp} data")
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")

    def auto_train(self):
        try:
            new_datasets = self.detect_new_datasets()
            if new_datasets:
                for file in new_datasets:
                    df = pd.read_parquet(f"{DATA_DIR}/processed/{self.infer_category(pd.read_parquet(file))}_new.parquet")
                    category = self.infer_category(df)
                    logger.info(f"Training {category} with new dataset")
                    train_component(category, f"{DATA_DIR}/processed/{category}_train.parquet")
                    self.db.store_update_log(category, "trained")
            else:
                self.initialize_datasets()
                for component in self.components:
                    self.preprocess(component)
                    train_component(component, f"{DATA_DIR}/processed/{component}_train.parquet")
                    self.db.store_update_log(component, "trained")
        except Exception as e:
            logger.error(f"Error auto-training: {str(e)}")