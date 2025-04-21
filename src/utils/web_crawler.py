import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from torpy.http.requests import TorRequests
from src.utils.database_manager import DatabaseManager
from logger import logger
from config import DATA_DIR, DEEP_WEB_CREDENTIALS
import os

class WebCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.db = DatabaseManager()
        logger.info("WebCrawler initialized with PostgreSQL support")

    def crawl_google(self, query, component):
        try:
            os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
            for url in search(query, num_results=10):
                try:
                    response = self.session.get(url, timeout=5)
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    self.db.store_crawled_data(component, url, text)
                    df = pd.DataFrame([{"url": url, "text": text}])
                    df.to_parquet(f"{DATA_DIR}/processed/crawled_{query.replace(' ', '_')}.parquet")
                except Exception as e:
                    logger.error(f"Error crawling {url}: {str(e)}")
            logger.info(f"Crawled Google for query: {query} (component: {component})")
        except Exception as e:
            logger.error(f"Error crawling Google: {str(e)}")

    def crawl_deep_web(self, url, component):
        try:
            credentials = DEEP_WEB_CREDENTIALS.get(url.split("/")[2], {})
            response = self.session.post(url, data=credentials, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            self.db.store_crawled_data(component, url, text)
            df = pd.DataFrame([{"url": url, "text": text}])
            df.to_parquet(f"{DATA_DIR}/processed/crawled_deep_{url.replace('/', '_')}.parquet")
            logger.info(f"Crawled deep web: {url} (component: {component})")
        except Exception as e:
            logger.error(f"Error crawling deep web: {str(e)}")

    def crawl_dark_web(self, url, component):
        try:
            with TorRequests() as tor:
                response = tor.get(url, timeout=10)
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                self.db.store_crawled_data(component, url, text)
                df = pd.DataFrame([{"url": url, "text": text}])
                df.to_parquet(f"{DATA_DIR}/processed/crawled_dark_{url.replace('/', '_')}.parquet")
                logger.info(f"Crawled dark web: {url} (component: {component})")
        except Exception as e:
            logger.error(f"Error crawling dark web: {str(e)}")

    def crawl_for_components(self):
        try:
            queries = {
                "dialog": ["natural language conversations", "reddit discussions", "twitter conversations"],
                "coding": ["github python snippets", "stack overflow solutions", "javascript framework updates"],
                "math": ["arxiv math papers", "math stack exchange problems", "calculus solutions"],
                "trading": ["bitcoin price data", "ethereum ohlcv", "solana market trends"],
                "captcha": ["captcha image datasets", "recaptcha examples", "hcaptcha samples"],
                "threat_intel": ["cve vulnerabilities", "dark web threat intel", "cybersecurity reports"],
                "network": ["network traffic datasets", "ids rules snort", "suricata logs"],
                "rag": ["technical documentation", "research papers", "blockchain articles"]
            }
            for component, component_queries in queries.items():
                for query in component_queries:
                    self.crawl_google(query, component)
                if component in ["threat_intel", "network"]:
                    self.crawl_dark_web("http://example.onion", component)  # Ganti dengan URL dark web
            logger.info("Crawled data for all components")
        except Exception as e:
            logger.error(f"Error crawling for components: {str(e)}")