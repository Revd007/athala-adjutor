import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from torpy.http.requests import TorRequests
from src.utils.database_manager import DatabaseManager
from logger import logger
import os
# from src.config import DATA_DIR, DEEP_WEB_CREDENTIALS
from src.config import Config

class WebCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.db = DatabaseManager()
        self.config = Config()
        self.DATA_DIR = self.config.DATA_DIR
        logger.info("WebCrawler initialized with PostgreSQL support")

    def crawl_google(self, query, component):
        try:
            os.makedirs(f"{self.DATA_DIR}/processed", exist_ok=True)
            for url in search(query, num_results=10):
                try:
                    response = self.session.get(url, timeout=5)
                    soup = BeautifulSoup(response.text, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                    self.db.store_crawled_data(component, url, text)
                    df = pd.DataFrame([{"url": url, "text": text}])
                    df.to_parquet(f"{self.DATA_DIR}/processed/crawled_{query.replace(' ', '_')}.parquet")
                except Exception as e:
                    logger.error(f"Error crawling {url}: {str(e)}")
            logger.info(f"Crawled Google for query: {query} (component: {component})")
        except Exception as e:
            logger.error(f"Error crawling Google: {str(e)}")

    def crawl_deep_web(self, url, component):
        try:
            # credentials = DEEP_WEB_CREDENTIALS.get(url.split("/")[2], {})
            response = self.session.post(url, data={}, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            self.db.store_crawled_data(component, url, text)
            df = pd.DataFrame([{"url": url, "text": text}])
            df.to_parquet(f"{self.DATA_DIR}/processed/crawled_deep_{url.replace('/', '_')}.parquet")
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
                df.to_parquet(f"{self.DATA_DIR}/processed/crawled_dark_{url.replace('/', '_')}.parquet")
                logger.info(f"Crawled dark web: {url} (component: {component})")
        except Exception as e:
            logger.error(f"Error crawling dark web: {str(e)}")

    def crawl_for_components(self):
        try:
            queries = {
                "dialog": ["natural language conversations", "reddit discussions", "twitter conversations"],
                "coding": ["github python snippets", "stack overflow solutions", "javascript framework updates"],
                "math": ["arxiv math papers", "math stack exchange problems", "calculus solutions"],
                "trading": [
                    # Crypto
                    "bitcoin price analysis technical",
                    "ethereum ohlcv data binance",
                    "solana market trends and news",
                    "altcoin market analysis",
                    "crypto arbitrage opportunities",
                    "DeFi yield farming strategies",
                    "NFT market trends opensea",
                    "stablecoin regulation updates",
                    # Forex
                    "forex trading strategies EURUSD",
                    "USDJPY technical analysis",
                    "GBPAUD market outlook",
                    "major currency pairs correlation",
                    "forex market sentiment analysis",
                    # Stocks & Indices (Global)
                    "S&P 500 technical analysis",
                    "NASDAQ composite index forecast",
                    "FTSE 100 market news",
                    "Nikkei 225 index trends",
                    "global stock market news reuters bloomberg",
                    "emerging markets ETF performance",
                    # Stocks & Indices (Indonesia - Kept from original)
                    "IHSG index analysis", # Indonesian Stock Exchange Index
                    "IDX stock recommendations", # Indonesian Stock Exchange
                    "saham blue chip Indonesia analysis", # Indonesian blue chip stocks
                    # Commodities
                    "gold price analysis XAUUSD",
                    "silver price forecast COMEX",
                    "crude oil WTI price trends",
                    "natural gas price analysis Henry Hub",
                    "commodity market news agriculture",
                    # General Trading/Finance
                    "stock market technical indicators explained",
                    "algorithmic trading strategies python",
                    "quantitative finance research papers",
                    "economic indicators impact on markets",
                    "Federal Reserve interest rate decisions",
                    "options trading basics",
                    "futures market data CME",
                ],
                "captcha": ["captcha image datasets", "recaptcha examples", "hcaptcha samples"],
                "threat_intel": ["cve vulnerabilities", "dark web threat intel", "cybersecurity reports", "malware analysis techniques", "phishing campaign indicators", "mitre att&ck framework updates"],
                "network": ["network traffic datasets pcap", "ids rules snort suricata", "network security monitoring logs", "wireshark analysis tutorials", "bgp routing analysis", "dns security best practices"],
                "rag": ["technical documentation APIs", "research papers machine learning", "blockchain whitepapers", "cloud computing best practices", "large language model research", "vector database comparisons"]
            }
            for component, component_queries in queries.items():
                for query in component_queries:
                    self.crawl_google(query, component)
                if component in ["threat_intel", "network"]:
                    self.crawl_dark_web("http://example.onion", component)  # Ganti dengan URL dark web
            logger.info("Crawled data for all components")
        except Exception as e:
            logger.error(f"Error crawling for components: {str(e)}")