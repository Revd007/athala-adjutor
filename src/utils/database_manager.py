import psycopg2
import pandas as pd
import numpy as np
from logger import logger
from src.config import Config

class DatabaseManager:
    def __init__(self):
        config = Config()
        self.conn = psycopg2.connect(**config.get("db_config"))
        self.cursor = self.conn.cursor()
        self.create_tables()
        logger.info("DatabaseManager initialized with PostgreSQL")

    def create_tables(self):
        try:
            # Tabel untuk hasil crawling
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS crawled_data (
                    id SERIAL PRIMARY KEY,
                    component VARCHAR(50),
                    url TEXT,
                    text TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Tabel untuk log update
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS update_logs (
                    id SERIAL PRIMARY KEY,
                    component VARCHAR(50),
                    action VARCHAR(50),
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Tabel untuk metadata RAG
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS rag_metadata (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER,
                    text TEXT,
                    embedding BYTEA,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
            logger.info("Database tables created")
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")

    def store_crawled_data(self, component, url, text):
        try:
            self.cursor.execute("""
                INSERT INTO crawled_data (component, url, text)
                VALUES (%s, %s, %s)
            """, (component, url, text))
            self.conn.commit()
            logger.info(f"Stored crawled data for {component}: {url}")
        except Exception as e:
            logger.error(f"Error storing crawled data: {str(e)}")

    def store_update_log(self, component, action):
        try:
            self.cursor.execute("""
                INSERT INTO update_logs (component, action)
                VALUES (%s, %s)
            """, (component, action))
            self.conn.commit()
            logger.info(f"Stored update log: {component} {action}")
        except Exception as e:
            logger.error(f"Error storing update log: {str(e)}")

    def store_rag_metadata(self, document_id, text, embedding):
        try:
            self.cursor.execute("""
                INSERT INTO rag_metadata (document_id, text, embedding)
                VALUES (%s, %s, %s)
            """, (document_id, text, psycopg2.Binary(embedding.tobytes())))
            self.conn.commit()
            logger.info(f"Stored RAG metadata for document {document_id}")
        except Exception as e:
            logger.error(f"Error storing RAG metadata: {str(e)}")

    def fetch_crawled_data(self, component):
        try:
            self.cursor.execute("SELECT url, text FROM crawled_data WHERE component = %s", (component,))
            results = self.cursor.fetchall()
            logger.info(f"Fetched {len(results)} crawled data for {component}")
            return pd.DataFrame(results, columns=["url", "text"])
        except Exception as e:
            logger.error(f"Error fetching crawled data: {str(e)}")
            return pd.DataFrame()

    def fetch_update_logs(self):
        try:
            self.cursor.execute("SELECT component, action, timestamp FROM update_logs")
            results = self.cursor.fetchall()
            logger.info(f"Fetched {len(results)} update logs")
            return pd.DataFrame(results, columns=["component", "action", "timestamp"])
        except Exception as e:
            logger.error(f"Error fetching update logs: {str(e)}")
            return pd.DataFrame()

    def fetch_rag_metadata(self, document_id):
        try:
            self.cursor.execute("SELECT text, embedding FROM rag_metadata WHERE document_id = %s", (document_id,))
            result = self.cursor.fetchone()
            if result:
                text, embedding = result
                embedding = np.frombuffer(embedding, dtype=np.float32)
                logger.info(f"Fetched RAG metadata for document {document_id}")
                return text, embedding
            return None, None
        except Exception as e:
            logger.error(f"Error fetching RAG metadata: {str(e)}")
            return None, None

    def close(self):
        self.conn.close()
        logger.info("Database connection closed")