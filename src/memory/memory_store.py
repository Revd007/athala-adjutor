import psycopg2
from logger import logger

class MemoryStore:
    def __init__(self, db_config):
        self.conn = psycopg2.connect(**db_config)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """Create table for prompt-response pairs."""
        try:
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id SERIAL PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.commit()
            logger.info("Memory table created or verified")
        except Exception as e:
            logger.error(f"Error creating memory table: {e}")
            raise

    def store(self, prompt, response):
        """Store a prompt-response pair."""
        try:
            self.cursor.execute(
                "INSERT INTO memory (prompt, response) VALUES (%s, %s) RETURNING id",
                (prompt, response)
            )
            memory_id = self.cursor.fetchone()[0]
            self.conn.commit()
            logger.info(f"Stored memory ID {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            raise

    def retrieve(self, prompt, limit=5):
        """Retrieve relevant responses for a prompt."""
        try:
            self.cursor.execute(
                "SELECT prompt, response FROM memory WHERE prompt ILIKE %s ORDER BY timestamp DESC LIMIT %s",
                (f"%{prompt}%", limit)
            )
            results = self.cursor.fetchall()
            logger.info(f"Retrieved {len(results)} memories for prompt: {prompt}")
            return [{"prompt": r[0], "response": r[1]} for r in results]
        except Exception as e:
            logger.error(f"Error retrieving memory: {e}")
            raise

    def close(self):
        """Close database connection."""
        self.cursor.close()
        self.conn.close()
        logger.info("Memory store connection closed")

if __name__ == "__main__":
    db_config = {
        "dbname": "athala_adjutor",
        "user": "postgres",
        "password": "password",
        "host": "localhost",
        "port": "5432"
    }
    memory = MemoryStore(db_config)
    memory.store("Test prompt", "Test response")
    print(memory.retrieve("Test"))
    memory.close()