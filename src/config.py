import yaml
from logger import logger

class Config:
    def __init__(self, config_file="config.yaml"):
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_file}")
        except Exception as e:
            logger.error(f"Config load error: {e}")
            raise

    def get(self, key, default=None):
        """Get config value by key."""
        return self.config.get(key, default)

if __name__ == "__main__":
    config = Config()
    print(config.get("db_config"))