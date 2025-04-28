import os
from logger import logger

class Config:
    def __init__(self):
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'athala_adjutor'),
            'user': os.getenv('DB_USER', 'revian_dbsiem'),
            'password': os.getenv('DB_PASSWORD', 'wokolcoy20'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432')
        }
        self.DATA_DIR = os.getenv('DATA_DIR', 'data')
        # self.DEEP_WEB_CREDENTIALS = {
        #     'example.com': {'username': 'user', 'password': 'pass'}
        # }
        self.KAGGLE_CREDENTIALS = {
            'username': os.getenv('KAGGLE_USERNAME', 'revianravilathala'),
            'key': os.getenv('KAGGLE_KEY', '91e0d3ec9d2f587c9f48ab100c028daf')
        }

    def get(self, key):
        if key == 'db_config':
            return self.db_config
        return getattr(self, key, None)

if __name__ == '__main__':
    config = Config()
    print(config.get('db_config'))