# Configuration for AI Agent Virtual Assistant
VOCAB_SIZE = 50000
N_LAYERS = 6
N_HEADS = 8
D_MODEL = 512
D_FF = 2048
MAX_LEN = 128
DROPOUT = 0.1
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPLAY_BUFFER_SIZE = 10000
MEMORY_STORE_SIZE = 1000
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
CODEGEN_MODEL = "THUDM/CodeGeeX"
SCRAPE_URLS = ["https://demo.eventloganalyzer.com/event/index2.do?url=emberapp#/home/dashboard/301"]
LOG_DIR = "logs"
LOG_LEVEL = "INFO"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX = "ai-agent-index"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
PROMETHEUS_PORT = 9090
TRELLO_API_KEY = "your-trello-api-key"
SLACK_WEBHOOK = "your-slack-webhook-url"

# DATA_DIR = "./data"
# DEVICE_CONFIGS = {
#     "device_a": {"host": "localhost", "port": 8766},
# }
# DEEP_WEB_CREDENTIALS = {
#     "researchgate": {"username": "your_username", "password": "your_password"}
# }
# POSTGRES_CONFIG = {
#     "host": "postgres",
#     "port": "5432",
#     "database": "athala_adjutor",
#     "user": "athala",
#     "password": "athala123"
# }