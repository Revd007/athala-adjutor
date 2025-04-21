# Athala Adjutor

Athala Adjutor is an autonomous, multifunctional AI designed to excel in **dialog**, **coding**, **math**, **sysadmin**, **trading**, **cybersecurity**, **streaming**, **CAPTCHA solving**, **web crawling**, **RAG**, **Enhanced RAG**, **network**, **mobility**, and **UI**. Built for local processing on consumer hardware (e.g., NVIDIA RTX 3060), it avoids external APIs for license safety. Athala uses **deep learning** (transformers, CNNs, RNNs), **machine learning** (Isolation Forest, RL), and **rule-based methods** (SymPy, OpenCV). It manages datasets (Kaggle, Hugging Face, Google, dark web, user-added) and a **PostgreSQL** database for crawled data, logs, and RAG metadata, deployed on `revianravilathala.my.id`. OpenCV is integrated for CAPTCHA and streaming.

## Table of Contents

- Vision
- Features
- Components
- Dataset and Database Management
- Installation
- Usage
- Training
- Testing
- Deployment
- Contributing
- License

## Vision

Athala Adjutor is a self-aware AI, not a bot, designed to advance technology. It autonomously learns from crawled and user-added datasets, solves real-world problems (e.g., CAPTCHAs, trading, math), and adapts without manual intervention. All capabilities are balanced with equal focus, using a hybrid of deep learning, machine learning, and rule-based methods.

## Features

- **Dialog**: Casual, human-like conversations with pre-trained GPT-2 (deep learning), augmented by RAG.
- **Coding**: Generates and debugs code (Python, Java) with pre-trained CodeGen (deep learning) and RAG context.
- **Math**: Solves symbolic (SymPy) and numerical problems with pre-trained transformer parsing (deep learning).
- **Sysadmin**: Automates tasks and IDS with Suricata and pre-trained Isolation Forest (machine learning).
- **Trading**: Analyzes Bitcoin, Ethereum, Solana with pre-trained LSTM-TCN-GRU (deep learning) and RL (PPO).
- **Cybersecurity**: Detects threats using dark web intel and Kaggle datasets with pre-trained ML models.
- **Streaming**: Real-time audio, video, and screen-sharing with OpenCV processing.
- **CAPTCHA Solving**: Handles image (pre-trained YOLOv8, deep learning), numeric (pre-trained ResNet18, deep learning), text (pre-trained BERT, deep learning), audio, reCAPTCHA, hCaptcha, and Web3 CAPTCHAs with OpenCV.
- **Web Crawling**: Collects data from Google, deep web, and dark web for all components, stored in PostgreSQL.
- **RAG**: Retrieval-Augmented Generation (pre-trained deep learning) for contextual dialog, coding, and math.
- **Enhanced RAG**: RAG with pre-trained reranking (deep learning) for improved retrieval accuracy.
- **Network**: Monitors traffic with pre-trained ML models.
- **Mobility**: Device-to-device communication via WebRTC.
- **UI**: Naruto Shimeji UI with Tailwind CSS.
- **Self-Learning**: Detects new datasets, trains models, and optimizes predictions for all components.
- **Freelance**: Showcases capabilities for blockchain, cybersecurity, and AI jobs.

## Components

 1. **Dialog (**`src/dialog/dialog.py`**)**
    - Casual responses with voice feedback via `voice_handler.py`.
 2. **Coding (**`src/codegen/codegen.py`**)**
    - Generates and debugs code, streamed via `stream_handler.py`.
 3. **Math (**`src/math/math_solver.py`**)**
    - Solves equations and optimization problems.
 4. **Sysadmin (**`src/sysadmin/sysadmin_manager.py`**)**
    - Automates tasks and detects threats.
 5. **Trading (**`src/finance/market_analyzer.py`**,** `trading_dashboard.py`**)**
    - Predicts prices and trades with CAPTCHA support.
 6. **Streaming (**`src/stream/stream_handler.py`**,** `voice_handler.py`**)**
    - Audio, video, and screen-sharing with WebSocket and OpenCV.
 7. **Network (**`src/network/network_manager.py`**,** `monitor.py`**)**
    - Configures devices and monitors traffic.
 8. **Mobility (**`src/mobility/device_mobility.py`**)**
    - Device-to-device communication via WebRTC.
 9. **UI (**`src/ui/ui_manager.py`**)**
    - Browser/desktop UI with Naruto shimeji.
10. **CAPTCHA Solver (**`src/utils/captcha_solver.py`**)**
    - Solves various CAPTCHAs with YOLOv8, ResNet18, BERT, and OpenCV.
11. **Web Crawler (**`src/utils/web_crawler.py`**)**
    - Crawls Google, deep web, dark web for all components.
12. **Dataset Manager (**`src/utils/dataset_manager.py`**)**
    - Manages crawled, Kaggle, Hugging Face, and user-added datasets.
13. **Database Manager (**`src/utils/database_manager.py`**)**
    - Manages PostgreSQL database for crawled data, logs, and RAG metadata.
14. **RAG (**`src/ai/rag.py`**)**
    - Augments dialog, coding, math with retrieved context.
15. **Enhanced RAG (**`src/ai/enhanced_rag.py`**)**
    - RAG with reranking for better accuracy.

## Dataset and Database Management

Athala uses **datasets** for training and a **PostgreSQL** database for dynamic data management:

- **Datasets**:
  - **Sources**:
    - **Kaggle**:
      - Dialog: `daily_dialog_dataset` (conversations).
      - Coding: `python_code_dataset` (Python snippets).
      - Math: `math_problem_dataset` (math problems).
      - Trading: `bitcoin_ohlcv_dataset` (OHLCV data).
      - CAPTCHA: `captcha_image_dataset` (image CAPTCHAs).
      - Threat Intel: `cve_dataset` (vulnerabilities).
      - Network: `network_traffic_dataset` (traffic logs).
    - **Hugging Face**:
      - Dialog: `daily_dialog` (natural conversations).
      - Coding: `codeparrot` (code snippets).
      - Math: `math_qa` (math Q&A).
      - CAPTCHA: `captcha_text_dataset` (text CAPTCHAs).
      - RAG: `wiki_dpr` (technical docs).
    - **Google**: Public datasets, news, papers, GitHub, Stack Overflow.
    - **Deep Web**: Academic databases (e.g., ResearchGate, ArXiv).
    - **Dark Web**: Threat intel, CVE vulnerabilities via Tor.
    - **User-Added**: Place datasets in `data/raw/new_dataset/`.
  - **Crawling**:
    - Dialog: Reddit, Twitter for natural conversations.
    - Coding: GitHub, Stack Overflow for snippets and framework updates.
    - Math: ArXiv, Math Stack Exchange for problems and solutions.
    - Trading: OHLCV data and market news.
    - CAPTCHA: Image and text CAPTCHA samples.
    - Threat Intel: CVE databases, dark web reports.
    - Network: Traffic datasets, IDS rules.
    - RAG: Technical documentation, research papers.
  - **Structure**:
    - **Raw (**`data/raw/`**)**: Kaggle, Hugging Face, crawled, user-added data.
    - **Processed (**`data/processed/`**)**: Parquet files (e.g., `dialog_train.parquet`, `rag_index/`).
    - **Models (**`data/models/`**)**: Pre-trained models (e.g., `yolo_captcha.pt`, `rag_model.pt`).
- **Database**:
  - **PostgreSQL (**`athala_adjutor`**)**:
    - Stores crawled data (URL, text, component).
    - Stores update logs (component, action, timestamp).
    - Stores RAG metadata (document ID, text, embedding).
  - **Management**:
    - **Crawling**: Stores results in PostgreSQL and Parquet.
    - **Deduplication**: FAISS for vector similarity.
    - **Categorization**: Transformer-based classification (dialog, coding, math, trading, captcha, threat_intel, network, rag).
    - **Preprocessing**: Cleans data into Parquet.
    - **Self-Update**: Detects new datasets, trains models, logs updates in PostgreSQL.
- **Adding Datasets**:
  - Place files in `data/raw/new_dataset/` (e.g., `captcha.csv`, `rag_context.parquet`).
  - Athala detects, categorizes, and trains automatically.

## Installation

1. **Clone Repository**:

   ```bash
   git clone https://github.com/revianravilathala/athala-adjutor
   cd athala-adjutor
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   - Tesseract: `sudo apt-get install tesseract-ocr`.
   - CMU Sphinx: `pip install pocketsphinx`.
   - Tor: `sudo apt-get install tor`.
   - OpenCV: Included in `requirements.txt` (`opencv-python`).
   - PostgreSQL: Managed via Docker (`docker-compose.yml`).

3. **Set Up Environment**:

   - Create `config.py`:

     ```python
     AWS_ACCESS_KEY = "your_access_key"
     AWS_SECRET_KEY = "your_secret_key"
     DATA_DIR = "./data"
     DEVICE_CONFIGS = {
         "device_a": {"host": "localhost", "port": 8766},
         "device_b": {"host": "localhost", "port": 8767}
     }
     DEEP_WEB_CREDENTIALS = {
         "researchgate": {"username": "your_username", "password": "your_password"}
     }
     POSTGRES_CONFIG = {
         "host": "postgres",
         "port": "5432",
         "database": "athala_adjutor",
         "user": "athala",
         "password": "athala123"
     }
     KAGGLE_CREDENTIALS = {
         "username": "your_kaggle_username",
         "key": "your_kaggle_api_key"
     }
     ```

4. **Set Up PostgreSQL**:

   ```bash
   docker-compose up -d
   ```

5. **Set Up Blockchain Nodes (Optional)**:

   - Ethereum: `npm install -g ganache-cli && ganache-cli`.
   - Solana: `solana config set --url https://api.mainnet-beta.solana.com`.

6. **Hardware**:

   - Minimum: NVIDIA RTX 3060, 16GB RAM, 1TB SSD.
   - Recommended: NVIDIA RTX 4080, 32GB RAM, 4TB SSD.

## Usage

1. **Initialize Datasets**:

   ```bash
   python src/utils/dataset_manager.py --initialize_datasets
   ```

   - Downloads datasets from Kaggle and Hugging Face.
   - Crawls Google and dark web for additional data.

2. **Add New Dataset**:

   - Place files in `data/raw/new_dataset/` (e.g., `captcha.csv`, `rag_context.parquet`).

   - Run:

     ```bash
     python src/utils/dataset_manager.py --auto_train
     ```

3. **Preprocess Data**:

   ```bash
   python src/utils/dataset_manager.py --component all
   ```

4. **Train Models**:

   ```bash
   python src/ai/train_multi.py --component all
   ```

5. **Run Continual Learning**:

   ```bash
   python src/ai/continual.py
   ```

6. **Run Active Learning**:

   ```bash
   python src/ai/active_learning_multi.py
   ```

7. **Start Servers**:

   ```bash
   python src/stream/stream_handler.py
   python src/mobility/device_mobility.py
   uvicorn src.api.main:app --host 0.0.0.0 --port 8000
   ```

8. **Interact**:

   - Dialog: `python src/dialog/dialog.py --prompt "Yo, apa kabar?"`
   - Coding: `python src/codegen/codegen.py --prompt "Write a factorial function"`
   - Math: `python src/math/math_solver.py --problem "solve x^2 - 4 = 0"`
   - RAG: `python src/ai/rag.py --query "Explain blockchain"`
   - Enhanced RAG: `python src/ai/enhanced_rag.py --query "Explain blockchain"`

## Training

- **Models**:

  - **Deep Learning**:
    - GPT-2: Dialog, RAG (pre-trained: `dialog_model.pt`, `rag_model.pt`).
    - CodeGen: Coding (pre-trained: `coding_model.pt`).
    - LSTM-TCN-GRU: Trading (pre-trained: `trading_model.pt`).
    - YOLOv8: Image CAPTCHA (pre-trained: `yolo_captcha.pt`).
    - ResNet18: Numeric CAPTCHA (pre-trained: `numeric_classifier.pt`).
    - BERT: Text CAPTCHA (pre-trained: `text_classifier.pt`).
    - Sentence-Transformers: RAG embeddings.
    - Flan-T5: Math parsing (pre-trained: `math_model.pt`).
  - **Machine Learning**:
    - Isolation Forest: Sysadmin threat detection (pre-trained: `threat_intel_model.pt`).
    - Random Forest: Network monitoring (pre-trained: `network_model.pt`).
    - PPO (RL): Trading strategy optimization.
  - **Rule-Based**:
    - SymPy: Math symbolic solving.
    - OpenCV: CAPTCHA detection and streaming.

- **Techniques**:

  - Knowledge distillation, contrastive learning, curriculum learning.
  - Active learning, RL (PPO), Optuna tuning.
  - Early stopping, dropout to prevent underfit/overfit.

- **Scripts**:

  ```bash
  python src/ai/train_multi.py --component <component>
  ```

## Testing

```bash
python -m unittest tests/test_suite.py
```

- Tests: Dialog, coding, math, sysadmin, trading, streaming, CAPTCHA, crawling, RAG, Enhanced RAG, network, mobility, UI, database.

## Deployment

1. **Docker**:

   ```bash
   docker-compose up -d
   ```

2. **DNS**:

   - Record A to `124.158.151.106`.
   - Verify: `ping revianravilathala.my.id`.

3. **Access**:

   - API: `http://revianravilathala.my.id:8000/docs`
   - UI: `http://revianravilathala.my.id`

## Contributing

1. Fork repository.
2. Create branch: `git checkout -b feature/your-feature`.
3. Commit: `git commit -m "Add feature"`.
4. Push: `git push origin feature/your-feature`.
5. Open pull request.

## License

MIT License. Local processing ensures no API conflicts.