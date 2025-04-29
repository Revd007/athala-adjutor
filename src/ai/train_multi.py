import torch
import torch.nn as nn
import sys
import os
from pathlib import Path
import json
import logging # Import modul logging standar
import time   # Untuk timestamp di nama file log

# --- Add project root to sys.path --- #
# Cari path root (dua level di atas file ini)
try:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
except IndexError:
    # Fallback jika struktur tidak seperti yang diharapkan (misal, dijalankan dari root)
    PROJECT_ROOT = Path(".").resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to sys.path")
# --- End sys.path modification --- #

# Define DATA_DIR setelah PROJECT_ROOT didefinisikan
DATA_DIR = PROJECT_ROOT / "data"
print(f"DATA_DIR set to: {DATA_DIR}")

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForImageClassification, AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoProcessor
from sklearn.ensemble import IsolationForest, RandomForestClassifier
# Import TCN for trading component
from src.ai.tcn import TCN # Impor ini SEHARUSNYA berhasil sekarang
try:
    from src.utils.database_manager import DatabaseManager
except ImportError:
    # Arahkan ke pesan error yang mungkin sebelumnya?
    print("Could not import DatabaseManager (src.utils.database_manager). Proceeding without database logging. Ensure src/utils/__init__.py exists.")
    DatabaseManager = None

from logger import logger # Asumsi logger.py ada di root atau PYTHONPATH
# Hapus definisi DATA_DIR lama jika ada di bawah
# DATA_DIR = os.path.join(PROJECT_ROOT, "data") # Duplikat, sudah di atas
import pandas as pd
import optuna
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import EarlyStoppingCallback

# --- Konfigurasi File Logging ---
log_filename = f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
log_filepath = PROJECT_ROOT / log_filename # Simpan log di root proyek

file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Tambahkan handler ke logger yang sudah ada
# Asumsi 'logger' adalah instance logger standar atau kompatibel
try:
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_filepath}")
except AttributeError:
    print(f"Could not add FileHandler directly to the imported 'logger'. Ensure it's a standard logging instance.")
# --- Akhir Konfigurasi File Logging ---

# --- Kelas dari train.py --- #
class AthalaDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        # Gunakan shape[0] untuk sparse matrix atau len() untuk list/numpy array biasa
        if hasattr(self.data, 'shape'):
            return self.data.shape[0]
        else:
            return len(self.data)

    def __getitem__(self, idx):
        # Pastikan data adalah tensor float dan label adalah tensor long
        item_data = self.data[idx]
        if hasattr(item_data, 'toarray'): # Cek jika sparse
            item_data = item_data.toarray()
        # Hapus squeeze(0) jika toarray menghasilkan (1, N)
        return torch.tensor(item_data, dtype=torch.float).squeeze(), torch.tensor(self.labels[idx], dtype=torch.long)

# --- Dataset untuk model Hugging Face --- #
class HFTextDataset(Dataset):
    def __init__(self, tokenizer, texts, labels=None, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        # Untuk model CausalLM seperti GPT2, labels biasanya input_ids
        self.labels = self.encodings.input_ids.clone() if labels is None else torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        # Pastikan ada key 'labels' yang diharapkan Trainer
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# --- Dataset untuk model Seq2Seq --- #
class HFSeq2SeqDataset(Dataset):
    def __init__(self, tokenizer, inputs, targets, max_input_length=512, max_target_length=128):
        self.inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=max_input_length, return_tensors="pt")
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            self.targets = tokenizer(targets, truncation=True, padding="max_length", max_length=max_target_length, return_tensors="pt")

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.inputs.items()}
        # Decoder input ids biasanya diambil dari target tokenizer
        item["labels"] = self.targets["input_ids"][idx].clone().detach()
        return item

    def __len__(self):
        return len(self.inputs["input_ids"])

# --- Dataset untuk Captcha (contoh sederhana) --- #
class CaptchaDataset(Dataset):
     # Implementasi dataset untuk gambar+teks captcha (lebih kompleks)
     # Perlu memuat gambar, memprosesnya, dan tokenize teks
     pass


class AthalaClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AthalaClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# --- Fungsi Pelatihan Utama yang Dimodifikasi --- #
def train_component(component, data_path):
    logger.info(f'--- Starting Training for Component: {component} ---')
    model_save_dir = Path(DATA_DIR) / 'models' # Direktori simpan model
    model_save_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Logika if/elif untuk setiap komponen dikembalikan --- #
        if component == 'dialog':
            logger.info('Configuring training for Dialog component...')
            data_file = Path(data_path)
            model_output_dir = model_save_dir / 'dialog_model'
            model_output_dir.mkdir(parents=True, exist_ok=True)

            if not data_file.exists():
                logger.error(f'Data file not found: {data_file}. Skipping dialog training.')
                return None

            df = pd.read_parquet(data_file)
            texts = df['text'].fillna('').tolist()
            if not texts: logger.error('No text data found in dialog file.'); return None

            # Split data into train and validation sets
            train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
            logger.info(f'Split data into {len(train_texts)} training and {len(val_texts)} validation samples')

            # Initialize tokenizer and model with custom name
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            
            # Set custom model name
            model.config.name_or_path = "Donquixote Athala"
            model.config.model_type = "gpt2"
            model.config.architectures = ["GPT2LMHeadModel"]
            
            # Create datasets
            train_dataset = HFTextDataset(tokenizer, train_texts)
            val_dataset = HFTextDataset(tokenizer, val_texts)

            # Training arguments for dialog (GPT-2)
            training_args = TrainingArguments(
                output_dir=str(model_output_dir),
                num_train_epochs=10,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=32,
                learning_rate=3e-5,
                weight_decay=0.01,
                warmup_steps=500,
                fp16=True,
                gradient_checkpointing=True,
                save_steps=500,
                logging_steps=100,
                save_total_limit=2,
                report_to='none',
                # Evaluasi dengan parameter yang kompatibel
                eval_steps=500,
                # Optimasi
                optim="adamw_torch",
                lr_scheduler_type="linear",
                max_grad_norm=1.0,
            )

            # Buat trainer dengan konfigurasi yang lebih lengkap
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=lambda eval_pred: {
                    "perplexity": torch.exp(torch.tensor(eval_pred.predictions)).mean().item(),
                    "eval_loss": eval_pred.predictions.mean().item()
                }
            )

            logger.info('Starting dialog model training...')
            # Tambahkan evaluasi manual
            best_eval_loss = float('inf')
            best_model_path = None
            
            for epoch in range(training_args.num_train_epochs):
                # Training
                train_result = trainer.train()
                train_loss = train_result.training_loss
                
                # Evaluasi
                eval_result = trainer.evaluate()
                eval_loss = eval_result['eval_loss']
                perplexity = eval_result['perplexity']
                
                logger.info(f'Epoch {epoch + 1}:')
                logger.info(f'  Train Loss: {train_loss:.4f}')
                logger.info(f'  Eval Loss: {eval_loss:.4f}')
                logger.info(f'  Perplexity: {perplexity:.4f}')
                
                # Simpan model terbaik
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model_path = model_output_dir / f"best_model_epoch_{epoch + 1}"
                    trainer.save_model(best_model_path)
                    logger.info(f'  New best model saved at {best_model_path}')
            
            # Load model terbaik
            if best_model_path:
                model = model.from_pretrained(best_model_path)
                logger.info(f'Loaded best model from {best_model_path}')
            
            # Save model with custom name and metadata
            model.save_pretrained(
                model_output_dir / "Donquixote_Athala",
                save_config=True,
                save_weights=True
            )
            tokenizer.save_pretrained(model_output_dir / "Donquixote_Athala")
            
            # Simpan metrik pelatihan
            metrics = {
                'model': model,
                'loss': train_loss,
                'eval_loss': eval_loss,
                'perplexity': perplexity,
                'epoch': epoch + 1,
                'best_model_path': str(best_model_path) if best_model_path else None,
                'training_config': {
                    'batch_size': training_args.per_device_train_batch_size,
                    'gradient_accumulation_steps': training_args.gradient_accumulation_steps,
                    'learning_rate': training_args.learning_rate,
                    'weight_decay': training_args.weight_decay,
                    'warmup_steps': training_args.warmup_steps,
                    'fp16': training_args.fp16,
                    'model_name': "Donquixote Athala"
                }
            }
            
            # Log metrik secara detail
            logger.info(f'Training metrics: {json.dumps(metrics, indent=2)}')
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return metrics

        elif component == "coding":
            logger.info("Configuring training for Coding component...")
            data_file = Path(data_path)
            model_output_dir = model_save_dir / "coding_model"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}. Skipping coding training.")
                return None

            df = pd.read_parquet(data_file)
            if 'content' not in df.columns: logger.error("Column 'content' not found. Skipping."); return None
            codes = df['content'].fillna('').tolist()
            if not codes: logger.error("No code data found."); return None

            # Split data into train and validation sets
            train_codes, val_codes = train_test_split(codes, test_size=0.1, random_state=42)
            logger.info(f'Split data into {len(train_codes)} training and {len(val_codes)} validation samples')

            # Initialize Phi-1.5 model with LoRA
            tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
            model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
            
            # Set custom model name
            model.config.name_or_path = "Donquixote Athala"
            model.config.model_type = "phi"
            model.config.architectures = ["PhiForCausalLM"]

            # Configure LoRA
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            
            # Create datasets
            train_dataset = HFTextDataset(tokenizer, train_codes)
            val_dataset = HFTextDataset(tokenizer, val_codes)

            # Training arguments for coding (Phi-1.5 with LoRA)
            training_args = TrainingArguments(
                output_dir=str(model_output_dir),
                num_train_epochs=5,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=16,
                learning_rate=2e-5,
                weight_decay=0.01,
                warmup_steps=500,
                fp16=True,
                gradient_checkpointing=True,
                save_steps=500,
                logging_steps=100,
                save_total_limit=2,
                report_to='none',
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=lambda eval_pred: {"perplexity": torch.exp(torch.tensor(eval_pred.predictions)).mean().item()}
            )

            logger.info("Starting coding model training...")
            trainer.train()
            trainer.save_model()
            
            # Save model with custom name
            model.save_pretrained(model_output_dir / "Donquixote_Athala")
            tokenizer.save_pretrained(model_output_dir / "Donquixote_Athala")
            
            logger.info(f"Coding model training complete. Model saved in {model_output_dir}")
            metrics = {
                'model': model,
                'loss': trainer.state.log_history[-1].get('train_loss', None) if trainer.state.log_history else None,
                'eval_loss': trainer.state.log_history[-1].get('eval_loss', None) if trainer.state.log_history else None,
                'perplexity': trainer.state.log_history[-1].get('perplexity', None) if trainer.state.log_history else None,
                'epoch': trainer.state.log_history[-1].get('epoch', None) if trainer.state.log_history else None
            }
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return metrics

        elif component == "general":
            logger.info("Configuring training for General Purpose component...")
            data_file = Path(data_path)
            model_output_dir = model_save_dir / "general_model"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}. Skipping general training.")
                return None

            df = pd.read_parquet(data_file)
            texts = df['text'].fillna('').tolist()
            if not texts: logger.error("No text data found."); return None

            # Split data into train and validation sets
            train_texts, val_texts = train_test_split(texts, test_size=0.1, random_state=42)
            logger.info(f'Split data into {len(train_texts)} training and {len(val_texts)} validation samples')

            # Initialize Mistral-7B with LoRA
            tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
            model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
            
            # Set custom model name
            model.config.name_or_path = "Donquixote Athala"
            model.config.model_type = "mistral"
            model.config.architectures = ["MistralForCausalLM"]

            # Configure LoRA
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, lora_config)
            
            # Create datasets
            train_dataset = HFTextDataset(tokenizer, train_texts)
            val_dataset = HFTextDataset(tokenizer, val_texts)

            # Training arguments for general purpose (Mistral-7B with LoRA)
            training_args = TrainingArguments(
                output_dir=str(model_output_dir),
                num_train_epochs=5,
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=32,
                learning_rate=2e-5,
                weight_decay=0.01,
                warmup_steps=500,
                fp16=True,
                gradient_checkpointing=True,
                save_steps=500,
                logging_steps=100,
                save_total_limit=2,
                report_to='none',
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=lambda eval_pred: {"perplexity": torch.exp(torch.tensor(eval_pred.predictions)).mean().item()}
            )

            logger.info("Starting general purpose model training...")
            trainer.train()
            trainer.save_model()
            
            # Save model with custom name
            model.save_pretrained(model_output_dir / "Donquixote_Athala")
            tokenizer.save_pretrained(model_output_dir / "Donquixote_Athala")
            
            logger.info(f"General purpose model training complete. Model saved in {model_output_dir}")
            metrics = {
                'model': model,
                'loss': trainer.state.log_history[-1].get('train_loss', None) if trainer.state.log_history else None,
                'eval_loss': trainer.state.log_history[-1].get('eval_loss', None) if trainer.state.log_history else None,
                'perplexity': trainer.state.log_history[-1].get('perplexity', None) if trainer.state.log_history else None,
                'epoch': trainer.state.log_history[-1].get('epoch', None) if trainer.state.log_history else None
            }
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return metrics

        elif component == "math":
            logger.info("Configuring training for Math component...")
            data_file = Path(data_path)
            model_output_dir = model_save_dir / "math_model"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}. Skipping math training.")
                return None

            df = pd.read_parquet(data_file)
            if 'problem' not in df.columns or 'solution' not in df.columns:
                 logger.error("Missing 'problem' or 'solution' column. Skipping."); return None
            problems = df["problem"].fillna('').tolist()
            solutions = df["solution"].fillna('').tolist()
            if not problems: logger.error("No math problems found."); return None

            # Gunakan google/flan-t5-small
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

            train_dataset = HFSeq2SeqDataset(tokenizer, problems, solutions)

            training_args = TrainingArguments(
                output_dir=str(model_output_dir),
                num_train_epochs=10, # Flan-T5 mungkin perlu lebih banyak epoch
                per_device_train_batch_size=4, # Sesuaikan
                gradient_accumulation_steps=4,
                save_steps=500,
                save_total_limit=2,
                logging_steps=50,
                fp16=torch.cuda.is_available(),
                report_to="none",
                predict_with_generate=True, # Penting untuk Seq2Seq
            )
            trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
            logger.info("Starting math model training...")
            trainer.train()
            trainer.save_model()
            logger.info(f"Math model training complete. Model saved in {model_output_dir}")
            metrics = {
                'model': model,
                'loss': trainer.state.log_history[-1].get('train_loss', None) if trainer.state.log_history else None,
                'epoch': trainer.state.log_history[-1].get('epoch', None) if trainer.state.log_history else None
            }
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return metrics
        elif component == "trading":
            logger.info("Configuring training for Trading component (TCN)...")
            data_file = Path(data_path)
            model_save_path = model_save_dir / "trading_model.pt"

            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}. Skipping trading training.")
                return None

            df = pd.read_parquet(data_file)
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in expected_cols):
                 logger.error(f"Missing columns {expected_cols}. Skipping."); return None

            # Ambil fitur dan target, bersihkan NaN
            try:
                features = df[expected_cols].apply(pd.to_numeric, errors='coerce').values
                targets_raw = df['Close'].apply(pd.to_numeric, errors='coerce').values
            except Exception as e:
                 logger.error(f"Failed to convert trading data to numeric: {e}. Skipping."); return None

            valid_feature_mask = ~np.isnan(features).any(axis=1)
            valid_target_mask = ~np.isnan(targets_raw)

            features_clean = features[valid_feature_mask]
            targets_clean = targets_raw[valid_target_mask]

            # Sesuaikan panjang untuk prediksi langkah berikutnya
            max_len = min(len(features_clean) - 1, len(targets_clean) -1)
            if max_len < 1:
                 logger.error("Not enough overlapping valid data points for TCN training. Skipping."); return None

            features_aligned = features_clean[:max_len]        # Data t=0..N-1
            targets_aligned = targets_clean[1 : max_len + 1]  # Data t=1..N

            # Persiapan Tensor untuk TCN (asumsi batch=1 untuk TCN sederhana ini)
            # TCN kita perlu input [batch, channels, sequence_length]
            inputs = torch.tensor(features_aligned, dtype=torch.float32).T.unsqueeze(0) # -> [1, 5, N-1]
            targets_tensor = torch.tensor(targets_aligned, dtype=torch.float32).unsqueeze(0) # -> [1, N-1]

            model = TCN(input_size=len(expected_cols), output_size=1, num_channels=[16, 32, 64])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Training TCN on device: {device}") # Log device
            model.to(device)
            inputs, targets_tensor = inputs.to(device), targets_tensor.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            num_epochs = 20 # TCN mungkin perlu lebih banyak epoch

            logger.info(f"Starting TCN training for {num_epochs} epochs...")
            model.train()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                # TCN output shape: [batch, output_size, sequence_length] perlu disesuaikan
                outputs = model(inputs).squeeze() # -> [N-1]
                loss = criterion(outputs, targets_tensor.squeeze()) # Target juga di-squeeze
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Trading Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

            torch.save(model.state_dict(), model_save_path)
            logger.info(f"Trading TCN model training complete. Model saved to {model_save_path}")
            metrics = {
                'model': model,
                'loss': loss.item(),
                'epoch': num_epochs
            }
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return metrics
        elif component == "captcha":
             # Implementasi pelatihan Captcha (lebih kompleks)
             # Perlu load gambar, proses dengan AutoProcessor, tokenize teks
             # Kemungkinan perlu model multi-modal atau dua model terpisah
             logger.warning(f"Training logic for Captcha component is not fully implemented. Skipping.")
             # Contoh placeholder:
             # image_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
             # text_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
             # ... (load data, dataset, dataloader, training loop) ...
             # torch.save(image_model.state_dict(), model_save_dir / "captcha_image_model.pt")
             # torch.save(text_model.state_dict(), model_save_dir / "captcha_text_model.pt")
             return None
        elif component == "threat_intel":
            logger.info("Configuring training for Threat Intel component (Isolation Forest)...")
            data_file = Path(data_path)
            model_save_path = model_save_dir / "threat_intel_model.pkl"

            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}. Skipping threat intel training.")
                return None

            df = pd.read_parquet(data_file)
            # Asumsi fitur sederhana: panjang teks (sesuaikan jika perlu fitur lebih kompleks)
            if 'text' not in df.columns: logger.error("'text' column needed for Isolation Forest. Skipping."); return None
            texts = df['text'].fillna('').tolist()
            if not texts: logger.error("No text data for Isolation Forest."); return None
            features = np.array([len(t) for t in texts]).reshape(-1, 1)

            model = IsolationForest(contamination='auto', random_state=42) # contamination='auto' lebih adaptif
            logger.info("Fitting Isolation Forest model...")
            model.fit(features)
            with open(model_save_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Threat Intel Isolation Forest model training complete. Model saved to {model_save_path}")
            metrics = {
                'model': model,
                'loss': None,
                'epoch': None
            }
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return metrics
        elif component == "network":
            logger.info("Configuring training for Network component (Random Forest)...")
            data_file = Path(data_path)
            model_save_path = model_save_dir / "network_model.pkl"
            label_encoder_path = model_save_dir / "network_label_encoder.pkl"

            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}. Skipping network training.")
                return None

            df = pd.read_parquet(data_file)
            # Asumsi fitur sederhana: panjang teks, dan ada kolom 'label'
            # Sesuaikan fitur jika data network lebih terstruktur
            if 'text' not in df.columns or 'label' not in df.columns:
                logger.error("'text' and 'label' columns needed for Random Forest. Skipping."); return None
            texts = df['text'].fillna('').tolist()
            labels_raw = df['label'].astype(str).fillna('unknown').tolist()
            if not texts: logger.error("No text data for Random Forest."); return None

            features = np.array([len(t) for t in texts]).reshape(-1, 1)
            encoder = LabelEncoder()
            labels = encoder.fit_transform(labels_raw)

            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Tambahkan class_weight
            logger.info("Fitting Random Forest model...")
            model.fit(features, labels)
            with open(model_save_path, 'wb') as f:
                pickle.dump(model, f)
            with open(label_encoder_path, 'wb') as f:
                pickle.dump(encoder, f) # Simpan juga encoder
            logger.info(f"Network Random Forest model training complete. Model and encoder saved.")
            metrics = {
                'model': model,
                'loss': None,
                'epoch': None
            }
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return metrics
        elif component == "rag":
            # Pelatihan RAG biasanya melibatkan fine-tuning retriever atau generator,
            # atau hanya indexing dokumen. Logika di sini mungkin hanya indexing.
            # Jika ingin fine-tuning GPT2 untuk RAG:
            logger.warning(f"Default RAG training logic assumes fine-tuning GPT-2 on combined context. Check if this is desired.")
            logger.info("Configuring training for RAG component (GPT-2 fine-tuning)... ")
            data_file = Path(data_path)
            model_output_dir = model_save_dir / "rag_model"
            model_output_dir.mkdir(parents=True, exist_ok=True)

            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}. Skipping RAG training.")
                return None

            df = pd.read_parquet(data_file)
            # Asumsi data sudah berisi teks yang siap untuk fine-tuning (mungkin context+query+answer)
            if 'text' not in df.columns: logger.error("'text' column needed for RAG GPT-2 fine-tuning. Skipping."); return None
            texts = df['text'].fillna('').tolist()
            if not texts: logger.error("No text data for RAG training."); return None

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            train_dataset = HFTextDataset(tokenizer, texts)

            training_args = TrainingArguments(
                output_dir=str(model_output_dir),
                num_train_epochs=10, # Sesuaikan
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                save_steps=1000,
                save_total_limit=2,
                logging_steps=100,
                fp16=torch.cuda.is_available(),
                report_to="none",
            )
            trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
            logger.info("Starting RAG model (GPT-2) training...")
            trainer.train()
            trainer.save_model()
            logger.info(f"RAG model training complete. Model saved in {model_output_dir}")
            metrics = {
                'model': model,
                'loss': trainer.state.log_history[-1].get('train_loss', None) if trainer.state.log_history else None,
                'epoch': trainer.state.log_history[-1].get('epoch', None) if trainer.state.log_history else None
            }
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return metrics
        elif component == "general_classifier":
            logger.info("Starting training for general_classifier...")
            processed_dir = Path(DATA_DIR) / "processed"
            all_texts = []
            all_labels = []
            model_save_path = model_save_dir / "general_classifier_model.pt"
            vectorizer_save_path = model_save_dir / "general_vectorizer.pkl"
            encoder_save_path = model_save_dir / "general_encoder.pkl"

            # --- Pemindaian File Lebih Luas --- #
            logger.info(f"Scanning for diverse processed files in {processed_dir}")
            # Gabungkan semua pola file yang relevan
            file_patterns = [
                "summary_*.json",
                "metadata_*.json", # Kembali aktifkan baris ini
                "extracted_*.txt",
                "*.parquet"  # <<<----- INI MASIH SALAH!
            ]
            # --- Akhir Revert --- #
            all_relevant_files = []
            for pattern in file_patterns:
                # Operasi glob ini HANYA menemukan file yang cocok dengan pola di atas
                all_relevant_files.extend(list(processed_dir.glob(pattern)))

            logger.info(f"Found {len(all_relevant_files)} potentially relevant processed files.")

            # --- Loop Melalui Semua File yang Ditemukan --- #
            processed_files_count = {"json": 0, "txt": 0, "parquet": 0, "skipped": 0}
            for file_path in all_relevant_files:
                extracted_text = None
                extracted_label = None
                logger.debug(f"Attempting to process: {file_path.name}")

                try:
                    # --- Logika untuk JSON (summary/metadata) --- #
                    if file_path.suffix == '.json': # Proses semua JSON
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Ekstrak teks dari field umum 
                        text_candidates = [
                            data.get('text'),
                            data.get('summary'), 
                            data.get('content'), 
                            data.get('sample_content'),
                            str(data.get('sample_data')) if data.get('sample_data') else None, 
                            data.get('metadata_comment'),
                            data.get('metadata_description')
                        ]
                        extracted_text = next((t for t in text_candidates if t and isinstance(t, str) and t.strip()), None)
                        
                        # Ekstrak label dari field umum, fallback ke nama file
                        label_candidates = [
                            data.get('label'),
                            data.get('category'),
                            data.get('metadata_category'),
                            data.get('type')
                        ]
                        extracted_label = next((l for l in label_candidates if l and isinstance(l, str) and l.strip()), None)
                        if not extracted_label:
                           # Fallback: Ambil dari nama file (setelah prefix)
                           if file_path.name.startswith("summary_"):
                               extracted_label = file_path.stem.replace("summary_", "")
                           elif file_path.name.startswith("metadata_"):
                                extracted_label = file_path.stem.replace("metadata_", "")
                           else:
                               extracted_label = file_path.stem 
                        extracted_label = extracted_label or "unknown_json" # Default jika tidak ada

                    # --- Logika untuk TXT (ekstrak PDF) --- #
                    elif file_path.suffix == '.txt' and file_path.name.startswith("extracted_"):
                         with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                             extracted_text = f.read().strip()
                         # Ambil label dari nama file
                         extracted_label = file_path.stem.replace("extracted_", "")
                         extracted_label = extracted_label or "unknown_pdf_extract"

                    # --- Logika untuk Parquet (dari manager_multi) --- #
                    elif file_path.suffix == '.parquet': # <-- Hapus syarat endswith("_train.parquet")
                        logger.debug(f"Processing Parquet file: {file_path.name}")
                        try:
                            df = pd.read_parquet(file_path)
                            if df.empty:
                                logger.warning(f"Skipping empty Parquet file: {file_path.name}")
                                processed_files_count["skipped"] += 1
                                continue

                            # --- Logika Pencarian Kolom Teks yang Lebih Fleksibel --- #
                            text_col_candidates = ['text', 'content', 'code', 'problem', 'question', 'answer', 'solution', 'selftext', 'body', 'article', 'title', 'summary', 'description']
                            text_col = next((col for col in text_col_candidates if col in df.columns), None)

                            if not text_col:
                                logger.warning(f"No suitable text column {text_col_candidates} found in Parquet: {file_path.name}. Skipping.")
                                processed_files_count["skipped"] += 1
                                continue
                            
                            # --- Logika Pencarian Kolom Label (Prioritaskan Kolom Internal) --- #
                            label_col_candidates = ['label', 'category', 'Symbol', 'subreddit', 'language', 'attack_cat', 'type'] # Tambahkan kandidat kolom label
                            label_col = next((col for col in label_col_candidates if col in df.columns), None)

                            # Ambil teks
                            texts_list = df[text_col].astype(str).fillna('').tolist()

                            # Ambil label
                            if label_col:
                                # Jika ada kolom label, gunakan itu per baris
                                labels_list = df[label_col].astype(str).fillna(f'unknown_{label_col}').tolist()
                                logger.debug(f"Using label column '{label_col}' from Parquet: {file_path.name}")
                            else:
                                # Jika tidak ada kolom label, gunakan nama file sebagai fallback untuk semua baris
                                fallback_label = file_path.stem.replace("_train", "").strip() # Tetap coba bersihkan _train jika ada
                                fallback_label = fallback_label if fallback_label else f"unknown_parquet_{file_path.stem}"
                                labels_list = [fallback_label] * len(texts_list)
                                logger.debug(f"Using filename as fallback label '{fallback_label}' for Parquet: {file_path.name}")

                            if texts_list and labels_list:
                                all_texts.extend(texts_list)
                                all_labels.extend(labels_list)
                                processed_files_count["parquet"] += 1
                                logger.debug(f"Added {len(texts_list)} samples from Parquet '{file_path.name}' with label source: {'column ' + label_col if label_col else 'filename'}")
                            else:
                                logger.warning(f"No text/labels extracted from Parquet: {file_path.name}")
                                processed_files_count["skipped"] += 1

                        except Exception as e_parquet:
                            logger.error(f"Failed to read or process Parquet file {file_path.name}: {e_parquet}", exc_info=True)
                            processed_files_count["skipped"] += 1
                        # Tidak perlu continue di sini, loop akan lanjut otomatis
                    # elif file_path.suffix == '.parquet' and file_path.name.endswith("_train.parquet"): <-- LOGIKA LAMA DIHAPUS
                    #     df = pd.read_parquet(file_path)
                    #     label = file_path.stem.replace("_train", "").strip()
                        
                    #     if not label:
                    #          logger.warning(f"Skipping Parquet file with empty label derived from filename: {file_path.name}")
                    #          continue
                        
                    #     text_series = None
                    #     common_text_cols = ['text', 'content', 'code', 'problem']
                    #     found_col = next((col for col in common_text_cols if col in df.columns), None)
                    #     if found_col:
                    #         text_series = df[found_col].astype(str).fillna('')
                    #         texts_list = text_series.tolist()
                    #         labels_list = [label] * len(texts_list)
                    #         all_texts.extend(texts_list)
                    #         all_labels.extend(labels_list)
                    #         logger.debug(f"Added {len(texts_list)} samples from Parquet: {label}")
                    #         continue
                    #     else:
                    #          logger.warning(f"No suitable text column {common_text_cols} found in Parquet: {file_path.name}. Skipping this file for general_classifier.")

                    # --- Tambahkan teks dan label tunggal dari JSON/TXT --- #
                    if file_path.suffix in ['.json', '.txt']:
                       if extracted_text and extracted_label:
                           all_texts.append(extracted_text)
                           all_labels.append(extracted_label)
                           if file_path.suffix == '.json': processed_files_count["json"] += 1
                           if file_path.suffix == '.txt': processed_files_count["txt"] += 1
                           logger.debug(f"Added 1 sample from {file_path.suffix}: {extracted_label}")
                       elif extracted_text is None:
                            logger.warning(f"Could not extract text from {file_path.name}")
                            processed_files_count["skipped"] += 1
                       elif extracted_label is None:
                            # Jika teks ada tapi label tidak ketemu dari field, label sudah di-fallback ke nama file sebelumnya
                            # Jadi, seharusnya tidak masuk sini kecuali fallback juga gagal (nama file kosong?)
                            logger.warning(f"Could not determine label for {file_path.name}")
                            processed_files_count["skipped"] += 1
                    # elif extracted_text is None and file_path.suffix in ['.json', '.txt']: <-- Logika lama dipindah ke atas
                    #      logger.warning(f"Could not extract text from {file_path.name}")

                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}")
                    processed_files_count["skipped"] += 1
             # --- Akhir Loop File --- #

            logger.info(f"File processing summary: JSON={processed_files_count['json']}, TXT={processed_files_count['txt']}, Parquet={processed_files_count['parquet']}, Skipped/Error={processed_files_count['skipped']}")

            if not all_texts:
                 logger.error("No text samples were collected from any source for general_classifier. Skipping.")
                 return None
            if len(set(all_labels)) < 1: # Cek minimal ada 1 label
                 logger.error("No labels were found for general_classifier. Skipping.")
                 return None
            
            logger.info(f"Total text samples collected from all sources: {len(all_texts)}, Total unique labels: {len(set(all_labels))}")

            # --- Lanjutkan dengan Vectorization, Encoding --- #
            logger.info("Vectorizing text data using TF-IDF...")
            # --- Reduce max_features to lower CPU/RAM usage --- #
            # vectorizer = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 2))
            vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2)) # Try 10k first
            logger.info(f"Using TF-IDF with max_features={vectorizer.max_features}")
            # --- End reduction --- #
            X_tfidf = vectorizer.fit_transform(all_texts)

            logger.info("Encoding labels...")
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(all_labels)
            num_classes = len(encoder.classes_)
            if num_classes < 50:
                 logger.info(f"Class mapping: {dict(zip(encoder.classes_, range(num_classes)))}")
            else:
                 logger.info(f"Total classes found: {num_classes}")

            # --- Filter kelas dengan hanya 1 anggota (tetap lakukan ini) --- #
            logger.info("Checking for and filtering classes with only one sample...")
            unique_labels, counts = np.unique(y_encoded, return_counts=True)
            single_sample_labels = unique_labels[counts == 1]
            if len(single_sample_labels) > 0:
                logger.warning(f"Found {len(single_sample_labels)} classes with only 1 sample. These will be removed before splitting.")
                keep_mask = np.isin(y_encoded, unique_labels[counts > 1])
                X_tfidf_filtered = X_tfidf[keep_mask]
                y_encoded_filtered = y_encoded[keep_mask]
                logger.info(f"Data filtered. Samples remaining: {X_tfidf_filtered.shape[0]}")
                
                # --- FIX: Remap filtered labels to be contiguous and 0-indexed --- #
                if X_tfidf_filtered.shape[0] > 0:
                    logger.info("Remapping filtered labels to start from 0...")
                    remapper = LabelEncoder() # Gunakan encoder baru untuk remapping
                    y_encoded_remapped = remapper.fit_transform(y_encoded_filtered)
                    final_num_classes = len(remapper.classes_) # Dapatkan jumlah kelas dari remapper
                    logger.info(f"Labels remapped. Final number of classes: {final_num_classes}")
                else:
                    logger.warning("No samples left after filtering, cannot remap labels.")
                    y_encoded_remapped = np.array([]) # Pastikan array kosong jika tidak ada data
                    final_num_classes = 0
                # --- Akhir FIX --- #

                # Cek ulang jumlah kelas setelah filter dan remap
                # num_classes_filtered = len(np.unique(y_encoded_filtered))
                # logger.info(f"Number of classes after filtering: {num_classes_filtered}") <-- Log lama, diganti di atas
                if X_tfidf_filtered.shape[0] == 0 or final_num_classes == 0:
                     logger.error("No data or no classes remaining after filtering single samples. Cannot train classifier. Skipping.")
                     return None
            else:
                logger.info("No classes with only one sample found. Proceeding with all data.")
                X_tfidf_filtered = X_tfidf
                y_encoded_remapped = y_encoded # Gunakan y_encoded asli jika tidak ada filter
                final_num_classes = num_classes # Gunakan num_classes asli
            # --- Akhir Filter & Remap Block --- #
            
            # --- Cek Ulang Sebelum Split --- #
            if X_tfidf_filtered.shape[0] == 0 or len(y_encoded_remapped) == 0:
                logger.error("No data available for splitting after potential filtering/remapping. Skipping general_classifier training.")
                return None
            if X_tfidf_filtered.shape[0] < 2:
                 logger.error("Need at least 2 samples total to perform train/test split. Skipping general_classifier training.")
                 return None
            
            # Tentukan num_classes aktual yang akan digunakan model (sudah dihitung sebagai final_num_classes)
            # final_num_classes = len(np.unique(y_encoded_remapped)) # <-- Dihitung di atas
            logger.info(f"Final number of classes for the model: {final_num_classes}")
            if final_num_classes < 1:
                 logger.error("No classes left to train on. Skipping.")
                 return None
                 
            logger.info(f"Splitting data (Random Split)... Ratio 80/20")
            # Gunakan data yang sudah difilter dan label yang sudah diremap
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf_filtered, 
                y_encoded_remapped, # <-- Gunakan label yang sudah diremap
                test_size=0.2, 
                random_state=42
            )

            logger.info("Creating DataLoaders...")
            train_dataset = AthalaDataset(X_train, y_train)
            test_dataset = AthalaDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

            input_size = X_tfidf_filtered.shape[1] # Gunakan shape dari data terfilter
            hidden_size = 256 
            # Gunakan jumlah kelas akhir setelah filter
            model = AthalaClassifier(input_size, hidden_size, final_num_classes)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Training general_classifier on device: {device}") # Log device
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # AdamW dan learning rate lebih kecil
            num_epochs = 15 # Tingkatkan epoch

            logger.info(f"Starting general_classifier training for {num_epochs} epochs on {device}...")
            best_accuracy = 0.0
            for epoch in range(num_epochs):
                model.train()
                train_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)
                train_loss /= len(train_loader.dataset)

                model.eval()
                test_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                test_loss /= len(test_loader.dataset)
                accuracy = 100 * correct / total
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

                # Simpan model terbaik berdasarkan akurasi validasi
                if accuracy > best_accuracy:
                     best_accuracy = accuracy
                     torch.save(model.state_dict(), model_save_path)
                     with open(vectorizer_save_path, 'wb') as f:
                         pickle.dump(vectorizer, f)
                     with open(encoder_save_path, 'wb') as f:
                         pickle.dump(encoder, f)
                     logger.info(f"Best model saved with accuracy: {accuracy:.2f}%")

            logger.info(f"General classifier training finished. Best model saved to {model_save_dir}")
            metrics = {
                'model': model,
                'loss': best_accuracy,
                'epoch': num_epochs
            }
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return metrics
        else:
            logger.warning(f'Component {component} not recognized for training in this script.')
            logger.info(f'--- Finished Training Attempt for Component: {component} ---')
            return None
    except FileNotFoundError as fnf_err:
        logger.error(f"Data file not found during training for {component}: {fnf_err}")
    except ImportError as imp_err:
         logger.error(f"Import error during training for {component}: {imp_err}. Make sure required libraries are installed.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training for {component}: {e}", exc_info=True)
        logger.info(f'--- Finished Training Attempt for Component: {component} ---')
        return None


# --- Bagian Eksekusi Utama --- #
if __name__ == "__main__":
    # Pastikan 'general_classifier' ada di daftar ini
    components_to_train = [
        "dialog", # Mungkin masih error jika trust_remote_code belum benar
        "coding", # Dikomentari oleh Anda
        "math",
        "trading",
        "captcha",       # Perlu implementasi logika captcha
        "threat_intel",
        "network",
        "rag",
        "general_classifier" # <-- PASTIKAN INI TIDAK DIKOMENTARI
    ]

    # Hapus deteksi dinamis dan pemanggilan importlib
    logger.info(f"=== Starting Centralized Training for Components: {components_to_train} ===")

    for component in components_to_train:
        # Untuk general_classifier, data_path tidak terlalu digunakan karena ia memindai seluruh direktori
        data_path_for_component = f"{DATA_DIR}/processed/{component}_train.parquet"
        train_component(component, data_path_for_component)

    logger.info("=== Finished Centralized Training Orchestration ===")
