import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# --- Add project root to sys.path --- 
# Assuming this script is in D:/athala-adjutor/src/ai/
# Go up two levels to get the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to sys.path")
# --- End sys.path modification ---

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForImageClassification, AutoModelForSequenceClassification
from sklearn.ensemble import IsolationForest, RandomForestClassifier
# Import TCN for trading component
from src.ai.tcn import TCN
try:
    from src.utils.database_manager import DatabaseManager
except ImportError:
    print("Could not import DatabaseManager. Proceeding without database logging.")
    DatabaseManager = None

from logger import logger
# Define DATA_DIR directly instead of importing from config
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
print(f"DATA_DIR set to: {DATA_DIR}")
import pandas as pd
import optuna

def train_component(component, data_path):
    try:
        db = DatabaseManager() if DatabaseManager is not None else None
        if component == "dialog":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            df = pd.read_parquet(data_path)
            texts = df["text"].tolist()
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            torch.save(model.state_dict(), f"{DATA_DIR}/models/dialog_model.pt")
            logger.info("Trained dialog model")

        elif component == "coding":
            tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
            model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
            df = pd.read_parquet(data_path)
            codes = df["code"].tolist()
            inputs = tokenizer(codes, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            torch.save(model.state_dict(), f"{DATA_DIR}/models/coding_model.pt")
            logger.info("Trained coding model")

        elif component == "math":
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
            df = pd.read_parquet(data_path)
            problems = df["problem"].tolist()
            solutions = df["solution"].tolist()
            inputs = tokenizer(problems, return_tensors="pt", padding=True, truncation=True)
            targets = tokenizer(solutions, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, labels=targets["input_ids"])
            loss = outputs.loss
            loss.backward()
            torch.save(model.state_dict(), f"{DATA_DIR}/models/math_model.pt")
            logger.info("Trained math model")

        elif component == "trading":
            model = TCN(input_size=5, output_size=1, num_channels=[16, 32, 64])
            df = pd.read_parquet(data_path)
            features = df[["open", "high", "low", "close", "volume"]].values
            inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1)

            if len(df) < 2:
                logger.error("Not enough data for trading component training (need at least 2 rows).")
                return
            targets = torch.tensor(df["close"].iloc[1:].values, dtype=torch.float32)
            inputs = inputs[:, :, :-1]

            if inputs.shape[2] != len(targets):
                logger.error(f"Input sequence length ({inputs.shape[2]}) does not match target length ({len(targets)}) after adjustments.")
                return

            optimizer = torch.optim.Adam(model.parameters())
            criterion = nn.MSELoss()
            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(inputs)
                if outputs.shape[0] != 1 or outputs.shape[1] != 1 or outputs.shape[2] != len(targets):
                    logger.error(f"Unexpected TCN output shape: {outputs.shape}. Expected: [1, 1, {len(targets)}]")
                    return
                loss = criterion(outputs.squeeze(0).squeeze(0), targets)
                loss.backward()
                optimizer.step()
                logger.info(f"Trading Epoch {epoch+1}, Loss: {loss.item():.4f}")
            torch.save(model.state_dict(), f"{DATA_DIR}/models/trading_model.pt")
            logger.info("Trained trading model using TCN")

        elif component == "captcha":
            image_model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18")
            text_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
            df = pd.read_parquet(data_path)
            images = df["image"].tolist()
            texts = df["text"].tolist()
            labels = df["label"].tolist()
            image_inputs = AutoProcessor.from_pretrained("microsoft/resnet-18")(images, return_tensors="pt")
            text_inputs = AutoProcessor.from_pretrained("bert-base-uncased")(texts, return_tensors="pt")
            image_outputs = image_model(**image_inputs, labels=torch.tensor(labels))
            text_outputs = text_model(**text_inputs, labels=torch.tensor(labels))
            loss = image_outputs.loss + text_outputs.loss
            loss.backward()
            torch.save(image_model.state_dict(), f"{DATA_DIR}/models/yolo_captcha.pt")
            torch.save(text_model.state_dict(), f"{DATA_DIR}/models/text_classifier.pt")
            logger.info("Trained captcha models")

        elif component == "threat_intel":
            model = IsolationForest(contamination=0.1)
            df = pd.read_parquet(data_path)
            features = df["text"].apply(len).values.reshape(-1, 1)
            model.fit(features)
            torch.save(model, f"{DATA_DIR}/models/threat_intel_model.pt")
            logger.info("Trained threat intel model")

        elif component == "network":
            model = RandomForestClassifier()
            df = pd.read_parquet(data_path)
            features = df["text"].apply(len).values.reshape(-1, 1)
            labels = df["label"].tolist()
            model.fit(features, labels)
            torch.save(model, f"{DATA_DIR}/models/network_model.pt")
            logger.info("Trained network model")

        elif component == "rag":
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            model = GPT2LMHeadModel.from_pretrained("gpt2")
            df = pd.read_parquet(data_path)
            texts = df["text"].tolist()
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            loss.backward()
            torch.save(model.state_dict(), f"{DATA_DIR}/models/rag_model.pt")
            logger.info("Trained RAG model")

        if db is not None:
            db.store_update_log(component, "trained")
    except Exception as e:
        logger.error(f"Error training {component}: {str(e)}")

if __name__ == "__main__":
    components = ["dialog", "coding", "math", "trading", "captcha", "threat_intel", "network", "rag"]
    for component in components:
        train_component(component, f"{DATA_DIR}/processed/{component}_train.parquet")