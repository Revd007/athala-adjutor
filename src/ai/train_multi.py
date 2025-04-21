import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForImageClassification, AutoModelForSequenceClassification
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from src.utils.database_manager import DatabaseManager
from logger import logger
from config import DATA_DIR
import pandas as pd
import os
import optuna

def train_component(component, data_path):
    try:
        db = DatabaseManager()
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
            model = LSTMTCNGRU()
            df = pd.read_parquet(data_path)
            inputs = torch.tensor(df[["open", "high", "low", "close", "volume"]].values, dtype=torch.float32)
            targets = torch.tensor(df["close"].shift(-1).dropna().values, dtype=torch.float32)
            optimizer = torch.optim.Adam(model.parameters())
            for epoch in range(10):
                outputs = model(inputs.unsqueeze(0))
                loss = nn.MSELoss()(outputs, targets)
                loss.backward()
                optimizer.step()
            torch.save(model.state_dict(), f"{DATA_DIR}/models/trading_model.pt")
            logger.info("Trained trading model")

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

        db.store_update_log(component, "trained")
    except Exception as e:
        logger.error(f"Error training {component}: {str(e)}")

if __name__ == "__main__":
    components = ["dialog", "coding", "math", "trading", "captcha", "threat_intel", "network", "rag"]
    for component in components:
        train_component(component, f"{DATA_DIR}/processed/{component}_train.parquet")