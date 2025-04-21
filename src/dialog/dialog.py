from transformers import GPT2LMHeadModel, GPT2Tokenizer
from src.utils.database_manager import DatabaseManager
from src.ai.rag import RAG
from logger import logger
from config import DATA_DIR
import os

class DialogModel:
    def __init__(self, model_path=f"{DATA_DIR}/models/dialog_model.pt"):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained(model_path if os.path.exists(model_path) else "gpt2")
        self.db = DatabaseManager()
        self.rag = RAG()
        logger.info("DialogModel initialized with pre-trained GPT-2")

    def generate_response(self, prompt):
        try:
            rag_response = self.rag.generate(prompt)
            inputs = self.tokenizer(f"{rag_response}\nUser: {prompt}\nBot:", return_tensors="pt")
            outputs = self.model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated dialog response: {response}")
            return response
        except Exception as e:
            logger.error(f"Error generating dialog response: {str(e)}")
            return "Maaf, ada error bro!"