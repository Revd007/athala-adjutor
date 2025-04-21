from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.database_manager import DatabaseManager
from src.ai.rag import RAG
from logger import logger
from config import DATA_DIR
import os

class CodeGenerator:
    def __init__(self, model_path=f"{DATA_DIR}/models/coding_model.pt"):
        self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
        self.model = AutoModelForCausalLM.from_pretrained(model_path if os.path.exists(model_path) else "Salesforce/codegen-350M-mono")
        self.db = DatabaseManager()
        self.rag = RAG()
        logger.info("CodeGenerator initialized with pre-trained CodeGen")

    def generate_code(self, prompt, language="python"):
        try:
            rag_response = self.rag.generate(prompt)
            inputs = self.tokenizer(f"{rag_response}\n# {language}\n{prompt}", return_tensors="pt")
            outputs = self.model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)
            code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"Generated code: {code}")
            return code
        except Exception as e:
            logger.error(f"Error generating code: {str(e)}")
            return "# Maaf, ada error bro!"