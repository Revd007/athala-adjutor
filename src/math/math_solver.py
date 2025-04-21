import sympy as sp
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.utils.database_manager import DatabaseManager
from src.ai.rag import RAG
from logger import logger
from config import DATA_DIR
import os

class MathSolver:
    def __init__(self, model_path=f"{DATA_DIR}/models/math_model.pt"):
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path if os.path.exists(model_path) else "google/flan-t5-small")
        self.db = DatabaseManager()
        self.rag = RAG()
        logger.info("MathSolver initialized with pre-trained transformer")

    def solve(self, problem):
        try:
            rag_response = self.rag.generate(problem)
            inputs = self.tokenizer(f"{rag_response}\nProblem: {problem}\nSolution:", return_tensors="pt")
            outputs = self.model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
            solution = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            try:
                x = sp.Symbol('x')
                eq = sp.sympify(problem.replace("solve ", ""))
                symbolic_solution = sp.solve(eq, x)
                solution += f"\nSymbolic solution: x = {symbolic_solution}"
            except:
                pass
            logger.info(f"Solved math problem: {solution}")
            return solution
        except Exception as e:
            logger.error(f"Error solving math problem: {str(e)}")
            return "Maaf, ada error bro!"