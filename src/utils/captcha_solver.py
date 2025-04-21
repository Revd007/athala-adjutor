import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoProcessor, AutoModelForSequenceClassification
from src.utils.browser_manager import BrowserManager
from src.utils.database_manager import DatabaseManager
from logger import logger
import numpy as np
from config import DATA_DIR
import os

class CaptchaSolver:
    def __init__(self, image_model_path=f"{DATA_DIR}/models/yolo_captcha.pt", numeric_model_path=f"{DATA_DIR}/models/numeric_classifier.pt", text_model_path=f"{DATA_DIR}/models/text_classifier.pt"):
        self.browser = BrowserManager()
        self.image_processor = AutoProcessor.from_pretrained("microsoft/resnet-18")
        self.image_model = AutoModelForImageClassification.from_pretrained(image_model_path if os.path.exists(image_model_path) else "microsoft/resnet-18")
        self.numeric_model = AutoModelForImageClassification.from_pretrained(numeric_model_path if os.path.exists(numeric_model_path) else "microsoft/resnet-18")
        self.text_processor = AutoProcessor.from_pretrained("bert-base-uncased")
        self.text_model = AutoModelForSequenceClassification.from_pretrained(text_model_path if os.path.exists(text_model_path) else "bert-base-uncased")
        self.db = DatabaseManager()
        logger.info("CaptchaSolver initialized with pre-trained models and PostgreSQL")

    def detect_captcha_type(self, driver):
        try:
            screenshot = driver.get_screenshot_as_png()
            img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            if edges.sum() > 10000:
                logger.info("Detected image-based CAPTCHA")
                return "image"
            logger.info("No CAPTCHA detected")
            return None
        except Exception as e:
            logger.error(f"Error detecting CAPTCHA type: {str(e)}")
            return None

    def solve_image_captcha(self, image_path):
        try:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            inputs = self.image_processor(images=img, return_tensors="pt")
            outputs = self.image_model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            logger.info(f"Solved image CAPTCHA with label: {predicted_label}")
            return predicted_label
        except Exception as e:
            logger.error(f"Error solving image CAPTCHA: {str(e)}")
            return None

    def solve_text_captcha(self, text):
        try:
            inputs = self.text_processor(text, return_tensors="pt")
            outputs = self.text_model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
            logger.info(f"Solved text CAPTCHA with label: {predicted_label}")
            return predicted_label
        except Exception as e:
            logger.error(f"Error solving text CAPTCHA: {str(e)}")
            return None