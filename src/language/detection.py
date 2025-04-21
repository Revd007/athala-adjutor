from langdetect import detect
from logger import logger

class LanguageDetector:
    def detect_language(self, text):
        """Detect the language of a text."""
        try:
            lang = detect(text)
            logger.info(f"Detected language: {lang}")
            return lang
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            raise

if __name__ == "__main__":
    detector = LanguageDetector()
    print(detector.detect_language("Hello, world!"))