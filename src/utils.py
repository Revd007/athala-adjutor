import os
from logger import logger

def ensure_dir(directory):
    """Ensure a directory exists."""
    try:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory: {directory}")
    except Exception as e:
        logger.error(f"Directory creation error: {directory}, {e}")
        raise

def clean_text(text):
    """Clean text by removing extra spaces."""
    try:
        cleaned = ' '.join(text.split())
        logger.info("Text cleaned")
        return cleaned
    except Exception as e:
        logger.error(f"Text cleaning error: {e}")
        raise

if __name__ == "__main__":
    ensure_dir("./data/test")
    print(clean_text("  Hello   world  "))