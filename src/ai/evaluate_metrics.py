from sklearn.metrics import accuracy_score, f1_score
from logger import logger

def compute_metrics(predictions, targets):
    """Compute accuracy and F1 score."""
    try:
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')
        metrics = {"accuracy": accuracy, "f1_score": f1}
        logger.info(f"Metrics: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"Metrics computation error: {e}")
        raise

if __name__ == "__main__":
    predictions = [0, 1, 1, 0]
    targets = [0, 1, 0, 0]
    print(compute_metrics(predictions, targets))