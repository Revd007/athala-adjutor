from src.network.monitor import NetworkMonitor
from logger import logger

class NetworkManager:
    def __init__(self):
        self.monitor = NetworkMonitor()

    def monitor_network(self):
        """Coordinate network monitoring."""
        try:
            result = self.monitor.monitor_network()
            logger.info(f"Network manager: {result}")
            return result
        except Exception as e:
            logger.error(f"Network manager error: {e}")
            raise

if __name__ == "__main__":
    manager = NetworkManager()
    print(manager.monitor_network())