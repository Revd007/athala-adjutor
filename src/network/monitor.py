import psutil
import time
from logger import logger

class NetworkMonitor:
    def __init__(self, interval=1):
        self.interval = interval

    def monitor_network(self):
        """Monitor network traffic in real-time."""
        try:
            net_io = psutil.net_io_counters()
            bytes_sent = net_io.bytes_sent
            bytes_recv = net_io.bytes_recv
            start_time = time.time()

            time.sleep(self.interval)

            net_io = psutil.net_io_counters()
            bytes_sent_new = net_io.bytes_sent
            bytes_recv_new = net_io.bytes_recv
            end_time = time.time()

            sent_speed = (bytes_sent_new - bytes_sent) / (end_time - start_time) / 1024  # KB/s
            recv_speed = (bytes_recv_new - bytes_recv) / (end_time - start_time) / 1024  # KB/s

            result = {
                "sent_speed_kbps": round(sent_speed, 2),
                "recv_speed_kbps": round(recv_speed, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            logger.info(f"Network monitor: {result}")
            return result
        except Exception as e:
            logger.error(f"Network monitor error: {e}")
            raise

if __name__ == "__main__":
    monitor = NetworkMonitor()
    print(monitor.monitor_network())