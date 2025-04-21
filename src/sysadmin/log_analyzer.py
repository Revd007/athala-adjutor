from src.sysadmin.parser import SysadminParser
from logger import logger

class LogAnalyzer:
    def __init__(self):
        self.parser = SysadminParser()

    def analyze_logs(self, log_file="./data/logs/sysadmin.log"):
        """Analyze logs for anomalies."""
        try:
            threats = self.parser.analyze(log_file)
            anomalies = [t for t in threats if "critical" in t["error"].lower()]
            logger.info(f"Detected anomalies: {anomalies}")
            return anomalies
        except Exception as e:
            logger.error(f"Log analyzer error: {e}")
            raise

if __name__ == "__main__":
    analyzer = LogAnalyzer()
    print(analyzer.analyze_logs())