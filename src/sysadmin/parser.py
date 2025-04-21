import re
from logger import logger

class SysadminParser:
    def __init__(self):
        self.pattern = r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*?ERROR:(.*)"

    def analyze(self, log_file="./data/logs/sysadmin.log"):
        """Parse log file for errors."""
        try:
            threats = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    for line in f:
                        match = re.search(self.pattern, line)
                        if match:
                            threats.append({"timestamp": match.group(1), "error": match.group(2)})
                logger.info(f"Parsed threats: {threats}")
            return threats
        except Exception as e:
            logger.error(f"Parser error: {e}")
            raise

if __name__ == "__main__":
    parser = SysadminParser()
    print(parser.analyze())