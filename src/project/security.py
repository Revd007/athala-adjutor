from scapy.all import sniff
from logger import logger

class SecurityAnalyzer:
    def analyze(self, count=10):
        """Analyze network packets for suspicious activity."""
        try:
            packets = sniff(count=count, filter="tcp")
            suspicious = [p for p in packets if p.haslayer("TCP") and p["TCP"].flags & 0x12]  # SYN-ACK
            logger.info(f"Detected {len(suspicious)} suspicious packets")
            return {"suspicious_count": len(suspicious)}
        except Exception as e:
            logger.error(f"Security analysis error: {e}")
            raise

if __name__ == "__main__":
    analyzer = SecurityAnalyzer()
    print(analyzer.analyze())