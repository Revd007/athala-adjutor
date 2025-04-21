from src.deepsearch.scraper import WebScraper
from logger import logger

class SysadminScraper(WebScraper):
    def scrape_sysadmin(self, url):
        """Scrape sysadmin-specific content."""
        try:
            text = self.scrape(url)
            # Filter sysadmin-related content (simplified)
            sysadmin_terms = ["error", "log", "server", "network", "configuration"]
            filtered = ' '.join([word for word in text.split() if any(term in word.lower() for term in sysadmin_terms)])
            logger.info(f"Scraped sysadmin content from {url}")
            return filtered
        except Exception as e:
            logger.error(f"Sysadmin scraping error for {url}: {e}")
            raise

if __name__ == "__main__":
    scraper = SysadminScraper()
    print(scraper.scrape_sysadmin("https://serverfault.com"))