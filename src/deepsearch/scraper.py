import requests
from bs4 import BeautifulSoup
from logger import logger

class WebScraper:
    def __init__(self):
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def scrape(self, url):
        """Scrape text content from a URL."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            logger.info(f"Scraped content from {url}")
            return text
        except Exception as e:
            logger.error(f"Scraping error for {url}: {e}")
            raise

if __name__ == "__main__":
    scraper = WebScraper()
    print(scraper.scrape("https://example.com"))