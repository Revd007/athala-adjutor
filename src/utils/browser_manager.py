from selenium import webdriver
from undetected_chromedriver import Chrome, ChromeOptions
from playwright.async_api import async_playwright
from logger import logger

class BrowserManager:
    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None
        self.playwright = None
        self.browser = None

    def launch_undetected_browser(self):
        """Launch Chrome browser with anti-detection."""
        try:
            options = ChromeOptions()
            if self.headless:
                options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = Chrome(options=options)
            logger.info("Undetected Chrome browser launched")
            return self.driver
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            raise

    async def launch_playwright_browser(self):
        """Launch Playwright browser for advanced crawling."""
        try:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
            logger.info("Playwright browser launched")
            return self.browser
        except Exception as e:
            logger.error(f"Failed to launch Playwright browser: {e}")
            raise

    def close_browser(self):
        """Close Selenium browser."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Selenium browser closed")

    async def close_playwright_browser(self):
        """Close Playwright browser."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
            self.browser = None
            self.playwright = None
            logger.info("Playwright browser closed")

if __name__ == "__main__":
    # Example usage
    bm = BrowserManager()
    driver = bm.launch_undetected_browser()
    driver.get("https://example.com")
    bm.close_browser()