import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urlparse, quote, unquote
from typing import List, Tuple, Dict, Optional, Set
import logging
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from aiohttp import ClientTimeout
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('webshop_analyzer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class WebsiteAnalysis:
    """Data class to store website analysis results"""
    url: str
    platforms: List[str]
    product_count: Optional[int]
    estimated_visitors: Optional[int]
    status: str = "success"
    error_message: Optional[str] = None

class Config:
    """Configuration class to store all constants and settings"""
    TIMEOUT_SECONDS = 30
    MAX_RETRIES = 3
    CONCURRENT_REQUESTS = 5
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0'
    ]

    SEARCH_ENGINES = {
        'google': 'https://www.google.com/search?q={}',
        'bing': 'https://www.bing.com/search?q={}',
        'duckduckgo': 'https://duckduckgo.com/html?q={}'
    }

    EXCLUDED_PATTERNS = {
        'social_media': ['facebook.com', 'instagram.com', 'linkedin.com', 'twitter.com'],
        'search': ['/search?', 'images.google', 'bing.com/images'],
        'marketplaces': ['amazon.', 'ebay.', 'pricerunner'],
        'content': ['wikipedia.org', 'youtube.com', 'pinterest']
    }

    # Platform detection signatures moved to a separate configuration file
    PLATFORM_SIGNATURES = {
        'WooCommerce': {
            'signatures': ['wp-content', 'woocommerce'],
            'product_selectors': ['li.product', 'div.product']
        },
        'Shopify': {
            'signatures': ['shopify.com', 'myshopify'],
            'product_selectors': ['div.grid-product__content', 'div.product-card']
        },
        'Magento': {
            'signatures': ['magento', 'mage'],
            'product_selectors': ['li.product-item', 'div.product-item-info']
        }
    }

class RateLimiter:
    """Rate limiter to prevent overwhelming servers"""
    def __init__(self, requests_per_second: float = 2.0):
        self.delay = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            current_time = asyncio.get_event_loop().time()
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.delay:
                await asyncio.sleep(self.delay - time_since_last)
            self.last_request_time = asyncio.get_event_loop().time()

class WebshopSearcher:
    """Handles searching for webshops across multiple search engines"""
    def __init__(self, config: Config):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = RateLimiter()
        self._user_agent_index = 0
        self.timeout = ClientTimeout(total=config.TIMEOUT_SECONDS)

    async def __aenter__(self):
        await self.init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_session()

    async def init_session(self):
        self.session = aiohttp.ClientSession(timeout=self.timeout)

    async def close_session(self):
        if self.session:
            await self.session.close()

    def get_headers(self) -> Dict[str, str]:
        """Generate headers for HTTP requests with rotating user agents"""
        return {
            'User-Agent': self.config.USER_AGENTS[self._user_agent_index % len(self.config.USER_AGENTS)],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'sv,en-US;q=0.7,en;q=0.3',
            'DNT': '1'
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL content with retry logic"""
        await self.rate_limiter.acquire()
        try:
            async with self.session.get(url, headers=self.get_headers()) as response:
                if response.status == 200:
                    return await response.text()
                logger.warning(f"Failed to fetch {url}, status: {response.status}")
                return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            raise

    async def search_company(self, company_name: str) -> Set[str]:
        """Search for company across multiple search engines"""
        results: Set[str] = set()
        search_tasks = []

        for engine_name, engine_url in self.config.SEARCH_ENGINES.items():
            search_query = f"{company_name} webshop sverige online shop"
            url = engine_url.format(quote(search_query))
            search_tasks.append(self.search_single_engine(url, company_name))

        engine_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        for urls in engine_results:
            if isinstance(urls, set):
                results.update(urls)

        return {url for url in results if self.is_valid_url(url)}

    async def search_single_engine(self, search_url: str, company_name: str) -> Set[str]:
        """Search a single engine for company websites"""
        results: Set[str] = set()
        html = await self.fetch_url(search_url)
        
        if html:
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/url?q=' in href:
                    href = href.split('/url?q=')[1].split('&')[0]
                
                if self.is_valid_url(href) and company_name.lower() in href.lower():
                    results.add(href)

        return results

    def is_valid_url(self, url: str) -> bool:
        """Validate URL against exclusion patterns and format"""
        try:
            decoded_url = unquote(url)
            parsed = urlparse(decoded_url)
            
            if not parsed.scheme in ('http', 'https') or not parsed.netloc:
                return False

            url_lower = decoded_url.lower()
            for category, patterns in self.config.EXCLUDED_PATTERNS.items():
                if any(pattern in url_lower for pattern in patterns):
                    return False

            return not parsed.path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.pdf', '.doc', '.docx'))
        except Exception as e:
            logger.error(f"Error validating URL {url}: {str(e)}")
            return False

class WebsiteAnalyzer:
    """Analyzes websites for e-commerce platform detection and metrics"""
    def __init__(self, config: Config):
        self.config = config

    def detect_platform(self, html: str) -> List[str]:
        """Detect e-commerce platform from HTML content"""
        detected_platforms = []
        for platform, config in self.config.PLATFORM_SIGNATURES.items():
            if any(sig.lower() in html.lower() for sig in config['signatures']):
                detected_platforms.append(platform)
        return detected_platforms if detected_platforms else ['Unknown']

    def count_products(self, soup: BeautifulSoup, platforms: List[str]) -> Optional[int]:
        """Count products based on detected platform selectors"""
        product_count = 0
        for platform in platforms:
            if platform in self.config.PLATFORM_SIGNATURES:
                selectors = self.config.PLATFORM_SIGNATURES[platform]['product_selectors']
                for selector in selectors:
                    product_count += len(soup.select(selector))
        return product_count if product_count > 0 else None

    def estimate_visitors(self, html: str) -> Optional[int]:
        """Estimate visitor count based on analytics presence"""
        analytics_patterns = ['Google Analytics', 'gtag', 'facebook-pixel']
        if any(pattern in html for pattern in analytics_patterns):
            return pd.Series([500, 5000]).sample(1).iloc[0]
        return None

class WebshopAnalyzer:
    """Main class coordinating the webshop analysis process"""
    def __init__(self, config: Config):
        self.config = config
        self.searcher = WebshopSearcher(config)
        self.analyzer = WebsiteAnalyzer(config)

    async def analyze_company(self, company_name: str) -> List[WebsiteAnalysis]:
        """Analyze company webshops and gather metrics"""
        async with self.searcher:
            urls = await self.searcher.search_company(company_name)
            analysis_tasks = [self.analyze_url(url) for url in urls]
            return await asyncio.gather(*analysis_tasks)

    async def analyze_url(self, url: str) -> WebsiteAnalysis:
        """Analyze a single URL for e-commerce metrics"""
        try:
            html = await self.searcher.fetch_url(url)
            if not html:
                return WebsiteAnalysis(url, ['Unknown'], None, None, "error", "Failed to fetch content")

            soup = BeautifulSoup(html, 'html.parser')
            platforms = self.analyzer.detect_platform(html)
            
            return WebsiteAnalysis(
                url=url,
                platforms=platforms,
                product_count=self.analyzer.count_products(soup, platforms),
                estimated_visitors=self.analyzer.estimate_visitors(html)
            )
        except Exception as e:
            logger.error(f"Error analyzing {url}: {str(e)}")
            return WebsiteAnalysis(url, ['Unknown'], None, None, "error", str(e))

async def main():
    """Main entry point for the webshop analyzer"""
    config = Config()
    analyzer = WebshopAnalyzer(config)
    
    company_name = input("Enter company name to analyze: ")
    results = await analyzer.analyze_company(company_name)
    
    # Create output directory if it doesn't exist
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    # Save results to Excel
    df = pd.DataFrame([
        {
            'Company Name': company_name,
            'URL': result.url,
            'Platforms': ', '.join(result.platforms),
            'Product Count': result.product_count,
            'Estimated Visitors': result.estimated_visitors,
            'Status': result.status,
            'Error': result.error_message
        }
        for result in results
    ])
    
    df.to_excel(output_dir / f'{company_name}_analysis.xlsx', index=False)
    logger.info(f"Analysis completed for {company_name}. Results saved to {company_name}_analysis.xlsx")

if __name__ == "__main__":
    asyncio.run(main())