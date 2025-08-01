#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Finnhub API Client
Enhanced integration for congressional trading data and market information.
"""

import os
import time
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Iterator, Union
from dataclasses import dataclass, asdict
from urllib.parse import urljoin
import json

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import yaml

logger = logging.getLogger(__name__)

@dataclass
class CongressionalTrade:
    """Data model for congressional trading transactions."""
    symbol: str
    transaction_date: str
    filing_date: str
    representative: str
    transaction_type: str  # Purchase, Sale, Exchange
    amount_min: int
    amount_max: int
    asset_description: str
    asset_type: str
    owner_type: str  # Self, Spouse, Dependent Child
    filing_id: Optional[str] = None

@dataclass
class StockPrice:
    """Data model for stock price information."""
    symbol: str
    date: str
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: Optional[float] = None

@dataclass
class CompanyProfile:
    """Data model for company profile information."""
    symbol: str
    name: str
    industry: str
    sector: str
    market_cap: int
    exchange: str
    ipo: Optional[str]
    website_url: Optional[str]
    description: Optional[str]

class FinnhubAPIClient:
    """
    Enhanced client for accessing Finnhub API data.
    Handles congressional trading data, stock prices, and company information.
    """
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """Initialize Finnhub API client."""
        self.api_key = api_key or os.getenv('FINNHUB_API_KEY')
        
        if not self.api_key:
            raise ValueError("Finnhub API key is required")
        
        self.config = self._load_config(config_path)
        self.session = self._setup_session()
        self.rate_limiter = RateLimiter(
            max_requests=self.config.get('rate_limit', 60),
            time_window=60  # 1 minute
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('api_sources', {}).get('finnhub', {})
        
        return {
            'base_url': self.BASE_URL,
            'rate_limit': 60,
            'timeout': 30,
            'retry_attempts': 3
        }
    
    def _setup_session(self) -> requests.Session:
        """Set up requests session with retry strategy."""
        session = requests.Session()
        
        # Retry strategy
        retry_strategy = Retry(
            total=self.config.get('retry_attempts', 3),
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"],
            backoff_factor=2
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Default headers
        session.headers.update({
            'User-Agent': 'Congressional-Trading-Intelligence-System/1.0',
            'Accept': 'application/json'
        })
        
        return session
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make authenticated request to Finnhub API."""
        self.rate_limiter.wait_if_needed()
        
        url = urljoin(self.config['base_url'], endpoint)
        
        # Add API key to parameters
        if params is None:
            params = {}
        params['token'] = self.api_key
        
        try:
            response = self.session.get(
                url, 
                params=params,
                timeout=self.config.get('timeout', 30)
            )
            response.raise_for_status()
            
            self.rate_limiter.record_request()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {url} - {e}")
            raise
    
    def get_congressional_trading(self, 
                                 symbol: Optional[str] = None,
                                 from_date: Optional[str] = None,
                                 to_date: Optional[str] = None) -> List[CongressionalTrade]:
        """
        Get congressional trading data.
        
        Args:
            symbol: Stock symbol to filter by (optional)
            from_date: Start date in YYYY-MM-DD format (optional)
            to_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            List of CongressionalTrade objects
        """
        logger.info(f"Fetching congressional trading data for symbol: {symbol}")
        
        endpoint = "/stock/congressional-trading"
        params = {}
        
        if symbol:
            params['symbol'] = symbol.upper()
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        try:
            response = self._make_request(endpoint, params)
            
            # Handle different response formats
            trades_data = response if isinstance(response, list) else response.get('data', [])
            
            trades = []
            for trade_data in trades_data:
                trade = self._parse_trade_data(trade_data)
                if trade:
                    trades.append(trade)
            
            logger.info(f"Successfully fetched {len(trades)} congressional trades")
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching congressional trading data: {e}")
            return []
    
    def _parse_trade_data(self, data: Dict[str, Any]) -> Optional[CongressionalTrade]:
        """Parse congressional trading data from API response."""
        try:
            return CongressionalTrade(
                symbol=data.get('symbol', '').upper(),
                transaction_date=data.get('transactionDate', ''),
                filing_date=data.get('filingDate', ''),
                representative=data.get('representative', ''),
                transaction_type=data.get('transactionType', ''),
                amount_min=data.get('minAmount', 0),
                amount_max=data.get('maxAmount', 0),
                asset_description=data.get('assetDescription', ''),
                asset_type=data.get('assetType', 'Stock'),
                owner_type=data.get('ownerType', 'Self'),
                filing_id=data.get('filingId')
            )
        except Exception as e:
            logger.warning(f"Failed to parse trade data: {e}")
            return None
    
    def get_stock_candles(self, 
                         symbol: str,
                         resolution: str = "D",
                         from_timestamp: Optional[int] = None,
                         to_timestamp: Optional[int] = None) -> List[StockPrice]:
        """
        Get stock price data (OHLCV).
        
        Args:
            symbol: Stock symbol
            resolution: Resolution ('1', '5', '15', '30', '60', 'D', 'W', 'M')
            from_timestamp: Start timestamp (Unix)
            to_timestamp: End timestamp (Unix)
            
        Returns:
            List of StockPrice objects
        """
        logger.info(f"Fetching stock candles for {symbol}")
        
        endpoint = "/stock/candle"
        
        # Default to last 365 days if no timestamps provided
        if not from_timestamp or not to_timestamp:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=365)
            from_timestamp = int(from_date.timestamp())
            to_timestamp = int(to_date.timestamp())
        
        params = {
            'symbol': symbol.upper(),
            'resolution': resolution,
            'from': from_timestamp,
            'to': to_timestamp
        }
        
        try:
            response = self._make_request(endpoint, params)
            
            if response.get('s') != 'ok':
                logger.warning(f"No data available for {symbol}")
                return []
            
            # Parse OHLCV data
            timestamps = response.get('t', [])
            opens = response.get('o', [])
            highs = response.get('h', [])
            lows = response.get('l', [])
            closes = response.get('c', [])
            volumes = response.get('v', [])
            
            prices = []
            for i in range(len(timestamps)):
                price = StockPrice(
                    symbol=symbol.upper(),
                    date=datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d'),
                    open_price=opens[i],
                    high_price=highs[i],
                    low_price=lows[i],
                    close_price=closes[i],
                    volume=int(volumes[i])
                )
                prices.append(price)
            
            logger.info(f"Successfully fetched {len(prices)} price records for {symbol}")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching stock candles for {symbol}: {e}")
            return []
    
    def get_company_profile(self, symbol: str) -> Optional[CompanyProfile]:
        """
        Get company profile information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            CompanyProfile object or None
        """
        logger.info(f"Fetching company profile for {symbol}")
        
        endpoint = "/stock/profile2"
        params = {'symbol': symbol.upper()}
        
        try:
            response = self._make_request(endpoint, params)
            
            if not response:
                logger.warning(f"No profile data available for {symbol}")
                return None
            
            return CompanyProfile(
                symbol=symbol.upper(),
                name=response.get('name', ''),
                industry=response.get('finnhubIndustry', ''),
                sector=response.get('ggroup', ''),  # GICS group
                market_cap=response.get('marketCapitalization', 0),
                exchange=response.get('exchange', ''),
                ipo=response.get('ipo'),
                website_url=response.get('weburl'),
                description=response.get('description')
            )
            
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            return None
    
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Quote data dictionary
        """
        endpoint = "/quote"
        params = {'symbol': symbol.upper()}
        
        try:
            response = self._make_request(endpoint, params)
            return response
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def get_insider_transactions(self, symbol: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """
        Get insider trading transactions.
        
        Args:
            symbol: Stock symbol
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            List of insider transaction data
        """
        endpoint = "/stock/insider-transactions"
        params = {
            'symbol': symbol.upper(),
            'from': from_date,
            'to': to_date
        }
        
        try:
            response = self._make_request(endpoint, params)
            return response.get('data', [])
            
        except Exception as e:
            logger.error(f"Error fetching insider transactions for {symbol}: {e}")
            return []
    
    def get_recommendation_trends(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get analyst recommendation trends.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of recommendation trend data
        """
        endpoint = "/stock/recommendation"
        params = {'symbol': symbol.upper()}
        
        try:
            response = self._make_request(endpoint, params)
            return response if isinstance(response, list) else []
            
        except Exception as e:
            logger.error(f"Error fetching recommendation trends for {symbol}: {e}")
            return []
    
    def get_earnings_calendar(self, from_date: str, to_date: str, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get earnings calendar.
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            symbol: Specific symbol (optional)
            
        Returns:
            List of earnings data
        """
        endpoint = "/calendar/earnings"
        params = {
            'from': from_date,
            'to': to_date
        }
        
        if symbol:
            params['symbol'] = symbol.upper()
        
        try:
            response = self._make_request(endpoint, params)
            return response.get('earningsCalendar', [])
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return []
    
    def get_sector_performance(self) -> Dict[str, Any]:
        """Get sector performance data."""
        endpoint = "/stock/sector-performance"
        
        try:
            response = self._make_request(endpoint)
            return response
            
        except Exception as e:
            logger.error(f"Error fetching sector performance: {e}")
            return {}
    
    def get_congressional_trades_by_representative(self, representative: str, 
                                                 from_date: Optional[str] = None,
                                                 to_date: Optional[str] = None) -> List[CongressionalTrade]:
        """
        Get all trades for a specific congressional representative.
        
        Args:
            representative: Name of the representative
            from_date: Start date (YYYY-MM-DD) (optional)
            to_date: End date (YYYY-MM-DD) (optional)
            
        Returns:
            List of CongressionalTrade objects
        """
        # Get all congressional trades and filter by representative
        all_trades = self.get_congressional_trading(from_date=from_date, to_date=to_date)
        
        # Filter by representative name (case-insensitive partial match)
        filtered_trades = [
            trade for trade in all_trades 
            if representative.lower() in trade.representative.lower()
        ]
        
        logger.info(f"Found {len(filtered_trades)} trades for {representative}")
        return filtered_trades
    
    def get_multiple_stock_prices(self, symbols: List[str], 
                                 resolution: str = "D",
                                 days_back: int = 365) -> Dict[str, List[StockPrice]]:
        """
        Get stock price data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            resolution: Price resolution
            days_back: Number of days to fetch
            
        Returns:
            Dictionary mapping symbols to price lists
        """
        logger.info(f"Fetching prices for {len(symbols)} symbols")
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        from_timestamp = int(from_date.timestamp())
        to_timestamp = int(to_date.timestamp())
        
        results = {}
        
        for symbol in symbols:
            try:
                prices = self.get_stock_candles(
                    symbol=symbol,
                    resolution=resolution,
                    from_timestamp=from_timestamp,
                    to_timestamp=to_timestamp
                )
                results[symbol] = prices
                
                # Add small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error fetching prices for {symbol}: {e}")
                results[symbol] = []
        
        return results

class RateLimiter:
    """Rate limiter for Finnhub API (60 requests per minute for free tier)."""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()
        
        # Remove old requests outside the time window
        self.requests = [req_time for req_time in self.requests if now - req_time < self.time_window]
        
        # Check if we're at the limit
        if len(self.requests) >= self.max_requests:
            # Wait until the oldest request is outside the window
            wait_time = self.time_window - (now - self.requests[0]) + 1
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
    
    def record_request(self):
        """Record a new request."""
        self.requests.append(time.time())

def main():
    """Test function for Finnhub API client."""
    logging.basicConfig(level=logging.INFO)
    
    client = FinnhubAPIClient()
    
    # Test congressional trading data
    print("Fetching congressional trading data...")
    trades = client.get_congressional_trading()
    
    if trades:
        print(f"Found {len(trades)} congressional trades")
        for i, trade in enumerate(trades[:5]):  # Show first 5 trades
            print(f"  {trade.representative}: {trade.transaction_type} {trade.symbol} "
                  f"${trade.amount_min:,}-${trade.amount_max:,} on {trade.transaction_date}")
    else:
        print("No congressional trading data found")
    
    # Test stock price data
    print("\nFetching stock price data for AAPL...")
    prices = client.get_stock_candles("AAPL")
    
    if prices:
        print(f"Found {len(prices)} price records")
        recent_price = prices[-1]  # Most recent
        print(f"  Most recent: {recent_price.date} - Close: ${recent_price.close_price:.2f}")
    
    # Test company profile
    print("\nFetching company profile for AAPL...")
    profile = client.get_company_profile("AAPL")
    
    if profile:
        print(f"  Company: {profile.name}")
        print(f"  Sector: {profile.sector}")
        print(f"  Market Cap: ${profile.market_cap:,}M")

if __name__ == "__main__":
    main()