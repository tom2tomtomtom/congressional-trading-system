#!/usr/bin/env python3
"""
Market Data Service - Real-Time and Historical Price Data
Integrates with Yahoo Finance (yfinance) for free, reliable market data.

Features:
- Historical price data for trade enrichment
- Real-time quotes for current valuations
- Sector and company classification
- Caching layer for performance
- Return calculations for trades
"""

import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import json
import time
import sqlite3
from pathlib import Path
from functools import lru_cache
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# yfinance is the primary data source (free, no API key)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logging.warning("yfinance not installed. Run: pip install yfinance")

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class Sector(Enum):
    """Market sector classifications"""
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCIAL = "Financial"
    CONSUMER_CYCLICAL = "Consumer Cyclical"
    CONSUMER_DEFENSIVE = "Consumer Defensive"
    INDUSTRIALS = "Industrials"
    ENERGY = "Energy"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    BASIC_MATERIALS = "Basic Materials"
    COMMUNICATION_SERVICES = "Communication Services"
    UNKNOWN = "Unknown"


@dataclass
class StockPrice:
    """Daily stock price data"""
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'date': str(self.date),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'adjusted_close': self.adjusted_close,
            'volume': self.volume
        }


@dataclass
class StockQuote:
    """Real-time stock quote"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[int] = None
    pe_ratio: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CompanyInfo:
    """Company profile information"""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: int
    exchange: str
    country: str
    website: str = ""
    description: str = ""
    employees: int = 0
    ceo: str = ""
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    beta: Optional[float] = None
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()


@dataclass
class TradeEnrichment:
    """Enrichment data for a trade"""
    symbol: str
    trade_date: date
    filing_date: Optional[date]

    # Prices at key dates
    price_at_trade: Optional[float] = None
    price_at_filing: Optional[float] = None
    price_current: Optional[float] = None

    # Returns
    return_trade_to_filing: Optional[float] = None  # % return from trade to filing
    return_trade_to_current: Optional[float] = None  # % return from trade to now
    return_1_day: Optional[float] = None  # 1-day return after trade
    return_1_week: Optional[float] = None  # 1-week return after trade
    return_1_month: Optional[float] = None  # 1-month return after trade

    # Company info
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: int = 0

    # Metadata
    enriched_at: str = ""
    data_source: str = "yfinance"


class MarketDataCache:
    """
    SQLite-based caching layer for market data.
    Reduces API calls and improves response times.
    """

    def __init__(self, cache_dir: Path, max_age_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_age_hours = max_age_hours
        self.db_path = cache_dir / "market_data_cache.db"
        self._init_db()
        self._lock = threading.Lock()

    def _init_db(self):
        """Initialize cache database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Price history cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_cache (
                symbol TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adjusted_close REAL,
                volume INTEGER,
                cached_at TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Quote cache (short TTL)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quote_cache (
                symbol TEXT PRIMARY KEY,
                price REAL,
                change REAL,
                change_percent REAL,
                volume INTEGER,
                market_cap INTEGER,
                cached_at TEXT
            )
        """)

        # Company info cache (long TTL)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS company_cache (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap INTEGER,
                exchange TEXT,
                country TEXT,
                website TEXT,
                description TEXT,
                cached_at TEXT
            )
        """)

        # Enrichment cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS enrichment_cache (
                cache_key TEXT PRIMARY KEY,
                data TEXT,
                cached_at TEXT
            )
        """)

        conn.commit()
        conn.close()

    def get_prices(self, symbol: str, start_date: date, end_date: date) -> List[StockPrice]:
        """Get cached prices for a symbol and date range"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT symbol, date, open, high, low, close, adjusted_close, volume
                FROM price_cache
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
            """, (symbol, str(start_date), str(end_date)))

            prices = []
            for row in cursor.fetchall():
                prices.append(StockPrice(
                    symbol=row[0],
                    date=datetime.strptime(row[1], '%Y-%m-%d').date(),
                    open=row[2],
                    high=row[3],
                    low=row[4],
                    close=row[5],
                    adjusted_close=row[6],
                    volume=row[7]
                ))

            conn.close()
            return prices

    def set_prices(self, prices: List[StockPrice]):
        """Cache price data"""
        if not prices:
            return

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            for price in prices:
                cursor.execute("""
                    INSERT OR REPLACE INTO price_cache
                    (symbol, date, open, high, low, close, adjusted_close, volume, cached_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    price.symbol, str(price.date), price.open, price.high,
                    price.low, price.close, price.adjusted_close, price.volume, now
                ))

            conn.commit()
            conn.close()

    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """Get cached quote (valid for 15 minutes)"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT price, change, change_percent, volume, market_cap, cached_at
                FROM quote_cache
                WHERE symbol = ?
            """, (symbol,))

            row = cursor.fetchone()
            conn.close()

            if row:
                cached_at = datetime.fromisoformat(row[5])
                if datetime.now() - cached_at < timedelta(minutes=15):
                    return StockQuote(
                        symbol=symbol,
                        price=row[0],
                        change=row[1],
                        change_percent=row[2],
                        volume=row[3],
                        market_cap=row[4],
                        timestamp=row[5]
                    )

            return None

    def set_quote(self, quote: StockQuote):
        """Cache quote data"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO quote_cache
                (symbol, price, change, change_percent, volume, market_cap, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                quote.symbol, quote.price, quote.change, quote.change_percent,
                quote.volume, quote.market_cap, datetime.now().isoformat()
            ))

            conn.commit()
            conn.close()

    def get_company(self, symbol: str) -> Optional[CompanyInfo]:
        """Get cached company info (valid for 7 days)"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT name, sector, industry, market_cap, exchange, country, website, description, cached_at
                FROM company_cache
                WHERE symbol = ?
            """, (symbol,))

            row = cursor.fetchone()
            conn.close()

            if row:
                cached_at = datetime.fromisoformat(row[8])
                if datetime.now() - cached_at < timedelta(days=7):
                    return CompanyInfo(
                        symbol=symbol,
                        name=row[0],
                        sector=row[1],
                        industry=row[2],
                        market_cap=row[3],
                        exchange=row[4],
                        country=row[5],
                        website=row[6],
                        description=row[7],
                        last_updated=row[8]
                    )

            return None

    def set_company(self, info: CompanyInfo):
        """Cache company info"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO company_cache
                (symbol, name, sector, industry, market_cap, exchange, country, website, description, cached_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                info.symbol, info.name, info.sector, info.industry, info.market_cap,
                info.exchange, info.country, info.website, info.description,
                datetime.now().isoformat()
            ))

            conn.commit()
            conn.close()

    def clear_old_cache(self, days: int = 30):
        """Clear cache entries older than specified days"""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cutoff = (datetime.now() - timedelta(days=days)).isoformat()

            cursor.execute("DELETE FROM price_cache WHERE cached_at < ?", (cutoff,))
            cursor.execute("DELETE FROM quote_cache WHERE cached_at < ?", (cutoff,))
            cursor.execute("DELETE FROM enrichment_cache WHERE cached_at < ?", (cutoff,))

            conn.commit()
            conn.close()


class MarketDataService:
    """
    Main market data service for fetching and caching stock information.

    Uses yfinance (Yahoo Finance) for free, reliable market data.
    Includes caching layer to minimize API calls.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".congressional_trading_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize cache
        self.cache = MarketDataCache(self.cache_dir)

        # Rate limiting
        self._last_request = 0
        self._min_request_interval = 0.1  # 100ms between requests
        self._request_lock = threading.Lock()

        # Stats
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls': 0,
            'errors': 0
        }

        if not YFINANCE_AVAILABLE:
            logger.error("yfinance is not installed. Market data will not be available.")

    def _rate_limit(self):
        """Enforce rate limiting"""
        with self._request_lock:
            now = time.time()
            elapsed = now - self._last_request
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
            self._last_request = time.time()

    def get_price_history(self, symbol: str, start_date: date, end_date: date) -> List[StockPrice]:
        """
        Get historical price data for a symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for history
            end_date: End date for history

        Returns:
            List of StockPrice objects
        """
        # Check cache first
        cached_prices = self.cache.get_prices(symbol, start_date, end_date)
        if cached_prices:
            # Check if we have complete data
            expected_days = (end_date - start_date).days
            if len(cached_prices) >= expected_days * 0.6:  # Allow for weekends/holidays
                self._stats['cache_hits'] += 1
                return cached_prices

        self._stats['cache_misses'] += 1

        if not YFINANCE_AVAILABLE:
            return cached_prices

        try:
            self._rate_limit()
            self._stats['api_calls'] += 1

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date + timedelta(days=1))

            if df.empty:
                logger.warning(f"No price data found for {symbol}")
                return cached_prices

            prices = []
            for idx, row in df.iterrows():
                price_date = idx.date() if hasattr(idx, 'date') else idx
                prices.append(StockPrice(
                    symbol=symbol,
                    date=price_date,
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    adjusted_close=float(row['Close']),  # yfinance returns adjusted
                    volume=int(row['Volume'])
                ))

            # Cache the results
            self.cache.set_prices(prices)

            return prices

        except Exception as e:
            logger.error(f"Error fetching price history for {symbol}: {e}")
            self._stats['errors'] += 1
            return cached_prices

    def get_price_on_date(self, symbol: str, target_date: date) -> Optional[float]:
        """
        Get closing price for a specific date.
        Returns the closest available price if market was closed.

        Args:
            symbol: Stock ticker symbol
            target_date: Target date for price

        Returns:
            Closing price or None if not found
        """
        # Get a range around the target date to handle weekends/holidays
        start = target_date - timedelta(days=5)
        end = target_date + timedelta(days=1)

        prices = self.get_price_history(symbol, start, end)

        if not prices:
            return None

        # Find closest price on or before target date
        valid_prices = [p for p in prices if p.date <= target_date]
        if valid_prices:
            return valid_prices[-1].close

        # If no price before target, use first available
        return prices[0].close if prices else None

    def get_current_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get current real-time quote for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            StockQuote object or None if not found
        """
        # Check cache first
        cached_quote = self.cache.get_quote(symbol)
        if cached_quote:
            self._stats['cache_hits'] += 1
            return cached_quote

        self._stats['cache_misses'] += 1

        if not YFINANCE_AVAILABLE:
            return None

        try:
            self._rate_limit()
            self._stats['api_calls'] += 1

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or 'currentPrice' not in info and 'regularMarketPrice' not in info:
                # Try getting fast info
                fast_info = ticker.fast_info
                if hasattr(fast_info, 'last_price'):
                    quote = StockQuote(
                        symbol=symbol,
                        price=float(fast_info.last_price),
                        change=0,
                        change_percent=0,
                        volume=int(fast_info.last_volume) if hasattr(fast_info, 'last_volume') else 0,
                        market_cap=int(fast_info.market_cap) if hasattr(fast_info, 'market_cap') else None
                    )
                    self.cache.set_quote(quote)
                    return quote
                return None

            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            previous_close = info.get('previousClose') or info.get('regularMarketPreviousClose', current_price)

            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close else 0

            quote = StockQuote(
                symbol=symbol,
                price=float(current_price),
                change=float(change),
                change_percent=float(change_percent),
                volume=int(info.get('volume', 0) or info.get('regularMarketVolume', 0)),
                market_cap=int(info.get('marketCap', 0)) if info.get('marketCap') else None,
                pe_ratio=float(info.get('trailingPE', 0)) if info.get('trailingPE') else None,
                fifty_two_week_high=float(info.get('fiftyTwoWeekHigh', 0)) if info.get('fiftyTwoWeekHigh') else None,
                fifty_two_week_low=float(info.get('fiftyTwoWeekLow', 0)) if info.get('fiftyTwoWeekLow') else None
            )

            self.cache.set_quote(quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            self._stats['errors'] += 1
            return None

    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        """
        Get company profile information.

        Args:
            symbol: Stock ticker symbol

        Returns:
            CompanyInfo object or None if not found
        """
        # Check cache first
        cached_info = self.cache.get_company(symbol)
        if cached_info:
            self._stats['cache_hits'] += 1
            return cached_info

        self._stats['cache_misses'] += 1

        if not YFINANCE_AVAILABLE:
            return None

        try:
            self._rate_limit()
            self._stats['api_calls'] += 1

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or 'shortName' not in info:
                return None

            company = CompanyInfo(
                symbol=symbol,
                name=info.get('shortName', '') or info.get('longName', ''),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=int(info.get('marketCap', 0)) if info.get('marketCap') else 0,
                exchange=info.get('exchange', ''),
                country=info.get('country', ''),
                website=info.get('website', ''),
                description=info.get('longBusinessSummary', ''),
                employees=int(info.get('fullTimeEmployees', 0)) if info.get('fullTimeEmployees') else 0,
                ceo=info.get('companyOfficers', [{}])[0].get('name', '') if info.get('companyOfficers') else '',
                dividend_yield=float(info.get('dividendYield', 0)) if info.get('dividendYield') else None,
                pe_ratio=float(info.get('trailingPE', 0)) if info.get('trailingPE') else None,
                beta=float(info.get('beta', 0)) if info.get('beta') else None
            )

            self.cache.set_company(company)
            return company

        except Exception as e:
            logger.error(f"Error fetching company info for {symbol}: {e}")
            self._stats['errors'] += 1
            return None

    def enrich_trade(self, symbol: str, trade_date: date,
                     filing_date: Optional[date] = None) -> TradeEnrichment:
        """
        Enrich a trade with market data.

        Args:
            symbol: Stock ticker symbol
            trade_date: Date of the trade
            filing_date: Date the trade was filed (optional)

        Returns:
            TradeEnrichment object with price and return data
        """
        enrichment = TradeEnrichment(
            symbol=symbol,
            trade_date=trade_date,
            filing_date=filing_date,
            enriched_at=datetime.now().isoformat()
        )

        today = date.today()

        # Get price at trade date
        enrichment.price_at_trade = self.get_price_on_date(symbol, trade_date)

        # Get price at filing date
        if filing_date:
            enrichment.price_at_filing = self.get_price_on_date(symbol, filing_date)

        # Get current price
        quote = self.get_current_quote(symbol)
        if quote:
            enrichment.price_current = quote.price

        # Calculate returns
        if enrichment.price_at_trade and enrichment.price_at_trade > 0:
            if enrichment.price_at_filing:
                enrichment.return_trade_to_filing = (
                    (enrichment.price_at_filing - enrichment.price_at_trade) /
                    enrichment.price_at_trade * 100
                )

            if enrichment.price_current:
                enrichment.return_trade_to_current = (
                    (enrichment.price_current - enrichment.price_at_trade) /
                    enrichment.price_at_trade * 100
                )

            # Calculate period returns
            # 1-day return
            day_after = trade_date + timedelta(days=1)
            price_1d = self.get_price_on_date(symbol, day_after)
            if price_1d:
                enrichment.return_1_day = (price_1d - enrichment.price_at_trade) / enrichment.price_at_trade * 100

            # 1-week return
            week_after = trade_date + timedelta(days=7)
            if week_after <= today:
                price_1w = self.get_price_on_date(symbol, week_after)
                if price_1w:
                    enrichment.return_1_week = (price_1w - enrichment.price_at_trade) / enrichment.price_at_trade * 100

            # 1-month return
            month_after = trade_date + timedelta(days=30)
            if month_after <= today:
                price_1m = self.get_price_on_date(symbol, month_after)
                if price_1m:
                    enrichment.return_1_month = (price_1m - enrichment.price_at_trade) / enrichment.price_at_trade * 100

        # Get company info
        company = self.get_company_info(symbol)
        if company:
            enrichment.company_name = company.name
            enrichment.sector = company.sector
            enrichment.industry = company.industry
            enrichment.market_cap = company.market_cap

        return enrichment

    def enrich_trades_batch(self, trades: List[Dict[str, Any]],
                           max_workers: int = 5) -> List[TradeEnrichment]:
        """
        Enrich multiple trades in parallel.

        Args:
            trades: List of trade dicts with 'symbol', 'trade_date', and optionally 'filing_date'
            max_workers: Maximum parallel workers

        Returns:
            List of TradeEnrichment objects
        """
        enrichments = []

        def enrich_single(trade: Dict) -> TradeEnrichment:
            symbol = trade.get('symbol', '')
            trade_date = trade.get('trade_date') or trade.get('transaction_date')
            filing_date = trade.get('filing_date')

            # Parse dates if strings
            if isinstance(trade_date, str):
                trade_date = datetime.strptime(trade_date, '%Y-%m-%d').date()
            if isinstance(filing_date, str):
                filing_date = datetime.strptime(filing_date, '%Y-%m-%d').date()

            return self.enrich_trade(symbol, trade_date, filing_date)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(enrich_single, trade): trade for trade in trades}

            for future in as_completed(futures):
                try:
                    enrichment = future.result()
                    enrichments.append(enrichment)
                except Exception as e:
                    logger.error(f"Error enriching trade: {e}")

        return enrichments

    def get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector classification for a symbol"""
        info = self.get_company_info(symbol)
        return info.sector if info else "Unknown"

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """Get quotes for multiple symbols"""
        quotes = {}

        for symbol in symbols:
            quote = self.get_current_quote(symbol)
            if quote:
                quotes[symbol] = quote

        return quotes

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self._stats,
            'cache_location': str(self.cache_dir),
            'yfinance_available': YFINANCE_AVAILABLE
        }


def main():
    """Test the market data service"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    service = MarketDataService()

    print("Market Data Service Test")
    print("=" * 60)

    # Test quote
    print("\nFetching AAPL quote...")
    quote = service.get_current_quote("AAPL")
    if quote:
        print(f"  Price: ${quote.price:.2f}")
        print(f"  Change: {quote.change_percent:+.2f}%")
        print(f"  Volume: {quote.volume:,}")

    # Test company info
    print("\nFetching NVDA company info...")
    company = service.get_company_info("NVDA")
    if company:
        print(f"  Name: {company.name}")
        print(f"  Sector: {company.sector}")
        print(f"  Industry: {company.industry}")
        print(f"  Market Cap: ${company.market_cap:,}")

    # Test price history
    print("\nFetching MSFT price history (last 30 days)...")
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    prices = service.get_price_history("MSFT", start_date, end_date)
    if prices:
        print(f"  Got {len(prices)} price records")
        print(f"  Latest: {prices[-1].date} - ${prices[-1].close:.2f}")

    # Test trade enrichment
    print("\nEnriching sample trade...")
    trade_date = date.today() - timedelta(days=60)
    filing_date = trade_date + timedelta(days=30)
    enrichment = service.enrich_trade("GOOGL", trade_date, filing_date)

    print(f"  Symbol: {enrichment.symbol}")
    print(f"  Trade Date: {enrichment.trade_date}")
    print(f"  Price at Trade: ${enrichment.price_at_trade:.2f}" if enrichment.price_at_trade else "  Price at Trade: N/A")
    print(f"  Price Current: ${enrichment.price_current:.2f}" if enrichment.price_current else "  Price Current: N/A")
    print(f"  Return to Current: {enrichment.return_trade_to_current:+.2f}%" if enrichment.return_trade_to_current else "  Return: N/A")
    print(f"  Sector: {enrichment.sector}")

    # Show stats
    print("\nService Stats:")
    stats = service.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
