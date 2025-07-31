"""
Congressional Trading Data Collection and API Integration System
Collects data from multiple sources for real-time monitoring
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import sqlite3
from bs4 import BeautifulSoup
import schedule
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """Configuration for API access"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    headers: Optional[Dict] = None

class DataCollector:
    """
    Main data collection system for congressional trading monitoring
    """
    
    def __init__(self, db_path: str = "congressional_data.db"):
        self.db_path = db_path
        self.setup_database()
        
        # API configurations
        self.api_configs = {
            "finnhub": APIConfig(
                name="Finnhub",
                base_url="https://finnhub.io/api/v1",
                api_key=None,  # User needs to provide
                rate_limit=60
            ),
            "fmp": APIConfig(
                name="Financial Modeling Prep",
                base_url="https://financialmodelingprep.com/api/v4",
                api_key=None,  # User needs to provide
                rate_limit=250
            ),
            "quiver": APIConfig(
                name="Quiver Quantitative",
                base_url="https://api.quiverquant.com/beta",
                api_key=None,  # User needs to provide
                rate_limit=300
            ),
            "alpha_vantage": APIConfig(
                name="Alpha Vantage",
                base_url="https://www.alphavantage.co/query",
                api_key=None,  # User needs to provide
                rate_limit=5  # Free tier limit
            )
        }
        
        self.last_request_time = {}
        
    def setup_database(self):
        """Initialize SQLite database for storing collected data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Congressional trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS congressional_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                member_name TEXT NOT NULL,
                stock_symbol TEXT NOT NULL,
                trade_date DATE NOT NULL,
                filing_date DATE NOT NULL,
                trade_type TEXT NOT NULL,
                amount_min REAL NOT NULL,
                amount_max REAL NOT NULL,
                owner_type TEXT,
                sector TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL
            )
        ''')
        
        # Legislative events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS legislative_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_date DATE NOT NULL,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                affected_sectors TEXT,
                affected_stocks TEXT,
                committees_involved TEXT,
                significance_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT NOT NULL
            )
        ''')
        
        # Member information table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS congress_members (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                chamber TEXT NOT NULL,
                party TEXT,
                state TEXT,
                committee_assignments TEXT,
                leadership_position TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                open_price REAL,
                close_price REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def set_api_key(self, service: str, api_key: str):
        """Set API key for a specific service"""
        if service in self.api_configs:
            self.api_configs[service].api_key = api_key
            logger.info(f"API key set for {service}")
        else:
            logger.error(f"Unknown service: {service}")
    
    def _rate_limit_check(self, service: str) -> bool:
        """Check if we can make a request without hitting rate limits"""
        config = self.api_configs.get(service)
        if not config:
            return False
        
        now = time.time()
        last_request = self.last_request_time.get(service, 0)
        
        # Calculate minimum time between requests
        min_interval = 60 / config.rate_limit  # seconds between requests
        
        if now - last_request < min_interval:
            sleep_time = min_interval - (now - last_request)
            logger.info(f"Rate limiting {service}, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time[service] = time.time()
        return True
    
    def collect_congressional_trades_finnhub(self, symbol: str = None) -> List[Dict]:
        """
        Collect congressional trading data from Finnhub API
        """
        if not self.api_configs["finnhub"].api_key:
            logger.warning("Finnhub API key not set, using demo data")
            return self._get_demo_congressional_data()
        
        self._rate_limit_check("finnhub")
        
        try:
            url = f"{self.api_configs['finnhub'].base_url}/stock/insider-transactions"
            params = {
                "token": self.api_configs["finnhub"].api_key,
                "symbol": symbol or "AAPL",  # Default to Apple if no symbol
                "from": (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                "to": datetime.now().strftime("%Y-%m-%d")
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            trades = []
            
            for trade in data.get("data", []):
                # Filter for congressional members (this would need a more sophisticated approach)
                if self._is_congressional_member(trade.get("name", "")):
                    trades.append({
                        "member_name": trade.get("name"),
                        "stock_symbol": symbol,
                        "trade_date": trade.get("transactionDate"),
                        "filing_date": trade.get("filingDate"),
                        "trade_type": "Purchase" if trade.get("transactionCode") == "P" else "Sale",
                        "amount_min": float(trade.get("transactionShares", 0)) * float(trade.get("transactionPrice", 0)),
                        "amount_max": float(trade.get("transactionShares", 0)) * float(trade.get("transactionPrice", 0)),
                        "owner_type": trade.get("ownershipNature", "Direct"),
                        "source": "finnhub"
                    })
            
            logger.info(f"Collected {len(trades)} congressional trades from Finnhub")
            return trades
            
        except requests.RequestException as e:
            logger.error(f"Error collecting data from Finnhub: {e}")
            return []
    
    def collect_congressional_trades_fmp(self) -> List[Dict]:
        """
        Collect congressional trading data from Financial Modeling Prep
        """
        if not self.api_configs["fmp"].api_key:
            logger.warning("FMP API key not set, using demo data")
            return self._get_demo_congressional_data()
        
        self._rate_limit_check("fmp")
        
        try:
            url = f"{self.api_configs['fmp'].base_url}/senate-trading"
            params = {
                "apikey": self.api_configs["fmp"].api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            trades = []
            
            for trade in data:
                trades.append({
                    "member_name": trade.get("representative"),
                    "stock_symbol": trade.get("ticker"),
                    "trade_date": trade.get("transactionDate"),
                    "filing_date": trade.get("dateRecieved"),
                    "trade_type": trade.get("transaction"),
                    "amount_min": self._parse_amount_range(trade.get("amount", ""))[0],
                    "amount_max": self._parse_amount_range(trade.get("amount", ""))[1],
                    "owner_type": trade.get("owner", "Self"),
                    "source": "fmp"
                })
            
            logger.info(f"Collected {len(trades)} congressional trades from FMP")
            return trades
            
        except requests.RequestException as e:
            logger.error(f"Error collecting data from FMP: {e}")
            return []
    
    def collect_market_data(self, symbol: str) -> Dict:
        """
        Collect market data for a specific symbol
        """
        if not self.api_configs["alpha_vantage"].api_key:
            logger.warning("Alpha Vantage API key not set, using demo data")
            return self._get_demo_market_data(symbol)
        
        self._rate_limit_check("alpha_vantage")
        
        try:
            url = self.api_configs["alpha_vantage"].base_url
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "apikey": self.api_configs["alpha_vantage"].api_key
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            time_series = data.get("Time Series (Daily)", {})
            
            if not time_series:
                logger.warning(f"No market data found for {symbol}")
                return {}
            
            # Get most recent data
            latest_date = max(time_series.keys())
            latest_data = time_series[latest_date]
            
            return {
                "symbol": symbol,
                "date": latest_date,
                "open_price": float(latest_data["1. open"]),
                "close_price": float(latest_data["4. close"]),
                "volume": int(latest_data["5. volume"])
            }
            
        except requests.RequestException as e:
            logger.error(f"Error collecting market data for {symbol}: {e}")
            return {}
    
    def scrape_congress_gov_legislation(self) -> List[Dict]:
        """
        Scrape recent legislative activity from Congress.gov
        """
        try:
            # This is a simplified example - real implementation would be more complex
            url = "https://www.congress.gov/search"
            params = {
                "q": "artificial intelligence OR semiconductor OR banking OR energy",
                "source": "legislation"
            }
            
            # Note: This would need proper scraping implementation
            # For now, return demo data
            return self._get_demo_legislative_data()
            
        except Exception as e:
            logger.error(f"Error scraping Congress.gov: {e}")
            return []
    
    def _get_demo_congressional_data(self) -> List[Dict]:
        """
        Generate demo congressional trading data for testing
        """
        demo_trades = [
            {
                "member_name": "Nancy Pelosi",
                "stock_symbol": "NVDA",
                "trade_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                "filing_date": datetime.now().strftime("%Y-%m-%d"),
                "trade_type": "Purchase",
                "amount_min": 1000000,
                "amount_max": 5000000,
                "owner_type": "Spouse",
                "source": "demo"
            },
            {
                "member_name": "Ron Wyden",
                "stock_symbol": "NVDA",
                "trade_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                "filing_date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                "trade_type": "Purchase",
                "amount_min": 250000,
                "amount_max": 500000,
                "owner_type": "Spouse",
                "source": "demo"
            },
            {
                "member_name": "Ro Khanna",
                "stock_symbol": "AAPL",
                "trade_date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                "filing_date": datetime.now().strftime("%Y-%m-%d"),
                "trade_type": "Sale",
                "amount_min": 15000,
                "amount_max": 50000,
                "owner_type": "Self",
                "source": "demo"
            }
        ]
        
        logger.info(f"Generated {len(demo_trades)} demo congressional trades")
        return demo_trades
    
    def _get_demo_legislative_data(self) -> List[Dict]:
        """
        Generate demo legislative data for testing
        """
        demo_events = [
            {
                "event_date": (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
                "event_type": "Committee Hearing",
                "title": "AI Safety and Regulation Hearing",
                "description": "House Oversight Committee hearing on AI safety regulations",
                "affected_sectors": "Technology",
                "affected_stocks": "NVDA,GOOGL,MSFT,AMZN",
                "committees_involved": "House Oversight",
                "significance_score": 8,
                "source": "demo"
            },
            {
                "event_date": (datetime.now() + timedelta(days=14)).strftime("%Y-%m-%d"),
                "event_type": "Bill Introduction",
                "title": "CHIPS Act Extension Bill",
                "description": "Extension of semiconductor manufacturing incentives",
                "affected_sectors": "Technology",
                "affected_stocks": "NVDA,AMD,INTC,TSM",
                "committees_involved": "Finance Committee",
                "significance_score": 9,
                "source": "demo"
            }
        ]
        
        logger.info(f"Generated {len(demo_events)} demo legislative events")
        return demo_events
    
    def _get_demo_market_data(self, symbol: str) -> Dict:
        """
        Generate demo market data for testing
        """
        import random
        
        base_price = {"NVDA": 800, "AAPL": 150, "GOOGL": 140, "MSFT": 300}.get(symbol, 100)
        
        return {
            "symbol": symbol,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "open_price": base_price + random.uniform(-5, 5),
            "close_price": base_price + random.uniform(-10, 10),
            "volume": random.randint(1000000, 10000000)
        }
    
    def _is_congressional_member(self, name: str) -> bool:
        """
        Check if a name belongs to a congressional member
        This would need a comprehensive database in production
        """
        known_members = [
            "Nancy Pelosi", "Ron Wyden", "Ro Khanna", "Josh Gottheimer",
            "Debbie Wasserman Schultz", "Michael McCaul", "Richard Burr"
        ]
        
        return any(member.lower() in name.lower() for member in known_members)
    
    def _parse_amount_range(self, amount_str: str) -> Tuple[float, float]:
        """
        Parse amount range strings like "$1,001 - $15,000"
        """
        if not amount_str:
            return 0.0, 0.0
        
        try:
            # Remove $ and commas, split on -
            clean_str = amount_str.replace("$", "").replace(",", "")
            if " - " in clean_str:
                parts = clean_str.split(" - ")
                return float(parts[0]), float(parts[1])
            else:
                value = float(clean_str)
                return value, value
        except:
            return 0.0, 0.0
    
    def store_trades(self, trades: List[Dict]):
        """Store congressional trades in database"""
        if not trades:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for trade in trades:
            cursor.execute('''
                INSERT OR REPLACE INTO congressional_trades 
                (member_name, stock_symbol, trade_date, filing_date, trade_type, 
                 amount_min, amount_max, owner_type, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade["member_name"],
                trade["stock_symbol"],
                trade["trade_date"],
                trade["filing_date"],
                trade["trade_type"],
                trade["amount_min"],
                trade["amount_max"],
                trade.get("owner_type", "Self"),
                trade["source"]
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(trades)} trades in database")
    
    def store_legislative_events(self, events: List[Dict]):
        """Store legislative events in database"""
        if not events:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for event in events:
            cursor.execute('''
                INSERT OR REPLACE INTO legislative_events 
                (event_date, event_type, title, description, affected_sectors,
                 affected_stocks, committees_involved, significance_score, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event["event_date"],
                event["event_type"],
                event["title"],
                event.get("description", ""),
                event.get("affected_sectors", ""),
                event.get("affected_stocks", ""),
                event.get("committees_involved", ""),
                event.get("significance_score", 5),
                event["source"]
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"Stored {len(events)} legislative events in database")
    
    def get_recent_trades(self, days: int = 30) -> pd.DataFrame:
        """Get recent congressional trades from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM congressional_trades 
            WHERE trade_date >= date('now', '-{} days')
            ORDER BY trade_date DESC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_upcoming_legislation(self, days: int = 30) -> pd.DataFrame:
        """Get upcoming legislative events from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM legislative_events 
            WHERE event_date >= date('now') AND event_date <= date('now', '+{} days')
            ORDER BY event_date ASC
        '''.format(days)
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def run_collection_cycle(self):
        """Run a complete data collection cycle"""
        logger.info("Starting data collection cycle")
        
        # Collect congressional trades
        trades = []
        trades.extend(self.collect_congressional_trades_finnhub())
        trades.extend(self.collect_congressional_trades_fmp())
        
        if trades:
            self.store_trades(trades)
        
        # Collect legislative data
        events = self.scrape_congress_gov_legislation()
        if events:
            self.store_legislative_events(events)
        
        # Collect market data for popular stocks
        popular_stocks = ["NVDA", "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        for symbol in popular_stocks:
            market_data = self.collect_market_data(symbol)
            if market_data:
                # Store market data (implementation would go here)
                pass
        
        logger.info("Data collection cycle completed")

class ScheduledCollector:
    """
    Scheduled data collection service
    """
    
    def __init__(self, collector: DataCollector):
        self.collector = collector
        self.running = False
        
    def start_scheduled_collection(self):
        """Start scheduled data collection"""
        self.running = True
        
        # Schedule collection every hour
        schedule.every().hour.do(self.collector.run_collection_cycle)
        
        # Schedule more frequent collection during market hours
        schedule.every(15).minutes.do(self._market_hours_collection)
        
        # Run scheduler in background thread
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Scheduled data collection started")
    
    def stop_scheduled_collection(self):
        """Stop scheduled data collection"""
        self.running = False
        schedule.clear()
        logger.info("Scheduled data collection stopped")
    
    def _market_hours_collection(self):
        """More frequent collection during market hours"""
        now = datetime.now()
        if 9 <= now.hour <= 16 and now.weekday() < 5:  # Market hours, weekdays
            logger.info("Running market hours collection")
            self.collector.run_collection_cycle()

# Example usage
if __name__ == "__main__":
    # Initialize data collector
    collector = DataCollector()
    
    # Set API keys (users would need to provide these)
    # collector.set_api_key("finnhub", "your_finnhub_key")
    # collector.set_api_key("fmp", "your_fmp_key")
    # collector.set_api_key("alpha_vantage", "your_av_key")
    
    # Run a collection cycle with demo data
    collector.run_collection_cycle()
    
    # Get recent data
    recent_trades = collector.get_recent_trades(30)
    upcoming_events = collector.get_upcoming_legislation(30)
    
    print(f"Collected {len(recent_trades)} recent trades")
    print(f"Found {len(upcoming_events)} upcoming legislative events")
    
    if not recent_trades.empty:
        print("\nRecent Trades:")
        print(recent_trades[['member_name', 'stock_symbol', 'trade_date', 'amount_min', 'amount_max']].head())
    
    if not upcoming_events.empty:
        print("\nUpcoming Events:")
        print(upcoming_events[['event_date', 'title', 'affected_sectors']].head())
    
    # Start scheduled collection (uncomment to run continuously)
    # scheduler = ScheduledCollector(collector)
    # scheduler.start_scheduled_collection()
    
    print("\nData collection system ready!")
    print("To run continuously, uncomment the scheduler lines above.")

