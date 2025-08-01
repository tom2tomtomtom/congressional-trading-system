#!/usr/bin/env python3
"""
Phase 1 Sample Data Population Test
Tests ETL pipeline with sample congressional data.
"""

import os
import sys
import logging
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime, timedelta
import random
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SampleDataPopulator:
    def __init__(self):
        """Initialize sample data populator."""
        self.config_path = Path("config/database.yml")
        self.config = self._load_config()
        self.db_config = self.config.get('development', {}).get('database', {})
        
    def _load_config(self):
        """Load database configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=self.db_config.get('host', 'localhost'),
            port=self.db_config.get('port', 5432),
            database=self.db_config.get('name', 'congressional_trading_dev'),
            user=self.db_config.get('user', 'postgres'),
            password=self.db_config.get('password', 'password')
        )
    
    def generate_sample_members(self, count=25):
        """Generate sample congressional members."""
        logger.info(f"Generating {count} sample congressional members...")
        
        first_names = ['Nancy', 'Kevin', 'Chuck', 'Mitch', 'Alexandria', 'Ted', 'Elizabeth', 'Josh', 
                      'John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Maria', 
                      'James', 'Jennifer', 'William', 'Patricia', 'Richard', 'Linda', 'Thomas', 'Barbara']
        
        last_names = ['Pelosi', 'McCarthy', 'Schumer', 'McConnell', 'Ocasio-Cortez', 'Cruz', 'Warren', 'Hawley',
                     'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 
                     'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson', 'Thomas']
        
        states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI', 'NJ', 'VA', 'WA', 'AZ', 'MA']
        parties = ['D', 'R', 'I']
        chambers = ['House', 'Senate']
        
        members = []
        used_ids = set()
        
        for i in range(count):
            # Generate unique bioguide ID
            bioguide_id = f"M{str(i+1).zfill(6)}"
            while bioguide_id in used_ids:
                bioguide_id = f"M{str(random.randint(1, 999999)).zfill(6)}"
            used_ids.add(bioguide_id)
            
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            party = random.choice(parties)
            state = random.choice(states)
            chamber = random.choice(chambers)
            district = random.randint(1, 20) if chamber == 'House' and random.random() > 0.2 else None
            
            member = {
                'bioguide_id': bioguide_id,
                'first_name': first_name,
                'last_name': last_name,
                'full_name': f"{first_name} {last_name}",
                'party': party,
                'state': state,
                'district': district,
                'chamber': chamber,
                'served_from': '2021-01-03',
                'birth_date': f"19{random.randint(45, 75)}-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
                'occupation': random.choice(['Lawyer', 'Business Owner', 'Teacher', 'Doctor', 'Military', 'Public Service']),
                'net_worth_estimate': f"${random.randint(1, 50)}M - ${random.randint(51, 100)}M" if random.random() > 0.7 else None
            }
            
            members.append(member)
        
        return members
    
    def generate_sample_trades(self, member_ids, count=100):
        """Generate sample trading transactions."""
        logger.info(f"Generating {count} sample trades...")
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V', 
                  'PG', 'UNH', 'MA', 'HD', 'DIS', 'PYPL', 'BAC', 'NFLX', 'KO', 'PFE']
        transaction_types = ['Purchase', 'Sale', 'Exchange']
        owner_types = ['Self', 'Spouse', 'Dependent Child']
        
        trades = []
        
        for i in range(count):
            member_id = random.choice(member_ids)
            transaction_date = datetime.now() - timedelta(days=random.randint(1, 730))  # Last 2 years
            filing_date = transaction_date + timedelta(days=random.randint(1, 45))  # Filing delay
            
            # Generate realistic amount ranges
            amount_min = random.choice([1000, 15000, 50000, 100000, 250000, 500000, 1000000])
            amount_max = amount_min + random.choice([14000, 35000, 50000, 150000, 250000, 500000, 4000000])
            
            trade = {
                'bioguide_id': member_id,
                'transaction_date': transaction_date.date(),
                'filing_date': filing_date.date(),
                'symbol': random.choice(symbols),
                'transaction_type': random.choice(transaction_types),
                'amount_min': amount_min,
                'amount_max': amount_max,
                'asset_name': f"{random.choice(symbols)} Common Stock",
                'owner_type': random.choice(owner_types),
                'filing_id': f"FD_{random.randint(10000, 99999)}_{i}"
            }
            
            trades.append(trade)
        
        return trades
    
    def generate_sample_committees(self):
        """Generate sample committees."""
        logger.info("Generating sample committees...")
        
        committees = [
            {'thomas_id': 'HSAG00', 'name': 'House Committee on Agriculture', 'chamber': 'House', 'committee_type': 'Standing'},
            {'thomas_id': 'HSAP00', 'name': 'House Committee on Appropriations', 'chamber': 'House', 'committee_type': 'Standing'},
            {'thomas_id': 'HSAS00', 'name': 'House Committee on Armed Services', 'chamber': 'House', 'committee_type': 'Standing'},
            {'thomas_id': 'HSBU00', 'name': 'House Committee on the Budget', 'chamber': 'House', 'committee_type': 'Standing'},
            {'thomas_id': 'HSED00', 'name': 'House Committee on Education and Labor', 'chamber': 'House', 'committee_type': 'Standing'},
            {'thomas_id': 'HSSY00', 'name': 'House Committee on Energy and Commerce', 'chamber': 'House', 'committee_type': 'Standing'},
            {'thomas_id': 'HSBA00', 'name': 'House Committee on Financial Services', 'chamber': 'House', 'committee_type': 'Standing'},
            {'thomas_id': 'HSFA00', 'name': 'House Committee on Foreign Affairs', 'chamber': 'House', 'committee_type': 'Standing'},
            {'thomas_id': 'SSAG00', 'name': 'Senate Committee on Agriculture, Nutrition, and Forestry', 'chamber': 'Senate', 'committee_type': 'Standing'},
            {'thomas_id': 'SSAP00', 'name': 'Senate Committee on Appropriations', 'chamber': 'Senate', 'committee_type': 'Standing'},
            {'thomas_id': 'SSAS00', 'name': 'Senate Committee on Armed Services', 'chamber': 'Senate', 'committee_type': 'Standing'},
            {'thomas_id': 'SSBU00', 'name': 'Senate Committee on Banking, Housing, and Urban Affairs', 'chamber': 'Senate', 'committee_type': 'Standing'},
            {'thomas_id': 'SSCM00', 'name': 'Senate Committee on Commerce, Science, and Transportation', 'chamber': 'Senate', 'committee_type': 'Standing'},
            {'thomas_id': 'SSEG00', 'name': 'Senate Committee on Energy and Natural Resources', 'chamber': 'Senate', 'committee_type': 'Standing'},
            {'thomas_id': 'SSFI00', 'name': 'Senate Committee on Finance', 'chamber': 'Senate', 'committee_type': 'Standing'}
        ]
        
        return committees
    
    def populate_sample_data(self):
        """Populate database with sample data."""
        logger.info("Starting sample data population...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Generate sample members
            members = self.generate_sample_members(25)
            member_ids = [m['bioguide_id'] for m in members]
            
            # Insert members
            member_insert_query = """
            INSERT INTO members (bioguide_id, first_name, last_name, full_name, party, state, district, chamber, served_from, birth_date, occupation, net_worth_estimate)
            VALUES (%(bioguide_id)s, %(first_name)s, %(last_name)s, %(full_name)s, %(party)s, %(state)s, %(district)s, %(chamber)s, %(served_from)s, %(birth_date)s, %(occupation)s, %(net_worth_estimate)s)
            ON CONFLICT (bioguide_id) DO UPDATE SET updated_at = NOW()
            """
            
            execute_batch(cursor, member_insert_query, members, page_size=50)
            logger.info(f"✅ Inserted {len(members)} members")
            
            # Generate and insert trades
            trades = self.generate_sample_trades(member_ids, 100)
            
            trade_insert_query = """
            INSERT INTO trades (bioguide_id, transaction_date, filing_date, symbol, transaction_type, amount_min, amount_max, asset_name, owner_type, filing_id)
            VALUES (%(bioguide_id)s, %(transaction_date)s, %(filing_date)s, %(symbol)s, %(transaction_type)s, %(amount_min)s, %(amount_max)s, %(asset_name)s, %(owner_type)s, %(filing_id)s)
            """
            
            execute_batch(cursor, trade_insert_query, trades, page_size=50)
            logger.info(f"✅ Inserted {len(trades)} trades")
            
            # Generate and insert committees
            committees = self.generate_sample_committees()
            
            committee_insert_query = """
            INSERT INTO committees (thomas_id, name, chamber, committee_type)
            VALUES (%(thomas_id)s, %(name)s, %(chamber)s, %(committee_type)s)
            ON CONFLICT (thomas_id) DO UPDATE SET updated_at = NOW()
            """
            
            execute_batch(cursor, committee_insert_query, committees, page_size=50)
            logger.info(f"✅ Inserted {len(committees)} committees")
            
            # Generate sample stock price data
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            stock_prices = []
            
            for symbol in symbols:
                base_price = random.uniform(50, 300)
                for days_back in range(30):  # Last 30 days
                    date = datetime.now() - timedelta(days=days_back)
                    daily_change = random.uniform(-0.05, 0.05)  # ±5% daily change
                    price = base_price * (1 + daily_change)
                    
                    stock_prices.append({
                        'symbol': symbol,
                        'date': date.date(),
                        'close_price': round(price, 2),
                        'volume': random.randint(1000000, 100000000)
                    })
            
            price_insert_query = """
            INSERT INTO stock_prices (symbol, date, close_price, volume)
            VALUES (%(symbol)s, %(date)s, %(close_price)s, %(volume)s)
            ON CONFLICT (symbol, date) DO NOTHING
            """
            
            execute_batch(cursor, price_insert_query, stock_prices, page_size=50)
            logger.info(f"✅ Inserted {len(stock_prices)} stock price records")
            
            conn.commit()
            
            # Get final statistics
            cursor.execute("SELECT COUNT(*) FROM members")
            member_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades")
            trade_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM committees")
            committee_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM stock_prices")
            price_count = cursor.fetchone()[0]
            
            logger.info("="*60)
            logger.info("SAMPLE DATA POPULATION COMPLETED!")
            logger.info("="*60)
            logger.info(f"Members: {member_count}")
            logger.info(f"Trades: {trade_count}")
            logger.info(f"Committees: {committee_count}")
            logger.info(f"Stock Prices: {price_count}")
            
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Sample data population failed: {e}")
            return False

def main():
    """Main entry point."""
    populator = SampleDataPopulator()
    success = populator.populate_sample_data()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()