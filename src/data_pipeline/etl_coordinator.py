#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - ETL Coordinator
Orchestrates data collection, validation, and loading from multiple sources.
"""

import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
import pandas as pd

# Import our API clients
from ..data_sources.congress_gov_client import CongressGovAPIClient, CongressionalMember
from ..data_sources.propublica_client import ProPublicaAPIClient, ProPublicaMember
from ..data_sources.finnhub_client import FinnhubAPIClient, CongressionalTrade

logger = logging.getLogger(__name__)

@dataclass
class ETLJobStatus:
    """Status tracking for ETL jobs."""
    job_id: str
    job_type: str
    status: str  # pending, running, completed, failed
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    records_processed: int
    errors: List[str]
    metadata: Dict[str, Any]

@dataclass
class DataQualityReport:
    """Data quality assessment report."""
    table_name: str
    total_records: int
    duplicate_records: int
    null_critical_fields: int
    accuracy_score: float
    completeness_score: float
    issues: List[str]
    recommendations: List[str]

class ETLCoordinator:
    """
    Coordinates Extract, Transform, Load operations for congressional trading data.
    Manages data collection from multiple APIs and ensures data quality.
    """
    
    def __init__(self, config_path: str = "config/database.yml"):
        """Initialize ETL coordinator with configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Database connection
        self.db_config = self.config.get('development', {}).get('database', {})
        
        # Initialize API clients
        self.congress_client = CongressGovAPIClient(config_path=config_path)
        self.propublica_client = ProPublicaAPIClient(config_path=config_path)
        self.finnhub_client = FinnhubAPIClient(config_path=config_path)
        
        # ETL status tracking
        self.jobs: Dict[str, ETLJobStatus] = {}
        
        # Data quality thresholds
        self.quality_thresholds = self.config.get('data_pipeline', {}).get('quality_thresholds', {
            'accuracy_minimum': 0.995,
            'completeness_minimum': 0.99,
            'freshness_maximum': 86400
        })
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.error(f"Configuration file not found: {self.config_path}")
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
    
    def create_etl_job(self, job_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new ETL job and return job ID."""
        job_id = f"{job_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.jobs[job_id] = ETLJobStatus(
            job_id=job_id,
            job_type=job_type,
            status='pending',
            started_at=None,
            completed_at=None,
            records_processed=0,
            errors=[],
            metadata=metadata or {}
        )
        
        logger.info(f"Created ETL job: {job_id}")
        return job_id
    
    def _update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status and metadata."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            job.status = status
            
            if status == 'running' and not job.started_at:
                job.started_at = datetime.now()
            elif status in ['completed', 'failed']:
                job.completed_at = datetime.now()
            
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            logger.info(f"Job {job_id} status updated: {status}")
    
    def extract_congressional_members(self, congress: int = 118) -> str:
        """
        Extract congressional member data from multiple sources.
        
        Args:
            congress: Congress number to extract data for
            
        Returns:
            Job ID for tracking
        """
        job_id = self.create_etl_job('extract_members', {'congress': congress})
        
        try:
            self._update_job_status(job_id, 'running')
            
            # Extract from Congress.gov
            logger.info("Extracting members from Congress.gov...")
            congress_members = list(self.congress_client.get_all_members(congress))
            
            # Extract from ProPublica
            logger.info("Extracting members from ProPublica...")
            propublica_members = list(self.propublica_client.get_all_members(congress))
            
            # Merge and deduplicate member data
            merged_members = self._merge_member_data(congress_members, propublica_members)
            
            # Load into database
            records_loaded = self._load_members_to_db(merged_members)
            
            self._update_job_status(
                job_id, 'completed',
                records_processed=records_loaded
            )
            
            logger.info(f"Successfully loaded {records_loaded} congressional members")
            
        except Exception as e:
            logger.error(f"Error in extract_congressional_members: {e}")
            self._update_job_status(job_id, 'failed', errors=[str(e)])
        
        return job_id
    
    def _merge_member_data(self, 
                          congress_members: List[CongressionalMember],
                          propublica_members: List[ProPublicaMember]) -> List[Dict[str, Any]]:
        """Merge member data from different sources."""
        # Create lookup dictionary for ProPublica data
        propublica_lookup = {}
        for member in propublica_members:
            # Use various IDs to match members
            if hasattr(member, 'govtrack_id') and member.govtrack_id:
                propublica_lookup[member.govtrack_id] = member
        
        merged_members = []
        
        for congress_member in congress_members:
            # Start with Congress.gov data as base
            member_data = {
                'bioguide_id': congress_member.bioguide_id,
                'first_name': congress_member.first_name,
                'last_name': congress_member.last_name,
                'full_name': congress_member.full_name,
                'party': congress_member.party,
                'state': congress_member.state,
                'district': congress_member.district,
                'chamber': congress_member.chamber,
                'served_from': congress_member.served_from,
                'served_to': congress_member.served_to,
                'birth_date': congress_member.birth_date,
                'official_full_name': congress_member.official_full_name,
                'nickname': congress_member.nickname
            }
            
            # Enhance with ProPublica data if available
            # Note: Would need proper ID matching in production
            for propublica_member in propublica_members:
                if (propublica_member.first_name == congress_member.first_name and
                    propublica_member.last_name == congress_member.last_name and
                    propublica_member.state == congress_member.state):
                    
                    member_data.update({
                        'leadership_position': propublica_member.leadership_role,
                        'net_worth_estimate': None,  # Not available in ProPublica
                        'education': [],  # Not available in ProPublica
                        'occupation': None,  # Not available in ProPublica
                    })
                    break
            
            merged_members.append(member_data)
        
        return merged_members
    
    def _load_members_to_db(self, members: List[Dict[str, Any]]) -> int:
        """Load member data into database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Prepare insert query
            insert_query = """
            INSERT INTO members (
                bioguide_id, first_name, last_name, full_name, party, state, district,
                chamber, served_from, served_to, birth_date, official_full_name, nickname,
                leadership_position, net_worth_estimate, education, occupation
            ) VALUES (
                %(bioguide_id)s, %(first_name)s, %(last_name)s, %(full_name)s, %(party)s,
                %(state)s, %(district)s, %(chamber)s, %(served_from)s, %(served_to)s,
                %(birth_date)s, %(official_full_name)s, %(nickname)s, %(leadership_position)s,
                %(net_worth_estimate)s, %(education)s, %(occupation)s
            ) ON CONFLICT (bioguide_id) DO UPDATE SET
                first_name = EXCLUDED.first_name,
                last_name = EXCLUDED.last_name,
                updated_at = NOW()
            """
            
            # Execute batch insert
            execute_batch(cursor, insert_query, members, page_size=100)
            conn.commit()
            
            logger.info(f"Loaded {len(members)} members to database")
            return len(members)
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error loading members to database: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def extract_congressional_trades(self, 
                                   days_back: int = 90,
                                   specific_symbols: Optional[List[str]] = None) -> str:
        """
        Extract congressional trading data from Finnhub.
        
        Args:
            days_back: Number of days back to extract data
            specific_symbols: Specific symbols to extract (optional)
            
        Returns:
            Job ID for tracking
        """
        job_id = self.create_etl_job('extract_trades', {
            'days_back': days_back,
            'symbols': specific_symbols
        })
        
        try:
            self._update_job_status(job_id, 'running')
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            logger.info(f"Extracting trades from {from_date.date()} to {to_date.date()}")
            
            # Extract trades
            if specific_symbols:
                all_trades = []
                for symbol in specific_symbols:
                    trades = self.finnhub_client.get_congressional_trading(
                        symbol=symbol,
                        from_date=from_date.strftime('%Y-%m-%d'),
                        to_date=to_date.strftime('%Y-%m-%d')
                    )
                    all_trades.extend(trades)
            else:
                all_trades = self.finnhub_client.get_congressional_trading(
                    from_date=from_date.strftime('%Y-%m-%d'),
                    to_date=to_date.strftime('%Y-%m-%d')
                )
            
            # Load trades into database
            records_loaded = self._load_trades_to_db(all_trades)
            
            self._update_job_status(
                job_id, 'completed',
                records_processed=records_loaded
            )
            
            logger.info(f"Successfully loaded {records_loaded} congressional trades")
            
        except Exception as e:
            logger.error(f"Error in extract_congressional_trades: {e}")
            self._update_job_status(job_id, 'failed', errors=[str(e)])
        
        return job_id
    
    def _load_trades_to_db(self, trades: List[CongressionalTrade]) -> int:
        """Load trading data into database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Convert trades to database format
            trade_records = []
            for trade in trades:
                # Map representative name to bioguide_id (simplified for now)
                bioguide_id = self._get_bioguide_id_from_name(trade.representative, cursor)
                
                if bioguide_id:
                    trade_records.append({
                        'bioguide_id': bioguide_id,
                        'transaction_date': trade.transaction_date,
                        'filing_date': trade.filing_date,
                        'symbol': trade.symbol,
                        'transaction_type': trade.transaction_type,
                        'amount_min': trade.amount_min,
                        'amount_max': trade.amount_max,
                        'asset_name': trade.asset_description,
                        'asset_type': trade.asset_type,
                        'owner_type': trade.owner_type,
                        'filing_id': trade.filing_id,
                        'source': 'Finnhub'
                    })
            
            # Insert trades
            insert_query = """
            INSERT INTO trades (
                bioguide_id, transaction_date, filing_date, symbol, transaction_type,
                amount_min, amount_max, asset_name, asset_type, owner_type, filing_id, source
            ) VALUES (
                %(bioguide_id)s, %(transaction_date)s, %(filing_date)s, %(symbol)s,
                %(transaction_type)s, %(amount_min)s, %(amount_max)s, %(asset_name)s,
                %(asset_type)s, %(owner_type)s, %(filing_id)s, %(source)s
            ) ON CONFLICT (filing_id) DO NOTHING
            """
            
            execute_batch(cursor, insert_query, trade_records, page_size=100)
            conn.commit()
            
            logger.info(f"Loaded {len(trade_records)} trades to database")
            return len(trade_records)
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error loading trades to database: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def _get_bioguide_id_from_name(self, representative_name: str, cursor) -> Optional[str]:
        """Map representative name to bioguide_id."""
        # Simple name matching - in production would use fuzzy matching
        cursor.execute("""
            SELECT bioguide_id 
            FROM members 
            WHERE LOWER(full_name) LIKE LOWER(%s)
            LIMIT 1
        """, (f"%{representative_name}%",))
        
        result = cursor.fetchone()
        return result[0] if result else None
    
    def extract_stock_prices(self, symbols: List[str], days_back: int = 365) -> str:
        """
        Extract stock price data for analysis.
        
        Args:
            symbols: List of stock symbols
            days_back: Number of days of historical data
            
        Returns:
            Job ID for tracking
        """
        job_id = self.create_etl_job('extract_prices', {
            'symbols': symbols,
            'days_back': days_back
        })
        
        try:
            self._update_job_status(job_id, 'running')
            
            logger.info(f"Extracting prices for {len(symbols)} symbols")
            
            # Extract prices in batches to respect rate limits
            all_prices = []
            for symbol in symbols:
                try:
                    prices = self.finnhub_client.get_stock_candles(symbol, days_back=days_back)
                    all_prices.extend(prices)
                    logger.debug(f"Extracted {len(prices)} prices for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to extract prices for {symbol}: {e}")
            
            # Load prices into database
            records_loaded = self._load_prices_to_db(all_prices)
            
            self._update_job_status(
                job_id, 'completed',
                records_processed=records_loaded
            )
            
            logger.info(f"Successfully loaded {records_loaded} price records")
            
        except Exception as e:
            logger.error(f"Error in extract_stock_prices: {e}")
            self._update_job_status(job_id, 'failed', errors=[str(e)])
        
        return job_id
    
    def _load_prices_to_db(self, prices: List) -> int:
        """Load stock price data into database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Convert prices to database format
            price_records = []
            for price in prices:
                price_records.append({
                    'symbol': price.symbol,
                    'date': price.date,
                    'open_price': price.open_price,
                    'high_price': price.high_price,
                    'low_price': price.low_price,
                    'close_price': price.close_price,
                    'volume': price.volume,
                    'source': 'Finnhub'
                })
            
            # Insert prices
            insert_query = """
            INSERT INTO stock_prices (
                symbol, date, open_price, high_price, low_price, close_price, volume, source
            ) VALUES (
                %(symbol)s, %(date)s, %(open_price)s, %(high_price)s, %(low_price)s,
                %(close_price)s, %(volume)s, %(source)s
            ) ON CONFLICT (symbol, date) DO UPDATE SET
                close_price = EXCLUDED.close_price,
                volume = EXCLUDED.volume
            """
            
            execute_batch(cursor, insert_query, price_records, page_size=100)
            conn.commit()
            
            return len(price_records)
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error loading prices to database: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def run_data_quality_assessment(self, table_name: str) -> DataQualityReport:
        """Run comprehensive data quality assessment on a table."""
        conn = self._get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        try:
            logger.info(f"Running data quality assessment for {table_name}")
            
            # Get total record count
            cursor.execute(f"SELECT COUNT(*) as total FROM {table_name}")
            total_records = cursor.fetchone()['total']
            
            issues = []
            recommendations = []
            
            # Check for duplicates based on primary key or unique constraints
            duplicate_records = 0
            if table_name == 'members':
                cursor.execute("SELECT COUNT(*) - COUNT(DISTINCT bioguide_id) as duplicates FROM members")
                duplicate_records = cursor.fetchone()['duplicates']
            elif table_name == 'trades':
                cursor.execute("SELECT COUNT(*) - COUNT(DISTINCT filing_id) as duplicates FROM trades WHERE filing_id IS NOT NULL")
                duplicate_records = cursor.fetchone()['duplicates']
            
            # Check for null critical fields
            null_critical_fields = 0
            if table_name == 'members':
                cursor.execute("""
                    SELECT COUNT(*) as nulls FROM members 
                    WHERE bioguide_id IS NULL OR full_name IS NULL OR party IS NULL
                """)
                null_critical_fields = cursor.fetchone()['nulls']
            elif table_name == 'trades':
                cursor.execute("""
                    SELECT COUNT(*) as nulls FROM trades 
                    WHERE bioguide_id IS NULL OR symbol IS NULL OR transaction_date IS NULL
                """)
                null_critical_fields = cursor.fetchone()['nulls']
            
            # Calculate quality scores
            accuracy_score = 1.0 - (duplicate_records / max(total_records, 1))
            completeness_score = 1.0 - (null_critical_fields / max(total_records, 1))
            
            # Generate issues and recommendations
            if duplicate_records > 0:
                issues.append(f"Found {duplicate_records} duplicate records")
                recommendations.append("Implement deduplication process")
            
            if null_critical_fields > 0:
                issues.append(f"Found {null_critical_fields} records with null critical fields")
                recommendations.append("Enhance data validation and cleanup")
            
            if accuracy_score < self.quality_thresholds['accuracy_minimum']:
                issues.append(f"Accuracy score {accuracy_score:.3f} below threshold")
            
            if completeness_score < self.quality_thresholds['completeness_minimum']:
                issues.append(f"Completeness score {completeness_score:.3f} below threshold")
            
            return DataQualityReport(
                table_name=table_name,
                total_records=total_records,
                duplicate_records=duplicate_records,
                null_critical_fields=null_critical_fields,
                accuracy_score=accuracy_score,
                completeness_score=completeness_score,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in data quality assessment: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    def run_full_etl_pipeline(self, congress: int = 118) -> Dict[str, str]:
        """
        Run the complete ETL pipeline.
        
        Args:
            congress: Congress number to process
            
        Returns:
            Dictionary of job IDs for each stage
        """
        logger.info(f"Starting full ETL pipeline for Congress {congress}")
        
        job_ids = {}
        
        # Stage 1: Extract congressional members
        job_ids['members'] = self.extract_congressional_members(congress)
        
        # Wait for members job to complete before proceeding
        while self.jobs[job_ids['members']].status not in ['completed', 'failed']:
            import time
            time.sleep(5)
        
        if self.jobs[job_ids['members']].status == 'failed':
            logger.error("Members extraction failed, aborting pipeline")
            return job_ids
        
        # Stage 2: Extract congressional trades
        job_ids['trades'] = self.extract_congressional_trades(days_back=365)
        
        # Stage 3: Get unique symbols from trades and extract prices
        conn = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM trades LIMIT 100")  # Limit for rate limiting
        symbols = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        
        if symbols:
            job_ids['prices'] = self.extract_stock_prices(symbols)
        
        logger.info(f"ETL pipeline initiated with jobs: {list(job_ids.keys())}")
        return job_ids
    
    def get_job_status(self, job_id: str) -> Optional[ETLJobStatus]:
        """Get status of a specific job."""
        return self.jobs.get(job_id)
    
    def get_all_job_statuses(self) -> Dict[str, ETLJobStatus]:
        """Get status of all jobs."""
        return self.jobs.copy()

def main():
    """Test function for ETL coordinator."""
    logging.basicConfig(level=logging.INFO)
    
    coordinator = ETLCoordinator()
    
    # Test member extraction
    print("Starting member extraction...")
    job_id = coordinator.extract_congressional_members(118)
    
    # Monitor job progress
    import time
    while True:
        status = coordinator.get_job_status(job_id)
        print(f"Job {job_id}: {status.status}")
        
        if status.status in ['completed', 'failed']:
            print(f"Final status: {status.status}")
            if status.errors:
                print(f"Errors: {status.errors}")
            break
        
        time.sleep(2)
    
    # Run data quality assessment
    print("\nRunning data quality assessment...")
    quality_report = coordinator.run_data_quality_assessment('members')
    print(f"Total records: {quality_report.total_records}")
    print(f"Accuracy score: {quality_report.accuracy_score:.3f}")
    print(f"Completeness score: {quality_report.completeness_score:.3f}")
    
    if quality_report.issues:
        print(f"Issues found: {quality_report.issues}")

if __name__ == "__main__":
    main()