#!/usr/bin/env python3
"""
Phase 1 Setup Test Script
Tests database connectivity, schema validation, and basic infrastructure.
"""

import os
import sys
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase1Tester:
    def __init__(self):
        """Initialize Phase 1 tester."""
        self.config_path = Path("config/database.yml")
        self.config = self._load_config()
        self.db_config = self.config.get('development', {}).get('database', {})
        
    def _load_config(self):
        """Load database configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def test_database_connection(self):
        """Test PostgreSQL database connection."""
        logger.info("Testing database connection...")
        
        try:
            conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('name', 'congressional_trading_dev'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', 'password')
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            
            logger.info(f"✅ Database connection successful")
            logger.info(f"PostgreSQL version: {version}")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def test_database_schema(self):
        """Test that all required tables exist."""
        logger.info("Testing database schema...")
        
        required_tables = [
            'members', 'trades', 'committees', 'committee_memberships',
            'stock_prices', 'benchmark_prices', 'trade_performance', 
            'bills', 'bill_cosponsors', 'stock_sectors'
        ]
        
        try:
            conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('name', 'congressional_trading_dev'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', 'password')
            )
            
            cursor = conn.cursor()
            
            # Check if tables exist
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            
            existing_tables = [row[0] for row in cursor.fetchall()]
            
            missing_tables = []
            for table in required_tables:
                if table in existing_tables:
                    logger.info(f"✅ Table '{table}' exists")
                else:
                    missing_tables.append(table)
                    logger.error(f"❌ Table '{table}' missing")
            
            if missing_tables:
                logger.error(f"Missing tables: {missing_tables}")
                logger.info("Run: psql -d congressional_trading_dev -f database/schema.sql")
                cursor.close()
                conn.close()
                return False
            
            logger.info("✅ All required tables exist")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Schema validation failed: {e}")
            return False
    
    def test_sample_data_insertion(self):
        """Test inserting sample data to validate schema."""
        logger.info("Testing sample data insertion...")
        
        try:
            conn = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('name', 'congressional_trading_dev'),
                user=self.db_config.get('user', 'postgres'),
                password=self.db_config.get('password', 'password')
            )
            
            cursor = conn.cursor()
            
            # Insert test member
            test_member = {
                'bioguide_id': 'TEST001',
                'first_name': 'Test',
                'last_name': 'Member',
                'full_name': 'Test Member',
                'party': 'D',
                'state': 'CA',
                'district': 1,
                'chamber': 'House',
                'served_from': '2023-01-03'
            }
            
            cursor.execute("""
                INSERT INTO members (bioguide_id, first_name, last_name, full_name, party, state, district, chamber, served_from)
                VALUES (%(bioguide_id)s, %(first_name)s, %(last_name)s, %(full_name)s, %(party)s, %(state)s, %(district)s, %(chamber)s, %(served_from)s)
                ON CONFLICT (bioguide_id) DO NOTHING
            """, test_member)
            
            # Insert test trade
            test_trade = {
                'bioguide_id': 'TEST001',
                'transaction_date': '2024-01-15',
                'filing_date': '2024-01-30',
                'symbol': 'AAPL',
                'transaction_type': 'Purchase',
                'amount_min': 1000,
                'amount_max': 15000,
                'owner_type': 'Self'
            }
            
            cursor.execute("""
                INSERT INTO trades (bioguide_id, transaction_date, filing_date, symbol, transaction_type, amount_min, amount_max, owner_type)
                VALUES (%(bioguide_id)s, %(transaction_date)s, %(filing_date)s, %(symbol)s, %(transaction_type)s, %(amount_min)s, %(amount_max)s, %(owner_type)s)
            """, test_trade)
            
            # Insert test committee
            test_committee = {
                'thomas_id': 'TEST01',
                'name': 'Test Committee',
                'chamber': 'House',
                'committee_type': 'Standing'
            }
            
            cursor.execute("""
                INSERT INTO committees (thomas_id, name, chamber, committee_type)
                VALUES (%(thomas_id)s, %(name)s, %(chamber)s, %(committee_type)s)
                ON CONFLICT (thomas_id) DO NOTHING
            """, test_committee)
            
            conn.commit()
            
            # Verify data was inserted
            cursor.execute("SELECT COUNT(*) FROM members WHERE bioguide_id = 'TEST001'")
            member_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM trades WHERE bioguide_id = 'TEST001'")
            trade_count = cursor.fetchone()[0]
            
            if member_count > 0 and trade_count > 0:
                logger.info("✅ Sample data insertion successful")
                logger.info(f"Test member records: {member_count}")
                logger.info(f"Test trade records: {trade_count}")
                result = True
            else:
                logger.error("❌ Sample data insertion failed")
                result = False
            
            cursor.close()
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ Sample data insertion failed: {e}")
            return False
    
    def test_api_configuration(self):
        """Test API configuration and environment variables.""" 
        logger.info("Testing API configuration...")
        
        api_sources = self.config.get('api_sources', {})
        
        tests_passed = 0
        total_tests = 0
        
        # Test Congress.gov API
        total_tests += 1
        congress_config = api_sources.get('congress_gov', {})
        if congress_config.get('base_url'):
            logger.info("✅ Congress.gov API configuration found")
            tests_passed += 1
        else:
            logger.error("❌ Congress.gov API configuration missing")
        
        # Test ProPublica API
        total_tests += 1
        propublica_config = api_sources.get('propublica', {})
        if propublica_config.get('base_url'):
            logger.info("✅ ProPublica API configuration found")
            tests_passed += 1
        else:
            logger.error("❌ ProPublica API configuration missing")
        
        # Test Finnhub API
        total_tests += 1
        finnhub_config = api_sources.get('finnhub', {})
        if finnhub_config.get('base_url'):
            logger.info("✅ Finnhub API configuration found")
            tests_passed += 1
        else:
            logger.error("❌ Finnhub API configuration missing")
        
        # Check for environment variables (optional for testing)
        env_vars = ['CONGRESS_API_KEY', 'PROPUBLICA_API_KEY', 'FINNHUB_API_KEY']
        env_found = 0
        
        for var in env_vars:
            if os.getenv(var):
                logger.info(f"✅ Environment variable {var} is set")
                env_found += 1
            else:
                logger.warning(f"⚠️  Environment variable {var} not set (optional for testing)")
        
        logger.info(f"API configuration tests: {tests_passed}/{total_tests} passed")
        logger.info(f"Environment variables: {env_found}/{len(env_vars)} set")
        
        return tests_passed >= 2  # At least 2 APIs configured
    
    def test_file_structure(self):
        """Test that all required Phase 1 files exist."""
        logger.info("Testing Phase 1 file structure...")
        
        required_files = [
            'database/schema.sql',
            'config/database.yml',
            'src/data_pipeline/etl_coordinator.py',
            'src/data_sources/congress_gov_client.py',
            'scripts/populate_congressional_database.py'
        ]
        
        missing_files = []
        
        for file_path in required_files:
            if Path(file_path).exists():
                logger.info(f"✅ {file_path} exists")
            else:
                missing_files.append(file_path)
                logger.error(f"❌ {file_path} missing")
        
        if missing_files:
            logger.error(f"Missing files: {missing_files}")
            return False
        
        logger.info("✅ All required Phase 1 files exist")
        return True
    
    def run_all_tests(self):
        """Run all Phase 1 tests."""
        logger.info("="*60)
        logger.info("PHASE 1 INFRASTRUCTURE TESTING")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        tests = [
            ("File Structure", self.test_file_structure),
            ("API Configuration", self.test_api_configuration),
            ("Database Connection", self.test_database_connection),
            ("Database Schema", self.test_database_schema),
            ("Sample Data Insertion", self.test_sample_data_insertion)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n--- {test_name} ---")
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {e}")
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("PHASE 1 TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"Duration: {duration}")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL PHASE 1 TESTS PASSED!")
            logger.info("Phase 1 infrastructure is ready for use.")
        else:
            logger.error("⚠️  SOME TESTS FAILED")
            logger.info("Please address the issues above before proceeding.")
        
        return passed_tests == total_tests

def main():
    """Main entry point."""
    tester = Phase1Tester()
    success = tester.run_all_tests()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()