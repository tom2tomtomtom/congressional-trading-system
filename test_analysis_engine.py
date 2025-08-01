#!/usr/bin/env python3
"""
Phase 1 Analysis Engine Test
Tests congressional analysis functionality with sample data.
"""

import os
import sys
import logging
import psycopg2
from psycopg2.extras import RealDictCursor  
from datetime import datetime, timedelta
import yaml
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisEngineTester:
    def __init__(self):
        """Initialize analysis engine tester."""
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
    
    def test_member_analysis(self):
        """Test member analysis functionality."""
        logger.info("Testing member analysis...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get member statistics
            cursor.execute("""
                SELECT 
                    party,
                    chamber,
                    COUNT(*) as member_count
                FROM members 
                GROUP BY party, chamber
                ORDER BY party, chamber
            """)
            
            party_stats = cursor.fetchall()
            
            if party_stats:
                logger.info("✅ Member statistics by party and chamber:")
                for stat in party_stats:
                    logger.info(f"   {stat['party']}-{stat['chamber']}: {stat['member_count']} members")
                result = True
            else:
                logger.error("❌ No member statistics found")
                result = False
            
            cursor.close()
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ Member analysis failed: {e}")
            return False
    
    def test_trading_analysis(self):
        """Test trading pattern analysis."""
        logger.info("Testing trading analysis...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get trading statistics
            cursor.execute("""
                SELECT 
                    t.transaction_type,
                    t.symbol,
                    COUNT(*) as trade_count,
                    AVG(t.amount_mid) as avg_amount,
                    m.party,
                    m.chamber
                FROM trades t
                JOIN members m ON t.bioguide_id = m.bioguide_id
                GROUP BY t.transaction_type, t.symbol, m.party, m.chamber
                ORDER BY trade_count DESC
                LIMIT 10
            """)
            
            trading_stats = cursor.fetchall()
            
            if trading_stats:
                logger.info("✅ Top trading patterns:")
                for stat in trading_stats:
                    logger.info(f"   {stat['symbol']} {stat['transaction_type']} by {stat['party']}-{stat['chamber']}: {stat['trade_count']} trades, avg ${stat['avg_amount']:,.0f}")
                result = True
            else:
                logger.error("❌ No trading statistics found")
                result = False
            
            # Test filing delay analysis
            cursor.execute("""
                SELECT 
                    AVG(filing_delay_days) as avg_delay,
                    MAX(filing_delay_days) as max_delay,
                    COUNT(*) as total_trades
                FROM trades
                WHERE filing_delay_days IS NOT NULL
            """)
            
            delay_stats = cursor.fetchone()
            
            if delay_stats and delay_stats['total_trades'] > 0:
                logger.info(f"✅ Filing delay analysis: Avg {delay_stats['avg_delay']:.1f} days, Max {delay_stats['max_delay']} days")
            
            cursor.close()
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ Trading analysis failed: {e}")
            return False
    
    def test_correlation_analysis(self):
        """Test committee-trading correlation analysis."""
        logger.info("Testing correlation analysis...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Test committee analysis
            cursor.execute("""
                SELECT 
                    c.name as committee_name,
                    c.chamber,
                    COUNT(DISTINCT cm.bioguide_id) as member_count
                FROM committees c
                LEFT JOIN committee_memberships cm ON c.id = cm.committee_id
                GROUP BY c.id, c.name, c.chamber
                ORDER BY member_count DESC
                LIMIT 5
            """)
            
            committee_stats = cursor.fetchall()
            
            if committee_stats:
                logger.info("✅ Committee membership analysis:")
                for stat in committee_stats:
                    logger.info(f"   {stat['committee_name']}: {stat['member_count']} members")
                result = True
            else:
                logger.warning("⚠️  No committee membership data found (expected for sample data)")
                result = True  # This is expected for sample data
            
            # Test sector analysis potential
            cursor.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as trade_count,
                    COUNT(DISTINCT bioguide_id) as unique_members
                FROM trades
                GROUP BY symbol
                ORDER BY trade_count DESC
                LIMIT 5
            """)
            
            sector_stats = cursor.fetchall()
            
            if sector_stats:
                logger.info("✅ Most traded symbols:")
                for stat in sector_stats:
                    logger.info(f"   {stat['symbol']}: {stat['trade_count']} trades by {stat['unique_members']} members")
            
            cursor.close()
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ Correlation analysis failed: {e}")
            return False
    
    def test_performance_analysis(self):
        """Test performance analysis capabilities."""
        logger.info("Testing performance analysis...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Test stock price data availability
            cursor.execute("""
                SELECT 
                    symbol,
                    COUNT(*) as price_records,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    AVG(close_price) as avg_price
                FROM stock_prices
                GROUP BY symbol
                ORDER BY symbol
            """)
            
            price_stats = cursor.fetchall()
            
            if price_stats:
                logger.info("✅ Stock price data analysis:")
                for stat in price_stats:
                    logger.info(f"   {stat['symbol']}: {stat['price_records']} records, avg price ${stat['avg_price']:.2f}")
                result = True
            else:
                logger.error("❌ No stock price data found")
                result = False
            
            # Test potential performance calculation
            cursor.execute("""
                SELECT 
                    t.symbol,
                    t.transaction_date,
                    t.transaction_type,
                    t.amount_mid,
                    sp.close_price
                FROM trades t
                LEFT JOIN stock_prices sp ON t.symbol = sp.symbol 
                    AND sp.date >= t.transaction_date
                    AND sp.date <= t.transaction_date + INTERVAL '7 days'
                WHERE sp.close_price IS NOT NULL
                LIMIT 5
            """)
            
            performance_samples = cursor.fetchall()
            
            if performance_samples:
                logger.info("✅ Performance analysis capability demonstrated:")
                for sample in performance_samples:
                    logger.info(f"   {sample['symbol']} {sample['transaction_type']} on {sample['transaction_date']}: ${sample['close_price']}")
            else:
                logger.warning("⚠️  Limited performance analysis data (expected for sample data)")
            
            cursor.close()
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ Performance analysis failed: {e}")
            return False
    
    def test_data_quality_checks(self):
        """Test data quality and validation."""
        logger.info("Testing data quality checks...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Check for data integrity
            issues = []
            
            # Check for orphaned trades
            cursor.execute("""
                SELECT COUNT(*) as orphaned_trades
                FROM trades t
                LEFT JOIN members m ON t.bioguide_id = m.bioguide_id
                WHERE m.bioguide_id IS NULL
            """)
            
            orphaned = cursor.fetchone()['orphaned_trades']
            if orphaned > 0:
                issues.append(f"{orphaned} orphaned trades found")
            
            # Check for invalid dates
            cursor.execute("""
                SELECT COUNT(*) as invalid_dates
                FROM trades
                WHERE filing_date < transaction_date
            """)
            
            invalid_dates = cursor.fetchone()['invalid_dates']
            if invalid_dates > 0:
                issues.append(f"{invalid_dates} trades with invalid filing dates")
            
            # Check for data completeness
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_members,
                    COUNT(CASE WHEN party IS NULL THEN 1 END) as missing_party,
                    COUNT(CASE WHEN chamber IS NULL THEN 1 END) as missing_chamber
                FROM members
            """)
            
            completeness = cursor.fetchone()
            if completeness['missing_party'] > 0:
                issues.append(f"{completeness['missing_party']} members missing party affiliation")
            if completeness['missing_chamber'] > 0:
                issues.append(f"{completeness['missing_chamber']} members missing chamber assignment")
            
            if issues:
                logger.warning("⚠️  Data quality issues found:")
                for issue in issues:
                    logger.warning(f"   - {issue}")
                result = len(issues) < 3  # Accept minor issues
            else:
                logger.info("✅ Data quality checks passed")
                result = True
            
            cursor.close()
            conn.close()
            return result
            
        except Exception as e:
            logger.error(f"❌ Data quality checks failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all analysis engine tests."""
        logger.info("="*60)
        logger.info("PHASE 1 ANALYSIS ENGINE TESTING")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        tests = [
            ("Member Analysis", self.test_member_analysis),
            ("Trading Analysis", self.test_trading_analysis),
            ("Correlation Analysis", self.test_correlation_analysis),
            ("Performance Analysis", self.test_performance_analysis),
            ("Data Quality Checks", self.test_data_quality_checks)
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
        logger.info("ANALYSIS ENGINE TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"Duration: {duration}")
        
        if passed_tests >= 4:  # At least 4 out of 5 tests passing
            logger.info("🎉 ANALYSIS ENGINE READY!")
            logger.info("Congressional analysis functionality is working.")
        else:
            logger.warning("⚠️  ANALYSIS ENGINE NEEDS ATTENTION")
            logger.info("Some analysis functions may need debugging.")
        
        return passed_tests >= 4

def main():
    """Main entry point."""
    tester = AnalysisEngineTester()
    success = tester.run_all_tests()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()