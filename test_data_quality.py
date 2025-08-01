#!/usr/bin/env python3
"""
Phase 1 Data Quality Validation
Comprehensive data quality assessment for Phase 1 implementation.
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

class DataQualityValidator:
    def __init__(self):
        """Initialize data quality validator."""
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
    
    def validate_member_data_quality(self):
        """Validate congressional member data quality."""
        logger.info("Validating member data quality...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            issues = []
            
            # Check total count
            cursor.execute("SELECT COUNT(*) as total FROM members")
            total_members = cursor.fetchone()['total']
            logger.info(f"Total members in database: {total_members}")
            
            if total_members < 10:
                issues.append(f"Low member count: {total_members} (expected 25+)")
            
            # Check required fields
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN bioguide_id IS NULL OR bioguide_id = '' THEN 1 END) as missing_id,
                    COUNT(CASE WHEN full_name IS NULL OR full_name = '' THEN 1 END) as missing_name,
                    COUNT(CASE WHEN party IS NULL OR party = '' THEN 1 END) as missing_party,
                    COUNT(CASE WHEN state IS NULL OR state = '' THEN 1 END) as missing_state,
                    COUNT(CASE WHEN chamber IS NULL OR chamber = '' THEN 1 END) as missing_chamber
                FROM members
            """)
            
            completeness = cursor.fetchone()
            
            for field, count in completeness.items():
                if field.startswith('missing_') and count > 0:
                    field_name = field.replace('missing_', '')
                    issues.append(f"{count} members missing {field_name}")
            
            # Check party distribution
            cursor.execute("""
                SELECT party, COUNT(*) as count
                FROM members
                GROUP BY party
                ORDER BY party
            """)
            
            party_dist = cursor.fetchall()
            logger.info("Party distribution:")
            for dist in party_dist:
                logger.info(f"  {dist['party']}: {dist['count']} members")
            
            # Check chamber distribution
            cursor.execute("""
                SELECT chamber, COUNT(*) as count
                FROM members
                GROUP BY chamber
                ORDER BY chamber
            """)
            
            chamber_dist = cursor.fetchall()
            logger.info("Chamber distribution:")
            for dist in chamber_dist:
                logger.info(f"  {dist['chamber']}: {dist['count']} members")
            
            cursor.close()
            conn.close()
            
            if issues:
                logger.warning("Member data quality issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                return len(issues) <= 2  # Accept minor issues
            else:
                logger.info("✅ Member data quality is good")
                return True
                
        except Exception as e:
            logger.error(f"❌ Member data validation failed: {e}")
            return False
    
    def validate_trading_data_quality(self):
        """Validate trading data quality."""
        logger.info("Validating trading data quality...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            issues = []
            
            # Check total count
            cursor.execute("SELECT COUNT(*) as total FROM trades")
            total_trades = cursor.fetchone()['total']
            logger.info(f"Total trades in database: {total_trades}")
            
            if total_trades < 50:
                issues.append(f"Low trade count: {total_trades} (expected 100+)")
            
            # Check data integrity
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN bioguide_id IS NULL THEN 1 END) as missing_member,
                    COUNT(CASE WHEN symbol IS NULL OR symbol = '' THEN 1 END) as missing_symbol,
                    COUNT(CASE WHEN transaction_date IS NULL THEN 1 END) as missing_trans_date,
                    COUNT(CASE WHEN filing_date IS NULL THEN 1 END) as missing_filing_date,
                    COUNT(CASE WHEN amount_min IS NULL OR amount_max IS NULL THEN 1 END) as missing_amounts
                FROM trades
            """)
            
            integrity = cursor.fetchone()
            
            for field, count in integrity.items():
                if field.startswith('missing_') and count > 0:
                    field_name = field.replace('missing_', '')
                    issues.append(f"{count} trades missing {field_name}")
            
            # Check date consistency
            cursor.execute("""
                SELECT COUNT(*) as invalid_dates
                FROM trades
                WHERE filing_date < transaction_date
            """)
            
            invalid_dates = cursor.fetchone()['invalid_dates']
            if invalid_dates > 0:
                issues.append(f"{invalid_dates} trades with filing date before transaction date")
            
            # Check amount consistency
            cursor.execute("""
                SELECT COUNT(*) as invalid_amounts
                FROM trades
                WHERE amount_min > amount_max
            """)
            
            invalid_amounts = cursor.fetchone()['invalid_amounts']
            if invalid_amounts > 0:
                issues.append(f"{invalid_amounts} trades with invalid amount ranges")
            
            # Check filing delays
            cursor.execute("""
                SELECT 
                    AVG(filing_delay_days) as avg_delay,
                    MAX(filing_delay_days) as max_delay,
                    COUNT(CASE WHEN filing_delay_days > 45 THEN 1 END) as late_filings
                FROM trades
            """)
            
            delays = cursor.fetchone()
            logger.info(f"Filing delays: Avg {delays['avg_delay']:.1f} days, Max {delays['max_delay']} days")
            
            if delays['late_filings'] > total_trades * 0.1:  # More than 10% late
                issues.append(f"{delays['late_filings']} trades filed late (>45 days)")
            
            # Check member-trade relationships
            cursor.execute("""
                SELECT COUNT(*) as orphaned_trades
                FROM trades t
                LEFT JOIN members m ON t.bioguide_id = m.bioguide_id
                WHERE m.bioguide_id IS NULL
            """)
            
            orphaned = cursor.fetchone()['orphaned_trades']
            if orphaned > 0:
                issues.append(f"{orphaned} trades with invalid member references")
            
            cursor.close()
            conn.close()
            
            if issues:
                logger.warning("Trading data quality issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                return len(issues) <= 3  # Accept minor issues for sample data
            else:
                logger.info("✅ Trading data quality is good")
                return True
                
        except Exception as e:
            logger.error(f"❌ Trading data validation failed: {e}")
            return False
    
    def validate_price_data_quality(self):
        """Validate stock price data quality."""
        logger.info("Validating stock price data quality...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            issues = []
            
            # Check total count
            cursor.execute("SELECT COUNT(*) as total FROM stock_prices")
            total_prices = cursor.fetchone()['total']
            logger.info(f"Total price records in database: {total_prices}")
            
            if total_prices < 50:
                issues.append(f"Low price record count: {total_prices} (expected 150+)")
            
            # Check price data completeness
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT symbol) as unique_symbols,
                    MIN(date) as earliest_date,
                    MAX(date) as latest_date,
                    COUNT(CASE WHEN close_price <= 0 THEN 1 END) as invalid_prices
                FROM stock_prices
            """)
            
            price_stats = cursor.fetchone()
            logger.info(f"Price data covers {price_stats['unique_symbols']} symbols")
            logger.info(f"Date range: {price_stats['earliest_date']} to {price_stats['latest_date']}")
            
            if price_stats['invalid_prices'] > 0:
                issues.append(f"{price_stats['invalid_prices']} records with invalid prices (≤0)")
            
            # Check for gaps in data
            cursor.execute("""
                SELECT symbol, COUNT(*) as record_count
                FROM stock_prices
                GROUP BY symbol
                HAVING COUNT(*) < 20
                ORDER BY record_count
            """)
            
            sparse_symbols = cursor.fetchall()
            if sparse_symbols:
                issues.append(f"{len(sparse_symbols)} symbols with limited price data")
            
            cursor.close()
            conn.close()
            
            if issues:
                logger.warning("Price data quality issues:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
                return len(issues) <= 2  # Accept minor issues for sample data
            else:
                logger.info("✅ Price data quality is good")
                return True
                
        except Exception as e:
            logger.error(f"❌ Price data validation failed: {e}")
            return False
    
    def validate_system_performance(self):
        """Validate system performance metrics."""
        logger.info("Validating system performance...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Test query performance
            start_time = datetime.now()
            
            cursor.execute("""
                SELECT 
                    m.full_name,
                    m.party,
                    COUNT(t.id) as trade_count,
                    AVG(t.amount_mid) as avg_amount
                FROM members m
                LEFT JOIN trades t ON m.bioguide_id = t.bioguide_id
                GROUP BY m.bioguide_id, m.full_name, m.party
                ORDER BY trade_count DESC
                LIMIT 10
            """)
            
            results = cursor.fetchall()
            end_time = datetime.now()
            
            query_time = (end_time - start_time).total_seconds()
            
            logger.info(f"Complex query executed in {query_time:.3f}s")
            logger.info(f"Top trading members:")
            for result in results[:5]:
                logger.info(f"  {result['full_name']} ({result['party']}): {result['trade_count']} trades")
            
            cursor.close()
            conn.close()
            
            if query_time < 1.0:
                logger.info("✅ System performance is good")
                return True
            elif query_time < 5.0:
                logger.warning(f"⚠️  System performance is acceptable ({query_time:.3f}s)")
                return True
            else:
                logger.error(f"❌ System performance is poor ({query_time:.3f}s)")
                return False
                
        except Exception as e:
            logger.error(f"❌ System performance validation failed: {e}")
            return False
    
    def generate_data_summary_report(self):
        """Generate comprehensive data summary report."""
        logger.info("Generating data summary report...")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get comprehensive statistics
            cursor.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM members) as total_members,
                    (SELECT COUNT(*) FROM trades) as total_trades,
                    (SELECT COUNT(*) FROM committees) as total_committees,
                    (SELECT COUNT(*) FROM stock_prices) as total_prices,
                    (SELECT COUNT(DISTINCT symbol) FROM trades) as unique_symbols_traded,
                    (SELECT COUNT(DISTINCT symbol) FROM stock_prices) as unique_symbols_priced,
                    (SELECT SUM(amount_mid) FROM trades) as total_trade_volume,
                    (SELECT AVG(filing_delay_days) FROM trades) as avg_filing_delay
            """)
            
            summary = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            logger.info("\n" + "="*60)
            logger.info("PHASE 1 DATA SUMMARY REPORT")
            logger.info("="*60)
            logger.info(f"Congressional Members: {summary['total_members']:,}")
            logger.info(f"Trading Records: {summary['total_trades']:,}")
            logger.info(f"Committees: {summary['total_committees']:,}")
            logger.info(f"Stock Price Records: {summary['total_prices']:,}")
            logger.info(f"Unique Symbols Traded: {summary['unique_symbols_traded']}")
            logger.info(f"Unique Symbols Priced: {summary['unique_symbols_priced']}")
            logger.info(f"Total Trade Volume: ${summary['total_trade_volume']:,.0f}")
            logger.info(f"Average Filing Delay: {summary['avg_filing_delay']:.1f} days")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Data summary generation failed: {e}")
            return False
    
    def run_all_validations(self):
        """Run all data quality validations."""
        logger.info("="*60)
        logger.info("PHASE 1 DATA QUALITY VALIDATION")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        tests = [
            ("Member Data Quality", self.validate_member_data_quality),
            ("Trading Data Quality", self.validate_trading_data_quality),
            ("Price Data Quality", self.validate_price_data_quality),
            ("System Performance", self.validate_system_performance)
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
        
        # Generate summary report
        logger.info(f"\n--- Data Summary Report ---")
        self.generate_data_summary_report()
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info("DATA QUALITY VALIDATION RESULTS")
        logger.info("="*60)
        logger.info(f"Validations passed: {passed_tests}/{total_tests}")
        logger.info(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info(f"Duration: {duration}")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL DATA QUALITY VALIDATIONS PASSED!")
            logger.info("Phase 1 data is ready for production use.")
        elif passed_tests >= 3:
            logger.warning("⚠️  MOST VALIDATIONS PASSED")
            logger.info("Phase 1 data is acceptable with minor issues.")
        else:
            logger.error("❌ MULTIPLE VALIDATION FAILURES")
            logger.info("Phase 1 data needs attention before production use.")
        
        return passed_tests >= 3

def main():
    """Main entry point."""
    validator = DataQualityValidator()
    success = validator.run_all_validations()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()