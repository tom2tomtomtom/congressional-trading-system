#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Database Setup Script
Creates and initializes PostgreSQL database with proper schema and validation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Handles Congressional Trading Intelligence System database setup."""
    
    def __init__(self, config_path: str = "config/database.yml"):
        """Initialize database setup with configuration."""
        self.config_path = Path(config_path)
        self.database_dir = Path(__file__).parent
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load database configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'database': {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': int(os.getenv('DB_PORT', '5432')),
                    'name': os.getenv('DB_NAME', 'congressional_trading'),
                    'user': os.getenv('DB_USER', 'postgres'),
                    'password': os.getenv('DB_PASSWORD', 'password'),
                    'admin_db': os.getenv('ADMIN_DB', 'postgres')
                }
            }
    
    def _get_admin_connection(self):
        """Get connection to admin database for database creation."""
        config = self.config['database']
        return psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['admin_db'],
            user=config['user'],
            password=config['password']
        )
    
    def _get_app_connection(self):
        """Get connection to application database."""
        config = self.config['database']
        return psycopg2.connect(
            host=config['host'],
            port=config['port'],
            database=config['name'],
            user=config['user'],
            password=config['password']
        )
    
    def create_database(self) -> bool:
        """Create the congressional trading database if it doesn't exist."""
        config = self.config['database']
        db_name = config['name']
        
        try:
            # Connect to admin database
            conn = self._get_admin_connection()
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
                (db_name,)
            )
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"Creating database: {db_name}")
                cursor.execute(f'CREATE DATABASE "{db_name}"')
                logger.info(f"Database {db_name} created successfully")
            else:
                logger.info(f"Database {db_name} already exists")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error creating database: {e}")
            return False
    
    def run_migration(self, migration_file: Path) -> bool:
        """Run a specific migration file."""
        try:
            conn = self._get_app_connection()
            cursor = conn.cursor()
            
            logger.info(f"Running migration: {migration_file.name}")
            
            # Read and execute migration file
            with open(migration_file, 'r') as f:
                migration_sql = f.read()
            
            cursor.execute(migration_sql)
            conn.commit()
            
            logger.info(f"Migration {migration_file.name} completed successfully")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error running migration {migration_file.name}: {e}")
            return False
    
    def run_all_migrations(self) -> bool:
        """Run all migration files in order."""
        migrations_dir = self.database_dir / "migrations"
        
        if not migrations_dir.exists():
            logger.error("Migrations directory not found")
            return False
        
        # Get all migration files and sort them
        migration_files = sorted(migrations_dir.glob("*.sql"))
        
        if not migration_files:
            logger.warning("No migration files found")
            return True
        
        success = True
        for migration_file in migration_files:
            if not self.run_migration(migration_file):
                success = False
                break
        
        return success
    
    def validate_schema(self) -> bool:
        """Validate that the database schema is properly set up."""
        try:
            conn = self._get_app_connection()
            cursor = conn.cursor()
            
            # Check for required tables
            required_tables = [
                'members', 'trades', 'committees', 'committee_memberships',
                'stock_prices', 'benchmark_prices', 'trade_performance',
                'bills', 'bill_cosponsors', 'stock_sectors', 'committee_sectors',
                'data_quality_metrics', 'api_calls'
            ]
            
            for table in required_tables:
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = %s
                    );
                """, (table,))
                
                if not cursor.fetchone()[0]:
                    logger.error(f"Required table '{table}' not found")
                    return False
            
            logger.info("Schema validation passed - all required tables exist")
            
            # Check for required extensions
            cursor.execute("SELECT extname FROM pg_extension WHERE extname IN ('uuid-ossp', 'pg_trgm');")
            extensions = [row[0] for row in cursor.fetchall()]
            
            required_extensions = ['uuid-ossp', 'pg_trgm']
            for ext in required_extensions:
                if ext not in extensions:
                    logger.error(f"Required extension '{ext}' not installed")
                    return False
            
            logger.info("Extensions validation passed")
            
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error validating schema: {e}")
            return False
    
    def insert_initial_data(self) -> bool:
        """Insert initial configuration and reference data."""
        try:
            conn = self._get_app_connection()
            cursor = conn.cursor()
            
            # Insert benchmark ETF data if not exists
            cursor.execute("SELECT COUNT(*) FROM stock_sectors WHERE sector = 'Benchmark'")
            benchmark_count = cursor.fetchone()[0]
            
            if benchmark_count == 0:
                logger.info("Inserting benchmark ETF data")
                benchmark_data = [
                    ('SPY', 'Benchmark', 'S&P 500 ETF', 'Large', 'NYSE'),
                    ('QQQ', 'Benchmark', 'NASDAQ 100 ETF', 'Large', 'NASDAQ'),
                    ('IWM', 'Benchmark', 'Russell 2000 ETF', 'Small', 'NYSE'),
                    ('VTI', 'Benchmark', 'Total Stock Market ETF', 'Large', 'NYSE')
                ]
                
                cursor.executemany("""
                    INSERT INTO stock_sectors (symbol, sector, industry, market_cap, exchange) 
                    VALUES (%s, %s, %s, %s, %s)
                """, benchmark_data)
                
                logger.info(f"Inserted {len(benchmark_data)} benchmark securities")
            
            # Update data quality metrics
            cursor.execute("""
                UPDATE data_quality_metrics 
                SET checked_at = NOW() 
                WHERE table_name IN ('members', 'trades', 'committees')
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("Initial data insertion completed")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting initial data: {e}")
            return False
    
    def create_app_user(self, username: str = "app_user") -> bool:
        """Create application user with appropriate permissions."""
        try:
            conn = self._get_admin_connection()
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if user exists
            cursor.execute(
                "SELECT 1 FROM pg_catalog.pg_user WHERE usename = %s",
                (username,)
            )
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"Creating application user: {username}")
                # Note: In production, use a strong password
                cursor.execute(f'CREATE USER "{username}" WITH PASSWORD \'app_password\'')
                
                # Grant permissions
                db_name = self.config['database']['name']
                cursor.execute(f'GRANT CONNECT ON DATABASE "{db_name}" TO "{username}"')
                
                # Connect to app database to grant table permissions
                cursor.close()
                conn.close()
                
                app_conn = self._get_app_connection()
                app_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                app_cursor = app_conn.cursor()
                
                app_cursor.execute(f'GRANT USAGE ON SCHEMA public TO "{username}"')
                app_cursor.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "{username}"')
                app_cursor.execute(f'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "{username}"')
                
                app_cursor.close()
                app_conn.close()
                
                logger.info(f"Application user {username} created with appropriate permissions")
            else:
                logger.info(f"Application user {username} already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating application user: {e}")
            return False
    
    def setup_complete(self) -> bool:
        """Run complete database setup process."""
        logger.info("Starting Congressional Trading Intelligence System database setup")
        
        steps = [
            ("Creating database", self.create_database),
            ("Running migrations", self.run_all_migrations),
            ("Validating schema", self.validate_schema),
            ("Inserting initial data", self.insert_initial_data),
            ("Creating application user", self.create_app_user)
        ]
        
        for step_name, step_function in steps:
            logger.info(f"Step: {step_name}")
            if not step_function():
                logger.error(f"Failed at step: {step_name}")
                return False
            logger.info(f"Completed: {step_name}")
        
        logger.info("Database setup completed successfully!")
        logger.info("You can now run the Congressional Trading Intelligence System")
        return True

def main():
    """Main entry point for database setup script."""
    parser = argparse.ArgumentParser(
        description="Congressional Trading Intelligence System - Database Setup"
    )
    parser.add_argument(
        "--config", 
        default="config/database.yml",
        help="Database configuration file path"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing schema without making changes"
    )
    
    args = parser.parse_args()
    
    setup = DatabaseSetup(config_path=args.config)
    
    if args.validate_only:
        logger.info("Running schema validation only")
        success = setup.validate_schema()
    else:
        success = setup.setup_complete()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()