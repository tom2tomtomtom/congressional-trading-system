#!/usr/bin/env python3
"""
Automated Congressional Trading Data Update System
Handles data refresh, validation, and pipeline coordination
"""

import os
import sys
import json
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for congressional trading data"""
    
    def __init__(self):
        self.validation_rules = {
            'members': {
                'required_fields': ['name', 'party', 'state', 'chamber'],
                'party_values': ['D', 'R', 'I'],
                'chamber_values': ['House', 'Senate'],
                'state_pattern': r'^[A-Z]{2}$'
            },
            'trades': {
                'required_fields': ['member_name', 'symbol', 'transaction_date', 'amount_from', 'amount_to'],
                'amount_range': (1, 50000000),  # $1 to $50M
                'date_range': (datetime(2012, 1, 1), datetime.now() + timedelta(days=30))
            }
        }
        
        self.validation_results = {}
    
    def validate_members_data(self, members_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate congressional members data"""
        if not isinstance(members_data, list):
            return {'valid': False, 'error': 'Data must be a list'}
        
        validation_result = {
            'valid': True,
            'total_records': len(members_data),
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        valid_records = 0
        party_counts = {'D': 0, 'R': 0, 'I': 0, 'Other': 0}
        chamber_counts = {'House': 0, 'Senate': 0, 'Unknown': 0}
        
        for i, member in enumerate(members_data):
            record_errors = []
            
            # Check required fields
            for field in self.validation_rules['members']['required_fields']:
                if field not in member or member[field] is None or member[field] == '':
                    record_errors.append(f"Missing required field: {field}")
            
            # Validate party
            if 'party' in member:
                party = str(member['party']).upper()
                if party in self.validation_rules['members']['party_values']:
                    party_counts[party] += 1
                else:
                    party_counts['Other'] += 1
                    record_errors.append(f"Invalid party value: {member['party']}")
            
            # Validate chamber
            if 'chamber' in member:
                chamber = str(member['chamber'])
                if chamber in self.validation_rules['members']['chamber_values']:
                    chamber_counts[chamber] += 1
                else:
                    chamber_counts['Unknown'] += 1
                    record_errors.append(f"Invalid chamber value: {member['chamber']}")
            
            # Validate state (if present)
            if 'state' in member and member['state']:
                state = str(member['state']).upper()
                if len(state) != 2 or not state.isalpha():
                    record_errors.append(f"Invalid state format: {member['state']}")
            
            if record_errors:
                validation_result['errors'].extend([
                    f"Record {i}: {error}" for error in record_errors
                ])
            else:
                valid_records += 1
        
        validation_result['statistics'] = {
            'valid_records': valid_records,
            'invalid_records': len(members_data) - valid_records,
            'validation_rate': (valid_records / len(members_data) * 100) if len(members_data) > 0 else 0,
            'party_distribution': party_counts,
            'chamber_distribution': chamber_counts
        }
        
        # Overall validation status
        if validation_result['statistics']['validation_rate'] < 90:
            validation_result['valid'] = False
            validation_result['warnings'].append(
                f"Low validation rate: {validation_result['statistics']['validation_rate']:.1f}%"
            )
        
        logger.info(f"✅ Members validation: {validation_result['statistics']['validation_rate']:.1f}% valid")
        return validation_result
    
    def validate_trades_data(self, trades_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate congressional trades data"""
        if not isinstance(trades_data, list):
            return {'valid': False, 'error': 'Data must be a list'}
        
        validation_result = {
            'valid': True,
            'total_records': len(trades_data),
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        valid_records = 0
        amount_stats = []
        date_errors = 0
        symbol_counts = {}
        
        for i, trade in enumerate(trades_data):
            record_errors = []
            
            # Check required fields
            for field in self.validation_rules['trades']['required_fields']:
                if field not in trade or trade[field] is None:
                    record_errors.append(f"Missing required field: {field}")
            
            # Validate amounts
            if 'amount_from' in trade and 'amount_to' in trade:
                try:
                    amount_from = float(trade['amount_from'])
                    amount_to = float(trade['amount_to'])
                    
                    # Check range
                    min_amount, max_amount = self.validation_rules['trades']['amount_range']
                    if not (min_amount <= amount_from <= max_amount):
                        record_errors.append(f"Amount_from out of range: {amount_from}")
                    if not (min_amount <= amount_to <= max_amount):
                        record_errors.append(f"Amount_to out of range: {amount_to}")
                    
                    # Check logical relationship
                    if amount_from > amount_to:
                        record_errors.append(f"Amount_from ({amount_from}) > Amount_to ({amount_to})")
                    
                    amount_stats.append((amount_from + amount_to) / 2)
                    
                except (ValueError, TypeError):
                    record_errors.append("Invalid amount values - must be numeric")
            
            # Validate transaction date
            if 'transaction_date' in trade and trade['transaction_date']:
                try:
                    if isinstance(trade['transaction_date'], str):
                        trans_date = pd.to_datetime(trade['transaction_date'])
                    else:
                        trans_date = trade['transaction_date']
                    
                    min_date, max_date = self.validation_rules['trades']['date_range']
                    if not (min_date <= trans_date <= max_date):
                        record_errors.append(f"Transaction date out of range: {trans_date}")
                        date_errors += 1
                        
                except (ValueError, TypeError):
                    record_errors.append("Invalid transaction date format")
                    date_errors += 1
            
            # Track symbols
            if 'symbol' in trade and trade['symbol']:
                symbol = str(trade['symbol']).upper()
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            if record_errors:
                validation_result['errors'].extend([
                    f"Record {i}: {error}" for error in record_errors
                ])
            else:
                valid_records += 1
        
        # Calculate statistics
        validation_result['statistics'] = {
            'valid_records': valid_records,
            'invalid_records': len(trades_data) - valid_records,
            'validation_rate': (valid_records / len(trades_data) * 100) if len(trades_data) > 0 else 0,
            'date_errors': date_errors,
            'unique_symbols': len(symbol_counts),
            'top_symbols': dict(sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        }
        
        if amount_stats:
            validation_result['statistics']['amount_statistics'] = {
                'mean': float(np.mean(amount_stats)),
                'median': float(np.median(amount_stats)),
                'std': float(np.std(amount_stats)),
                'min': float(np.min(amount_stats)),
                'max': float(np.max(amount_stats))
            }
        
        # Overall validation status
        if validation_result['statistics']['validation_rate'] < 85:
            validation_result['valid'] = False
            validation_result['warnings'].append(
                f"Low validation rate: {validation_result['statistics']['validation_rate']:.1f}%"
            )
        
        logger.info(f"✅ Trades validation: {validation_result['statistics']['validation_rate']:.1f}% valid")
        return validation_result
    
    def cross_validate_data(self, members_data: List[Dict], trades_data: List[Dict]) -> Dict[str, Any]:
        """Cross-validate members and trades data for consistency"""
        result = {
            'valid': True,
            'cross_validation_checks': {}
        }
        
        # Extract member names from both datasets
        member_names_members = {member.get('name', '').strip() for member in members_data if member.get('name')}
        member_names_trades = {trade.get('member_name', '').strip() for trade in trades_data if trade.get('member_name')}
        
        # Check for orphaned trades (trades without corresponding members)
        orphaned_trades = member_names_trades - member_names_members
        
        # Check for members without trades
        members_without_trades = member_names_members - member_names_trades
        
        result['cross_validation_checks'] = {
            'total_members_in_members_data': len(member_names_members),
            'total_members_in_trades_data': len(member_names_trades),
            'orphaned_trades_count': len(orphaned_trades),
            'members_without_trades_count': len(members_without_trades),
            'orphaned_trade_members': list(orphaned_trades)[:10],  # First 10
            'members_without_trades': list(members_without_trades)[:10]  # First 10
        }
        
        # Validation thresholds
        if len(orphaned_trades) > len(member_names_trades) * 0.1:  # More than 10% orphaned
            result['valid'] = False
            result['error'] = f"Too many orphaned trades: {len(orphaned_trades)}"
        
        logger.info(f"✅ Cross-validation: {len(orphaned_trades)} orphaned trades, {len(members_without_trades)} members without trades")
        return result

class CongressionalDataUpdater:
    """Automated data update system for congressional trading intelligence"""
    
    def __init__(self, config_file: str = 'config/data_updater.json'):
        self.config_file = config_file
        self.config = self._load_config()
        self.validator = DataValidator()
        
        # Data paths
        self.data_dir = self.config.get('data_directory', 'src/data')
        self.backup_dir = self.config.get('backup_directory', 'data_backups')
        self.output_dir = self.config.get('output_directory', 'analysis_output')
        
        # Create directories
        for directory in [self.data_dir, self.backup_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Update tracking
        self.last_update = None
        self.update_history = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default"""
        default_config = {
            'data_directory': 'src/data',
            'backup_directory': 'data_backups',
            'output_directory': 'analysis_output',
            'update_schedule': {
                'enabled': True,
                'frequency': 'daily',
                'time': '02:00'
            },
            'data_sources': {
                'members': {
                    'url': None,  # To be configured
                    'format': 'json',
                    'timeout': 30
                },
                'trades': {
                    'url': None,  # To be configured
                    'format': 'json',
                    'timeout': 60
                }
            },
            'validation': {
                'enabled': True,
                'strict_mode': False,
                'backup_on_failure': True
            },
            'notifications': {
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',
                    'password': '',
                    'recipients': []
                }
            },
            'quality_thresholds': {
                'minimum_validation_rate': 85.0,
                'maximum_orphaned_trades_rate': 10.0,
                'minimum_data_freshness_hours': 24
            }
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                
                # Merge with defaults
                default_config.update(loaded_config)
                return default_config
            else:
                # Create default config file
                os.makedirs(os.path.dirname(self.config_file) if os.path.dirname(self.config_file) else '.', exist_ok=True)
                with open(self.config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                logger.info(f"📝 Created default configuration file: {self.config_file}")
                
        except Exception as e:
            logger.warning(f"⚠️ Error loading config: {e}. Using defaults.")
        
        return default_config
    
    def backup_existing_data(self) -> Dict[str, str]:
        """Create backup of existing data files"""
        backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_files = {}
        
        # Files to backup
        files_to_backup = [
            'congressional_members_full.json',
            'congressional_trades_full.json'
        ]
        
        for filename in files_to_backup:
            source_path = os.path.join(self.data_dir, filename)
            if os.path.exists(source_path):
                backup_filename = f"{filename.replace('.json', '')}_{backup_timestamp}.json"
                backup_path = os.path.join(self.backup_dir, backup_filename)
                
                try:
                    import shutil
                    shutil.copy2(source_path, backup_path)
                    backup_files[filename] = backup_path
                    logger.info(f"📦 Backed up {filename} to {backup_path}")
                except Exception as e:
                    logger.error(f"❌ Failed to backup {filename}: {e}")
        
        return backup_files
    
    def fetch_data_from_source(self, source_type: str) -> Optional[List[Dict[str, Any]]]:
        """Fetch data from configured source URL"""
        if source_type not in self.config['data_sources']:
            logger.error(f"❌ Unknown data source type: {source_type}")
            return None
        
        source_config = self.config['data_sources'][source_type]
        url = source_config.get('url')
        
        if not url:
            logger.warning(f"⚠️ No URL configured for {source_type} data source")
            return None
        
        try:
            logger.info(f"🌐 Fetching {source_type} data from {url}")
            
            response = requests.get(
                url,
                timeout=source_config.get('timeout', 30),
                headers={'User-Agent': 'Congressional-Trading-Intelligence/1.0'}
            )
            response.raise_for_status()
            
            if source_config.get('format', 'json') == 'json':
                data = response.json()
                if isinstance(data, list):
                    logger.info(f"✅ Fetched {len(data)} {source_type} records")
                    return data
                else:
                    logger.error(f"❌ Expected list format for {source_type} data")
                    return None
            else:
                logger.error(f"❌ Unsupported format: {source_config.get('format')}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to fetch {source_type} data: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in {source_type} data: {e}")
            return None
    
    def load_existing_data(self) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
        """Load existing data files"""
        members_data = None
        trades_data = None
        
        # Load members data
        members_path = os.path.join(self.data_dir, 'congressional_members_full.json')
        if os.path.exists(members_path):
            try:
                with open(members_path, 'r') as f:
                    members_data = json.load(f)
                logger.info(f"📂 Loaded {len(members_data)} existing members")
            except Exception as e:
                logger.error(f"❌ Failed to load existing members data: {e}")
        
        # Load trades data
        trades_path = os.path.join(self.data_dir, 'congressional_trades_full.json')
        if os.path.exists(trades_path):
            try:
                with open(trades_path, 'r') as f:
                    trades_data = json.load(f)
                logger.info(f"📂 Loaded {len(trades_data)} existing trades")
            except Exception as e:
                logger.error(f"❌ Failed to load existing trades data: {e}")
        
        return members_data, trades_data
    
    def merge_data(self, existing_data: List[Dict], new_data: List[Dict], key_field: str) -> List[Dict]:
        """Merge existing and new data, avoiding duplicates"""
        if not existing_data:
            return new_data
        
        if not new_data:
            return existing_data
        
        # Create lookup for existing data
        existing_keys = {str(item.get(key_field, '')).strip().lower() for item in existing_data}
        
        # Filter new data to avoid duplicates
        unique_new_data = []
        for item in new_data:
            key = str(item.get(key_field, '')).strip().lower()
            if key not in existing_keys:
                unique_new_data.append(item)
                existing_keys.add(key)
        
        merged_data = existing_data + unique_new_data
        logger.info(f"🔄 Merged data: {len(existing_data)} existing + {len(unique_new_data)} new = {len(merged_data)} total")
        
        return merged_data
    
    def save_data(self, members_data: List[Dict], trades_data: List[Dict]) -> bool:
        """Save data to files"""
        try:
            # Save members data
            members_path = os.path.join(self.data_dir, 'congressional_members_full.json')
            with open(members_path, 'w') as f:
                json.dump(members_data, f, indent=2, default=str)
            
            # Save trades data
            trades_path = os.path.join(self.data_dir, 'congressional_trades_full.json')
            with open(trades_path, 'w') as f:
                json.dump(trades_data, f, indent=2, default=str)
            
            logger.info(f"💾 Saved {len(members_data)} members and {len(trades_data)} trades")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save data: {e}")
            return False
    
    def run_data_update(self) -> Dict[str, Any]:
        """Run complete data update process"""
        update_start = datetime.now()
        update_result = {
            'success': False,
            'timestamp': update_start.isoformat(),
            'steps_completed': [],
            'errors': [],
            'statistics': {}
        }
        
        logger.info("🚀 Starting automated data update process...")
        
        try:
            # Step 1: Backup existing data
            logger.info("📦 Step 1: Backing up existing data...")
            backup_files = self.backup_existing_data()
            update_result['steps_completed'].append('backup')
            update_result['backup_files'] = backup_files
            
            # Step 2: Load existing data
            logger.info("📂 Step 2: Loading existing data...")
            existing_members, existing_trades = self.load_existing_data()
            update_result['steps_completed'].append('load_existing')
            
            # Step 3: Fetch new data (if sources configured)
            new_members = None
            new_trades = None
            
            if self.config['data_sources']['members']['url']:
                logger.info("🌐 Step 3a: Fetching new members data...")
                new_members = self.fetch_data_from_source('members')
                if new_members:
                    update_result['steps_completed'].append('fetch_members')
            
            if self.config['data_sources']['trades']['url']:
                logger.info("🌐 Step 3b: Fetching new trades data...")
                new_trades = self.fetch_data_from_source('trades')
                if new_trades:
                    update_result['steps_completed'].append('fetch_trades')
            
            # Step 4: Merge data
            logger.info("🔄 Step 4: Merging data...")
            final_members = existing_members or []
            final_trades = existing_trades or []
            
            if new_members:
                final_members = self.merge_data(final_members, new_members, 'name')
            
            if new_trades:
                final_trades = self.merge_data(final_trades, new_trades, 'id')
            
            update_result['steps_completed'].append('merge')
            
            # Step 5: Validate data
            if self.config['validation']['enabled']:
                logger.info("✅ Step 5: Validating data...")
                
                members_validation = self.validator.validate_members_data(final_members)
                trades_validation = self.validator.validate_trades_data(final_trades)
                cross_validation = self.validator.cross_validate_data(final_members, final_trades)
                
                update_result['validation'] = {
                    'members': members_validation,
                    'trades': trades_validation,
                    'cross_validation': cross_validation
                }
                
                # Check if validation passes quality thresholds
                quality_check = self._check_quality_thresholds(update_result['validation'])
                
                if not quality_check['passed'] and self.config['validation']['strict_mode']:
                    update_result['errors'].append(f"Data quality check failed: {quality_check['reason']}")
                    logger.error(f"❌ Data quality check failed: {quality_check['reason']}")
                    return update_result
                
                update_result['steps_completed'].append('validation')
            
            # Step 6: Save data
            logger.info("💾 Step 6: Saving updated data...")
            if self.save_data(final_members, final_trades):
                update_result['steps_completed'].append('save')
                update_result['success'] = True
            else:
                update_result['errors'].append("Failed to save data")
                return update_result
            
            # Step 7: Update statistics
            update_result['statistics'] = {
                'members_count': len(final_members),
                'trades_count': len(final_trades),
                'update_duration_seconds': (datetime.now() - update_start).total_seconds()
            }
            
            # Step 8: Send notifications
            if self.config['notifications']['email']['enabled']:
                self._send_update_notification(update_result)
            
            self.last_update = update_start
            self.update_history.append(update_result)
            
            logger.info(f"✅ Data update completed successfully in {update_result['statistics']['update_duration_seconds']:.1f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Data update failed: {e}")
            update_result['errors'].append(str(e))
        
        return update_result
    
    def _check_quality_thresholds(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if data meets quality thresholds"""
        thresholds = self.config['quality_thresholds']
        
        # Check members validation rate
        members_rate = validation_results['members']['statistics']['validation_rate']
        if members_rate < thresholds['minimum_validation_rate']:
            return {
                'passed': False,
                'reason': f"Members validation rate {members_rate:.1f}% below threshold {thresholds['minimum_validation_rate']:.1f}%"
            }
        
        # Check trades validation rate
        trades_rate = validation_results['trades']['statistics']['validation_rate']
        if trades_rate < thresholds['minimum_validation_rate']:
            return {
                'passed': False,
                'reason': f"Trades validation rate {trades_rate:.1f}% below threshold {thresholds['minimum_validation_rate']:.1f}%"
            }
        
        # Check orphaned trades rate
        cross_val = validation_results['cross_validation']['cross_validation_checks']
        total_members_trades = cross_val['total_members_in_trades_data']
        orphaned_count = cross_val['orphaned_trades_count']
        
        if total_members_trades > 0:
            orphaned_rate = (orphaned_count / total_members_trades) * 100
            if orphaned_rate > thresholds['maximum_orphaned_trades_rate']:
                return {
                    'passed': False,
                    'reason': f"Orphaned trades rate {orphaned_rate:.1f}% above threshold {thresholds['maximum_orphaned_trades_rate']:.1f}%"
                }
        
        return {'passed': True}
    
    def _send_update_notification(self, update_result: Dict[str, Any]) -> None:
        """Send email notification about update status"""
        try:
            email_config = self.config['notifications']['email']
            
            if not email_config.get('recipients'):
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"Congressional Trading Data Update - {'Success' if update_result['success'] else 'Failed'}"
            
            # Create email body
            body = f"""
Congressional Trading Intelligence Data Update Report

Status: {'SUCCESS' if update_result['success'] else 'FAILED'}
Timestamp: {update_result['timestamp']}
Duration: {update_result.get('statistics', {}).get('update_duration_seconds', 'N/A')} seconds

Steps Completed: {', '.join(update_result['steps_completed'])}

Statistics:
- Members: {update_result.get('statistics', {}).get('members_count', 'N/A')}
- Trades: {update_result.get('statistics', {}).get('trades_count', 'N/A')}

            """
            
            if update_result.get('errors'):
                body += f"\nErrors:\n" + '\n'.join(f"- {error}" for error in update_result['errors'])
            
            if 'validation' in update_result:
                validation = update_result['validation']
                body += f"""
Validation Results:
- Members Validation Rate: {validation['members']['statistics']['validation_rate']:.1f}%
- Trades Validation Rate: {validation['trades']['statistics']['validation_rate']:.1f}%
- Orphaned Trades: {validation['cross_validation']['cross_validation_checks']['orphaned_trades_count']}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"📧 Update notification sent to {len(email_config['recipients'])} recipients")
            
        except Exception as e:
            logger.error(f"❌ Failed to send email notification: {e}")
    
    def schedule_updates(self) -> None:
        """Schedule automated updates based on configuration"""
        if not self.config['update_schedule']['enabled']:
            logger.info("⏰ Automated updates disabled in configuration")
            return
        
        frequency = self.config['update_schedule']['frequency']
        update_time = self.config['update_schedule']['time']
        
        if frequency == 'daily':
            schedule.every().day.at(update_time).do(self.run_data_update)
            logger.info(f"⏰ Scheduled daily updates at {update_time}")
        elif frequency == 'hourly':
            schedule.every().hour.do(self.run_data_update)
            logger.info("⏰ Scheduled hourly updates")
        elif frequency == 'weekly':
            schedule.every().week.do(self.run_data_update)
            logger.info("⏰ Scheduled weekly updates")
        
        logger.info("⏰ Update scheduler started. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("⏰ Update scheduler stopped")
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get current update status and history"""
        return {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_history_count': len(self.update_history),
            'recent_updates': self.update_history[-5:],  # Last 5 updates
            'scheduler_status': 'running' if schedule.jobs else 'stopped',
            'next_scheduled_update': str(schedule.next_run()) if schedule.jobs else None
        }

def main():
    """Command-line interface for data updater"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Congressional Trading Data Updater')
    parser.add_argument('--update', action='store_true', help='Run data update once')
    parser.add_argument('--schedule', action='store_true', help='Start scheduled updates')
    parser.add_argument('--status', action='store_true', help='Show update status')
    parser.add_argument('--config', default='config/data_updater.json', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = CongressionalDataUpdater(args.config)
    
    if args.update:
        # Run single update
        result = updater.run_data_update()
        
        print("\n" + "="*60)
        print("📊 DATA UPDATE REPORT")
        print("="*60)
        print(f"Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Steps Completed: {', '.join(result['steps_completed'])}")
        
        if 'statistics' in result:
            stats = result['statistics']
            print(f"Members: {stats.get('members_count', 'N/A')}")
            print(f"Trades: {stats.get('trades_count', 'N/A')}")
            print(f"Duration: {stats.get('update_duration_seconds', 'N/A'):.1f} seconds")
        
        if result.get('errors'):
            print(f"\nErrors:")
            for error in result['errors']:
                print(f"  • {error}")
        
        print("="*60)
    
    elif args.schedule:
        # Start scheduler
        updater.schedule_updates()
    
    elif args.status:
        # Show status
        status = updater.get_update_status()
        
        print("\n" + "="*60)
        print("📊 DATA UPDATE STATUS")
        print("="*60)
        print(f"Last Update: {status['last_update'] or 'Never'}")
        print(f"Update History: {status['update_history_count']} updates")
        print(f"Scheduler: {status['scheduler_status']}")
        print(f"Next Update: {status['next_scheduled_update'] or 'Not scheduled'}")
        print("="*60)
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main()