#!/usr/bin/env python3
"""
Filing Parser and Validator Pipeline
Processes raw disclosure filings through validation, parsing, and enrichment stages.

Pipeline: Scraper → Parser → Validator → Database → Alert Trigger
"""

import logging
import re
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"  # Critical issue, record cannot be processed
    WARNING = "warning"  # Non-critical issue, proceed with caution
    INFO = "info"  # Informational note


class TransactionType(Enum):
    """Standardized transaction types"""
    PURCHASE = "purchase"
    SALE = "sale"
    SALE_PARTIAL = "sale_partial"
    EXCHANGE = "exchange"
    UNKNOWN = "unknown"


class AssetType(Enum):
    """Asset type classifications"""
    STOCK = "stock"
    OPTION = "option"
    BOND = "bond"
    MUTUAL_FUND = "mutual_fund"
    ETF = "etf"
    CRYPTOCURRENCY = "cryptocurrency"
    REAL_ESTATE = "real_estate"
    OTHER = "other"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    field: str
    level: ValidationLevel
    message: str
    original_value: Any = None
    corrected_value: Any = None


@dataclass
class ParsedTrade:
    """Fully parsed and validated trade record"""
    # Identifiers
    trade_id: str
    filing_id: str

    # Member info
    member_name: str
    member_first_name: str
    member_last_name: str
    bioguide_id: Optional[str] = None
    chamber: str = ""
    state: str = ""
    party: Optional[str] = None

    # Trade details
    symbol: str = ""
    asset_name: str = ""
    asset_type: AssetType = AssetType.STOCK
    transaction_type: TransactionType = TransactionType.UNKNOWN

    # Amounts
    amount_min: int = 0
    amount_max: int = 0
    amount_mid: int = 0
    shares: Optional[float] = None
    price_per_share: Optional[float] = None

    # Dates
    transaction_date: Optional[date] = None
    notification_date: Optional[date] = None
    filing_date: Optional[date] = None
    filing_delay_days: int = 0

    # Ownership
    owner: str = "Self"  # Self, Spouse, Dependent Child, Joint

    # Source
    source_document: str = ""
    source_url: str = ""
    data_source: str = ""

    # Metadata
    raw_data: Dict[str, Any] = field(default_factory=dict)
    validation_results: List[ValidationResult] = field(default_factory=list)
    is_valid: bool = True
    parsed_at: str = ""
    comment: str = ""

    def __post_init__(self):
        if not self.parsed_at:
            self.parsed_at = datetime.now().isoformat()
        if not self.trade_id:
            self.trade_id = self._generate_trade_id()

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        content = f"{self.filing_id}{self.symbol}{self.transaction_date}{self.amount_mid}"
        hash_part = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"T{hash_part.upper()}"

    @property
    def has_errors(self) -> bool:
        return any(v.level == ValidationLevel.ERROR for v in self.validation_results)

    @property
    def has_warnings(self) -> bool:
        return any(v.level == ValidationLevel.WARNING for v in self.validation_results)


class BaseValidator(ABC):
    """Base class for validators"""

    @abstractmethod
    def validate(self, trade: ParsedTrade) -> List[ValidationResult]:
        """Validate a trade and return validation results"""
        pass


class RequiredFieldsValidator(BaseValidator):
    """Validate that required fields are present"""

    REQUIRED_FIELDS = ['member_name', 'symbol', 'transaction_type', 'amount_mid']

    def validate(self, trade: ParsedTrade) -> List[ValidationResult]:
        results = []

        for field_name in self.REQUIRED_FIELDS:
            value = getattr(trade, field_name, None)
            if value is None or value == "" or value == 0:
                results.append(ValidationResult(
                    field=field_name,
                    level=ValidationLevel.ERROR,
                    message=f"Required field '{field_name}' is missing or empty",
                    original_value=value
                ))

        return results


class SymbolValidator(BaseValidator):
    """Validate and normalize stock symbols"""

    SYMBOL_PATTERN = re.compile(r'^[A-Z]{1,5}$')

    # Common symbol corrections
    SYMBOL_CORRECTIONS = {
        'GOOGL': 'GOOGL',
        'GOOG': 'GOOG',
        'FB': 'META',  # Facebook renamed to Meta
        'BRKB': 'BRK.B',
        'BRKA': 'BRK.A',
    }

    def validate(self, trade: ParsedTrade) -> List[ValidationResult]:
        results = []

        if not trade.symbol:
            return results

        symbol = trade.symbol.upper().strip()

        # Apply corrections
        if symbol in self.SYMBOL_CORRECTIONS:
            corrected = self.SYMBOL_CORRECTIONS[symbol]
            results.append(ValidationResult(
                field='symbol',
                level=ValidationLevel.INFO,
                message=f"Symbol corrected from '{symbol}' to '{corrected}'",
                original_value=symbol,
                corrected_value=corrected
            ))
            trade.symbol = corrected

        # Validate format
        if not self.SYMBOL_PATTERN.match(symbol.replace('.', '')):
            results.append(ValidationResult(
                field='symbol',
                level=ValidationLevel.WARNING,
                message=f"Symbol '{symbol}' may not be a valid stock ticker",
                original_value=symbol
            ))

        return results


class DateValidator(BaseValidator):
    """Validate and normalize dates"""

    def validate(self, trade: ParsedTrade) -> List[ValidationResult]:
        results = []
        today = date.today()

        # Validate transaction date
        if trade.transaction_date:
            if trade.transaction_date > today:
                results.append(ValidationResult(
                    field='transaction_date',
                    level=ValidationLevel.ERROR,
                    message=f"Transaction date {trade.transaction_date} is in the future",
                    original_value=trade.transaction_date
                ))
            elif trade.transaction_date < date(2012, 1, 1):  # STOCK Act effective date
                results.append(ValidationResult(
                    field='transaction_date',
                    level=ValidationLevel.WARNING,
                    message=f"Transaction date {trade.transaction_date} is before STOCK Act",
                    original_value=trade.transaction_date
                ))

        # Validate filing date
        if trade.filing_date:
            if trade.filing_date > today:
                results.append(ValidationResult(
                    field='filing_date',
                    level=ValidationLevel.ERROR,
                    message=f"Filing date {trade.filing_date} is in the future",
                    original_value=trade.filing_date
                ))

        # Check filing delay
        if trade.transaction_date and trade.filing_date:
            delay = (trade.filing_date - trade.transaction_date).days
            trade.filing_delay_days = delay

            if delay > 45:  # STOCK Act requires filing within 45 days
                results.append(ValidationResult(
                    field='filing_delay_days',
                    level=ValidationLevel.WARNING,
                    message=f"Filing delayed {delay} days (STOCK Act requires 45 days)",
                    original_value=delay
                ))

            if delay < 0:
                results.append(ValidationResult(
                    field='filing_delay_days',
                    level=ValidationLevel.ERROR,
                    message=f"Filing date is before transaction date",
                    original_value=delay
                ))

        return results


class AmountValidator(BaseValidator):
    """Validate trade amounts"""

    STOCK_ACT_RANGES = [
        (1001, 15000),
        (15001, 50000),
        (50001, 100000),
        (100001, 250000),
        (250001, 500000),
        (500001, 1000000),
        (1000001, 5000000),
        (5000001, 25000000),
        (25000001, 50000000),
        (50000001, float('inf'))
    ]

    def validate(self, trade: ParsedTrade) -> List[ValidationResult]:
        results = []

        # Check amount range validity
        if trade.amount_min > trade.amount_max:
            results.append(ValidationResult(
                field='amount_range',
                level=ValidationLevel.ERROR,
                message=f"Minimum amount ({trade.amount_min}) exceeds maximum ({trade.amount_max})",
                original_value=(trade.amount_min, trade.amount_max)
            ))

        # Check if amounts are in valid STOCK Act ranges
        if trade.amount_min > 0 and trade.amount_max > 0:
            in_valid_range = False
            for min_range, max_range in self.STOCK_ACT_RANGES:
                if trade.amount_min >= min_range and trade.amount_max <= max_range:
                    in_valid_range = True
                    break

            if not in_valid_range:
                results.append(ValidationResult(
                    field='amount_range',
                    level=ValidationLevel.INFO,
                    message=f"Amount range may not match standard STOCK Act ranges",
                    original_value=(trade.amount_min, trade.amount_max)
                ))

        # Calculate mid if not set
        if trade.amount_mid == 0 and trade.amount_min > 0 and trade.amount_max > 0:
            trade.amount_mid = (trade.amount_min + trade.amount_max) // 2
            results.append(ValidationResult(
                field='amount_mid',
                level=ValidationLevel.INFO,
                message=f"Calculated mid amount: {trade.amount_mid}",
                corrected_value=trade.amount_mid
            ))

        return results


class TransactionTypeValidator(BaseValidator):
    """Validate and normalize transaction types"""

    TYPE_MAPPINGS = {
        'p': TransactionType.PURCHASE,
        'purchase': TransactionType.PURCHASE,
        'buy': TransactionType.PURCHASE,
        's': TransactionType.SALE,
        'sale': TransactionType.SALE,
        'sell': TransactionType.SALE,
        'sale (full)': TransactionType.SALE,
        's (full)': TransactionType.SALE,
        'sale (partial)': TransactionType.SALE_PARTIAL,
        's (partial)': TransactionType.SALE_PARTIAL,
        'e': TransactionType.EXCHANGE,
        'exchange': TransactionType.EXCHANGE,
    }

    def validate(self, trade: ParsedTrade) -> List[ValidationResult]:
        results = []

        if trade.transaction_type == TransactionType.UNKNOWN:
            # Try to determine from raw data
            raw_type = trade.raw_data.get('transaction_type', '')
            if isinstance(raw_type, str):
                normalized = raw_type.lower().strip()
                if normalized in self.TYPE_MAPPINGS:
                    trade.transaction_type = self.TYPE_MAPPINGS[normalized]
                    results.append(ValidationResult(
                        field='transaction_type',
                        level=ValidationLevel.INFO,
                        message=f"Normalized transaction type to {trade.transaction_type.value}",
                        original_value=raw_type,
                        corrected_value=trade.transaction_type.value
                    ))
                else:
                    results.append(ValidationResult(
                        field='transaction_type',
                        level=ValidationLevel.WARNING,
                        message=f"Unknown transaction type: {raw_type}",
                        original_value=raw_type
                    ))

        return results


class DuplicateDetector(BaseValidator):
    """Detect potential duplicate trades"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self._seen_trades = set()
        self._load_seen_trades()

    def _load_seen_trades(self):
        """Load previously seen trade hashes"""
        db_path = self.cache_dir / "pipeline.db"
        if not db_path.exists():
            return

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("SELECT trade_hash FROM processed_trades")
            self._seen_trades = {row[0] for row in cursor.fetchall()}
        except:
            pass
        finally:
            conn.close()

    def _get_trade_hash(self, trade: ParsedTrade) -> str:
        """Generate a hash for duplicate detection"""
        content = f"{trade.member_name}{trade.symbol}{trade.transaction_date}{trade.amount_mid}{trade.transaction_type.value}"
        return hashlib.md5(content.encode()).hexdigest()

    def validate(self, trade: ParsedTrade) -> List[ValidationResult]:
        results = []

        trade_hash = self._get_trade_hash(trade)

        if trade_hash in self._seen_trades:
            results.append(ValidationResult(
                field='trade_id',
                level=ValidationLevel.WARNING,
                message="This trade appears to be a duplicate",
                original_value=trade.trade_id
            ))

        return results


class FilingPipeline:
    """
    Main pipeline for processing disclosure filings.

    Stages:
    1. Parse raw disclosure data
    2. Validate all fields
    3. Normalize and enrich data
    4. Store in database
    5. Trigger alerts for valid filings
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".congressional_trading_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize validators
        self.validators = [
            RequiredFieldsValidator(),
            SymbolValidator(),
            DateValidator(),
            AmountValidator(),
            TransactionTypeValidator(),
            DuplicateDetector(self.cache_dir),
        ]

        # Initialize database
        self._init_db()

        # Stats
        self._stats = {
            'total_processed': 0,
            'valid_trades': 0,
            'invalid_trades': 0,
            'warnings': 0,
            'errors': 0
        }

    def _init_db(self):
        """Initialize pipeline database"""
        db_path = self.cache_dir / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_trades (
                trade_id TEXT PRIMARY KEY,
                filing_id TEXT,
                trade_hash TEXT UNIQUE,
                member_name TEXT,
                symbol TEXT,
                transaction_type TEXT,
                transaction_date TEXT,
                filing_date TEXT,
                amount_min INTEGER,
                amount_max INTEGER,
                amount_mid INTEGER,
                owner TEXT,
                is_valid BOOLEAN,
                validation_errors INTEGER,
                validation_warnings INTEGER,
                processed_at TEXT,
                raw_data TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                completed_at TEXT,
                records_processed INTEGER,
                valid_records INTEGER,
                invalid_records INTEGER,
                source TEXT
            )
        """)

        conn.commit()
        conn.close()

    def parse_raw_disclosure(self, raw_data: Dict[str, Any]) -> ParsedTrade:
        """
        Parse raw disclosure data into a ParsedTrade object.

        Args:
            raw_data: Dictionary containing raw disclosure data

        Returns:
            ParsedTrade object with parsed data
        """
        # Extract member name
        member_name = raw_data.get('member_name', '') or raw_data.get('representative', '')
        first_name, last_name = self._parse_name(member_name)

        # Parse dates
        transaction_date = self._parse_date(
            raw_data.get('transaction_date') or raw_data.get('transactionDate')
        )
        filing_date = self._parse_date(
            raw_data.get('filing_date') or raw_data.get('filingDate')
        )
        notification_date = self._parse_date(raw_data.get('notification_date'))

        # Parse amounts
        amount_min, amount_max, amount_mid = self._parse_amount(raw_data)

        # Parse transaction type
        transaction_type = self._parse_transaction_type(
            raw_data.get('transaction_type') or raw_data.get('transactionType')
        )

        # Parse asset type
        asset_type = self._parse_asset_type(
            raw_data.get('asset_type') or raw_data.get('assetType')
        )

        return ParsedTrade(
            trade_id='',  # Will be generated
            filing_id=raw_data.get('filing_id', ''),
            member_name=member_name,
            member_first_name=first_name,
            member_last_name=last_name,
            bioguide_id=raw_data.get('bioguide_id'),
            chamber=raw_data.get('chamber', ''),
            state=raw_data.get('state', ''),
            party=raw_data.get('party'),
            symbol=str(raw_data.get('symbol', '')).upper().strip(),
            asset_name=raw_data.get('asset_name', '') or raw_data.get('asset_description', ''),
            asset_type=asset_type,
            transaction_type=transaction_type,
            amount_min=amount_min,
            amount_max=amount_max,
            amount_mid=amount_mid,
            transaction_date=transaction_date,
            notification_date=notification_date,
            filing_date=filing_date,
            owner=raw_data.get('owner', 'Self') or raw_data.get('owner_type', 'Self'),
            source_document=raw_data.get('source_document', ''),
            source_url=raw_data.get('document_url', '') or raw_data.get('source_url', ''),
            data_source=raw_data.get('data_source', 'unknown'),
            raw_data=raw_data,
            comment=raw_data.get('comment', '')
        )

    def _parse_name(self, full_name: str) -> Tuple[str, str]:
        """Parse full name into first and last name"""
        if not full_name:
            return '', ''

        if ',' in full_name:
            parts = full_name.split(',')
            last_name = parts[0].strip()
            first_name = parts[1].strip().split()[0] if len(parts) > 1 else ''
        else:
            parts = full_name.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = parts[-1]
            else:
                first_name = ''
                last_name = full_name

        return first_name, last_name

    def _parse_date(self, date_value: Any) -> Optional[date]:
        """Parse date from various formats"""
        if not date_value:
            return None

        if isinstance(date_value, date):
            return date_value

        if isinstance(date_value, datetime):
            return date_value.date()

        if isinstance(date_value, str):
            # Try common formats
            formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%m-%d-%Y',
                '%Y/%m/%d',
                '%d-%b-%Y',
                '%B %d, %Y',
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date_value.strip(), fmt).date()
                except ValueError:
                    continue

        return None

    def _parse_amount(self, raw_data: Dict) -> Tuple[int, int, int]:
        """Parse amount range from raw data"""
        amount_min = 0
        amount_max = 0
        amount_mid = 0

        # Try direct numeric values
        if 'amount_min' in raw_data:
            try:
                amount_min = int(raw_data['amount_min'])
            except (ValueError, TypeError):
                pass

        if 'amount_max' in raw_data:
            try:
                amount_max = int(raw_data['amount_max'])
            except (ValueError, TypeError):
                pass

        if 'amount_mid' in raw_data:
            try:
                amount_mid = int(raw_data['amount_mid'])
            except (ValueError, TypeError):
                pass

        # Try amount_range string
        if amount_min == 0 and 'amount_range' in raw_data:
            amount_str = raw_data['amount_range']
            amount_min, amount_max, amount_mid = self._parse_amount_string(amount_str)

        # Calculate mid if not set
        if amount_mid == 0 and amount_min > 0 and amount_max > 0:
            amount_mid = (amount_min + amount_max) // 2

        return amount_min, amount_max, amount_mid

    def _parse_amount_string(self, amount_str: str) -> Tuple[int, int, int]:
        """Parse amount range string"""
        AMOUNT_RANGES = {
            "$1,001 - $15,000": (1001, 15000),
            "$15,001 - $50,000": (15001, 50000),
            "$50,001 - $100,000": (50001, 100000),
            "$100,001 - $250,000": (100001, 250000),
            "$250,001 - $500,000": (250001, 500000),
            "$500,001 - $1,000,000": (500001, 1000000),
            "$1,000,001 - $5,000,000": (1000001, 5000000),
            "$5,000,001 - $25,000,000": (5000001, 25000000),
            "$25,000,001 - $50,000,000": (25000001, 50000000),
            "Over $50,000,000": (50000001, 100000000),
        }

        amount_str = amount_str.strip()

        if amount_str in AMOUNT_RANGES:
            min_amt, max_amt = AMOUNT_RANGES[amount_str]
            return min_amt, max_amt, (min_amt + max_amt) // 2

        # Try to parse numbers
        numbers = re.findall(r'[\d,]+', amount_str)
        if len(numbers) >= 2:
            min_amt = int(numbers[0].replace(',', ''))
            max_amt = int(numbers[1].replace(',', ''))
            return min_amt, max_amt, (min_amt + max_amt) // 2

        return 0, 0, 0

    def _parse_transaction_type(self, type_value: Any) -> TransactionType:
        """Parse transaction type"""
        if not type_value:
            return TransactionType.UNKNOWN

        type_str = str(type_value).lower().strip()

        mappings = {
            'p': TransactionType.PURCHASE,
            'purchase': TransactionType.PURCHASE,
            'buy': TransactionType.PURCHASE,
            's': TransactionType.SALE,
            'sale': TransactionType.SALE,
            'sell': TransactionType.SALE,
            'e': TransactionType.EXCHANGE,
            'exchange': TransactionType.EXCHANGE,
        }

        return mappings.get(type_str, TransactionType.UNKNOWN)

    def _parse_asset_type(self, type_value: Any) -> AssetType:
        """Parse asset type"""
        if not type_value:
            return AssetType.STOCK

        type_str = str(type_value).lower().strip()

        mappings = {
            'stock': AssetType.STOCK,
            'common stock': AssetType.STOCK,
            'option': AssetType.OPTION,
            'call option': AssetType.OPTION,
            'put option': AssetType.OPTION,
            'bond': AssetType.BOND,
            'municipal bond': AssetType.BOND,
            'corporate bond': AssetType.BOND,
            'mutual fund': AssetType.MUTUAL_FUND,
            'etf': AssetType.ETF,
            'exchange traded fund': AssetType.ETF,
            'crypto': AssetType.CRYPTOCURRENCY,
            'cryptocurrency': AssetType.CRYPTOCURRENCY,
            'bitcoin': AssetType.CRYPTOCURRENCY,
        }

        return mappings.get(type_str, AssetType.OTHER)

    def validate(self, trade: ParsedTrade) -> ParsedTrade:
        """
        Run all validators on a trade.

        Args:
            trade: ParsedTrade to validate

        Returns:
            Trade with validation results populated
        """
        all_results = []

        for validator in self.validators:
            try:
                results = validator.validate(trade)
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Validator {validator.__class__.__name__} failed: {e}")
                all_results.append(ValidationResult(
                    field='validator',
                    level=ValidationLevel.ERROR,
                    message=f"Validator {validator.__class__.__name__} failed: {str(e)}"
                ))

        trade.validation_results = all_results
        trade.is_valid = not any(r.level == ValidationLevel.ERROR for r in all_results)

        return trade

    def process(self, raw_data: Dict[str, Any]) -> ParsedTrade:
        """
        Process a single raw disclosure record through the full pipeline.

        Args:
            raw_data: Raw disclosure data dictionary

        Returns:
            Fully processed and validated ParsedTrade
        """
        # Parse
        trade = self.parse_raw_disclosure(raw_data)

        # Validate
        trade = self.validate(trade)

        # Store
        self._store_trade(trade)

        # Update stats
        self._stats['total_processed'] += 1
        if trade.is_valid:
            self._stats['valid_trades'] += 1
        else:
            self._stats['invalid_trades'] += 1

        self._stats['warnings'] += sum(1 for r in trade.validation_results
                                       if r.level == ValidationLevel.WARNING)
        self._stats['errors'] += sum(1 for r in trade.validation_results
                                     if r.level == ValidationLevel.ERROR)

        return trade

    def process_batch(self, raw_records: List[Dict[str, Any]]) -> List[ParsedTrade]:
        """
        Process a batch of raw disclosure records.

        Args:
            raw_records: List of raw disclosure data dictionaries

        Returns:
            List of processed ParsedTrade objects
        """
        processed = []

        for raw_data in raw_records:
            try:
                trade = self.process(raw_data)
                processed.append(trade)
            except Exception as e:
                logger.error(f"Failed to process record: {e}")

        return processed

    def _store_trade(self, trade: ParsedTrade):
        """Store processed trade in database"""
        db_path = self.cache_dir / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Generate trade hash
        content = f"{trade.member_name}{trade.symbol}{trade.transaction_date}{trade.amount_mid}{trade.transaction_type.value}"
        trade_hash = hashlib.md5(content.encode()).hexdigest()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO processed_trades
                (trade_id, filing_id, trade_hash, member_name, symbol, transaction_type,
                 transaction_date, filing_date, amount_min, amount_max, amount_mid,
                 owner, is_valid, validation_errors, validation_warnings, processed_at, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.trade_id,
                trade.filing_id,
                trade_hash,
                trade.member_name,
                trade.symbol,
                trade.transaction_type.value,
                str(trade.transaction_date) if trade.transaction_date else None,
                str(trade.filing_date) if trade.filing_date else None,
                trade.amount_min,
                trade.amount_max,
                trade.amount_mid,
                trade.owner,
                trade.is_valid,
                sum(1 for r in trade.validation_results if r.level == ValidationLevel.ERROR),
                sum(1 for r in trade.validation_results if r.level == ValidationLevel.WARNING),
                trade.parsed_at,
                json.dumps(trade.raw_data)
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to store trade: {e}")
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            **self._stats,
            'cache_location': str(self.cache_dir)
        }

    def get_recent_trades(self, limit: int = 100, valid_only: bool = False) -> List[Dict[str, Any]]:
        """Get recently processed trades"""
        db_path = self.cache_dir / "pipeline.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        query = """
            SELECT trade_id, filing_id, member_name, symbol, transaction_type,
                   transaction_date, filing_date, amount_mid, is_valid, processed_at
            FROM processed_trades
        """

        if valid_only:
            query += " WHERE is_valid = 1"

        query += " ORDER BY processed_at DESC LIMIT ?"

        cursor.execute(query, (limit,))

        trades = []
        for row in cursor.fetchall():
            trades.append({
                'trade_id': row[0],
                'filing_id': row[1],
                'member_name': row[2],
                'symbol': row[3],
                'transaction_type': row[4],
                'transaction_date': row[5],
                'filing_date': row[6],
                'amount_mid': row[7],
                'is_valid': bool(row[8]),
                'processed_at': row[9]
            })

        conn.close()
        return trades


def main():
    """Test the filing pipeline"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    pipeline = FilingPipeline()

    # Test with sample data
    sample_trades = [
        {
            'filing_id': 'H20240101ABC12345',
            'member_name': 'Pelosi, Nancy',
            'symbol': 'NVDA',
            'transaction_type': 'Purchase',
            'transaction_date': '2024-01-15',
            'filing_date': '2024-02-10',
            'amount_min': 500001,
            'amount_max': 1000000,
            'chamber': 'house',
            'state': 'CA',
            'owner': 'Spouse',
            'data_source': 'house_clerk_website'
        },
        {
            'filing_id': 'S20240102DEF67890',
            'member_name': 'Warren, Elizabeth',
            'symbol': 'AAPL',
            'transaction_type': 'Sale',
            'transaction_date': '2024-01-20',
            'filing_date': '2024-03-15',  # Late filing
            'amount_min': 15001,
            'amount_max': 50000,
            'chamber': 'senate',
            'state': 'MA',
            'owner': 'Self',
            'data_source': 'senate_efd'
        }
    ]

    print("Filing Pipeline Test")
    print("=" * 60)

    for raw_trade in sample_trades:
        trade = pipeline.process(raw_trade)

        print(f"\nProcessed: {trade.member_name} - {trade.symbol}")
        print(f"  Valid: {trade.is_valid}")
        print(f"  Trade ID: {trade.trade_id}")
        print(f"  Amount: ${trade.amount_mid:,}")
        print(f"  Filing Delay: {trade.filing_delay_days} days")

        if trade.validation_results:
            print("  Validation Results:")
            for result in trade.validation_results:
                print(f"    [{result.level.value}] {result.field}: {result.message}")

    print("\nPipeline Stats:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
