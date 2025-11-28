#!/usr/bin/env python3
"""
House Financial Disclosure Scraper - Enhanced Real-Time Detection
Detects new STOCK Act filings within 1 hour of publication.
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set, Tuple
import re
import json
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin, urlparse, parse_qs
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """STOCK Act report types"""
    PERIODIC_TRANSACTION = "PTR"  # Periodic Transaction Report
    ANNUAL = "Annual"
    NEW_FILER = "New Filer"
    TERMINATION = "Termination"
    AMENDMENT = "Amendment"
    BLIND_TRUST = "Blind Trust"
    OTHER = "Other"


class TransactionType(Enum):
    """Transaction types in disclosures"""
    PURCHASE = "P"
    SALE_FULL = "S"
    SALE_PARTIAL = "S (Partial)"
    EXCHANGE = "E"


@dataclass
class DisclosureRecord:
    """Represents a single disclosure filing"""
    filing_id: str
    member_name: str
    first_name: str
    last_name: str
    state: str
    district: str
    filing_date: str
    report_type: str
    document_url: str
    year: int
    chamber: str = "house"
    is_amendment: bool = False
    amendment_date: Optional[str] = None
    checksum: str = ""

    def __post_init__(self):
        """Generate checksum for deduplication"""
        if not self.checksum:
            content = f"{self.member_name}{self.filing_date}{self.report_type}{self.document_url}"
            self.checksum = hashlib.md5(content.encode()).hexdigest()


@dataclass
class TradeRecord:
    """Represents a single trade from a disclosure"""
    filing_id: str
    member_name: str
    symbol: str
    asset_name: str
    transaction_type: str
    transaction_date: str
    filing_date: str
    amount_min: int
    amount_max: int
    amount_mid: int
    owner: str  # Self, Spouse, Dependent Child, Joint
    filing_delay_days: int
    source_document: str
    data_source: str = "house_clerk_website"


class HouseDisclosureScraper:
    """
    Scrapes financial disclosure data directly from House Clerk website.
    Enhanced for real-time detection with polling support.

    Public records - completely legal and ethical data collection.
    """

    BASE_URL = "https://disclosures-clerk.house.gov"
    SEARCH_URL = f"{BASE_URL}/FinancialDisclosure"
    PTR_SEARCH_URL = f"{BASE_URL}/FinancialDisclosure/Search"

    # Amount range mappings from STOCK Act
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

    def __init__(self, delay_seconds: float = 1.0, cache_dir: Optional[str] = None):
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Congressional Trading Intelligence Research Tool/2.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': self.BASE_URL,
        })

        # Cache for tracking seen filings
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".congressional_trading_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_cache_db()

        # Track seen filing IDs for real-time detection
        self._seen_filings: Set[str] = set()
        self._load_seen_filings()

    def _init_cache_db(self):
        """Initialize SQLite cache database"""
        db_path = self.cache_dir / "house_disclosures.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS seen_filings (
                filing_id TEXT PRIMARY KEY,
                checksum TEXT,
                first_seen_at TEXT,
                member_name TEXT,
                filing_date TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scrape_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                completed_at TEXT,
                filings_found INTEGER,
                new_filings INTEGER,
                status TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_seen_filings(self):
        """Load previously seen filings from cache"""
        db_path = self.cache_dir / "house_disclosures.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT filing_id FROM seen_filings")
        self._seen_filings = {row[0] for row in cursor.fetchall()}

        conn.close()
        logger.info(f"Loaded {len(self._seen_filings)} seen filings from cache")

    def _mark_filing_seen(self, disclosure: DisclosureRecord):
        """Mark a filing as seen in cache"""
        db_path = self.cache_dir / "house_disclosures.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO seen_filings (filing_id, checksum, first_seen_at, member_name, filing_date)
            VALUES (?, ?, ?, ?, ?)
        """, (disclosure.filing_id, disclosure.checksum, datetime.now().isoformat(),
              disclosure.member_name, disclosure.filing_date))

        conn.commit()
        conn.close()

        self._seen_filings.add(disclosure.filing_id)

    def _log_scrape_run(self, filings_found: int, new_filings: int, status: str) -> int:
        """Log a scraping run"""
        db_path = self.cache_dir / "house_disclosures.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO scrape_runs (started_at, completed_at, filings_found, new_filings, status)
            VALUES (?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), datetime.now().isoformat(), filings_found, new_filings, status))

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return run_id

    def get_available_years(self) -> List[int]:
        """Get list of years with available disclosure data"""
        try:
            response = self.session.get(self.SEARCH_URL, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            years = []
            # Look for year dropdown
            year_select = soup.find('select', {'id': 'FilingYear'}) or soup.find('select', {'name': 'FilingYear'})

            if year_select:
                for option in year_select.find_all('option'):
                    year_text = option.get('value', '') or option.get_text(strip=True)
                    year_match = re.search(r'(20\d{2})', str(year_text))
                    if year_match:
                        years.append(int(year_match.group(1)))

            # Fallback: look for year links
            if not years:
                year_elements = soup.find_all(['option', 'a'], text=re.compile(r'20\d{2}'))
                for element in year_elements:
                    year_match = re.search(r'(20\d{2})', element.get_text())
                    if year_match:
                        years.append(int(year_match.group(1)))

            # Default to recent years if none found
            if not years:
                current_year = datetime.now().year
                years = list(range(current_year - 2, current_year + 1))

            return sorted(set(years), reverse=True)

        except Exception as e:
            logger.error(f"Error getting available years: {e}")
            current_year = datetime.now().year
            return list(range(current_year - 2, current_year + 1))

    def search_ptr_disclosures(self, year: int = None,
                               last_name: str = None,
                               state: str = None,
                               filing_date_start: str = None,
                               filing_date_end: str = None) -> List[DisclosureRecord]:
        """
        Search for Periodic Transaction Reports (PTR) - the key STOCK Act filings.

        Args:
            year: Year to search (default: current year)
            last_name: Member last name to filter (optional)
            state: State abbreviation to filter (optional)
            filing_date_start: Start date for filing search (YYYY-MM-DD)
            filing_date_end: End date for filing search (YYYY-MM-DD)

        Returns:
            List of DisclosureRecord objects
        """
        if year is None:
            year = datetime.now().year

        try:
            # Build search parameters for PTR search
            search_params = {
                'FilingYear': year,
                'ReportType': 'PTR',  # Periodic Transaction Report
                'LastName': last_name or '',
                'State': state or '',
            }

            if filing_date_start:
                search_params['FilingDateStart'] = filing_date_start
            if filing_date_end:
                search_params['FilingDateEnd'] = filing_date_end

            logger.info(f"Searching House PTR disclosures for year {year}")

            response = self.session.get(
                self.PTR_SEARCH_URL,
                params=search_params,
                timeout=60
            )
            response.raise_for_status()

            disclosures = self._parse_disclosure_results(response.content, year)

            time.sleep(self.delay_seconds)

            return disclosures

        except Exception as e:
            logger.error(f"Error searching PTR disclosures for year {year}: {e}")
            return []

    def search_all_disclosures(self, year: int = None,
                               report_type: str = None) -> List[DisclosureRecord]:
        """
        Search for all types of financial disclosures.

        Args:
            year: Year to search (default: current year)
            report_type: Type of report to filter (PTR, Annual, etc.)

        Returns:
            List of DisclosureRecord objects
        """
        if year is None:
            year = datetime.now().year

        try:
            search_params = {
                'FilingYear': year,
                'ReportType': report_type or '',
            }

            logger.info(f"Searching House disclosures for year {year}, type: {report_type or 'All'}")

            response = self.session.get(
                self.SEARCH_URL,
                params=search_params,
                timeout=60
            )
            response.raise_for_status()

            disclosures = self._parse_disclosure_results(response.content, year)

            time.sleep(self.delay_seconds)

            return disclosures

        except Exception as e:
            logger.error(f"Error searching disclosures for year {year}: {e}")
            return []

    def _parse_disclosure_results(self, content: bytes, year: int) -> List[DisclosureRecord]:
        """Parse disclosure search results from HTML"""
        soup = BeautifulSoup(content, 'html.parser')
        results = []

        # Look for results table
        table = soup.find('table', {'class': 'table'}) or soup.find('table', {'id': 'DataTable'})

        if not table:
            # Try finding any table with relevant content
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th', string=re.compile(r'Name|Filer', re.I)):
                    table = t
                    break

        if table:
            rows = table.find_all('tr')[1:]  # Skip header row

            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 4:  # Minimum expected columns
                    try:
                        disclosure = self._parse_disclosure_row(cells, year)
                        if disclosure:
                            results.append(disclosure)
                    except Exception as e:
                        logger.warning(f"Error parsing disclosure row: {e}")

        logger.info(f"Found {len(results)} disclosure records for year {year}")
        return results

    def _parse_disclosure_row(self, cells, year: int) -> Optional[DisclosureRecord]:
        """Parse a single disclosure row into a DisclosureRecord"""
        try:
            # Extract member name (usually first column)
            full_name = cells[0].get_text(strip=True)

            # Parse name components
            first_name, last_name = self._parse_name(full_name)

            # Extract other fields
            state = cells[1].get_text(strip=True) if len(cells) > 1 else ''
            district = cells[2].get_text(strip=True) if len(cells) > 2 else ''
            report_type = cells[3].get_text(strip=True) if len(cells) > 3 else ''
            filing_date = cells[4].get_text(strip=True) if len(cells) > 4 else ''

            # Extract document link
            document_url = ''
            link = cells[0].find('a') or row.find('a', href=re.compile(r'\.pdf|View|Download', re.I))
            if link and link.get('href'):
                document_url = urljoin(self.BASE_URL, link.get('href'))

            # Generate filing ID
            filing_id = self._generate_filing_id(full_name, filing_date, report_type)

            # Check for amendment
            is_amendment = 'amendment' in report_type.lower() if report_type else False

            return DisclosureRecord(
                filing_id=filing_id,
                member_name=full_name,
                first_name=first_name,
                last_name=last_name,
                state=state,
                district=district,
                filing_date=filing_date,
                report_type=report_type,
                document_url=document_url,
                year=year,
                chamber="house",
                is_amendment=is_amendment
            )

        except Exception as e:
            logger.warning(f"Error parsing disclosure row: {e}")
            return None

    def _parse_name(self, full_name: str) -> Tuple[str, str]:
        """Parse full name into first and last name"""
        if ',' in full_name:
            parts = full_name.split(',')
            last_name = parts[0].strip()
            first_name = parts[1].strip() if len(parts) > 1 else ''
        else:
            parts = full_name.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = parts[-1]
            else:
                first_name = ''
                last_name = full_name

        return first_name, last_name

    def _generate_filing_id(self, member_name: str, filing_date: str, report_type: str) -> str:
        """Generate a unique filing ID"""
        content = f"{member_name}{filing_date}{report_type}"
        hash_part = hashlib.md5(content.encode()).hexdigest()[:8]
        date_part = re.sub(r'[^\d]', '', filing_date)[:8] if filing_date else datetime.now().strftime('%Y%m%d')
        return f"H{date_part}{hash_part.upper()}"

    def check_for_new_filings(self, hours_back: int = 1) -> Tuple[List[DisclosureRecord], List[DisclosureRecord]]:
        """
        Check for new filings in the last N hours.

        Args:
            hours_back: Number of hours to look back

        Returns:
            Tuple of (new_filings, all_filings)
        """
        logger.info(f"Checking for new filings in the last {hours_back} hour(s)")

        current_year = datetime.now().year

        # Get recent filings
        all_filings = self.search_ptr_disclosures(year=current_year)

        # Filter to find truly new filings
        new_filings = []
        for filing in all_filings:
            if filing.filing_id not in self._seen_filings:
                new_filings.append(filing)
                self._mark_filing_seen(filing)
                logger.info(f"NEW FILING: {filing.member_name} - {filing.report_type} on {filing.filing_date}")

        # Log the scrape run
        self._log_scrape_run(len(all_filings), len(new_filings), "completed")

        if new_filings:
            logger.info(f"Found {len(new_filings)} new filings!")
        else:
            logger.info("No new filings detected")

        return new_filings, all_filings

    def get_recent_filings(self, days: int = 7) -> List[DisclosureRecord]:
        """
        Get all filings from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of DisclosureRecord objects
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        return self.search_ptr_disclosures(
            year=end_date.year,
            filing_date_start=start_date.strftime('%Y-%m-%d'),
            filing_date_end=end_date.strftime('%Y-%m-%d')
        )

    def parse_amount_range(self, amount_str: str) -> Tuple[int, int, int]:
        """
        Parse STOCK Act amount range string.

        Args:
            amount_str: Amount range string (e.g., "$15,001 - $50,000")

        Returns:
            Tuple of (min_amount, max_amount, mid_amount)
        """
        amount_str = amount_str.strip()

        # Check known ranges
        if amount_str in self.AMOUNT_RANGES:
            min_amt, max_amt = self.AMOUNT_RANGES[amount_str]
            mid_amt = (min_amt + max_amt) // 2
            return min_amt, max_amt, mid_amt

        # Try to parse custom range
        numbers = re.findall(r'[\d,]+', amount_str)
        if len(numbers) >= 2:
            min_amt = int(numbers[0].replace(',', ''))
            max_amt = int(numbers[1].replace(',', ''))
            mid_amt = (min_amt + max_amt) // 2
            return min_amt, max_amt, mid_amt
        elif len(numbers) == 1:
            amt = int(numbers[0].replace(',', ''))
            return amt, amt, amt

        return 0, 0, 0

    def get_all_recent_disclosures(self, years: int = 2) -> List[DisclosureRecord]:
        """
        Get all disclosure records from recent years.

        Args:
            years: Number of recent years to fetch

        Returns:
            Combined list of all disclosure records
        """
        all_disclosures = []
        available_years = self.get_available_years()
        target_years = available_years[:years]

        for year in target_years:
            logger.info(f"Fetching disclosures for year {year}")

            # Get PTR (Periodic Transaction Reports)
            ptr_disclosures = self.search_ptr_disclosures(year=year)
            all_disclosures.extend(ptr_disclosures)

            time.sleep(self.delay_seconds * 2)

        logger.info(f"Total disclosures collected: {len(all_disclosures)}")
        return all_disclosures

    def get_disclosure_stats(self) -> Dict[str, Any]:
        """Get statistics about scraped disclosures"""
        db_path = self.cache_dir / "house_disclosures.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM seen_filings")
        total_filings = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM seen_filings
            WHERE first_seen_at > datetime('now', '-24 hours')
        """)
        filings_last_24h = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(*) FROM seen_filings
            WHERE first_seen_at > datetime('now', '-1 hour')
        """)
        filings_last_hour = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*), status FROM scrape_runs GROUP BY status")
        run_stats = {row[1]: row[0] for row in cursor.fetchall()}

        conn.close()

        return {
            "total_filings_tracked": total_filings,
            "filings_last_24_hours": filings_last_24h,
            "filings_last_hour": filings_last_hour,
            "scrape_run_stats": run_stats,
            "cache_location": str(self.cache_dir)
        }

    def to_dataframe(self, disclosures: List[DisclosureRecord]) -> pd.DataFrame:
        """Convert disclosure records to pandas DataFrame"""
        if not disclosures:
            return pd.DataFrame()

        return pd.DataFrame([asdict(d) for d in disclosures])

    def save_to_csv(self, disclosures: List[DisclosureRecord], filename: str):
        """Save disclosure data to CSV file"""
        try:
            if not disclosures:
                logger.warning("No disclosures to save")
                return

            df = self.to_dataframe(disclosures)
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(disclosures)} disclosure records to {filename}")

        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")


def main():
    """Test the enhanced House disclosure scraper"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    scraper = HouseDisclosureScraper()

    print("House Financial Disclosure Scraper - Enhanced Real-Time Detection")
    print("=" * 70)

    # Test getting available years
    years = scraper.get_available_years()
    print(f"Available years: {years}")

    # Test checking for new filings
    print("\nChecking for new filings...")
    new_filings, all_filings = scraper.check_for_new_filings(hours_back=24)

    print(f"Total filings found: {len(all_filings)}")
    print(f"New filings: {len(new_filings)}")

    # Show first few new filings
    for i, filing in enumerate(new_filings[:5]):
        print(f"\n{i+1}. {filing.member_name} ({filing.state})")
        print(f"   Report: {filing.report_type}")
        print(f"   Filed: {filing.filing_date}")
        print(f"   ID: {filing.filing_id}")

    # Show stats
    print("\nScraping Statistics:")
    stats = scraper.get_disclosure_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test recent filings
    print("\nGetting filings from last 7 days...")
    recent = scraper.get_recent_filings(days=7)
    print(f"Found {len(recent)} filings in the last 7 days")


if __name__ == "__main__":
    main()
