#!/usr/bin/env python3
"""
Senate Financial Disclosure Scraper - Real-Time Detection
Detects new STOCK Act filings from Senate within 1 hour of publication.

Senate disclosures are accessed through the Electronic Financial Disclosures (eFD) system:
https://efdsearch.senate.gov/
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
from urllib.parse import urljoin, urlencode
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class SenateReportType(Enum):
    """Senate STOCK Act report types"""
    PERIODIC_TRANSACTION = "Periodic Transaction Report"
    ANNUAL = "Annual Report"
    AMENDMENT = "Amendment"
    TERMINATION = "Termination Report"
    NEW_FILER = "New Filer Report"
    BLIND_TRUST = "Blind Trust"
    EXTENSION = "Extension"
    DUE_DATE_EXTENSION = "Due Date Extension"


@dataclass
class SenateDisclosureRecord:
    """Represents a single Senate disclosure filing"""
    filing_id: str
    member_name: str
    first_name: str
    last_name: str
    state: str
    office: str  # Senator's office designation
    filing_date: str
    report_type: str
    document_url: str
    year: int
    chamber: str = "senate"
    is_amendment: bool = False
    filer_type: str = "Senator"  # Could be Senator, Candidate, Former Senator
    checksum: str = ""

    def __post_init__(self):
        """Generate checksum for deduplication"""
        if not self.checksum:
            content = f"{self.member_name}{self.filing_date}{self.report_type}{self.document_url}"
            self.checksum = hashlib.md5(content.encode()).hexdigest()


@dataclass
class SenateTradeRecord:
    """Represents a single trade from a Senate disclosure"""
    filing_id: str
    member_name: str
    symbol: str
    asset_name: str
    asset_type: str  # Stock, Bond, Option, etc.
    transaction_type: str  # Purchase, Sale, Exchange
    transaction_date: str
    filing_date: str
    notification_date: str
    amount_min: int
    amount_max: int
    amount_mid: int
    owner: str  # Self, Spouse, Dependent Child, Joint
    comment: str
    filing_delay_days: int
    source_document: str
    data_source: str = "senate_efd"


class SenateDisclosureScraper:
    """
    Scrapes financial disclosure data from the Senate eFD (Electronic Financial Disclosures) system.
    Enhanced for real-time detection with polling support.

    Public records - completely legal and ethical data collection.

    Note: The Senate eFD system has specific search requirements and may have
    rate limiting in place. This scraper respects those limits.
    """

    BASE_URL = "https://efdsearch.senate.gov"
    SEARCH_URL = f"{BASE_URL}/search/"
    REPORT_URL = f"{BASE_URL}/search/view/"
    AGREEMENT_URL = f"{BASE_URL}/search/home/"

    # Amount range mappings (Senate uses same ranges as House)
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

    # State abbreviations for validation
    STATES = [
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
        'DC', 'PR', 'VI', 'GU', 'AS'
    ]

    def __init__(self, delay_seconds: float = 2.0, cache_dir: Optional[str] = None):
        """
        Initialize Senate disclosure scraper.

        Args:
            delay_seconds: Delay between requests (Senate eFD is sensitive to rapid requests)
            cache_dir: Directory for caching scraped data
        """
        self.delay_seconds = delay_seconds
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Congressional Trading Intelligence Research Tool/2.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })

        # Cache setup
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".congressional_trading_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_cache_db()

        # Track seen filings
        self._seen_filings: Set[str] = set()
        self._load_seen_filings()

        # Agreement acceptance status
        self._agreement_accepted = False

    def _init_cache_db(self):
        """Initialize SQLite cache database for Senate disclosures"""
        db_path = self.cache_dir / "senate_disclosures.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS seen_filings (
                filing_id TEXT PRIMARY KEY,
                checksum TEXT,
                first_seen_at TEXT,
                member_name TEXT,
                filing_date TEXT,
                report_type TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scrape_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                completed_at TEXT,
                filings_found INTEGER,
                new_filings INTEGER,
                status TEXT,
                error_message TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filing_id TEXT,
                symbol TEXT,
                asset_name TEXT,
                transaction_type TEXT,
                transaction_date TEXT,
                amount_min INTEGER,
                amount_max INTEGER,
                owner TEXT,
                created_at TEXT,
                UNIQUE(filing_id, symbol, transaction_date, transaction_type)
            )
        """)

        conn.commit()
        conn.close()

    def _load_seen_filings(self):
        """Load previously seen filings from cache"""
        db_path = self.cache_dir / "senate_disclosures.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT filing_id FROM seen_filings")
        self._seen_filings = {row[0] for row in cursor.fetchall()}

        conn.close()
        logger.info(f"Loaded {len(self._seen_filings)} seen Senate filings from cache")

    def _mark_filing_seen(self, disclosure: SenateDisclosureRecord):
        """Mark a filing as seen in cache"""
        db_path = self.cache_dir / "senate_disclosures.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO seen_filings
            (filing_id, checksum, first_seen_at, member_name, filing_date, report_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (disclosure.filing_id, disclosure.checksum, datetime.now().isoformat(),
              disclosure.member_name, disclosure.filing_date, disclosure.report_type))

        conn.commit()
        conn.close()

        self._seen_filings.add(disclosure.filing_id)

    def _log_scrape_run(self, filings_found: int, new_filings: int,
                        status: str, error_message: str = None) -> int:
        """Log a scraping run"""
        db_path = self.cache_dir / "senate_disclosures.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO scrape_runs
            (started_at, completed_at, filings_found, new_filings, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now().isoformat(), datetime.now().isoformat(),
              filings_found, new_filings, status, error_message))

        run_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return run_id

    def _accept_agreement(self) -> bool:
        """
        Accept the Senate eFD user agreement.
        The eFD system requires accepting terms before searching.
        """
        if self._agreement_accepted:
            return True

        try:
            # First, get the agreement page to get any CSRF tokens
            response = self.session.get(self.AGREEMENT_URL, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Look for CSRF token
            csrf_token = None
            csrf_input = soup.find('input', {'name': 'csrfmiddlewaretoken'})
            if csrf_input:
                csrf_token = csrf_input.get('value')

            # Submit agreement form
            agreement_data = {
                'prohibition_agreement': '1',
                'csrfmiddlewaretoken': csrf_token or '',
            }

            headers = {
                'Referer': self.AGREEMENT_URL,
                'Content-Type': 'application/x-www-form-urlencoded',
            }

            response = self.session.post(
                self.AGREEMENT_URL,
                data=agreement_data,
                headers=headers,
                timeout=30
            )

            if response.status_code == 200:
                self._agreement_accepted = True
                logger.info("Senate eFD agreement accepted")
                return True

            logger.warning(f"Failed to accept Senate eFD agreement: {response.status_code}")
            return False

        except Exception as e:
            logger.error(f"Error accepting Senate eFD agreement: {e}")
            return False

    def search_periodic_transaction_reports(self,
                                           year: int = None,
                                           last_name: str = None,
                                           state: str = None,
                                           from_date: str = None,
                                           to_date: str = None) -> List[SenateDisclosureRecord]:
        """
        Search for Periodic Transaction Reports (PTR) from Senate.

        Args:
            year: Filing year (default: current year)
            last_name: Senator's last name to filter
            state: State abbreviation to filter
            from_date: Start date for search (MM/DD/YYYY format)
            to_date: End date for search (MM/DD/YYYY format)

        Returns:
            List of SenateDisclosureRecord objects
        """
        if year is None:
            year = datetime.now().year

        # Accept agreement first
        if not self._accept_agreement():
            logger.warning("Could not accept Senate eFD agreement, proceeding anyway")

        time.sleep(self.delay_seconds)

        try:
            # Build search parameters
            search_params = {
                'submitted': 'true',
                'filer_type': 'Senator',
                'report_type': 'ptr',  # Periodic Transaction Report
                'first_name': '',
                'last_name': last_name or '',
                'state': state if state in self.STATES else '',
                'senator_state': state if state in self.STATES else '',
            }

            # Add date filters if provided
            if from_date:
                search_params['submitted_start_date'] = from_date
            if to_date:
                search_params['submitted_end_date'] = to_date

            logger.info(f"Searching Senate PTR disclosures for year {year}")

            response = self.session.get(
                self.SEARCH_URL,
                params=search_params,
                timeout=60
            )
            response.raise_for_status()

            disclosures = self._parse_search_results(response.content, year)

            time.sleep(self.delay_seconds)

            return disclosures

        except Exception as e:
            logger.error(f"Error searching Senate PTR disclosures: {e}")
            return []

    def search_all_disclosures(self,
                               year: int = None,
                               report_type: str = None,
                               filer_type: str = "Senator") -> List[SenateDisclosureRecord]:
        """
        Search for all types of Senate financial disclosures.

        Args:
            year: Filing year (default: current year)
            report_type: Type of report to filter (ptr, annual, amendment)
            filer_type: Type of filer (Senator, Candidate, Former Senator)

        Returns:
            List of SenateDisclosureRecord objects
        """
        if year is None:
            year = datetime.now().year

        if not self._accept_agreement():
            logger.warning("Could not accept Senate eFD agreement")

        time.sleep(self.delay_seconds)

        try:
            search_params = {
                'submitted': 'true',
                'filer_type': filer_type,
                'report_type': report_type or '',
            }

            logger.info(f"Searching Senate disclosures: year={year}, type={report_type or 'All'}")

            response = self.session.get(
                self.SEARCH_URL,
                params=search_params,
                timeout=60
            )
            response.raise_for_status()

            disclosures = self._parse_search_results(response.content, year)

            time.sleep(self.delay_seconds)

            return disclosures

        except Exception as e:
            logger.error(f"Error searching Senate disclosures: {e}")
            return []

    def _parse_search_results(self, content: bytes, year: int) -> List[SenateDisclosureRecord]:
        """Parse search results from Senate eFD"""
        soup = BeautifulSoup(content, 'html.parser')
        results = []

        # Look for results table
        table = soup.find('table', {'class': 'table'}) or soup.find('table', {'id': 'results'})

        if not table:
            # Try finding data table
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th', string=re.compile(r'Name|Filer|Senator', re.I)):
                    table = t
                    break

        if table:
            rows = table.find_all('tr')[1:]  # Skip header

            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    try:
                        disclosure = self._parse_result_row(cells, row, year)
                        if disclosure:
                            results.append(disclosure)
                    except Exception as e:
                        logger.warning(f"Error parsing Senate disclosure row: {e}")

        # Also check for JSON data (some Senate pages use JavaScript rendering)
        script_data = soup.find('script', string=re.compile(r'var\s+tableData'))
        if script_data:
            try:
                json_match = re.search(r'var\s+tableData\s*=\s*(\[.*?\]);', script_data.string, re.DOTALL)
                if json_match:
                    table_data = json.loads(json_match.group(1))
                    for row_data in table_data:
                        disclosure = self._parse_json_row(row_data, year)
                        if disclosure and disclosure.filing_id not in {r.filing_id for r in results}:
                            results.append(disclosure)
            except Exception as e:
                logger.warning(f"Error parsing JSON table data: {e}")

        logger.info(f"Found {len(results)} Senate disclosure records")
        return results

    def _parse_result_row(self, cells, row, year: int) -> Optional[SenateDisclosureRecord]:
        """Parse a single result row into a SenateDisclosureRecord"""
        try:
            # Extract name (usually first column with link)
            name_cell = cells[0]
            full_name = name_cell.get_text(strip=True)
            first_name, last_name = self._parse_name(full_name)

            # Extract other fields based on Senate eFD layout
            office = cells[1].get_text(strip=True) if len(cells) > 1 else ''
            report_type = cells[2].get_text(strip=True) if len(cells) > 2 else ''
            filing_date = cells[3].get_text(strip=True) if len(cells) > 3 else ''

            # Extract state from office or name
            state = self._extract_state(office, full_name)

            # Get document URL
            document_url = ''
            link = name_cell.find('a') or row.find('a', href=True)
            if link and link.get('href'):
                href = link.get('href')
                if href.startswith('/'):
                    document_url = urljoin(self.BASE_URL, href)
                else:
                    document_url = href

            # Generate filing ID
            filing_id = self._generate_filing_id(full_name, filing_date, report_type)

            # Check for amendment
            is_amendment = 'amendment' in report_type.lower() if report_type else False

            # Determine filer type
            filer_type = "Senator"
            if 'candidate' in office.lower():
                filer_type = "Candidate"
            elif 'former' in office.lower():
                filer_type = "Former Senator"

            return SenateDisclosureRecord(
                filing_id=filing_id,
                member_name=full_name,
                first_name=first_name,
                last_name=last_name,
                state=state,
                office=office,
                filing_date=filing_date,
                report_type=report_type,
                document_url=document_url,
                year=year,
                chamber="senate",
                is_amendment=is_amendment,
                filer_type=filer_type
            )

        except Exception as e:
            logger.warning(f"Error parsing Senate result row: {e}")
            return None

    def _parse_json_row(self, row_data: Dict, year: int) -> Optional[SenateDisclosureRecord]:
        """Parse a JSON row from JavaScript-rendered table"""
        try:
            full_name = row_data.get('name', '') or row_data.get('filer_name', '')
            first_name, last_name = self._parse_name(full_name)

            office = row_data.get('office', '') or row_data.get('senator_office', '')
            report_type = row_data.get('report_type', '') or row_data.get('type', '')
            filing_date = row_data.get('date_received', '') or row_data.get('filing_date', '')

            state = self._extract_state(office, full_name)
            state = row_data.get('state', state) or state

            document_url = row_data.get('report_url', '') or row_data.get('url', '')
            if document_url and document_url.startswith('/'):
                document_url = urljoin(self.BASE_URL, document_url)

            filing_id = self._generate_filing_id(full_name, filing_date, report_type)
            is_amendment = 'amendment' in report_type.lower() if report_type else False

            return SenateDisclosureRecord(
                filing_id=filing_id,
                member_name=full_name,
                first_name=first_name,
                last_name=last_name,
                state=state,
                office=office,
                filing_date=filing_date,
                report_type=report_type,
                document_url=document_url,
                year=year,
                chamber="senate",
                is_amendment=is_amendment,
                filer_type=row_data.get('filer_type', 'Senator')
            )

        except Exception as e:
            logger.warning(f"Error parsing JSON row: {e}")
            return None

    def _parse_name(self, full_name: str) -> Tuple[str, str]:
        """Parse full name into first and last name"""
        full_name = full_name.strip()

        # Handle "Last, First" format
        if ',' in full_name:
            parts = full_name.split(',')
            last_name = parts[0].strip()
            first_name = parts[1].strip().split()[0] if len(parts) > 1 else ''
        else:
            # Handle "First Last" format
            parts = full_name.split()
            if len(parts) >= 2:
                first_name = parts[0]
                last_name = parts[-1]
            else:
                first_name = ''
                last_name = full_name

        return first_name, last_name

    def _extract_state(self, office: str, name: str) -> str:
        """Extract state abbreviation from office or name string"""
        # Check for state abbreviation pattern
        for state in self.STATES:
            if f"({state})" in office or f"({state})" in name:
                return state
            if f"-{state}" in office or f"-{state}" in name:
                return state
            if f" {state} " in office:
                return state

        # Try to find state in office string
        state_match = re.search(r'\b([A-Z]{2})\b', office)
        if state_match and state_match.group(1) in self.STATES:
            return state_match.group(1)

        return ''

    def _generate_filing_id(self, member_name: str, filing_date: str, report_type: str) -> str:
        """Generate a unique filing ID"""
        content = f"{member_name}{filing_date}{report_type}"
        hash_part = hashlib.md5(content.encode()).hexdigest()[:8]
        date_part = re.sub(r'[^\d]', '', filing_date)[:8] if filing_date else datetime.now().strftime('%Y%m%d')
        return f"S{date_part}{hash_part.upper()}"

    def check_for_new_filings(self, hours_back: int = 1) -> Tuple[List[SenateDisclosureRecord], List[SenateDisclosureRecord]]:
        """
        Check for new Senate filings in the last N hours.

        Args:
            hours_back: Number of hours to look back

        Returns:
            Tuple of (new_filings, all_filings)
        """
        logger.info(f"Checking for new Senate filings in the last {hours_back} hour(s)")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back * 24)  # Get a wider range to be safe

        # Search for PTR disclosures
        all_filings = self.search_periodic_transaction_reports(
            from_date=start_date.strftime('%m/%d/%Y'),
            to_date=end_date.strftime('%m/%d/%Y')
        )

        # Filter to find new filings
        new_filings = []
        for filing in all_filings:
            if filing.filing_id not in self._seen_filings:
                new_filings.append(filing)
                self._mark_filing_seen(filing)
                logger.info(f"NEW SENATE FILING: {filing.member_name} - {filing.report_type} on {filing.filing_date}")

        # Log the run
        self._log_scrape_run(len(all_filings), len(new_filings), "completed")

        if new_filings:
            logger.info(f"Found {len(new_filings)} new Senate filings!")
        else:
            logger.info("No new Senate filings detected")

        return new_filings, all_filings

    def get_recent_filings(self, days: int = 7) -> List[SenateDisclosureRecord]:
        """
        Get all Senate filings from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of SenateDisclosureRecord objects
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        return self.search_periodic_transaction_reports(
            from_date=start_date.strftime('%m/%d/%Y'),
            to_date=end_date.strftime('%m/%d/%Y')
        )

    def get_all_recent_disclosures(self, years: int = 2) -> List[SenateDisclosureRecord]:
        """
        Get all Senate disclosure records from recent years.

        Args:
            years: Number of years to fetch

        Returns:
            Combined list of all disclosure records
        """
        all_disclosures = []
        current_year = datetime.now().year

        for year in range(current_year, current_year - years, -1):
            logger.info(f"Fetching Senate disclosures for year {year}")

            # Calculate date range for the year
            start_date = f"01/01/{year}"
            end_date = f"12/31/{year}" if year < current_year else datetime.now().strftime('%m/%d/%Y')

            disclosures = self.search_periodic_transaction_reports(
                year=year,
                from_date=start_date,
                to_date=end_date
            )
            all_disclosures.extend(disclosures)

            time.sleep(self.delay_seconds * 2)

        logger.info(f"Total Senate disclosures collected: {len(all_disclosures)}")
        return all_disclosures

    def get_disclosure_stats(self) -> Dict[str, Any]:
        """Get statistics about scraped Senate disclosures"""
        db_path = self.cache_dir / "senate_disclosures.db"
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

        cursor.execute("""
            SELECT report_type, COUNT(*) FROM seen_filings
            GROUP BY report_type ORDER BY COUNT(*) DESC
        """)
        type_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "total_filings_tracked": total_filings,
            "filings_last_24_hours": filings_last_24h,
            "filings_last_hour": filings_last_hour,
            "scrape_run_stats": run_stats,
            "report_type_breakdown": type_breakdown,
            "cache_location": str(self.cache_dir)
        }

    def parse_amount_range(self, amount_str: str) -> Tuple[int, int, int]:
        """Parse STOCK Act amount range string"""
        amount_str = amount_str.strip()

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

    def to_dataframe(self, disclosures: List[SenateDisclosureRecord]) -> pd.DataFrame:
        """Convert disclosure records to pandas DataFrame"""
        if not disclosures:
            return pd.DataFrame()

        return pd.DataFrame([asdict(d) for d in disclosures])

    def save_to_csv(self, disclosures: List[SenateDisclosureRecord], filename: str):
        """Save disclosure data to CSV file"""
        try:
            if not disclosures:
                logger.warning("No disclosures to save")
                return

            df = self.to_dataframe(disclosures)
            df.to_csv(filename, index=False)
            logger.info(f"Saved {len(disclosures)} Senate disclosure records to {filename}")

        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")


def main():
    """Test the Senate disclosure scraper"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    scraper = SenateDisclosureScraper()

    print("Senate Financial Disclosure Scraper - Real-Time Detection")
    print("=" * 70)

    # Test checking for new filings
    print("\nChecking for new Senate filings...")
    new_filings, all_filings = scraper.check_for_new_filings(hours_back=24)

    print(f"Total filings found: {len(all_filings)}")
    print(f"New filings: {len(new_filings)}")

    # Show first few filings
    for i, filing in enumerate((new_filings or all_filings)[:5]):
        print(f"\n{i+1}. {filing.member_name} ({filing.state})")
        print(f"   Office: {filing.office}")
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
    print(f"Found {len(recent)} Senate filings in the last 7 days")


if __name__ == "__main__":
    main()
