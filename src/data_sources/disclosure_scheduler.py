#!/usr/bin/env python3
"""
Disclosure Polling Scheduler
Polls House and Senate disclosure websites every 15 minutes for new STOCK Act filings.
Triggers alerts when new filings are detected.
"""

import logging
import time
import threading
import schedule
import signal
import sys
from datetime import datetime
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
from queue import Queue
import sqlite3

from .house_disclosure_scraper import HouseDisclosureScraper, DisclosureRecord
from .senate_disclosure_scraper import SenateDisclosureScraper, SenateDisclosureRecord

logger = logging.getLogger(__name__)


@dataclass
class FilingAlert:
    """Represents an alert for a new filing"""
    filing_id: str
    chamber: str  # 'house' or 'senate'
    member_name: str
    state: str
    report_type: str
    filing_date: str
    document_url: str
    detected_at: str
    priority: str = "normal"  # normal, high (for late filings or large amounts)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DisclosureScheduler:
    """
    Automated scheduler for polling congressional disclosure websites.

    Features:
    - Polls House and Senate every 15 minutes
    - Detects new filings within 1 hour of publication
    - Triggers configurable callbacks on new filings
    - Maintains filing history and deduplication
    - Supports graceful shutdown
    """

    DEFAULT_POLL_INTERVAL = 15  # minutes

    def __init__(self,
                 poll_interval_minutes: int = DEFAULT_POLL_INTERVAL,
                 cache_dir: Optional[str] = None,
                 on_new_filing: Optional[Callable[[FilingAlert], None]] = None):
        """
        Initialize the disclosure scheduler.

        Args:
            poll_interval_minutes: How often to poll for new filings (default: 15)
            cache_dir: Directory for caching data
            on_new_filing: Callback function called when new filing is detected
        """
        self.poll_interval = poll_interval_minutes
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".congressional_trading_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scrapers
        self.house_scraper = HouseDisclosureScraper(cache_dir=str(self.cache_dir))
        self.senate_scraper = SenateDisclosureScraper(cache_dir=str(self.cache_dir))

        # Callback for new filings
        self.on_new_filing = on_new_filing

        # Alert queue for async processing
        self.alert_queue: Queue = Queue()

        # Running state
        self._running = False
        self._stop_event = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._alert_thread: Optional[threading.Thread] = None

        # Stats tracking
        self._stats = {
            'total_runs': 0,
            'total_house_filings': 0,
            'total_senate_filings': 0,
            'new_house_filings': 0,
            'new_senate_filings': 0,
            'last_run': None,
            'errors': 0,
            'started_at': None
        }

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize the scheduler database"""
        db_path = self.cache_dir / "scheduler.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filing_id TEXT UNIQUE,
                chamber TEXT,
                member_name TEXT,
                state TEXT,
                report_type TEXT,
                filing_date TEXT,
                document_url TEXT,
                detected_at TEXT,
                priority TEXT,
                metadata TEXT,
                processed BOOLEAN DEFAULT FALSE,
                processed_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scheduler_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT,
                completed_at TEXT,
                house_filings_found INTEGER,
                house_new_filings INTEGER,
                senate_filings_found INTEGER,
                senate_new_filings INTEGER,
                status TEXT,
                error_message TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _save_alert(self, alert: FilingAlert):
        """Save an alert to the database"""
        db_path = self.cache_dir / "scheduler.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        try:
            cursor.execute("""
                INSERT OR REPLACE INTO alerts
                (filing_id, chamber, member_name, state, report_type, filing_date,
                 document_url, detected_at, priority, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.filing_id, alert.chamber, alert.member_name, alert.state,
                alert.report_type, alert.filing_date, alert.document_url,
                alert.detected_at, alert.priority, json.dumps(alert.metadata)
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
        finally:
            conn.close()

    def _log_run(self, house_found: int, house_new: int,
                 senate_found: int, senate_new: int,
                 status: str, error_message: str = None):
        """Log a scheduler run"""
        db_path = self.cache_dir / "scheduler.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO scheduler_runs
            (started_at, completed_at, house_filings_found, house_new_filings,
             senate_filings_found, senate_new_filings, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(), datetime.now().isoformat(),
            house_found, house_new, senate_found, senate_new,
            status, error_message
        ))

        conn.commit()
        conn.close()

    def _create_alert(self, disclosure, chamber: str) -> FilingAlert:
        """Create an alert from a disclosure record"""
        if chamber == "house":
            return FilingAlert(
                filing_id=disclosure.filing_id,
                chamber="house",
                member_name=disclosure.member_name,
                state=disclosure.state,
                report_type=disclosure.report_type,
                filing_date=disclosure.filing_date,
                document_url=disclosure.document_url,
                detected_at=datetime.now().isoformat(),
                priority="normal",
                metadata={
                    'district': disclosure.district,
                    'is_amendment': disclosure.is_amendment
                }
            )
        else:  # senate
            return FilingAlert(
                filing_id=disclosure.filing_id,
                chamber="senate",
                member_name=disclosure.member_name,
                state=disclosure.state,
                report_type=disclosure.report_type,
                filing_date=disclosure.filing_date,
                document_url=disclosure.document_url,
                detected_at=datetime.now().isoformat(),
                priority="normal",
                metadata={
                    'office': disclosure.office,
                    'filer_type': disclosure.filer_type,
                    'is_amendment': disclosure.is_amendment
                }
            )

    def poll_disclosures(self):
        """
        Poll both House and Senate for new disclosures.
        This is the main polling function called by the scheduler.
        """
        logger.info("Starting disclosure poll...")
        self._stats['total_runs'] += 1
        self._stats['last_run'] = datetime.now().isoformat()

        house_found = 0
        house_new = 0
        senate_found = 0
        senate_new = 0
        error_message = None

        try:
            # Poll House disclosures
            logger.info("Polling House disclosures...")
            new_house, all_house = self.house_scraper.check_for_new_filings(hours_back=1)
            house_found = len(all_house)
            house_new = len(new_house)

            for disclosure in new_house:
                alert = self._create_alert(disclosure, "house")
                self._save_alert(alert)
                self.alert_queue.put(alert)
                self._stats['new_house_filings'] += 1
                logger.info(f"House alert: {alert.member_name} - {alert.report_type}")

            self._stats['total_house_filings'] += house_found

        except Exception as e:
            logger.error(f"Error polling House disclosures: {e}")
            error_message = f"House error: {str(e)}"
            self._stats['errors'] += 1

        try:
            # Poll Senate disclosures
            logger.info("Polling Senate disclosures...")
            new_senate, all_senate = self.senate_scraper.check_for_new_filings(hours_back=1)
            senate_found = len(all_senate)
            senate_new = len(new_senate)

            for disclosure in new_senate:
                alert = self._create_alert(disclosure, "senate")
                self._save_alert(alert)
                self.alert_queue.put(alert)
                self._stats['new_senate_filings'] += 1
                logger.info(f"Senate alert: {alert.member_name} - {alert.report_type}")

            self._stats['total_senate_filings'] += senate_found

        except Exception as e:
            logger.error(f"Error polling Senate disclosures: {e}")
            if error_message:
                error_message += f"; Senate error: {str(e)}"
            else:
                error_message = f"Senate error: {str(e)}"
            self._stats['errors'] += 1

        # Log the run
        status = "completed" if not error_message else "completed_with_errors"
        self._log_run(house_found, house_new, senate_found, senate_new, status, error_message)

        logger.info(f"Poll complete. House: {house_new}/{house_found} new, Senate: {senate_new}/{senate_found} new")

    def _alert_processor(self):
        """Process alerts from the queue and call callbacks"""
        while not self._stop_event.is_set():
            try:
                # Wait for alert with timeout to check stop event
                try:
                    alert = self.alert_queue.get(timeout=1.0)
                except:
                    continue

                # Call the callback if configured
                if self.on_new_filing:
                    try:
                        self.on_new_filing(alert)
                    except Exception as e:
                        logger.error(f"Error in new filing callback: {e}")

                # Mark alert as processed
                db_path = self.cache_dir / "scheduler.db"
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE alerts SET processed = TRUE, processed_at = ?
                    WHERE filing_id = ?
                """, (datetime.now().isoformat(), alert.filing_id))
                conn.commit()
                conn.close()

            except Exception as e:
                logger.error(f"Error processing alert: {e}")

    def _scheduler_loop(self):
        """Run the scheduler loop"""
        # Schedule the poll job
        schedule.every(self.poll_interval).minutes.do(self.poll_disclosures)

        # Run initial poll
        self.poll_disclosures()

        # Run the scheduler
        while not self._stop_event.is_set():
            schedule.run_pending()
            time.sleep(1)

        logger.info("Scheduler loop stopped")

    def start(self, blocking: bool = False):
        """
        Start the disclosure scheduler.

        Args:
            blocking: If True, block the main thread. If False, run in background.
        """
        if self._running:
            logger.warning("Scheduler is already running")
            return

        logger.info(f"Starting disclosure scheduler (poll every {self.poll_interval} minutes)")
        self._running = True
        self._stop_event.clear()
        self._stats['started_at'] = datetime.now().isoformat()

        # Start alert processor thread
        self._alert_thread = threading.Thread(target=self._alert_processor, daemon=True)
        self._alert_thread.start()

        if blocking:
            # Run in main thread
            self._scheduler_loop()
        else:
            # Run in background thread
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._scheduler_thread.start()

        logger.info("Disclosure scheduler started")

    def stop(self):
        """Stop the disclosure scheduler gracefully"""
        if not self._running:
            logger.warning("Scheduler is not running")
            return

        logger.info("Stopping disclosure scheduler...")
        self._running = False
        self._stop_event.set()

        # Wait for threads to finish
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=5.0)

        if self._alert_thread and self._alert_thread.is_alive():
            self._alert_thread.join(timeout=5.0)

        schedule.clear()
        logger.info("Disclosure scheduler stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        return {
            **self._stats,
            'is_running': self._running,
            'poll_interval_minutes': self.poll_interval,
            'pending_alerts': self.alert_queue.qsize(),
            'house_scraper_stats': self.house_scraper.get_disclosure_stats(),
            'senate_scraper_stats': self.senate_scraper.get_disclosure_stats()
        }

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts from the database"""
        db_path = self.cache_dir / "scheduler.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("""
            SELECT filing_id, chamber, member_name, state, report_type,
                   filing_date, document_url, detected_at, priority, processed
            FROM alerts
            ORDER BY detected_at DESC
            LIMIT ?
        """, (limit,))

        alerts = []
        for row in cursor.fetchall():
            alerts.append({
                'filing_id': row[0],
                'chamber': row[1],
                'member_name': row[2],
                'state': row[3],
                'report_type': row[4],
                'filing_date': row[5],
                'document_url': row[6],
                'detected_at': row[7],
                'priority': row[8],
                'processed': bool(row[9])
            })

        conn.close()
        return alerts

    def run_once(self) -> Dict[str, Any]:
        """
        Run a single poll cycle (useful for testing or manual triggering).

        Returns:
            Dictionary with results of the poll
        """
        logger.info("Running single disclosure poll...")

        results = {
            'house': {'found': 0, 'new': 0, 'filings': []},
            'senate': {'found': 0, 'new': 0, 'filings': []},
            'timestamp': datetime.now().isoformat(),
            'errors': []
        }

        try:
            new_house, all_house = self.house_scraper.check_for_new_filings(hours_back=1)
            results['house']['found'] = len(all_house)
            results['house']['new'] = len(new_house)
            results['house']['filings'] = [
                {'member': f.member_name, 'type': f.report_type, 'date': f.filing_date}
                for f in new_house
            ]
        except Exception as e:
            results['errors'].append(f"House: {str(e)}")

        try:
            new_senate, all_senate = self.senate_scraper.check_for_new_filings(hours_back=1)
            results['senate']['found'] = len(all_senate)
            results['senate']['new'] = len(new_senate)
            results['senate']['filings'] = [
                {'member': f.member_name, 'type': f.report_type, 'date': f.filing_date}
                for f in new_senate
            ]
        except Exception as e:
            results['errors'].append(f"Senate: {str(e)}")

        return results


def default_alert_handler(alert: FilingAlert):
    """Default handler for new filing alerts - prints to console"""
    print(f"\n{'='*60}")
    print(f"NEW {alert.chamber.upper()} FILING DETECTED!")
    print(f"{'='*60}")
    print(f"Member: {alert.member_name} ({alert.state})")
    print(f"Report Type: {alert.report_type}")
    print(f"Filing Date: {alert.filing_date}")
    print(f"Document: {alert.document_url}")
    print(f"Detected At: {alert.detected_at}")
    print(f"{'='*60}\n")


def main():
    """Run the disclosure scheduler"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create scheduler with default alert handler
    scheduler = DisclosureScheduler(
        poll_interval_minutes=15,
        on_new_filing=default_alert_handler
    )

    # Handle graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal")
        scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("Congressional Disclosure Scheduler")
    print("=" * 50)
    print(f"Poll Interval: {scheduler.poll_interval} minutes")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    # Start scheduler in blocking mode
    scheduler.start(blocking=True)


if __name__ == "__main__":
    main()
