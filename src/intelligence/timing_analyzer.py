#!/usr/bin/env python3
"""
Congressional Trading Intelligence System
Track A - Task A2: Timing Analysis Engine

This module identifies trades occurring suspiciously close to market-moving events
the member had plausible access to.

Event Types Analyzed:
- Committee hearings
- Bill introductions/votes affecting sector
- Classified briefings (by committee membership)
- Earnings announcements
- FDA decisions
- Defense contract announcements
- Major regulatory announcements

Output includes:
- TimingAnalysis with suspicious events correlation
- Timing score (0-1) indicating suspicion level
- Human-readable summary
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of events that can correlate with trades."""
    COMMITTEE_HEARING = "committee_hearing"
    BILL_INTRODUCTION = "bill_introduction"
    BILL_VOTE = "bill_vote"
    CLASSIFIED_BRIEFING = "classified_briefing"
    EARNINGS_ANNOUNCEMENT = "earnings_announcement"
    FDA_DECISION = "fda_decision"
    DEFENSE_CONTRACT = "defense_contract"
    REGULATORY_ANNOUNCEMENT = "regulatory_announcement"
    EXECUTIVE_ORDER = "executive_order"
    PANDEMIC_BRIEFING = "pandemic_briefing"
    CRISIS_BRIEFING = "crisis_briefing"


class AccessLevel(Enum):
    """Member's level of access to event information."""
    DIRECT = "direct"          # Member directly involved (chair, sponsor, briefed)
    COMMITTEE = "committee"    # Member on relevant committee
    CHAMBER = "chamber"        # All members of chamber had access
    PUBLIC = "public"          # Public information


@dataclass
class SuspiciousEvent:
    """An event that may correlate with a trade."""
    event_type: EventType
    event_date: date
    days_before_trade: int      # Negative = event before trade
    days_after_trade: int       # Positive = event after trade
    member_access_level: AccessLevel
    description: str
    affected_symbols: List[str]
    affected_sectors: List[str]
    information_advantage_score: float  # 0-1, how much advantage member had
    event_id: Optional[str] = None

    @property
    def is_suspicious_timing(self) -> bool:
        """Check if timing is suspicious (trade before event)."""
        # Suspicious if trade occurred 0-30 days before a positive event
        return self.days_before_trade >= 0 and self.days_before_trade <= 30

    @property
    def time_delta(self) -> int:
        """Return days between trade and event (negative = trade before event)."""
        return -self.days_before_trade if self.days_before_trade > 0 else self.days_after_trade


@dataclass
class TimingAnalysis:
    """Complete timing analysis for a trade."""
    trade_id: str
    member_id: str
    member_name: str
    symbol: str
    transaction_date: date
    suspicious_events: List[SuspiciousEvent]
    timing_score: float  # 0-1, higher = more suspicious
    summary: str
    risk_factors: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_suspicious_timing(self) -> bool:
        """Check if any suspicious timing was detected."""
        return len(self.suspicious_events) > 0 and self.timing_score > 0.5

    @property
    def most_suspicious_event(self) -> Optional[SuspiciousEvent]:
        """Return the most suspicious event correlation."""
        if not self.suspicious_events:
            return None
        return max(self.suspicious_events, key=lambda e: e.information_advantage_score)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "member_id": self.member_id,
            "member_name": self.member_name,
            "symbol": self.symbol,
            "transaction_date": str(self.transaction_date),
            "suspicious_events": [
                {
                    "event_type": e.event_type.value,
                    "event_date": str(e.event_date),
                    "days_before_trade": e.days_before_trade,
                    "days_after_trade": e.days_after_trade,
                    "member_access_level": e.member_access_level.value,
                    "description": e.description,
                    "information_advantage_score": e.information_advantage_score
                }
                for e in self.suspicious_events
            ],
            "timing_score": round(self.timing_score, 3),
            "summary": self.summary,
            "risk_factors": self.risk_factors,
            "timestamp": self.timestamp.isoformat()
        }


# Stock to sector mapping for event correlation
STOCK_SECTORS = {
    # Technology
    "AAPL": ["Technology", "Consumer Electronics"],
    "MSFT": ["Technology", "Software", "Cloud"],
    "GOOGL": ["Technology", "Advertising", "AI"],
    "AMZN": ["Technology", "E-commerce", "Cloud"],
    "META": ["Technology", "Social Media", "Advertising"],
    "NVDA": ["Technology", "Semiconductors", "AI"],
    "TSLA": ["Technology", "Automotive", "Energy"],

    # Financials
    "JPM": ["Financials", "Banking"],
    "BAC": ["Financials", "Banking"],
    "WFC": ["Financials", "Banking"],
    "GS": ["Financials", "Investment Banking"],
    "MS": ["Financials", "Investment Banking"],
    "COIN": ["Financials", "Cryptocurrency"],

    # Healthcare/Pharma
    "UNH": ["Healthcare", "Insurance"],
    "JNJ": ["Healthcare", "Pharmaceuticals"],
    "PFE": ["Healthcare", "Pharmaceuticals"],
    "MRNA": ["Healthcare", "Biotechnology", "Vaccines"],
    "GILD": ["Healthcare", "Biotechnology"],
    "ABBV": ["Healthcare", "Pharmaceuticals"],

    # Energy
    "XOM": ["Energy", "Oil & Gas"],
    "CVX": ["Energy", "Oil & Gas"],
    "COP": ["Energy", "Oil & Gas"],

    # Defense
    "LMT": ["Defense", "Aerospace"],
    "RTX": ["Defense", "Aerospace"],
    "NOC": ["Defense", "Aerospace"],
    "BA": ["Defense", "Aerospace", "Commercial Aviation"],
    "GD": ["Defense", "Aerospace"],
}

# Sector to committee mapping for access determination
SECTOR_COMMITTEE_MAP = {
    "Technology": ["Intelligence", "Commerce", "Judiciary", "Energy and Commerce"],
    "Semiconductors": ["Intelligence", "Commerce", "Armed Services"],
    "AI": ["Intelligence", "Commerce", "Science"],
    "Financials": ["Banking", "Financial Services", "Finance"],
    "Banking": ["Banking", "Financial Services"],
    "Healthcare": ["Health", "HELP", "Energy and Commerce"],
    "Pharmaceuticals": ["Health", "HELP", "Energy and Commerce"],
    "Biotechnology": ["Health", "HELP", "Energy and Commerce"],
    "Energy": ["Energy", "Energy and Natural Resources"],
    "Oil & Gas": ["Energy", "Energy and Natural Resources"],
    "Defense": ["Armed Services", "Intelligence", "Appropriations"],
    "Aerospace": ["Armed Services", "Intelligence", "Transportation"],
    "Cryptocurrency": ["Banking", "Financial Services", "Finance"],
}


# Known significant events database (expandable)
KNOWN_EVENTS = [
    # COVID-19 Related Events
    {
        "event_id": "COVID_BRIEFING_2020_01",
        "event_type": EventType.PANDEMIC_BRIEFING,
        "event_date": date(2020, 1, 24),
        "description": "Senate classified COVID-19 briefing",
        "affected_sectors": ["Healthcare", "Travel", "Hospitality", "Retail"],
        "affected_symbols": [],  # Broad market impact
        "committees_briefed": ["Intelligence", "HELP"],
        "access_level": AccessLevel.COMMITTEE
    },
    {
        "event_id": "COVID_BRIEFING_2020_02",
        "event_type": EventType.PANDEMIC_BRIEFING,
        "event_date": date(2020, 2, 13),
        "description": "All-senators COVID-19 briefing",
        "affected_sectors": ["Healthcare", "Travel", "Hospitality", "Retail"],
        "affected_symbols": ["HCA", "MAR", "H", "CCL", "RCL", "UAL", "DAL", "AAL"],
        "committees_briefed": ["ALL_SENATORS"],
        "access_level": AccessLevel.CHAMBER
    },
    # AI/Tech Legislation
    {
        "event_id": "AI_ACT_INTRO_2024",
        "event_type": EventType.BILL_INTRODUCTION,
        "event_date": date(2024, 3, 15),
        "description": "AI Safety and Innovation Act introduction",
        "affected_sectors": ["Technology", "AI", "Semiconductors"],
        "affected_symbols": ["NVDA", "GOOGL", "MSFT", "META", "AMD"],
        "committees_briefed": ["Commerce", "Energy and Commerce"],
        "access_level": AccessLevel.COMMITTEE
    },
    # Banking/Crypto Regulation
    {
        "event_id": "CRYPTO_REG_2024",
        "event_type": EventType.BILL_INTRODUCTION,
        "event_date": date(2024, 5, 1),
        "description": "Cryptocurrency Regulation Framework introduction",
        "affected_sectors": ["Financials", "Cryptocurrency"],
        "affected_symbols": ["COIN", "MSTR", "SQ", "PYPL"],
        "committees_briefed": ["Banking", "Financial Services"],
        "access_level": AccessLevel.COMMITTEE
    },
    # Energy Policy
    {
        "event_id": "ENERGY_ACT_2024",
        "event_type": EventType.BILL_VOTE,
        "event_date": date(2024, 6, 15),
        "description": "Energy Security and Independence Act vote",
        "affected_sectors": ["Energy", "Oil & Gas"],
        "affected_symbols": ["XOM", "CVX", "COP", "EOG", "SLB"],
        "committees_briefed": ["Energy", "Energy and Natural Resources"],
        "access_level": AccessLevel.COMMITTEE
    },
    # Defense Contracts
    {
        "event_id": "F35_CONTRACT_2023",
        "event_type": EventType.DEFENSE_CONTRACT,
        "event_date": date(2023, 10, 15),
        "description": "Major F-35 contract award announcement",
        "affected_sectors": ["Defense", "Aerospace"],
        "affected_symbols": ["LMT", "RTX", "NOC"],
        "committees_briefed": ["Armed Services", "Appropriations"],
        "access_level": AccessLevel.COMMITTEE
    },
    # FDA Decisions
    {
        "event_id": "FDA_COVID_VAX_2020",
        "event_type": EventType.FDA_DECISION,
        "event_date": date(2020, 12, 11),
        "description": "FDA Emergency Use Authorization for COVID vaccine",
        "affected_sectors": ["Healthcare", "Pharmaceuticals", "Biotechnology"],
        "affected_symbols": ["PFE", "MRNA", "JNJ", "AZN"],
        "committees_briefed": ["HELP", "Health"],
        "access_level": AccessLevel.COMMITTEE
    },
]


class TimingAnalyzer:
    """
    Engine for analyzing trade timing relative to market-moving events.

    Identifies trades occurring suspiciously close to events the member
    had plausible access to.
    """

    # Time windows for analysis (days)
    SUSPICIOUS_WINDOW_BEFORE = 30  # Days before event to flag as suspicious
    SUSPICIOUS_WINDOW_AFTER = 7    # Days after event (late filing detected)
    MAX_WINDOW = 60                # Maximum window to consider

    # Scoring weights
    TIMING_WEIGHTS = {
        "days_proximity": 0.35,     # How close to event
        "access_level": 0.30,       # Direct vs committee vs public
        "event_magnitude": 0.20,    # Importance of event
        "pattern_match": 0.15       # Trade direction matches expected outcome
    }

    def __init__(
        self,
        events_database: Optional[List[Dict]] = None,
        committee_calendar: Optional[Dict] = None
    ):
        """
        Initialize the TimingAnalyzer.

        Args:
            events_database: Optional list of known events to correlate
            committee_calendar: Optional dictionary of committee hearing schedules
        """
        self.events_database = events_database or KNOWN_EVENTS
        self.committee_calendar = committee_calendar or {}

    def analyze_trade_timing(
        self,
        trade: Dict,
        member: Dict,
        additional_events: Optional[List[Dict]] = None
    ) -> TimingAnalysis:
        """
        Analyze timing of a trade relative to potential market-moving events.

        Args:
            trade: Trade data dictionary
            member: Member data dictionary
            additional_events: Optional list of additional events to check

        Returns:
            TimingAnalysis with correlation results
        """
        trade_id = str(trade.get("id") or trade.get("trade_id", "unknown"))
        member_id = str(member.get("id") or member.get("member_id", "unknown"))
        member_name = member.get("name") or member.get("full_name", "Unknown")
        symbol = trade.get("symbol", "").upper()
        transaction_date = self._parse_date(trade.get("transaction_date"))
        transaction_type = trade.get("transaction_type", "").lower()

        if not transaction_date:
            return TimingAnalysis(
                trade_id=trade_id,
                member_id=member_id,
                member_name=member_name,
                symbol=symbol,
                transaction_date=date.today(),
                suspicious_events=[],
                timing_score=0.0,
                summary="Unable to analyze timing (missing transaction date)",
                risk_factors=[]
            )

        # Get member's committee assignments
        member_committees = self._get_member_committees(member)

        # Get stock sectors
        stock_sectors = STOCK_SECTORS.get(symbol, ["Unknown"])

        # Combine all events to check
        all_events = self.events_database.copy()
        if additional_events:
            all_events.extend(additional_events)

        # Find suspicious event correlations
        suspicious_events = []
        for event in all_events:
            correlation = self._check_event_correlation(
                event=event,
                transaction_date=transaction_date,
                symbol=symbol,
                stock_sectors=stock_sectors,
                member_committees=member_committees,
                transaction_type=transaction_type
            )
            if correlation:
                suspicious_events.append(correlation)

        # Also check for generic timing patterns
        generic_events = self._check_generic_timing_patterns(
            transaction_date=transaction_date,
            symbol=symbol,
            member_committees=member_committees
        )
        suspicious_events.extend(generic_events)

        # Calculate overall timing score
        timing_score = self._calculate_timing_score(
            suspicious_events=suspicious_events,
            transaction_type=transaction_type
        )

        # Generate risk factors list
        risk_factors = self._identify_risk_factors(
            suspicious_events=suspicious_events,
            member=member,
            trade=trade
        )

        # Generate summary
        summary = self._generate_summary(
            suspicious_events=suspicious_events,
            timing_score=timing_score,
            member_name=member_name,
            symbol=symbol,
            transaction_date=transaction_date
        )

        return TimingAnalysis(
            trade_id=trade_id,
            member_id=member_id,
            member_name=member_name,
            symbol=symbol,
            transaction_date=transaction_date,
            suspicious_events=suspicious_events,
            timing_score=timing_score,
            summary=summary,
            risk_factors=risk_factors
        )

    def _check_event_correlation(
        self,
        event: Dict,
        transaction_date: date,
        symbol: str,
        stock_sectors: List[str],
        member_committees: List[str],
        transaction_type: str
    ) -> Optional[SuspiciousEvent]:
        """
        Check if an event correlates with the trade.

        Returns SuspiciousEvent if correlation found, None otherwise.
        """
        event_date = self._parse_date(event.get("event_date"))
        if not event_date:
            return None

        # Calculate time delta
        days_diff = (event_date - transaction_date).days
        days_before_trade = -days_diff if days_diff < 0 else 0
        days_after_trade = days_diff if days_diff > 0 else 0

        # Check if within analysis window
        if abs(days_diff) > self.MAX_WINDOW:
            return None

        # Check symbol/sector match
        affected_symbols = event.get("affected_symbols", [])
        affected_sectors = event.get("affected_sectors", [])

        symbol_match = symbol in affected_symbols
        sector_match = any(s in affected_sectors for s in stock_sectors)

        if not symbol_match and not sector_match and affected_symbols:
            # No sector match and specific symbols listed
            return None

        # Determine member's access level
        access_level = self._determine_access_level(
            event=event,
            member_committees=member_committees
        )

        # Calculate information advantage score
        info_advantage = self._calculate_info_advantage(
            days_before_trade=days_before_trade,
            days_after_trade=days_after_trade,
            access_level=access_level,
            symbol_match=symbol_match,
            sector_match=sector_match,
            transaction_type=transaction_type,
            event_type=event.get("event_type", EventType.REGULATORY_ANNOUNCEMENT)
        )

        # Only report if there's meaningful advantage
        if info_advantage < 0.2:
            return None

        event_type = event.get("event_type")
        if isinstance(event_type, str):
            event_type = EventType(event_type) if event_type in [e.value for e in EventType] else EventType.REGULATORY_ANNOUNCEMENT

        return SuspiciousEvent(
            event_type=event_type,
            event_date=event_date,
            days_before_trade=days_before_trade,
            days_after_trade=days_after_trade,
            member_access_level=access_level,
            description=event.get("description", "Unknown event"),
            affected_symbols=affected_symbols,
            affected_sectors=affected_sectors,
            information_advantage_score=info_advantage,
            event_id=event.get("event_id")
        )

    def _check_generic_timing_patterns(
        self,
        transaction_date: date,
        symbol: str,
        member_committees: List[str]
    ) -> List[SuspiciousEvent]:
        """
        Check for generic suspicious timing patterns.

        Returns list of SuspiciousEvent for any patterns detected.
        """
        events = []

        # Check for COVID-19 timing (Jan-March 2020)
        covid_start = date(2020, 1, 15)
        covid_crash = date(2020, 3, 23)

        if covid_start <= transaction_date <= covid_crash:
            # Check if member had intelligence access
            has_intel_access = any(
                "intelligence" in c.lower() or "health" in c.lower() or "help" in c.lower()
                for c in member_committees
            )

            if has_intel_access:
                events.append(SuspiciousEvent(
                    event_type=EventType.PANDEMIC_BRIEFING,
                    event_date=date(2020, 1, 24),  # First briefing date
                    days_before_trade=max(0, (transaction_date - date(2020, 1, 24)).days),
                    days_after_trade=0,
                    member_access_level=AccessLevel.COMMITTEE,
                    description="Trade during COVID-19 early briefing period",
                    affected_symbols=[symbol],
                    affected_sectors=["Healthcare", "Travel", "Hospitality"],
                    information_advantage_score=0.85
                ))

        # Check for end-of-quarter patterns (options expiration, rebalancing)
        month = transaction_date.month
        day = transaction_date.day
        is_quarter_end = month in [3, 6, 9, 12] and day >= 15

        if is_quarter_end:
            # This is informational, not directly suspicious
            pass

        # Check for earnings season timing (simplified)
        # Major earnings typically Jan, Apr, Jul, Oct
        is_earnings_month = month in [1, 4, 7, 10]
        if is_earnings_month and 10 <= day <= 25:
            sectors = STOCK_SECTORS.get(symbol, [])
            if "Technology" in sectors or "Financials" in sectors:
                events.append(SuspiciousEvent(
                    event_type=EventType.EARNINGS_ANNOUNCEMENT,
                    event_date=transaction_date + timedelta(days=7),  # Approximate
                    days_before_trade=0,
                    days_after_trade=7,
                    member_access_level=AccessLevel.PUBLIC,
                    description=f"Trade during earnings announcement season for {symbol}",
                    affected_symbols=[symbol],
                    affected_sectors=sectors,
                    information_advantage_score=0.3  # Lower - public timing
                ))

        return events

    def _determine_access_level(
        self,
        event: Dict,
        member_committees: List[str]
    ) -> AccessLevel:
        """Determine member's access level to event information."""
        committees_briefed = event.get("committees_briefed", [])
        default_level = event.get("access_level", AccessLevel.PUBLIC)

        if isinstance(default_level, str):
            default_level = AccessLevel(default_level)

        # Check for ALL_SENATORS or ALL_HOUSE
        if "ALL_SENATORS" in committees_briefed or "ALL_HOUSE" in committees_briefed:
            return AccessLevel.CHAMBER

        # Check committee match
        for member_comm in member_committees:
            member_comm_lower = member_comm.lower()
            for briefed_comm in committees_briefed:
                if briefed_comm.lower() in member_comm_lower or member_comm_lower in briefed_comm.lower():
                    return AccessLevel.COMMITTEE

        return AccessLevel.PUBLIC

    def _calculate_info_advantage(
        self,
        days_before_trade: int,
        days_after_trade: int,
        access_level: AccessLevel,
        symbol_match: bool,
        sector_match: bool,
        transaction_type: str,
        event_type: EventType
    ) -> float:
        """
        Calculate information advantage score (0-1).

        Higher score = more suspicious timing pattern.
        """
        # Base score from timing
        if days_before_trade > 0:
            # Trade happened before event - suspicious if member knew
            if days_before_trade <= 7:
                timing_score = 1.0
            elif days_before_trade <= 14:
                timing_score = 0.8
            elif days_before_trade <= 30:
                timing_score = 0.6
            else:
                timing_score = 0.3
        elif days_after_trade <= 7:
            # Trade right after event - less suspicious
            timing_score = 0.4
        else:
            timing_score = 0.2

        # Access level multiplier
        access_multipliers = {
            AccessLevel.DIRECT: 1.0,
            AccessLevel.COMMITTEE: 0.85,
            AccessLevel.CHAMBER: 0.6,
            AccessLevel.PUBLIC: 0.3
        }
        access_mult = access_multipliers.get(access_level, 0.3)

        # Symbol/sector match multiplier
        if symbol_match:
            match_mult = 1.0
        elif sector_match:
            match_mult = 0.7
        else:
            match_mult = 0.3

        # Event type importance
        event_importance = {
            EventType.CLASSIFIED_BRIEFING: 1.0,
            EventType.PANDEMIC_BRIEFING: 0.95,
            EventType.CRISIS_BRIEFING: 0.9,
            EventType.FDA_DECISION: 0.85,
            EventType.DEFENSE_CONTRACT: 0.8,
            EventType.BILL_VOTE: 0.75,
            EventType.BILL_INTRODUCTION: 0.65,
            EventType.COMMITTEE_HEARING: 0.6,
            EventType.REGULATORY_ANNOUNCEMENT: 0.55,
            EventType.EXECUTIVE_ORDER: 0.5,
            EventType.EARNINGS_ANNOUNCEMENT: 0.4
        }
        event_mult = event_importance.get(event_type, 0.5)

        # Combine scores
        score = timing_score * access_mult * match_mult * event_mult

        return min(1.0, score)

    def _calculate_timing_score(
        self,
        suspicious_events: List[SuspiciousEvent],
        transaction_type: str
    ) -> float:
        """Calculate overall timing suspicion score (0-1)."""
        if not suspicious_events:
            return 0.0

        # Use highest information advantage score
        max_score = max(e.information_advantage_score for e in suspicious_events)

        # Boost for multiple correlating events
        if len(suspicious_events) > 1:
            max_score = min(1.0, max_score * 1.15)
        if len(suspicious_events) > 2:
            max_score = min(1.0, max_score * 1.1)

        return max_score

    def _identify_risk_factors(
        self,
        suspicious_events: List[SuspiciousEvent],
        member: Dict,
        trade: Dict
    ) -> List[str]:
        """Identify specific risk factors for the trade."""
        risk_factors = []

        if not suspicious_events:
            return ["No significant timing correlations detected"]

        for event in suspicious_events:
            if event.information_advantage_score > 0.7:
                if event.member_access_level == AccessLevel.DIRECT:
                    risk_factors.append(f"Direct access to {event.event_type.value} information")
                elif event.member_access_level == AccessLevel.COMMITTEE:
                    risk_factors.append(f"Committee access before {event.event_type.value}")

                if event.days_before_trade <= 7:
                    risk_factors.append(f"Trade {event.days_before_trade} days before major event")

        # Check filing delay correlation
        filing_date = self._parse_date(trade.get("filing_date"))
        transaction_date = self._parse_date(trade.get("transaction_date"))
        if filing_date and transaction_date:
            delay = (filing_date - transaction_date).days
            if delay > 45:
                risk_factors.append(f"Late filing ({delay} days) may indicate concealment")

        return risk_factors if risk_factors else ["Minor timing correlation detected"]

    def _generate_summary(
        self,
        suspicious_events: List[SuspiciousEvent],
        timing_score: float,
        member_name: str,
        symbol: str,
        transaction_date: date
    ) -> str:
        """Generate human-readable summary of timing analysis."""
        if not suspicious_events:
            return f"No suspicious timing patterns detected for {member_name}'s {symbol} trade on {transaction_date}"

        if timing_score > 0.8:
            severity = "highly suspicious"
        elif timing_score > 0.6:
            severity = "suspicious"
        elif timing_score > 0.4:
            severity = "potentially concerning"
        else:
            severity = "minor"

        most_suspicious = max(suspicious_events, key=lambda e: e.information_advantage_score)

        summary = (
            f"{member_name}'s {symbol} trade on {transaction_date} has {severity} timing "
            f"(score: {timing_score:.2f}). "
        )

        if most_suspicious.days_before_trade > 0:
            summary += (
                f"Trade occurred {most_suspicious.days_before_trade} days before "
                f"{most_suspicious.description}. "
            )
        else:
            summary += (
                f"Trade occurred {most_suspicious.days_after_trade} days after "
                f"{most_suspicious.description}. "
            )

        summary += f"Member had {most_suspicious.member_access_level.value} access to related information."

        return summary

    def _get_member_committees(self, member: Dict) -> List[str]:
        """Extract committee list from member data."""
        committees = member.get("committees", [])
        if isinstance(committees, str):
            committees = [committees]

        committee = member.get("committee", "")
        if committee and committee not in committees:
            committees.append(committee)

        return committees

    def _parse_date(self, date_value) -> Optional[date]:
        """Parse date from various formats."""
        if date_value is None:
            return None
        if isinstance(date_value, date):
            return date_value
        if isinstance(date_value, datetime):
            return date_value.date()
        if isinstance(date_value, str):
            for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d"]:
                try:
                    return datetime.strptime(date_value, fmt).date()
                except ValueError:
                    continue
        return None

    def analyze_batch(
        self,
        trades: List[Dict],
        members: Dict[str, Dict],
        additional_events: Optional[List[Dict]] = None
    ) -> List[TimingAnalysis]:
        """
        Analyze timing for multiple trades.

        Args:
            trades: List of trade dictionaries
            members: Dictionary mapping member_id -> member data
            additional_events: Optional additional events to check

        Returns:
            List of TimingAnalysis results sorted by timing_score (highest first)
        """
        results = []

        for trade in trades:
            member_id = str(trade.get("member_id", ""))
            member = members.get(member_id, {
                "id": member_id,
                "name": trade.get("member_name", "Unknown")
            })

            analysis = self.analyze_trade_timing(trade, member, additional_events)
            results.append(analysis)

        # Sort by timing score (highest first)
        results.sort(key=lambda x: x.timing_score, reverse=True)

        return results

    def get_covid_trades_analysis(
        self,
        trades: List[Dict],
        members: Dict[str, Dict]
    ) -> List[TimingAnalysis]:
        """
        Specialized analysis for COVID-19 period trades (Jan-Mar 2020).

        These trades are of particular interest due to known classified briefings.
        """
        covid_start = date(2020, 1, 1)
        covid_end = date(2020, 3, 31)

        # Filter to COVID period trades
        covid_trades = []
        for trade in trades:
            trans_date = self._parse_date(trade.get("transaction_date"))
            if trans_date and covid_start <= trans_date <= covid_end:
                covid_trades.append(trade)

        if not covid_trades:
            return []

        # Analyze with COVID-specific events
        return self.analyze_batch(covid_trades, members)


def main():
    """Example usage and testing of the TimingAnalyzer."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Congressional Trading Timing Analysis Engine - Test Run")
    print("=" * 70)

    # Sample trade data
    sample_trades = [
        {
            "id": "T001",
            "member_id": "M001",
            "symbol": "HCA",
            "transaction_date": "2020-02-13",
            "filing_date": "2020-03-19",
            "transaction_type": "Sale",
            "amount_from": 628000,
            "amount_to": 1720000
        },
        {
            "id": "T002",
            "member_id": "M002",
            "symbol": "NVDA",
            "transaction_date": "2024-03-01",
            "filing_date": "2024-04-10",
            "transaction_type": "Purchase",
            "amount_from": 500000,
            "amount_to": 1000000
        },
        {
            "id": "T003",
            "member_id": "M003",
            "symbol": "XOM",
            "transaction_date": "2024-06-01",
            "filing_date": "2024-07-01",
            "transaction_type": "Purchase",
            "amount_from": 250000,
            "amount_to": 500000
        }
    ]

    # Sample member data
    sample_members = {
        "M001": {
            "id": "M001",
            "name": "Richard Burr",
            "party": "R",
            "state": "NC",
            "committees": ["Senate Intelligence", "Senate HELP"],
            "leadership_role": "Former Intelligence Chair"
        },
        "M002": {
            "id": "M002",
            "name": "Nancy Pelosi",
            "party": "D",
            "state": "CA",
            "committees": ["House Intelligence", "House Financial Services"]
        },
        "M003": {
            "id": "M003",
            "name": "Joe Manchin",
            "party": "D",
            "state": "WV",
            "committees": ["Senate Energy and Natural Resources"],
            "leadership_role": "Energy Committee Chair"
        }
    }

    # Initialize analyzer
    analyzer = TimingAnalyzer()

    # Analyze all trades
    results = analyzer.analyze_batch(sample_trades, sample_members)

    print("\nTiming Analysis Results:")
    print("-" * 70)

    for result in results:
        print(f"\n{result.member_name} - {result.symbol} on {result.transaction_date}")
        print(f"  Timing Score: {result.timing_score:.2f}")
        print(f"  Summary: {result.summary}")

        if result.suspicious_events:
            print("  Suspicious Events:")
            for event in result.suspicious_events:
                print(f"    - {event.event_type.value}: {event.description}")
                print(f"      Access: {event.member_access_level.value}, "
                      f"Advantage Score: {event.information_advantage_score:.2f}")

        if result.risk_factors:
            print(f"  Risk Factors: {', '.join(result.risk_factors)}")

    # Summary
    print("\n" + "=" * 70)
    print("Analysis Summary:")
    print("-" * 70)
    suspicious_count = sum(1 for r in results if r.timing_score > 0.5)
    print(f"Total trades analyzed: {len(results)}")
    print(f"Trades with suspicious timing: {suspicious_count}")
    if results:
        print(f"Highest timing score: {max(r.timing_score for r in results):.2f}")


if __name__ == "__main__":
    main()
