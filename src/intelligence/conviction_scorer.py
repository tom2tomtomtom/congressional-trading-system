#!/usr/bin/env python3
"""
Congressional Trading Intelligence System
Track A - Task A1: Conviction Score Algorithm

This module implements the core conviction scoring engine that analyzes trades
and outputs a 0-100 score indicating how likely a trade appears to be based on
non-public information.

Scoring Factors (weights totaling 100):
- Committee Access (25 points): Does member sit on committee overseeing this sector?
- Timing Proximity (25 points): How close to relevant announcements/hearings?
- Filing Delay (15 points): How late was the disclosure filed?
- Trade Size Anomaly (15 points): Is this unusually large for this member?
- Historical Pattern (10 points): Does this deviate from normal behavior?
- Sector Concentration (10 points): Over-weighted in oversight sectors?
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import statistics
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classification based on conviction score."""
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConvictionFactor:
    """Individual factor contributing to conviction score."""
    name: str
    score: float  # 0-100 normalized
    weight: float  # Factor weight (0-1)
    raw_value: Any  # Original value before normalization
    explanation: str

    @property
    def weighted_score(self) -> float:
        """Calculate weighted contribution to total score."""
        return self.score * self.weight


@dataclass
class ConvictionResult:
    """Complete conviction analysis result for a trade."""
    trade_id: str
    member_id: str
    member_name: str
    score: float  # 0-100
    risk_level: RiskLevel
    factors: Dict[str, ConvictionFactor]
    explanation: str
    top_factors: List[str]  # Top 3 contributing factors
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert result to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "member_id": self.member_id,
            "member_name": self.member_name,
            "score": round(self.score, 1),
            "risk_level": self.risk_level.value,
            "factors": {
                name: {
                    "score": round(f.score, 1),
                    "weight": f.weight,
                    "weighted_score": round(f.weighted_score, 1),
                    "explanation": f.explanation
                }
                for name, f in self.factors.items()
            },
            "explanation": self.explanation,
            "top_factors": self.top_factors,
            "timestamp": self.timestamp.isoformat()
        }


# Committee to sector/stock mapping for oversight detection
COMMITTEE_SECTOR_MAP = {
    "Financial Services": {
        "sectors": ["XLF", "Financials", "Banking", "Insurance", "Fintech"],
        "stocks": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "COIN", "PYPL", "V", "MA", "AXP"],
        "keywords": ["bank", "financial", "credit", "insurance", "fintech", "payment"]
    },
    "Armed Services": {
        "sectors": ["XLI", "Defense", "Aerospace"],
        "stocks": ["LMT", "RTX", "NOC", "GD", "BA", "HII", "LHX", "TXT"],
        "keywords": ["defense", "military", "aerospace", "weapons", "contractor"]
    },
    "Energy and Commerce": {
        "sectors": ["XLE", "XLU", "Energy", "Utilities"],
        "stocks": ["XOM", "CVX", "COP", "EOG", "SLB", "NEE", "DUK", "SO", "D", "AEP"],
        "keywords": ["energy", "oil", "gas", "utility", "power", "electric"]
    },
    "Energy and Natural Resources": {
        "sectors": ["XLE", "Energy", "Natural Resources"],
        "stocks": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "KMI", "OKE"],
        "keywords": ["energy", "oil", "gas", "mining", "natural resources"]
    },
    "Health, Education, Labor and Pensions": {
        "sectors": ["XLV", "Healthcare", "Pharma", "Biotech"],
        "stocks": ["UNH", "JNJ", "PFE", "ABT", "TMO", "DHR", "BMY", "AMGN", "GILD", "BIIB", "MRK", "LLY"],
        "keywords": ["health", "pharma", "biotech", "medical", "drug", "hospital"]
    },
    "Intelligence": {
        "sectors": ["XLK", "Technology", "Cybersecurity", "Defense"],
        "stocks": ["MSFT", "GOOG", "GOOGL", "AMZN", "ORCL", "PANW", "CRWD", "ZS", "NET", "LMT", "RTX", "NOC"],
        "keywords": ["cyber", "security", "intelligence", "surveillance", "tech", "data"]
    },
    "Banking": {
        "sectors": ["XLF", "Financials", "Banking"],
        "stocks": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF"],
        "keywords": ["bank", "financial", "credit", "lending", "mortgage"]
    },
    "Commerce": {
        "sectors": ["XLK", "Technology", "Communication Services"],
        "stocks": ["AMZN", "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS"],
        "keywords": ["commerce", "tech", "media", "telecom", "communications"]
    },
    "Agriculture": {
        "sectors": ["Commodities", "Agriculture", "Food"],
        "stocks": ["ADM", "BG", "DE", "AGCO", "CF", "MOS", "NTR", "TSN", "HRL"],
        "keywords": ["agriculture", "farm", "food", "commodity", "grain"]
    },
    "Judiciary": {
        "sectors": ["XLK", "Technology"],
        "stocks": ["GOOGL", "META", "AMZN", "AAPL", "MSFT"],
        "keywords": ["antitrust", "tech", "privacy", "legal"]
    },
    "Oversight": {
        "sectors": [],  # General oversight - no specific sectors
        "stocks": [],
        "keywords": ["government", "contractor", "spending"]
    },
    "Transportation": {
        "sectors": ["XLI", "Transportation"],
        "stocks": ["UNP", "CSX", "NSC", "FDX", "UPS", "DAL", "UAL", "LUV", "AAL"],
        "keywords": ["transport", "rail", "airline", "shipping", "logistics"]
    },
    "Ways and Means": {
        "sectors": ["Broad Market"],  # Tax policy affects all sectors
        "stocks": [],
        "keywords": ["tax", "trade", "tariff"]
    }
}

# Stock sector classification
STOCK_SECTORS = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "META": "Technology", "NVDA": "Technology",
    "TSLA": "Technology", "ORCL": "Technology", "CRM": "Technology",
    "ADBE": "Technology", "NFLX": "Technology", "INTC": "Technology",
    "AMD": "Technology", "CSCO": "Technology", "IBM": "Technology",

    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "USB": "Financials", "PNC": "Financials", "TFC": "Financials",
    "COF": "Financials", "V": "Financials", "MA": "Financials",
    "AXP": "Financials", "COIN": "Financials", "PYPL": "Financials",

    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare",
    "ABT": "Healthcare", "TMO": "Healthcare", "DHR": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "BIIB": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "EOG": "Energy", "SLB": "Energy", "MPC": "Energy",
    "VLO": "Energy", "PSX": "Energy", "KMI": "Energy", "OKE": "Energy",

    # Defense/Industrial
    "BA": "Defense", "LMT": "Defense", "RTX": "Defense",
    "NOC": "Defense", "GD": "Defense", "CAT": "Industrial",
    "GE": "Industrial", "MMM": "Industrial", "HON": "Industrial",

    # Consumer
    "WMT": "Consumer", "HD": "Consumer", "PG": "Consumer",
    "KO": "Consumer", "PEP": "Consumer", "COST": "Consumer",
    "LOW": "Consumer", "TGT": "Consumer", "SBUX": "Consumer",
    "MCD": "Consumer", "DIS": "Consumer", "NKE": "Consumer",
}


class ConvictionScorer:
    """
    Core conviction scoring engine for congressional trade analysis.

    Analyzes trades against multiple factors to generate a 0-100 conviction
    score indicating likelihood of informed trading.
    """

    # Factor weights (must sum to 1.0)
    FACTOR_WEIGHTS = {
        "committee_access": 0.25,
        "timing_proximity": 0.25,
        "filing_delay": 0.15,
        "trade_size_anomaly": 0.15,
        "historical_pattern": 0.10,
        "sector_concentration": 0.10
    }

    # Filing delay thresholds (STOCK Act requires 45 days)
    FILING_DELAY_THRESHOLDS = {
        "on_time": 30,       # Reasonable filing
        "borderline": 45,    # At deadline
        "late": 60,          # Past deadline
        "very_late": 90,     # Significantly late
        "extreme": 180       # Extremely late
    }

    def __init__(self, member_trading_history: Optional[Dict] = None):
        """
        Initialize the ConvictionScorer.

        Args:
            member_trading_history: Optional dict of member_id -> list of historical trades
        """
        self.member_trading_history = member_trading_history or {}
        self._validate_weights()

    def _validate_weights(self):
        """Ensure factor weights sum to 1.0."""
        total = sum(self.FACTOR_WEIGHTS.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Factor weights must sum to 1.0, got {total}")

    def score_trade(self, trade: Dict, member: Dict) -> ConvictionResult:
        """
        Calculate conviction score for a single trade.

        Args:
            trade: Trade data dictionary containing:
                - id/trade_id: Unique trade identifier
                - symbol: Stock symbol traded
                - transaction_date: Date of trade
                - filing_date: Date trade was disclosed
                - amount_from/amount_to: Trade amount range
                - transaction_type: "Purchase" or "Sale"
            member: Member data dictionary containing:
                - id/member_id: Unique member identifier
                - name: Member's full name
                - party: Political party
                - state: State represented
                - committee/committees: Committee assignments

        Returns:
            ConvictionResult with score and detailed breakdown
        """
        trade_id = str(trade.get("id") or trade.get("trade_id", "unknown"))
        member_id = str(member.get("id") or member.get("member_id", "unknown"))
        member_name = member.get("name") or member.get("full_name", "Unknown")

        # Calculate each factor
        factors = {}

        factors["committee_access"] = self._score_committee_access(trade, member)
        factors["timing_proximity"] = self._score_timing_proximity(trade, member)
        factors["filing_delay"] = self._score_filing_delay(trade)
        factors["trade_size_anomaly"] = self._score_trade_size_anomaly(trade, member)
        factors["historical_pattern"] = self._score_historical_pattern(trade, member)
        factors["sector_concentration"] = self._score_sector_concentration(trade, member)

        # Calculate total weighted score
        total_score = sum(f.weighted_score for f in factors.values())

        # Determine risk level
        risk_level = self._determine_risk_level(total_score)

        # Get top contributing factors
        sorted_factors = sorted(
            factors.items(),
            key=lambda x: x[1].weighted_score,
            reverse=True
        )
        top_factors = [name for name, _ in sorted_factors[:3]]

        # Generate explanation
        explanation = self._generate_explanation(total_score, factors, member_name, trade)

        return ConvictionResult(
            trade_id=trade_id,
            member_id=member_id,
            member_name=member_name,
            score=total_score,
            risk_level=risk_level,
            factors=factors,
            explanation=explanation,
            top_factors=top_factors
        )

    def _score_committee_access(self, trade: Dict, member: Dict) -> ConvictionFactor:
        """
        Score based on whether member's committee has oversight of traded sector.

        25 points maximum weight.
        """
        symbol = trade.get("symbol", "").upper()
        committees = self._get_member_committees(member)

        stock_sector = STOCK_SECTORS.get(symbol, "Unknown")

        # Check for committee-sector overlap
        overlap_score = 0
        matching_committee = None

        for committee in committees:
            # Normalize committee name for matching
            committee_normalized = committee.lower()

            for comm_name, comm_data in COMMITTEE_SECTOR_MAP.items():
                if comm_name.lower() in committee_normalized or committee_normalized in comm_name.lower():
                    # Check if traded stock is in oversight area
                    if symbol in comm_data.get("stocks", []):
                        overlap_score = 100  # Direct stock match
                        matching_committee = comm_name
                        break

                    # Check sector match
                    if stock_sector in comm_data.get("sectors", []):
                        overlap_score = max(overlap_score, 80)
                        matching_committee = comm_name

                    # Check keyword match in stock name/sector
                    for keyword in comm_data.get("keywords", []):
                        if keyword in stock_sector.lower() or keyword in symbol.lower():
                            overlap_score = max(overlap_score, 60)
                            matching_committee = comm_name

            if overlap_score == 100:
                break

        # Leadership positions increase score
        leadership = member.get("leadership_role") or member.get("committee", "")
        if any(term in str(leadership).lower() for term in ["chair", "ranking", "leader"]):
            overlap_score = min(100, overlap_score * 1.2)

        if overlap_score > 0:
            explanation = f"Member serves on {matching_committee} which has oversight of {stock_sector} sector"
        else:
            explanation = f"No direct committee oversight of {symbol} ({stock_sector})"

        return ConvictionFactor(
            name="committee_access",
            score=overlap_score,
            weight=self.FACTOR_WEIGHTS["committee_access"],
            raw_value={"committees": committees, "symbol": symbol, "sector": stock_sector},
            explanation=explanation
        )

    def _score_timing_proximity(self, trade: Dict, member: Dict) -> ConvictionFactor:
        """
        Score based on timing relative to known events.

        25 points maximum weight.
        This is a simplified version - the full TimingAnalyzer (A2) provides detailed analysis.
        """
        # For now, use a heuristic based on filing patterns
        # The full timing analysis is in timing_analyzer.py (A2)

        transaction_date = self._parse_date(trade.get("transaction_date"))
        filing_date = self._parse_date(trade.get("filing_date"))

        if not transaction_date:
            return ConvictionFactor(
                name="timing_proximity",
                score=50,  # Neutral score when we can't determine
                weight=self.FACTOR_WEIGHTS["timing_proximity"],
                raw_value=None,
                explanation="Unable to analyze timing (missing transaction date)"
            )

        # Baseline timing score - will be enhanced by TimingAnalyzer
        timing_score = 30  # Baseline

        # Check for known suspicious timing periods
        suspicious_periods = [
            # COVID-19 early trading window
            (date(2020, 1, 15), date(2020, 3, 15), "COVID-19 briefing period", 40),
            # Pre-earnings seasons (simplified)
        ]

        for start, end, period_name, bonus in suspicious_periods:
            if transaction_date and start <= transaction_date <= end:
                timing_score += bonus
                explanation = f"Trade occurred during {period_name}"
                break
        else:
            explanation = "Standard timing - detailed analysis available via timing_analyzer"

        timing_score = min(100, timing_score)

        return ConvictionFactor(
            name="timing_proximity",
            score=timing_score,
            weight=self.FACTOR_WEIGHTS["timing_proximity"],
            raw_value={"transaction_date": str(transaction_date)},
            explanation=explanation
        )

    def _score_filing_delay(self, trade: Dict) -> ConvictionFactor:
        """
        Score based on how late the disclosure was filed.

        15 points maximum weight.
        STOCK Act requires disclosure within 45 days.
        """
        transaction_date = self._parse_date(trade.get("transaction_date"))
        filing_date = self._parse_date(trade.get("filing_date"))

        if not transaction_date or not filing_date:
            # Check if filing_delay_days is provided directly
            delay_days = trade.get("filing_delay_days")
            if delay_days is None:
                return ConvictionFactor(
                    name="filing_delay",
                    score=50,
                    weight=self.FACTOR_WEIGHTS["filing_delay"],
                    raw_value=None,
                    explanation="Unable to calculate filing delay (missing dates)"
                )
        else:
            delay_days = (filing_date - transaction_date).days

        # Score based on delay
        if delay_days <= self.FILING_DELAY_THRESHOLDS["on_time"]:
            score = 10
            explanation = f"Filed on time ({delay_days} days)"
        elif delay_days <= self.FILING_DELAY_THRESHOLDS["borderline"]:
            score = 30
            explanation = f"Filed near deadline ({delay_days} days)"
        elif delay_days <= self.FILING_DELAY_THRESHOLDS["late"]:
            score = 60
            explanation = f"Filed late - STOCK Act violation ({delay_days} days, limit is 45)"
        elif delay_days <= self.FILING_DELAY_THRESHOLDS["very_late"]:
            score = 80
            explanation = f"Significantly late filing ({delay_days} days)"
        else:
            score = 100
            explanation = f"Extremely late filing ({delay_days} days) - potential concealment"

        return ConvictionFactor(
            name="filing_delay",
            score=score,
            weight=self.FACTOR_WEIGHTS["filing_delay"],
            raw_value=delay_days,
            explanation=explanation
        )

    def _score_trade_size_anomaly(self, trade: Dict, member: Dict) -> ConvictionFactor:
        """
        Score based on whether trade size is unusual for this member.

        15 points maximum weight.
        """
        # Get trade amount
        amount_from = trade.get("amount_from") or trade.get("amountFrom") or 0
        amount_to = trade.get("amount_to") or trade.get("amountTo") or amount_from
        avg_amount = (amount_from + amount_to) / 2 if amount_to else amount_from

        if avg_amount == 0:
            return ConvictionFactor(
                name="trade_size_anomaly",
                score=50,
                weight=self.FACTOR_WEIGHTS["trade_size_anomaly"],
                raw_value=None,
                explanation="Unable to determine trade size"
            )

        # Get member's net worth for context
        net_worth = member.get("net_worth", 1000000)  # Default 1M if not specified

        # Get historical trading amounts for this member
        member_id = str(member.get("id") or member.get("member_id", "unknown"))
        historical_trades = self.member_trading_history.get(member_id, [])

        if historical_trades:
            historical_amounts = [
                ((t.get("amount_from", 0) or t.get("amountFrom", 0)) +
                 (t.get("amount_to", 0) or t.get("amountTo", 0))) / 2
                for t in historical_trades
                if (t.get("amount_from") or t.get("amountFrom"))
            ]
            if historical_amounts:
                avg_historical = statistics.mean(historical_amounts)
                std_historical = statistics.stdev(historical_amounts) if len(historical_amounts) > 1 else avg_historical * 0.5

                # Calculate z-score
                if std_historical > 0:
                    z_score = (avg_amount - avg_historical) / std_historical
                else:
                    z_score = 0

                # Higher z-score = more unusual
                if abs(z_score) > 3:
                    score = 100
                    explanation = f"Extremely unusual size (${avg_amount:,.0f}) - {abs(z_score):.1f} standard deviations from normal"
                elif abs(z_score) > 2:
                    score = 75
                    explanation = f"Significantly larger than usual (${avg_amount:,.0f})"
                elif abs(z_score) > 1:
                    score = 50
                    explanation = f"Somewhat larger than typical trades (${avg_amount:,.0f})"
                else:
                    score = 25
                    explanation = f"Normal trade size for this member (${avg_amount:,.0f})"

                return ConvictionFactor(
                    name="trade_size_anomaly",
                    score=score,
                    weight=self.FACTOR_WEIGHTS["trade_size_anomaly"],
                    raw_value={"amount": avg_amount, "z_score": z_score},
                    explanation=explanation
                )

        # Fallback: compare to net worth
        pct_of_net_worth = (avg_amount / net_worth) * 100 if net_worth > 0 else 0

        if pct_of_net_worth > 10:
            score = 90
            explanation = f"Very large trade relative to net worth (${avg_amount:,.0f} = {pct_of_net_worth:.1f}%)"
        elif pct_of_net_worth > 5:
            score = 70
            explanation = f"Large trade relative to net worth (${avg_amount:,.0f} = {pct_of_net_worth:.1f}%)"
        elif pct_of_net_worth > 1:
            score = 40
            explanation = f"Moderate trade size (${avg_amount:,.0f})"
        else:
            score = 20
            explanation = f"Small trade relative to net worth (${avg_amount:,.0f})"

        return ConvictionFactor(
            name="trade_size_anomaly",
            score=score,
            weight=self.FACTOR_WEIGHTS["trade_size_anomaly"],
            raw_value={"amount": avg_amount, "pct_net_worth": pct_of_net_worth},
            explanation=explanation
        )

    def _score_historical_pattern(self, trade: Dict, member: Dict) -> ConvictionFactor:
        """
        Score based on deviation from member's historical trading behavior.

        10 points maximum weight.
        """
        member_id = str(member.get("id") or member.get("member_id", "unknown"))
        historical_trades = self.member_trading_history.get(member_id, [])
        symbol = trade.get("symbol", "").upper()

        if not historical_trades:
            return ConvictionFactor(
                name="historical_pattern",
                score=50,  # Neutral when no history
                weight=self.FACTOR_WEIGHTS["historical_pattern"],
                raw_value=None,
                explanation="No historical trading data available for comparison"
            )

        # Check if this is a new symbol for the member
        historical_symbols = set(t.get("symbol", "").upper() for t in historical_trades)
        is_new_symbol = symbol not in historical_symbols

        # Check trading frequency
        num_historical = len(historical_trades)

        # Check sector patterns
        historical_sectors = [STOCK_SECTORS.get(t.get("symbol", "").upper(), "Unknown")
                            for t in historical_trades]
        current_sector = STOCK_SECTORS.get(symbol, "Unknown")
        sector_frequency = historical_sectors.count(current_sector) / len(historical_sectors) if historical_sectors else 0

        # Calculate pattern deviation score
        if is_new_symbol and sector_frequency < 0.1:
            score = 80
            explanation = f"First trade in {symbol} and rare trading in {current_sector} sector"
        elif is_new_symbol:
            score = 60
            explanation = f"First trade in {symbol} (has traded in {current_sector} sector before)"
        elif sector_frequency < 0.2:
            score = 50
            explanation = f"Uncommon sector for this member ({current_sector})"
        else:
            score = 25
            explanation = f"Consistent with historical trading pattern"

        return ConvictionFactor(
            name="historical_pattern",
            score=score,
            weight=self.FACTOR_WEIGHTS["historical_pattern"],
            raw_value={
                "is_new_symbol": is_new_symbol,
                "sector_frequency": sector_frequency,
                "num_historical_trades": num_historical
            },
            explanation=explanation
        )

    def _score_sector_concentration(self, trade: Dict, member: Dict) -> ConvictionFactor:
        """
        Score based on over-concentration in oversight sectors.

        10 points maximum weight.
        """
        member_id = str(member.get("id") or member.get("member_id", "unknown"))
        historical_trades = self.member_trading_history.get(member_id, [])
        committees = self._get_member_committees(member)

        # Get oversight sectors from committees
        oversight_sectors = set()
        oversight_stocks = set()
        for committee in committees:
            committee_lower = committee.lower()
            for comm_name, comm_data in COMMITTEE_SECTOR_MAP.items():
                if comm_name.lower() in committee_lower or committee_lower in comm_name.lower():
                    oversight_sectors.update(comm_data.get("sectors", []))
                    oversight_stocks.update(comm_data.get("stocks", []))

        if not oversight_sectors:
            return ConvictionFactor(
                name="sector_concentration",
                score=30,  # Lower score if no clear oversight area
                weight=self.FACTOR_WEIGHTS["sector_concentration"],
                raw_value=None,
                explanation="No specific oversight sectors identified"
            )

        # Calculate what percentage of trades are in oversight areas
        all_trades = historical_trades + [trade]
        trades_in_oversight = 0

        for t in all_trades:
            t_symbol = t.get("symbol", "").upper()
            t_sector = STOCK_SECTORS.get(t_symbol, "Unknown")

            if t_symbol in oversight_stocks or t_sector in oversight_sectors:
                trades_in_oversight += 1

        concentration_pct = (trades_in_oversight / len(all_trades)) * 100 if all_trades else 0

        if concentration_pct > 70:
            score = 100
            explanation = f"Extremely high concentration in oversight sectors ({concentration_pct:.0f}% of trades)"
        elif concentration_pct > 50:
            score = 80
            explanation = f"High concentration in oversight sectors ({concentration_pct:.0f}% of trades)"
        elif concentration_pct > 30:
            score = 50
            explanation = f"Moderate concentration in oversight sectors ({concentration_pct:.0f}% of trades)"
        else:
            score = 20
            explanation = f"Low concentration in oversight sectors ({concentration_pct:.0f}% of trades)"

        return ConvictionFactor(
            name="sector_concentration",
            score=score,
            weight=self.FACTOR_WEIGHTS["sector_concentration"],
            raw_value={
                "concentration_pct": concentration_pct,
                "oversight_sectors": list(oversight_sectors)
            },
            explanation=explanation
        )

    def _get_member_committees(self, member: Dict) -> List[str]:
        """Extract committee list from member data."""
        committees = member.get("committees", [])
        if isinstance(committees, str):
            committees = [committees]

        # Also check 'committee' field
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
            try:
                return datetime.strptime(date_value, "%Y-%m-%d").date()
            except ValueError:
                try:
                    return datetime.strptime(date_value, "%m/%d/%Y").date()
                except ValueError:
                    return None
        return None

    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level from conviction score."""
        if score >= 80:
            return RiskLevel.CRITICAL
        elif score >= 65:
            return RiskLevel.HIGH
        elif score >= 50:
            return RiskLevel.ELEVATED
        elif score >= 30:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW

    def _generate_explanation(
        self,
        score: float,
        factors: Dict[str, ConvictionFactor],
        member_name: str,
        trade: Dict
    ) -> str:
        """Generate human-readable explanation of the conviction score."""
        symbol = trade.get("symbol", "Unknown")

        # Sort factors by weighted contribution
        sorted_factors = sorted(
            factors.items(),
            key=lambda x: x[1].weighted_score,
            reverse=True
        )

        # Build explanation
        risk_level = self._determine_risk_level(score)

        explanation = f"This trade has a conviction score of {score:.0f}/100 ({risk_level.value} risk) because: "

        # Add top 3 contributing factors
        factor_explanations = []
        for name, factor in sorted_factors[:3]:
            if factor.weighted_score > 5:  # Only include significant factors
                factor_explanations.append(factor.explanation)

        if factor_explanations:
            explanation += "; ".join(factor_explanations) + "."
        else:
            explanation += "No significant risk factors identified."

        return explanation

    def score_all_trades(
        self,
        trades: List[Dict],
        members: Dict[str, Dict]
    ) -> List[ConvictionResult]:
        """
        Batch score all trades.

        Args:
            trades: List of trade dictionaries
            members: Dictionary mapping member_id -> member data

        Returns:
            List of ConvictionResult objects sorted by score (highest first)
        """
        results = []

        for trade in trades:
            member_id = str(trade.get("member_id", ""))
            member = members.get(member_id, {"id": member_id, "name": trade.get("member_name", "Unknown")})

            result = self.score_trade(trade, member)
            results.append(result)

        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def get_summary_statistics(self, results: List[ConvictionResult]) -> Dict:
        """
        Generate summary statistics from conviction results.

        Args:
            results: List of ConvictionResult objects

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {"error": "No results to analyze"}

        scores = [r.score for r in results]

        # Count by risk level
        risk_counts = {}
        for level in RiskLevel:
            risk_counts[level.value] = sum(1 for r in results if r.risk_level == level)

        # Top suspicious trades
        top_trades = [
            {
                "trade_id": r.trade_id,
                "member_name": r.member_name,
                "score": r.score,
                "risk_level": r.risk_level.value,
                "top_factors": r.top_factors
            }
            for r in results[:10]
        ]

        return {
            "total_trades": len(results),
            "average_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0,
            "risk_distribution": risk_counts,
            "top_suspicious_trades": top_trades
        }


def main():
    """Example usage and testing of the ConvictionScorer."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Congressional Trading Conviction Score Engine - Test Run")
    print("=" * 60)

    # Sample trade data
    sample_trades = [
        {
            "id": "T001",
            "member_id": "M001",
            "symbol": "NVDA",
            "transaction_date": "2023-12-15",
            "filing_date": "2024-01-20",
            "amount_from": 1000000,
            "amount_to": 5000000,
            "transaction_type": "Purchase"
        },
        {
            "id": "T002",
            "member_id": "M002",
            "symbol": "XOM",
            "transaction_date": "2024-02-14",
            "filing_date": "2024-06-15",  # Very late filing
            "amount_from": 250000,
            "amount_to": 500000,
            "transaction_type": "Purchase"
        },
        {
            "id": "T003",
            "member_id": "M003",
            "symbol": "JPM",
            "transaction_date": "2023-11-10",
            "filing_date": "2023-12-20",
            "amount_from": 100000,
            "amount_to": 250000,
            "transaction_type": "Purchase"
        }
    ]

    # Sample member data
    sample_members = {
        "M001": {
            "id": "M001",
            "name": "Nancy Pelosi",
            "party": "D",
            "state": "CA",
            "committees": ["House Intelligence", "House Financial Services"],
            "net_worth": 114000000
        },
        "M002": {
            "id": "M002",
            "name": "Joe Manchin",
            "party": "D",
            "state": "WV",
            "committees": ["Senate Energy and Natural Resources"],
            "leadership_role": "Energy Committee Chair",
            "net_worth": 7600000
        },
        "M003": {
            "id": "M003",
            "name": "Pat Toomey",
            "party": "R",
            "state": "PA",
            "committees": ["Senate Banking"],
            "leadership_role": "Former Banking Committee Chair",
            "net_worth": 3000000
        }
    }

    # Initialize scorer
    scorer = ConvictionScorer()

    # Score all trades
    results = scorer.score_all_trades(sample_trades, sample_members)

    print("\nConviction Score Results:")
    print("-" * 60)

    for result in results:
        print(f"\n{result.member_name} - {result.trade_id}")
        print(f"  Score: {result.score:.0f}/100 ({result.risk_level.value})")
        print(f"  Explanation: {result.explanation}")
        print(f"  Top Factors: {', '.join(result.top_factors)}")
        print("  Factor Breakdown:")
        for name, factor in sorted(result.factors.items(), key=lambda x: x[1].weighted_score, reverse=True):
            print(f"    - {name}: {factor.score:.0f} (weighted: {factor.weighted_score:.1f})")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics:")
    print("-" * 60)
    stats = scorer.get_summary_statistics(results)
    print(f"Total trades analyzed: {stats['total_trades']}")
    print(f"Average conviction score: {stats['average_score']:.1f}")
    print(f"Median conviction score: {stats['median_score']:.1f}")
    print(f"Risk distribution: {stats['risk_distribution']}")


if __name__ == "__main__":
    main()
