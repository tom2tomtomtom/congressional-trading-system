#!/usr/bin/env python3
"""
Congressional Trading Intelligence System
Track A - Task A3: Committee-Trade Correlation

This module maps every member's committee assignments to sectors and flags
trades in their oversight areas.

Key Features:
- Maps committees to sectors and specific stocks
- Calculates oversight correlation scores
- Generates member rankings by oversight trading percentage
- Provides visualization-ready data for dashboard
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from collections import defaultdict
import statistics
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class Sector:
    """Represents a market sector with associated stocks and ETFs."""
    name: str
    etf: Optional[str]  # Primary ETF tracking the sector
    keywords: List[str]  # Keywords for matching
    stocks: List[str]    # Sample stocks in sector

    def matches_stock(self, symbol: str, company_name: str = "") -> bool:
        """Check if a stock belongs to this sector."""
        if symbol.upper() in [s.upper() for s in self.stocks]:
            return True
        company_lower = company_name.lower()
        return any(kw.lower() in company_lower for kw in self.keywords)


@dataclass
class CorrelationResult:
    """Result of checking if a trade is in member's oversight area."""
    is_oversight: bool
    committee: str
    sectors: List[str]
    overlap_score: float  # 0-1
    explanation: str
    trade_id: Optional[str] = None
    symbol: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "is_oversight": self.is_oversight,
            "committee": self.committee,
            "sectors": self.sectors,
            "overlap_score": round(self.overlap_score, 3),
            "explanation": self.explanation,
            "trade_id": self.trade_id,
            "symbol": self.symbol
        }


@dataclass
class MemberOversightProfile:
    """Complete oversight profile for a member."""
    member_id: str
    member_name: str
    committees: List[str]
    oversight_sectors: List[str]
    oversight_stocks: List[str]
    total_trades: int
    oversight_trades: int
    oversight_percentage: float
    rank: Optional[int] = None
    top_oversight_stocks: List[Tuple[str, int]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "member_id": self.member_id,
            "member_name": self.member_name,
            "committees": self.committees,
            "oversight_sectors": self.oversight_sectors,
            "total_trades": self.total_trades,
            "oversight_trades": self.oversight_trades,
            "oversight_percentage": round(self.oversight_percentage, 2),
            "rank": self.rank,
            "top_oversight_stocks": [
                {"symbol": s, "count": c} for s, c in self.top_oversight_stocks[:5]
            ],
            "timestamp": self.timestamp.isoformat()
        }


# Comprehensive Committee to Sector Mapping
COMMITTEE_SECTORS: Dict[str, Dict] = {
    # Financial Committees
    "Financial Services": {
        "sectors": ["Financials", "Banking", "Insurance", "Fintech", "Real Estate"],
        "etfs": ["XLF", "KBE", "KIE", "KBWB"],
        "stocks": [
            "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
            "AXP", "V", "MA", "PYPL", "SQ", "COIN", "SCHW", "BLK", "SPGI",
            "MET", "PRU", "AIG", "TRV", "ALL", "PGR", "CB", "AFL", "HIG"
        ],
        "keywords": ["bank", "financial", "credit", "insurance", "fintech", "payment", "mortgage", "asset", "capital"]
    },
    "Banking": {
        "sectors": ["Financials", "Banking", "Fintech"],
        "etfs": ["XLF", "KBE", "KBWB"],
        "stocks": [
            "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
            "SCHW", "BLK", "SPGI", "CME", "ICE", "NDAQ", "COIN", "HOOD"
        ],
        "keywords": ["bank", "financial", "credit", "lending", "deposit", "fintech", "crypto"]
    },
    "Finance": {
        "sectors": ["Financials", "Banking"],
        "etfs": ["XLF"],
        "stocks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW"],
        "keywords": ["bank", "financial", "credit", "tax", "revenue"]
    },

    # Armed Services/Defense
    "Armed Services": {
        "sectors": ["Defense", "Aerospace", "Industrial"],
        "etfs": ["ITA", "XAR", "PPA"],
        "stocks": [
            "LMT", "RTX", "NOC", "GD", "BA", "HII", "LHX", "TXT", "LDOS",
            "BAH", "KTOS", "MRCY", "AXON", "TDY", "HEI", "TDG", "SPR"
        ],
        "keywords": ["defense", "military", "aerospace", "weapons", "contractor", "aircraft", "missile", "navy", "army"]
    },

    # Energy Committees
    "Energy and Commerce": {
        "sectors": ["Energy", "Utilities", "Healthcare", "Technology", "Telecommunications"],
        "etfs": ["XLE", "XLU", "XLV", "XLK"],
        "stocks": [
            "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "MPC", "VLO", "PSX",
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PEG",
            "UNH", "JNJ", "PFE", "ABT", "TMO", "DHR", "BMY",
            "AAPL", "MSFT", "GOOGL", "META", "AMZN",
            "VZ", "T", "TMUS", "CMCSA"
        ],
        "keywords": ["energy", "oil", "gas", "utility", "power", "electric", "pharma", "drug", "telecom", "wireless"]
    },
    "Energy and Natural Resources": {
        "sectors": ["Energy", "Natural Resources", "Mining", "Utilities"],
        "etfs": ["XLE", "XOP", "OIH", "GDX", "XLU"],
        "stocks": [
            "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "KMI", "OKE",
            "PXD", "DVN", "HAL", "BKR", "FANG", "OXY", "HES", "MRO",
            "NEM", "FCX", "GOLD", "WPM", "AEM"
        ],
        "keywords": ["energy", "oil", "gas", "mining", "natural resources", "drilling", "pipeline", "coal"]
    },

    # Healthcare Committees
    "Health, Education, Labor and Pensions": {
        "sectors": ["Healthcare", "Pharmaceuticals", "Biotechnology", "Education"],
        "etfs": ["XLV", "XBI", "IBB", "VHT"],
        "stocks": [
            "UNH", "JNJ", "PFE", "ABT", "TMO", "DHR", "BMY", "AMGN", "GILD",
            "BIIB", "MRK", "LLY", "ABBV", "VRTX", "REGN", "MRNA", "ISRG",
            "ELV", "HUM", "CI", "CVS", "WBA", "HCA", "CHNG", "CERN"
        ],
        "keywords": ["health", "pharma", "biotech", "medical", "drug", "hospital", "vaccine", "FDA", "education"]
    },
    "Health": {
        "sectors": ["Healthcare", "Pharmaceuticals", "Biotechnology"],
        "etfs": ["XLV", "XBI"],
        "stocks": ["UNH", "JNJ", "PFE", "ABT", "TMO", "DHR", "BMY", "AMGN", "GILD", "MRK", "LLY"],
        "keywords": ["health", "pharma", "biotech", "medical", "drug", "hospital"]
    },

    # Intelligence/Security
    "Intelligence": {
        "sectors": ["Technology", "Cybersecurity", "Defense", "Telecommunications"],
        "etfs": ["CIBR", "HACK", "BUG", "ITA"],
        "stocks": [
            "MSFT", "GOOGL", "AMZN", "ORCL", "PANW", "CRWD", "ZS", "NET",
            "FTNT", "CYBR", "OKTA", "VRNS", "S", "TENB",
            "LMT", "RTX", "NOC", "BAH", "LDOS", "PLTR"
        ],
        "keywords": ["cyber", "security", "intelligence", "surveillance", "tech", "data", "cloud", "classified"]
    },

    # Commerce/Technology
    "Commerce": {
        "sectors": ["Technology", "Telecommunications", "Consumer", "Transportation"],
        "etfs": ["XLK", "XLC", "XLY"],
        "stocks": [
            "AMZN", "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
            "AAPL", "MSFT", "CRM", "ADBE", "NOW", "SHOP", "PYPL", "SQ",
            "WMT", "TGT", "COST", "HD", "LOW"
        ],
        "keywords": ["commerce", "tech", "media", "telecom", "communications", "streaming", "e-commerce", "retail"]
    },
    "Science": {
        "sectors": ["Technology", "Biotechnology", "Aerospace"],
        "etfs": ["XLK", "XBI", "ITA"],
        "stocks": ["MSFT", "GOOGL", "IBM", "AMD", "INTC", "NVDA", "MRNA", "BA", "LMT", "SPCE"],
        "keywords": ["science", "technology", "research", "space", "AI", "quantum"]
    },

    # Agriculture
    "Agriculture": {
        "sectors": ["Agriculture", "Food", "Commodities"],
        "etfs": ["MOO", "DBA", "CORN", "WEAT", "SOYB"],
        "stocks": [
            "ADM", "BG", "DE", "AGCO", "CF", "MOS", "NTR", "FMC", "CTVA",
            "TSN", "HRL", "CAG", "GIS", "K", "CPB", "SJM", "KHC"
        ],
        "keywords": ["agriculture", "farm", "food", "commodity", "grain", "fertilizer", "seed", "meat"]
    },

    # Judiciary/Antitrust
    "Judiciary": {
        "sectors": ["Technology", "Telecommunications", "Media"],
        "etfs": ["XLK", "XLC"],
        "stocks": [
            "GOOGL", "META", "AMZN", "AAPL", "MSFT",
            "NFLX", "DIS", "CMCSA", "T", "VZ"
        ],
        "keywords": ["antitrust", "tech", "privacy", "legal", "monopoly", "merger", "acquisition"]
    },

    # Oversight
    "Oversight": {
        "sectors": ["Government Services", "Defense"],
        "etfs": [],
        "stocks": ["BAH", "LDOS", "CACI", "SAIC", "GD", "LMT", "RTX", "NOC"],
        "keywords": ["government", "contractor", "federal", "spending", "oversight"]
    },

    # Transportation
    "Transportation": {
        "sectors": ["Transportation", "Logistics", "Airlines"],
        "etfs": ["IYT", "XTN", "JETS"],
        "stocks": [
            "UNP", "CSX", "NSC", "UPS", "FDX", "CHRW", "EXPD", "JBHT",
            "DAL", "UAL", "LUV", "AAL", "JBLU",
            "GM", "F", "TSLA", "RIVN", "LCID"
        ],
        "keywords": ["transport", "rail", "airline", "shipping", "logistics", "aviation", "trucking"]
    },

    # Ways and Means (Tax Policy)
    "Ways and Means": {
        "sectors": ["Broad Market", "International Trade"],
        "etfs": ["SPY", "EEM", "EFA"],
        "stocks": [],  # Tax policy affects all sectors
        "keywords": ["tax", "trade", "tariff", "revenue", "customs", "import", "export"]
    },

    # Appropriations
    "Appropriations": {
        "sectors": ["Defense", "Government Services", "Healthcare"],
        "etfs": ["ITA", "XLV"],
        "stocks": [
            "LMT", "RTX", "NOC", "GD", "BA", "BAH", "LDOS",
            "UNH", "HCA", "CNC", "ANTM"
        ],
        "keywords": ["spending", "budget", "appropriation", "funding", "government", "federal"]
    },

    # Foreign Affairs/Relations
    "Foreign Affairs": {
        "sectors": ["Defense", "International", "Energy"],
        "etfs": ["ITA", "EEM", "EFA"],
        "stocks": ["LMT", "RTX", "NOC", "BA", "XOM", "CVX"],
        "keywords": ["foreign", "international", "diplomatic", "sanction", "trade", "defense"]
    },
    "Foreign Relations": {
        "sectors": ["Defense", "International", "Energy"],
        "etfs": ["ITA", "EEM"],
        "stocks": ["LMT", "RTX", "NOC", "BA", "XOM", "CVX"],
        "keywords": ["foreign", "international", "diplomatic", "sanction", "treaty"]
    },

    # Homeland Security
    "Homeland Security": {
        "sectors": ["Cybersecurity", "Defense", "Technology"],
        "etfs": ["CIBR", "ITA"],
        "stocks": [
            "PANW", "CRWD", "ZS", "FTNT", "NET",
            "LMT", "RTX", "NOC", "AXON", "OSK"
        ],
        "keywords": ["security", "cyber", "border", "homeland", "emergency", "TSA"]
    },

    # Small Business
    "Small Business": {
        "sectors": ["Small Cap", "Regional Banking"],
        "etfs": ["IWM", "KRE"],
        "stocks": ["SQ", "PYPL", "INTU", "ADP"],
        "keywords": ["small business", "SBA", "lending", "entrepreneur"]
    },

    # Veterans Affairs
    "Veterans Affairs": {
        "sectors": ["Healthcare", "Government Services"],
        "etfs": ["XLV"],
        "stocks": ["CVS", "WBA", "UNH", "HCA", "CNC"],
        "keywords": ["veteran", "VA", "healthcare", "military"]
    },

    # Budget
    "Budget": {
        "sectors": ["Broad Market"],
        "etfs": ["SPY", "TLT"],
        "stocks": [],
        "keywords": ["budget", "deficit", "debt", "spending", "fiscal"]
    },

    # Rules
    "Rules": {
        "sectors": [],
        "etfs": [],
        "stocks": [],
        "keywords": []
    },

    # Ethics
    "Ethics": {
        "sectors": [],
        "etfs": [],
        "stocks": [],
        "keywords": []
    }
}


# Stock to sector classification
STOCK_SECTORS: Dict[str, List[str]] = {
    # Technology
    "AAPL": ["Technology", "Consumer Electronics"],
    "MSFT": ["Technology", "Software", "Cloud"],
    "GOOGL": ["Technology", "Advertising", "AI"],
    "GOOG": ["Technology", "Advertising", "AI"],
    "AMZN": ["Technology", "E-commerce", "Cloud"],
    "META": ["Technology", "Social Media", "Advertising"],
    "NVDA": ["Technology", "Semiconductors", "AI"],
    "TSLA": ["Technology", "Automotive", "Energy"],
    "ORCL": ["Technology", "Software", "Cloud"],
    "CRM": ["Technology", "Software"],
    "ADBE": ["Technology", "Software"],
    "INTC": ["Technology", "Semiconductors"],
    "AMD": ["Technology", "Semiconductors"],
    "AVGO": ["Technology", "Semiconductors"],

    # Financials
    "JPM": ["Financials", "Banking"],
    "BAC": ["Financials", "Banking"],
    "WFC": ["Financials", "Banking"],
    "GS": ["Financials", "Investment Banking"],
    "MS": ["Financials", "Investment Banking"],
    "C": ["Financials", "Banking"],
    "USB": ["Financials", "Banking"],
    "PNC": ["Financials", "Banking"],
    "COIN": ["Financials", "Cryptocurrency"],
    "V": ["Financials", "Payments"],
    "MA": ["Financials", "Payments"],
    "PYPL": ["Financials", "Fintech"],
    "SQ": ["Financials", "Fintech"],

    # Healthcare
    "UNH": ["Healthcare", "Insurance"],
    "JNJ": ["Healthcare", "Pharmaceuticals"],
    "PFE": ["Healthcare", "Pharmaceuticals"],
    "ABT": ["Healthcare", "Medical Devices"],
    "TMO": ["Healthcare", "Life Sciences"],
    "DHR": ["Healthcare", "Life Sciences"],
    "BMY": ["Healthcare", "Pharmaceuticals"],
    "AMGN": ["Healthcare", "Biotechnology"],
    "GILD": ["Healthcare", "Biotechnology"],
    "MRNA": ["Healthcare", "Biotechnology", "Vaccines"],
    "MRK": ["Healthcare", "Pharmaceuticals"],
    "LLY": ["Healthcare", "Pharmaceuticals"],
    "ABBV": ["Healthcare", "Pharmaceuticals"],
    "HCA": ["Healthcare", "Hospitals"],

    # Energy
    "XOM": ["Energy", "Oil & Gas"],
    "CVX": ["Energy", "Oil & Gas"],
    "COP": ["Energy", "Oil & Gas"],
    "EOG": ["Energy", "Oil & Gas"],
    "SLB": ["Energy", "Oil Services"],
    "MPC": ["Energy", "Refining"],
    "VLO": ["Energy", "Refining"],
    "PSX": ["Energy", "Refining"],
    "KMI": ["Energy", "Pipelines"],
    "OKE": ["Energy", "Pipelines"],

    # Defense
    "LMT": ["Defense", "Aerospace"],
    "RTX": ["Defense", "Aerospace"],
    "NOC": ["Defense", "Aerospace"],
    "GD": ["Defense", "Aerospace"],
    "BA": ["Defense", "Aerospace", "Commercial Aviation"],
    "HII": ["Defense", "Shipbuilding"],
    "LHX": ["Defense", "Technology"],

    # Consumer
    "WMT": ["Consumer", "Retail"],
    "HD": ["Consumer", "Retail"],
    "TGT": ["Consumer", "Retail"],
    "COST": ["Consumer", "Retail"],
    "LOW": ["Consumer", "Retail"],
    "NKE": ["Consumer", "Apparel"],
    "MCD": ["Consumer", "Restaurants"],
    "SBUX": ["Consumer", "Restaurants"],
    "DIS": ["Consumer", "Entertainment"],

    # Transportation
    "UNP": ["Transportation", "Railroads"],
    "CSX": ["Transportation", "Railroads"],
    "UPS": ["Transportation", "Logistics"],
    "FDX": ["Transportation", "Logistics"],
    "DAL": ["Transportation", "Airlines"],
    "UAL": ["Transportation", "Airlines"],
    "LUV": ["Transportation", "Airlines"],

    # Telecommunications
    "VZ": ["Telecommunications"],
    "T": ["Telecommunications"],
    "TMUS": ["Telecommunications"],
    "CMCSA": ["Telecommunications", "Media"],

    # Utilities
    "NEE": ["Utilities"],
    "DUK": ["Utilities"],
    "SO": ["Utilities"],
    "D": ["Utilities"],
    "AEP": ["Utilities"],
}


class CommitteeCorrelator:
    """
    Maps committee assignments to sectors and detects oversight trading.
    """

    def __init__(self, custom_mappings: Optional[Dict] = None):
        """
        Initialize the CommitteeCorrelator.

        Args:
            custom_mappings: Optional custom committee-sector mappings
        """
        self.committee_sectors = COMMITTEE_SECTORS.copy()
        if custom_mappings:
            self.committee_sectors.update(custom_mappings)

        self.stock_sectors = STOCK_SECTORS.copy()

    def get_oversight_sectors(self, member: Dict) -> List[str]:
        """
        Get all sectors under member's committee oversight.

        Args:
            member: Member data with committee assignments

        Returns:
            List of sector names member has oversight of
        """
        committees = self._get_member_committees(member)
        oversight_sectors = set()

        for committee in committees:
            committee_data = self._find_committee_data(committee)
            if committee_data:
                oversight_sectors.update(committee_data.get("sectors", []))

        return list(oversight_sectors)

    def get_oversight_stocks(self, member: Dict) -> List[str]:
        """
        Get all stocks under member's committee oversight.

        Args:
            member: Member data with committee assignments

        Returns:
            List of stock symbols member has oversight of
        """
        committees = self._get_member_committees(member)
        oversight_stocks = set()

        for committee in committees:
            committee_data = self._find_committee_data(committee)
            if committee_data:
                oversight_stocks.update(committee_data.get("stocks", []))

        return list(oversight_stocks)

    def is_oversight_trade(
        self,
        trade: Dict,
        member: Dict
    ) -> CorrelationResult:
        """
        Check if a trade is in the member's committee oversight area.

        Args:
            trade: Trade data with symbol
            member: Member data with committees

        Returns:
            CorrelationResult with details
        """
        symbol = trade.get("symbol", "").upper()
        trade_id = str(trade.get("id") or trade.get("trade_id", ""))
        committees = self._get_member_committees(member)

        if not symbol:
            return CorrelationResult(
                is_oversight=False,
                committee="",
                sectors=[],
                overlap_score=0.0,
                explanation="No symbol provided",
                trade_id=trade_id,
                symbol=symbol
            )

        # Get stock's sectors
        stock_sectors = self.stock_sectors.get(symbol, ["Unknown"])

        # Check each committee for overlap
        best_match = None
        best_score = 0.0

        for committee in committees:
            committee_data = self._find_committee_data(committee)
            if not committee_data:
                continue

            committee_stocks = committee_data.get("stocks", [])
            committee_sectors = committee_data.get("sectors", [])
            committee_keywords = committee_data.get("keywords", [])

            # Calculate overlap score
            score = 0.0

            # Direct stock match (strongest)
            if symbol in committee_stocks:
                score = 1.0

            # Sector match
            sector_overlap = set(stock_sectors) & set(committee_sectors)
            if sector_overlap:
                score = max(score, 0.7)

            # Keyword match (weakest)
            company_name = trade.get("asset_name", "").lower()
            if any(kw.lower() in company_name for kw in committee_keywords):
                score = max(score, 0.4)

            if score > best_score:
                best_score = score
                best_match = {
                    "committee": committee,
                    "sectors": list(sector_overlap) if sector_overlap else committee_sectors[:3],
                    "score": score
                }

        if best_match and best_score > 0:
            is_oversight = best_score >= 0.4

            if best_score >= 0.9:
                explanation = f"Direct stock match: {symbol} is specifically under {best_match['committee']} oversight"
            elif best_score >= 0.6:
                explanation = f"Sector overlap: {symbol} is in {', '.join(best_match['sectors'])} sector(s) overseen by {best_match['committee']}"
            else:
                explanation = f"Weak correlation: {symbol} has tangential connection to {best_match['committee']}"

            return CorrelationResult(
                is_oversight=is_oversight,
                committee=best_match["committee"],
                sectors=best_match["sectors"],
                overlap_score=best_score,
                explanation=explanation,
                trade_id=trade_id,
                symbol=symbol
            )

        return CorrelationResult(
            is_oversight=False,
            committee="",
            sectors=stock_sectors,
            overlap_score=0.0,
            explanation=f"No committee oversight found for {symbol}",
            trade_id=trade_id,
            symbol=symbol
        )

    def analyze_member_oversight_trading(
        self,
        member: Dict,
        trades: List[Dict]
    ) -> MemberOversightProfile:
        """
        Analyze all of a member's trades for oversight correlation.

        Args:
            member: Member data
            trades: List of member's trades

        Returns:
            MemberOversightProfile with statistics
        """
        member_id = str(member.get("id") or member.get("member_id", "unknown"))
        member_name = member.get("name") or member.get("full_name", "Unknown")
        committees = self._get_member_committees(member)

        oversight_sectors = self.get_oversight_sectors(member)
        oversight_stocks = self.get_oversight_stocks(member)

        total_trades = len(trades)
        oversight_trade_count = 0
        oversight_stock_counts = defaultdict(int)

        for trade in trades:
            result = self.is_oversight_trade(trade, member)
            if result.is_oversight:
                oversight_trade_count += 1
                symbol = trade.get("symbol", "").upper()
                if symbol:
                    oversight_stock_counts[symbol] += 1

        oversight_percentage = (
            (oversight_trade_count / total_trades * 100)
            if total_trades > 0 else 0.0
        )

        # Get top oversight stocks
        top_stocks = sorted(
            oversight_stock_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        return MemberOversightProfile(
            member_id=member_id,
            member_name=member_name,
            committees=committees,
            oversight_sectors=oversight_sectors,
            oversight_stocks=oversight_stocks,
            total_trades=total_trades,
            oversight_trades=oversight_trade_count,
            oversight_percentage=oversight_percentage,
            top_oversight_stocks=top_stocks
        )

    def rank_members_by_oversight_trading(
        self,
        members: Dict[str, Dict],
        member_trades: Dict[str, List[Dict]]
    ) -> List[MemberOversightProfile]:
        """
        Rank all members by their oversight trading percentage.

        Args:
            members: Dict mapping member_id -> member data
            member_trades: Dict mapping member_id -> list of trades

        Returns:
            List of MemberOversightProfile sorted by oversight_percentage (highest first)
        """
        profiles = []

        for member_id, member in members.items():
            trades = member_trades.get(member_id, [])
            if not trades:
                continue

            profile = self.analyze_member_oversight_trading(member, trades)
            profiles.append(profile)

        # Sort by oversight percentage (highest first)
        profiles.sort(key=lambda p: p.oversight_percentage, reverse=True)

        # Assign ranks
        for i, profile in enumerate(profiles):
            profile.rank = i + 1

        return profiles

    def get_visualization_data(
        self,
        profiles: List[MemberOversightProfile]
    ) -> Dict:
        """
        Generate data ready for dashboard visualizations.

        Returns data for:
        - Network graph (member -> committee -> sector -> stocks)
        - Heat map (committee x sector trading volume)
        - Leaderboard (members ranked by oversight trading %)
        """
        # Network nodes
        member_nodes = []
        committee_nodes = []
        sector_nodes = set()
        stock_nodes = set()

        # Network edges
        member_committee_edges = []
        committee_sector_edges = []

        # Heat map data
        committee_sector_matrix = defaultdict(lambda: defaultdict(int))

        for profile in profiles:
            # Member node
            member_nodes.append({
                "id": f"member_{profile.member_id}",
                "type": "member",
                "label": profile.member_name,
                "oversight_pct": profile.oversight_percentage
            })

            for committee in profile.committees:
                # Committee node
                committee_id = f"committee_{committee.replace(' ', '_')}"
                if not any(c["id"] == committee_id for c in committee_nodes):
                    committee_nodes.append({
                        "id": committee_id,
                        "type": "committee",
                        "label": committee
                    })

                # Member -> Committee edge
                member_committee_edges.append({
                    "source": f"member_{profile.member_id}",
                    "target": committee_id
                })

                # Get committee sectors
                committee_data = self._find_committee_data(committee)
                if committee_data:
                    for sector in committee_data.get("sectors", []):
                        sector_id = f"sector_{sector.replace(' ', '_')}"
                        sector_nodes.add(sector)

                        # Committee -> Sector edge
                        committee_sector_edges.append({
                            "source": committee_id,
                            "target": sector_id
                        })

            # Update heat map with oversight trades
            for committee in profile.committees:
                for sector in profile.oversight_sectors:
                    committee_sector_matrix[committee][sector] += profile.oversight_trades

        # Leaderboard data
        leaderboard = [
            {
                "rank": p.rank,
                "member_name": p.member_name,
                "oversight_percentage": round(p.oversight_percentage, 1),
                "oversight_trades": p.oversight_trades,
                "total_trades": p.total_trades,
                "committees": p.committees[:2]  # Top 2 committees
            }
            for p in profiles[:20]  # Top 20
        ]

        return {
            "network": {
                "nodes": member_nodes + committee_nodes + [
                    {"id": f"sector_{s.replace(' ', '_')}", "type": "sector", "label": s}
                    for s in sector_nodes
                ],
                "edges": member_committee_edges + committee_sector_edges
            },
            "heat_map": {
                committee: dict(sectors)
                for committee, sectors in committee_sector_matrix.items()
            },
            "leaderboard": leaderboard
        }

    def _get_member_committees(self, member: Dict) -> List[str]:
        """Extract committee list from member data."""
        committees = member.get("committees", [])
        if isinstance(committees, str):
            committees = [committees]

        committee = member.get("committee", "")
        if committee and committee not in committees:
            committees.append(committee)

        return [c for c in committees if c]  # Filter empty

    def _find_committee_data(self, committee_name: str) -> Optional[Dict]:
        """Find committee data by name (case-insensitive partial match)."""
        committee_lower = committee_name.lower()

        # Direct match first
        for comm_key in self.committee_sectors:
            if comm_key.lower() == committee_lower:
                return self.committee_sectors[comm_key]

        # Partial match
        for comm_key in self.committee_sectors:
            if comm_key.lower() in committee_lower or committee_lower in comm_key.lower():
                return self.committee_sectors[comm_key]

        return None

    def get_summary_statistics(
        self,
        profiles: List[MemberOversightProfile]
    ) -> Dict:
        """
        Generate summary statistics from oversight analysis.

        Args:
            profiles: List of MemberOversightProfile objects

        Returns:
            Dictionary with summary statistics
        """
        if not profiles:
            return {"error": "No profiles to analyze"}

        oversight_pcts = [p.oversight_percentage for p in profiles]
        trading_members = [p for p in profiles if p.total_trades > 0]
        high_oversight = [p for p in profiles if p.oversight_percentage > 50]

        return {
            "total_members_analyzed": len(profiles),
            "members_with_trades": len(trading_members),
            "average_oversight_pct": round(statistics.mean(oversight_pcts), 1) if oversight_pcts else 0,
            "median_oversight_pct": round(statistics.median(oversight_pcts), 1) if oversight_pcts else 0,
            "max_oversight_pct": round(max(oversight_pcts), 1) if oversight_pcts else 0,
            "members_over_50_pct_oversight": len(high_oversight),
            "top_5_oversight_traders": [
                {
                    "name": p.member_name,
                    "oversight_pct": round(p.oversight_percentage, 1),
                    "committees": p.committees[:2]
                }
                for p in profiles[:5]
            ]
        }


def main():
    """Example usage and testing of the CommitteeCorrelator."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Congressional Trading Committee Correlation Engine - Test Run")
    print("=" * 70)

    # Sample member data
    sample_members = {
        "M001": {
            "id": "M001",
            "name": "Nancy Pelosi",
            "party": "D",
            "state": "CA",
            "committees": ["House Intelligence", "House Financial Services"]
        },
        "M002": {
            "id": "M002",
            "name": "Joe Manchin",
            "party": "D",
            "state": "WV",
            "committees": ["Senate Energy and Natural Resources"],
            "leadership_role": "Energy Committee Chair"
        },
        "M003": {
            "id": "M003",
            "name": "Pat Toomey",
            "party": "R",
            "state": "PA",
            "committees": ["Senate Banking", "Senate Finance"],
            "leadership_role": "Former Banking Committee Chair"
        },
        "M004": {
            "id": "M004",
            "name": "Richard Burr",
            "party": "R",
            "state": "NC",
            "committees": ["Senate Intelligence", "Senate Health, Education, Labor and Pensions"]
        }
    }

    # Sample trades by member
    sample_trades = {
        "M001": [
            {"id": "T001", "symbol": "NVDA", "asset_name": "NVIDIA Corporation"},
            {"id": "T002", "symbol": "GOOGL", "asset_name": "Alphabet Inc"},
            {"id": "T003", "symbol": "JPM", "asset_name": "JPMorgan Chase"},
            {"id": "T004", "symbol": "AAPL", "asset_name": "Apple Inc"},
        ],
        "M002": [
            {"id": "T005", "symbol": "XOM", "asset_name": "Exxon Mobil"},
            {"id": "T006", "symbol": "CVX", "asset_name": "Chevron"},
            {"id": "T007", "symbol": "SLB", "asset_name": "Schlumberger"},
            {"id": "T008", "symbol": "AAPL", "asset_name": "Apple Inc"},
        ],
        "M003": [
            {"id": "T009", "symbol": "JPM", "asset_name": "JPMorgan Chase"},
            {"id": "T010", "symbol": "BAC", "asset_name": "Bank of America"},
            {"id": "T011", "symbol": "COIN", "asset_name": "Coinbase"},
            {"id": "T012", "symbol": "WMT", "asset_name": "Walmart"},
        ],
        "M004": [
            {"id": "T013", "symbol": "HCA", "asset_name": "HCA Healthcare"},
            {"id": "T014", "symbol": "PFE", "asset_name": "Pfizer"},
            {"id": "T015", "symbol": "PANW", "asset_name": "Palo Alto Networks"},
            {"id": "T016", "symbol": "CRWD", "asset_name": "CrowdStrike"},
        ]
    }

    # Initialize correlator
    correlator = CommitteeCorrelator()

    # Test individual trade correlation
    print("\n1. Testing Individual Trade Correlations:")
    print("-" * 70)

    for member_id, trades in sample_trades.items():
        member = sample_members[member_id]
        print(f"\n{member['name']} ({', '.join(member['committees'][:2])}):")

        for trade in trades:
            result = correlator.is_oversight_trade(trade, member)
            status = "YES" if result.is_oversight else "No"
            print(f"  {trade['symbol']}: {status} (score: {result.overlap_score:.2f}) - {result.explanation[:50]}")

    # Test member oversight profiles
    print("\n" + "=" * 70)
    print("2. Member Oversight Trading Profiles:")
    print("-" * 70)

    profiles = correlator.rank_members_by_oversight_trading(sample_members, sample_trades)

    for profile in profiles:
        print(f"\n{profile.member_name} (Rank #{profile.rank}):")
        print(f"  Committees: {', '.join(profile.committees)}")
        print(f"  Oversight Sectors: {', '.join(profile.oversight_sectors[:3])}")
        print(f"  Trades: {profile.oversight_trades}/{profile.total_trades} in oversight areas")
        print(f"  Oversight %: {profile.oversight_percentage:.1f}%")
        if profile.top_oversight_stocks:
            print(f"  Top Oversight Stocks: {', '.join(s for s, _ in profile.top_oversight_stocks[:3])}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("3. Summary Statistics:")
    print("-" * 70)

    stats = correlator.get_summary_statistics(profiles)
    print(f"Total members analyzed: {stats['total_members_analyzed']}")
    print(f"Members with trades: {stats['members_with_trades']}")
    print(f"Average oversight %: {stats['average_oversight_pct']}%")
    print(f"Median oversight %: {stats['median_oversight_pct']}%")
    print(f"Members >50% oversight: {stats['members_over_50_pct_oversight']}")

    print("\nTop 5 Oversight Traders:")
    for trader in stats['top_5_oversight_traders']:
        print(f"  {trader['name']}: {trader['oversight_pct']}%")

    # Visualization data
    print("\n" + "=" * 70)
    print("4. Visualization Data (Sample):")
    print("-" * 70)

    viz_data = correlator.get_visualization_data(profiles)
    print(f"Network nodes: {len(viz_data['network']['nodes'])}")
    print(f"Network edges: {len(viz_data['network']['edges'])}")
    print(f"Leaderboard entries: {len(viz_data['leaderboard'])}")


if __name__ == "__main__":
    main()
