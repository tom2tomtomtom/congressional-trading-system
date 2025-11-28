"""
Data services for the Congressional Trading Intelligence API.
Provides sample data and business logic for API endpoints.
"""

import random
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any, Tuple
from decimal import Decimal

from .schemas import (
    MemberSummary, MemberDetail, CommitteeResponse,
    TradeSummary, TradeDetail,
    ConvictionAnalysis, ConvictionFactor, TimingAnalysis, SuspiciousEvent,
    SwampScore, SwampScoreComponents,
    LeaderboardEntry, Leaderboard,
    AggregateStats,
    StoryFormat, StoryResponse,
    PartyEnum, ChamberEnum, TransactionTypeEnum, AssetTypeEnum, AlertLevelEnum,
    AccessTierEnum
)


# Sample data - In production, this would come from the database
SAMPLE_MEMBERS = [
    {
        "id": 1, "bioguide_id": "P000197", "full_name": "Nancy Pelosi",
        "first_name": "Nancy", "last_name": "Pelosi",
        "party": "D", "state": "CA", "chamber": "house", "district": "11",
        "is_active": True, "website": "https://pelosi.house.gov",
        "twitter_handle": "SpeakerPelosi",
        "leadership_role": "Former Speaker",
        "committees": ["Financial Services"],
        "trade_count": 45, "total_trade_volume": 15000000
    },
    {
        "id": 2, "bioguide_id": "M001157", "full_name": "Michael McCaul",
        "first_name": "Michael", "last_name": "McCaul",
        "party": "R", "state": "TX", "chamber": "house", "district": "10",
        "is_active": True, "website": "https://mccaul.house.gov",
        "leadership_role": "Foreign Affairs Chair",
        "committees": ["Foreign Affairs", "Homeland Security"],
        "trade_count": 32, "total_trade_volume": 8500000
    },
    {
        "id": 3, "bioguide_id": "T000476", "full_name": "Tommy Tuberville",
        "first_name": "Tommy", "last_name": "Tuberville",
        "party": "R", "state": "AL", "chamber": "senate", "district": None,
        "is_active": True, "website": "https://tuberville.senate.gov",
        "committees": ["Armed Services", "Agriculture"],
        "trade_count": 132, "total_trade_volume": 3200000
    },
    {
        "id": 4, "bioguide_id": "B001310", "full_name": "Richard Burr",
        "first_name": "Richard", "last_name": "Burr",
        "party": "R", "state": "NC", "chamber": "senate", "district": None,
        "is_active": False, "website": "https://burr.senate.gov",
        "leadership_role": "Former Intelligence Chair",
        "committees": ["Intelligence", "Health"],
        "trade_count": 28, "total_trade_volume": 1700000
    },
    {
        "id": 5, "bioguide_id": "L000579", "full_name": "Kelly Loeffler",
        "first_name": "Kelly", "last_name": "Loeffler",
        "party": "R", "state": "GA", "chamber": "senate", "district": None,
        "is_active": False,
        "committees": ["Agriculture", "Health"],
        "trade_count": 156, "total_trade_volume": 25000000
    },
    {
        "id": 6, "bioguide_id": "G000576", "full_name": "Josh Gottheimer",
        "first_name": "Josh", "last_name": "Gottheimer",
        "party": "D", "state": "NJ", "chamber": "house", "district": "5",
        "is_active": True, "website": "https://gottheimer.house.gov",
        "committees": ["Financial Services"],
        "trade_count": 67, "total_trade_volume": 4200000
    },
    {
        "id": 7, "bioguide_id": "M001183", "full_name": "Joe Manchin",
        "first_name": "Joe", "last_name": "Manchin",
        "party": "D", "state": "WV", "chamber": "senate", "district": None,
        "is_active": True, "website": "https://manchin.senate.gov",
        "committees": ["Energy and Natural Resources", "Appropriations"],
        "trade_count": 23, "total_trade_volume": 5600000
    },
    {
        "id": 8, "bioguide_id": "W000805", "full_name": "Mark Warner",
        "first_name": "Mark", "last_name": "Warner",
        "party": "D", "state": "VA", "chamber": "senate", "district": None,
        "is_active": True, "website": "https://warner.senate.gov",
        "leadership_role": "Intelligence Chair",
        "committees": ["Intelligence", "Banking", "Finance"],
        "trade_count": 18, "total_trade_volume": 12500000
    },
    {
        "id": 9, "bioguide_id": "C001120", "full_name": "Dan Crenshaw",
        "first_name": "Dan", "last_name": "Crenshaw",
        "party": "R", "state": "TX", "chamber": "house", "district": "2",
        "is_active": True, "website": "https://crenshaw.house.gov",
        "committees": ["Energy and Commerce", "Intelligence"],
        "trade_count": 42, "total_trade_volume": 2100000
    },
    {
        "id": 10, "bioguide_id": "G000553", "full_name": "Marjorie Taylor Greene",
        "first_name": "Marjorie", "last_name": "Greene",
        "party": "R", "state": "GA", "chamber": "house", "district": "14",
        "is_active": True, "website": "https://greene.house.gov",
        "committees": ["Oversight", "Homeland Security"],
        "trade_count": 38, "total_trade_volume": 890000
    },
]

SAMPLE_TRADES = [
    {
        "id": 1, "member_id": 1, "symbol": "NVDA", "asset_type": "stock",
        "transaction_type": "purchase", "transaction_date": "2024-06-15",
        "filing_date": "2024-07-12", "filing_delay_days": 27,
        "amount_min": 1000000, "amount_max": 5000000, "amount_mid": 2500000,
        "description": "NVIDIA Corporation - Common Stock",
        "conviction_score": 78
    },
    {
        "id": 2, "member_id": 1, "symbol": "AAPL", "asset_type": "stock",
        "transaction_type": "sale", "transaction_date": "2024-05-20",
        "filing_date": "2024-06-28", "filing_delay_days": 39,
        "amount_min": 500000, "amount_max": 1000000, "amount_mid": 750000,
        "description": "Apple Inc - Common Stock",
        "conviction_score": 45
    },
    {
        "id": 3, "member_id": 3, "symbol": "TSLA", "asset_type": "stock",
        "transaction_type": "purchase", "transaction_date": "2024-07-01",
        "filing_date": "2024-08-15", "filing_delay_days": 45,
        "amount_min": 50000, "amount_max": 100000, "amount_mid": 75000,
        "description": "Tesla Inc - Common Stock",
        "conviction_score": 62
    },
    {
        "id": 4, "member_id": 5, "symbol": "MSFT", "asset_type": "stock",
        "transaction_type": "sale", "transaction_date": "2020-02-14",
        "filing_date": "2020-03-20", "filing_delay_days": 35,
        "amount_min": 250000, "amount_max": 500000, "amount_mid": 375000,
        "description": "Microsoft Corporation - Common Stock (COVID sale)",
        "conviction_score": 92
    },
    {
        "id": 5, "member_id": 4, "symbol": "IHG", "asset_type": "stock",
        "transaction_type": "sale", "transaction_date": "2020-02-13",
        "filing_date": "2020-02-27", "filing_delay_days": 14,
        "amount_min": 50000, "amount_max": 100000, "amount_mid": 75000,
        "description": "InterContinental Hotels - Common Stock (COVID sale)",
        "conviction_score": 95
    },
    {
        "id": 6, "member_id": 2, "symbol": "RTX", "asset_type": "stock",
        "transaction_type": "purchase", "transaction_date": "2024-03-10",
        "filing_date": "2024-04-08", "filing_delay_days": 29,
        "amount_min": 100000, "amount_max": 250000, "amount_mid": 175000,
        "description": "Raytheon Technologies - Common Stock",
        "conviction_score": 71
    },
    {
        "id": 7, "member_id": 7, "symbol": "XOM", "asset_type": "stock",
        "transaction_type": "purchase", "transaction_date": "2024-01-22",
        "filing_date": "2024-02-28", "filing_delay_days": 37,
        "amount_min": 500000, "amount_max": 1000000, "amount_mid": 750000,
        "description": "Exxon Mobil Corporation - Common Stock",
        "conviction_score": 83
    },
    {
        "id": 8, "member_id": 6, "symbol": "JPM", "asset_type": "stock",
        "transaction_type": "purchase", "transaction_date": "2024-04-05",
        "filing_date": "2024-05-10", "filing_delay_days": 35,
        "amount_min": 100000, "amount_max": 250000, "amount_mid": 175000,
        "description": "JPMorgan Chase & Co - Common Stock",
        "conviction_score": 68
    },
    {
        "id": 9, "member_id": 9, "symbol": "CVX", "asset_type": "stock",
        "transaction_type": "sale", "transaction_date": "2024-05-15",
        "filing_date": "2024-06-20", "filing_delay_days": 36,
        "amount_min": 15000, "amount_max": 50000, "amount_mid": 32500,
        "description": "Chevron Corporation - Common Stock",
        "conviction_score": 55
    },
    {
        "id": 10, "member_id": 10, "symbol": "META", "asset_type": "stock",
        "transaction_type": "purchase", "transaction_date": "2024-06-01",
        "filing_date": "2024-07-15", "filing_delay_days": 44,
        "amount_min": 15000, "amount_max": 50000, "amount_mid": 32500,
        "description": "Meta Platforms Inc - Common Stock",
        "conviction_score": 41
    },
]


class MemberService:
    """Service for member-related operations."""

    @staticmethod
    def get_members(
        page: int = 1,
        per_page: int = 20,
        party: Optional[str] = None,
        chamber: Optional[str] = None,
        state: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Tuple[List[MemberSummary], int]:
        """Get paginated list of members with optional filters."""
        filtered = SAMPLE_MEMBERS.copy()

        # Apply filters
        if party:
            filtered = [m for m in filtered if m["party"] == party.upper()]
        if chamber:
            filtered = [m for m in filtered if m["chamber"] == chamber.lower()]
        if state:
            filtered = [m for m in filtered if m["state"] == state.upper()]
        if search:
            search_lower = search.lower()
            filtered = [m for m in filtered if search_lower in m["full_name"].lower()]

        total = len(filtered)

        # Paginate
        start = (page - 1) * per_page
        end = start + per_page
        paginated = filtered[start:end]

        members = [
            MemberSummary(
                id=m["id"],
                bioguide_id=m["bioguide_id"],
                full_name=m["full_name"],
                party=PartyEnum(m["party"]),
                state=m["state"],
                chamber=ChamberEnum(m["chamber"]),
                district=m.get("district"),
                is_active=m.get("is_active", True),
                trade_count=m.get("trade_count", 0),
            )
            for m in paginated
        ]

        return members, total

    @staticmethod
    def get_member(member_id: int) -> Optional[MemberDetail]:
        """Get detailed member information."""
        member = next((m for m in SAMPLE_MEMBERS if m["id"] == member_id), None)
        if not member:
            return None

        committees = [
            CommitteeResponse(id=i+1, name=name, is_active=True)
            for i, name in enumerate(member.get("committees", []))
        ]

        return MemberDetail(
            id=member["id"],
            bioguide_id=member["bioguide_id"],
            full_name=member["full_name"],
            first_name=member["first_name"],
            last_name=member["last_name"],
            party=PartyEnum(member["party"]),
            state=member["state"],
            chamber=ChamberEnum(member["chamber"]),
            district=member.get("district"),
            is_active=member.get("is_active", True),
            website=member.get("website"),
            twitter_handle=member.get("twitter_handle"),
            leadership_role=member.get("leadership_role"),
            committees=committees,
            trade_count=member.get("trade_count", 0),
            total_trade_volume=member.get("total_trade_volume", 0),
        )


class TradeService:
    """Service for trade-related operations."""

    @staticmethod
    def get_trades(
        page: int = 1,
        per_page: int = 20,
        member_id: Optional[int] = None,
        symbol: Optional[str] = None,
        transaction_type: Optional[str] = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> Tuple[List[TradeSummary], int]:
        """Get paginated list of trades with optional filters."""
        filtered = SAMPLE_TRADES.copy()

        # Apply filters
        if member_id:
            filtered = [t for t in filtered if t["member_id"] == member_id]
        if symbol:
            filtered = [t for t in filtered if t["symbol"] == symbol.upper()]
        if transaction_type:
            filtered = [t for t in filtered if t["transaction_type"] == transaction_type.lower()]
        if min_amount:
            filtered = [t for t in filtered if t["amount_mid"] >= min_amount]
        if max_amount:
            filtered = [t for t in filtered if t["amount_mid"] <= max_amount]
        if start_date:
            filtered = [t for t in filtered if t["transaction_date"] >= str(start_date)]
        if end_date:
            filtered = [t for t in filtered if t["transaction_date"] <= str(end_date)]

        total = len(filtered)

        # Paginate
        start = (page - 1) * per_page
        end = start + per_page
        paginated = filtered[start:end]

        # Get member names
        member_map = {m["id"]: m["full_name"] for m in SAMPLE_MEMBERS}

        trades = [
            TradeSummary(
                id=t["id"],
                member_id=t["member_id"],
                member_name=member_map.get(t["member_id"]),
                symbol=t["symbol"],
                asset_type=AssetTypeEnum(t["asset_type"]),
                transaction_type=TransactionTypeEnum(t["transaction_type"]),
                transaction_date=date.fromisoformat(t["transaction_date"]),
                filing_date=date.fromisoformat(t["filing_date"]) if t.get("filing_date") else None,
                filing_delay_days=t.get("filing_delay_days"),
                amount_min=t.get("amount_min"),
                amount_max=t.get("amount_max"),
                amount_mid=t.get("amount_mid"),
            )
            for t in paginated
        ]

        return trades, total

    @staticmethod
    def get_trade(trade_id: int) -> Optional[TradeDetail]:
        """Get detailed trade information."""
        trade = next((t for t in SAMPLE_TRADES if t["id"] == trade_id), None)
        if not trade:
            return None

        member_map = {m["id"]: m["full_name"] for m in SAMPLE_MEMBERS}

        return TradeDetail(
            id=trade["id"],
            member_id=trade["member_id"],
            member_name=member_map.get(trade["member_id"]),
            symbol=trade["symbol"],
            asset_type=AssetTypeEnum(trade["asset_type"]),
            transaction_type=TransactionTypeEnum(trade["transaction_type"]),
            transaction_date=date.fromisoformat(trade["transaction_date"]),
            filing_date=date.fromisoformat(trade["filing_date"]) if trade.get("filing_date") else None,
            filing_delay_days=trade.get("filing_delay_days"),
            amount_min=trade.get("amount_min"),
            amount_max=trade.get("amount_max"),
            amount_mid=trade.get("amount_mid"),
            description=trade.get("description"),
            is_late_filing=trade.get("filing_delay_days", 0) > 45,
            is_large_trade=trade.get("amount_mid", 0) >= 50000,
            conviction_score=trade.get("conviction_score"),
        )

    @staticmethod
    def get_member_trades(
        member_id: int,
        page: int = 1,
        per_page: int = 20
    ) -> Tuple[List[TradeSummary], int]:
        """Get trades for a specific member."""
        return TradeService.get_trades(page=page, per_page=per_page, member_id=member_id)


class AnalysisService:
    """Service for analysis-related operations."""

    @staticmethod
    def get_conviction_analysis(trade_id: int) -> Optional[ConvictionAnalysis]:
        """Get conviction score analysis for a trade."""
        trade = next((t for t in SAMPLE_TRADES if t["id"] == trade_id), None)
        if not trade:
            return None

        member = next((m for m in SAMPLE_MEMBERS if m["id"] == trade["member_id"]), None)
        base_score = trade.get("conviction_score", 50)

        # Generate factors
        factors = [
            ConvictionFactor(
                name="Committee Access",
                weight=25,
                score=base_score * 0.25 * random.uniform(0.8, 1.2),
                max_score=25,
                explanation=f"Member serves on committees with potential oversight of {trade['symbol']}"
            ),
            ConvictionFactor(
                name="Timing Proximity",
                weight=25,
                score=base_score * 0.25 * random.uniform(0.8, 1.2),
                max_score=25,
                explanation="Trade occurred within suspicious window of market-moving events"
            ),
            ConvictionFactor(
                name="Filing Delay",
                weight=15,
                score=min(15, trade.get("filing_delay_days", 0) / 3),
                max_score=15,
                explanation=f"Filing was delayed {trade.get('filing_delay_days', 0)} days (45 day limit)"
            ),
            ConvictionFactor(
                name="Trade Size Anomaly",
                weight=15,
                score=base_score * 0.15 * random.uniform(0.7, 1.3),
                max_score=15,
                explanation="Trade size compared to member's typical trading patterns"
            ),
            ConvictionFactor(
                name="Historical Pattern",
                weight=10,
                score=base_score * 0.10 * random.uniform(0.6, 1.4),
                max_score=10,
                explanation="Deviation from member's historical trading behavior"
            ),
            ConvictionFactor(
                name="Sector Concentration",
                weight=10,
                score=base_score * 0.10 * random.uniform(0.5, 1.5),
                max_score=10,
                explanation="Concentration of trades in sectors under member's oversight"
            ),
        ]

        # Determine risk level
        if base_score >= 80:
            risk_level = AlertLevelEnum.CRITICAL
        elif base_score >= 60:
            risk_level = AlertLevelEnum.HIGH
        elif base_score >= 40:
            risk_level = AlertLevelEnum.MEDIUM
        else:
            risk_level = AlertLevelEnum.LOW

        member_name = member["full_name"] if member else "Unknown"
        explanation = (
            f"This trade by {member_name} scored {base_score}/100 on our conviction analysis. "
            f"The primary factors were committee access ({factors[0].score:.1f}/25) and "
            f"timing proximity ({factors[1].score:.1f}/25). "
            f"The filing was made {trade.get('filing_delay_days', 0)} days after the transaction."
        )

        return ConvictionAnalysis(
            trade_id=trade_id,
            score=base_score,
            factors=factors,
            explanation=explanation,
            risk_level=risk_level,
            analyzed_at=datetime.utcnow()
        )

    @staticmethod
    def get_timing_analysis(trade_id: int) -> Optional[TimingAnalysis]:
        """Get timing analysis for a trade."""
        trade = next((t for t in SAMPLE_TRADES if t["id"] == trade_id), None)
        if not trade:
            return None

        trade_date = date.fromisoformat(trade["transaction_date"])

        # Generate sample suspicious events
        events = []

        # Committee hearing event
        if random.random() > 0.3:
            events.append(SuspiciousEvent(
                event_type="committee_hearing",
                event_date=trade_date + timedelta(days=random.randint(3, 14)),
                days_before_trade=None,
                days_after_trade=random.randint(3, 14),
                member_access_level="committee",
                description=f"Committee hearing on {trade['symbol']} sector scheduled",
                relevance_score=random.uniform(0.6, 0.95)
            ))

        # Earnings announcement
        if random.random() > 0.4:
            events.append(SuspiciousEvent(
                event_type="earnings_announcement",
                event_date=trade_date + timedelta(days=random.randint(5, 21)),
                days_before_trade=None,
                days_after_trade=random.randint(5, 21),
                member_access_level="public",
                description=f"{trade['symbol']} quarterly earnings announcement",
                relevance_score=random.uniform(0.4, 0.8)
            ))

        # Regulatory action
        if trade.get("conviction_score", 0) > 70:
            events.append(SuspiciousEvent(
                event_type="regulatory_action",
                event_date=trade_date + timedelta(days=random.randint(7, 30)),
                days_before_trade=None,
                days_after_trade=random.randint(7, 30),
                member_access_level="direct",
                description=f"Regulatory announcement affecting {trade['symbol']}",
                relevance_score=random.uniform(0.7, 0.98)
            ))

        timing_score = sum(e.relevance_score for e in events) / max(len(events), 1)

        summary = (
            f"Analysis identified {len(events)} potentially related events "
            f"within 30 days of this trade. "
            + ("Timing patterns suggest possible advance knowledge." if timing_score > 0.7 else
               "No conclusive evidence of information advantage." if timing_score < 0.4 else
               "Some timing correlations warrant further investigation.")
        )

        return TimingAnalysis(
            trade_id=trade_id,
            suspicious_events=events,
            timing_score=timing_score,
            summary=summary,
            analyzed_at=datetime.utcnow()
        )

    @staticmethod
    def get_member_score(member_id: int) -> Optional[SwampScore]:
        """Get swamp score for a member."""
        member = next((m for m in SAMPLE_MEMBERS if m["id"] == member_id), None)
        if not member:
            return None

        # Calculate component scores
        member_trades = [t for t in SAMPLE_TRADES if t["member_id"] == member_id]
        avg_conviction = sum(t.get("conviction_score", 50) for t in member_trades) / max(len(member_trades), 1)

        components = SwampScoreComponents(
            avg_conviction_score=avg_conviction,
            filing_compliance_rate=random.uniform(0.7, 0.98),
            oversight_trading_pct=random.uniform(0.2, 0.6),
            timing_suspicion=random.uniform(0.1, 0.8),
            volume_anomaly=random.uniform(0.1, 0.5)
        )

        # Calculate total score
        total_score = int(
            components.avg_conviction_score * 0.40 +
            (1 - components.filing_compliance_rate) * 100 * 0.20 +
            components.oversight_trading_pct * 100 * 0.20 +
            components.timing_suspicion * 100 * 0.10 +
            components.volume_anomaly * 100 * 0.10
        )

        # Determine rank and percentile
        all_scores = [(m["id"], random.randint(20, 90)) for m in SAMPLE_MEMBERS]
        all_scores.append((member_id, total_score))
        all_scores.sort(key=lambda x: x[1], reverse=True)
        rank = next(i+1 for i, (mid, _) in enumerate(all_scores) if mid == member_id)
        percentile = int((len(all_scores) - rank) / len(all_scores) * 100)

        # Determine trend
        trend = random.choice(["improving", "worsening", "stable"])

        explanation = (
            f"{member['full_name']} ranks #{rank} out of {len(SAMPLE_MEMBERS)} tracked members "
            f"with a Swamp Score of {total_score}. "
            f"Average conviction score on trades: {avg_conviction:.1f}. "
            f"Filing compliance: {components.filing_compliance_rate*100:.1f}%. "
            f"Trend: {trend}."
        )

        return SwampScore(
            member_id=member_id,
            total_score=total_score,
            rank=rank,
            percentile=percentile,
            components=components,
            trend=trend,
            explanation=explanation,
            calculated_at=datetime.utcnow()
        )

    @staticmethod
    def get_leaderboard(category: str = "conviction") -> Leaderboard:
        """Get leaderboard rankings."""
        entries = []

        if category == "conviction":
            # Rank by average conviction score
            for i, member in enumerate(sorted(SAMPLE_MEMBERS, key=lambda m: m.get("trade_count", 0), reverse=True)[:20]):
                member_trades = [t for t in SAMPLE_TRADES if t["member_id"] == member["id"]]
                avg_score = sum(t.get("conviction_score", 50) for t in member_trades) / max(len(member_trades), 1)
                entries.append(LeaderboardEntry(
                    rank=i+1,
                    member_id=member["id"],
                    full_name=member["full_name"],
                    party=PartyEnum(member["party"]),
                    state=member["state"],
                    chamber=ChamberEnum(member["chamber"]),
                    score=avg_score,
                    metric_value=len(member_trades)
                ))
            description = "Members ranked by average conviction score on trades"

        elif category == "volume":
            # Rank by trade volume
            for i, member in enumerate(sorted(SAMPLE_MEMBERS, key=lambda m: m.get("total_trade_volume", 0), reverse=True)[:20]):
                entries.append(LeaderboardEntry(
                    rank=i+1,
                    member_id=member["id"],
                    full_name=member["full_name"],
                    party=PartyEnum(member["party"]),
                    state=member["state"],
                    chamber=ChamberEnum(member["chamber"]),
                    score=member.get("total_trade_volume", 0),
                    metric_value=member.get("trade_count", 0)
                ))
            description = "Members ranked by total trade volume"

        elif category == "activity":
            # Rank by trade count
            for i, member in enumerate(sorted(SAMPLE_MEMBERS, key=lambda m: m.get("trade_count", 0), reverse=True)[:20]):
                entries.append(LeaderboardEntry(
                    rank=i+1,
                    member_id=member["id"],
                    full_name=member["full_name"],
                    party=PartyEnum(member["party"]),
                    state=member["state"],
                    chamber=ChamberEnum(member["chamber"]),
                    score=member.get("trade_count", 0),
                    metric_value=member.get("total_trade_volume", 0)
                ))
            description = "Members ranked by number of trades"

        else:
            description = f"Unknown category: {category}"

        return Leaderboard(
            category=category,
            description=description,
            entries=entries,
            total_count=len(entries),
            as_of=datetime.utcnow()
        )

    @staticmethod
    def get_stats() -> AggregateStats:
        """Get aggregate statistics."""
        # Count by party
        party_counts = {}
        for member in SAMPLE_MEMBERS:
            party = member["party"]
            party_counts[party] = party_counts.get(party, 0) + 1

        # Count by chamber
        chamber_counts = {}
        for member in SAMPLE_MEMBERS:
            chamber = member["chamber"]
            chamber_counts[chamber] = chamber_counts.get(chamber, 0) + 1

        # Count by transaction type
        type_counts = {}
        for trade in SAMPLE_TRADES:
            t_type = trade["transaction_type"]
            type_counts[t_type] = type_counts.get(t_type, 0) + 1

        # Calculate averages
        total_volume = sum(t.get("amount_mid", 0) for t in SAMPLE_TRADES)
        avg_conviction = sum(t.get("conviction_score", 50) for t in SAMPLE_TRADES) / len(SAMPLE_TRADES)
        high_conviction = len([t for t in SAMPLE_TRADES if t.get("conviction_score", 0) >= 70])
        late_filings = len([t for t in SAMPLE_TRADES if t.get("filing_delay_days", 0) > 45])

        return AggregateStats(
            total_members=len(SAMPLE_MEMBERS),
            total_trades=len(SAMPLE_TRADES),
            total_trade_volume=total_volume,
            members_by_party=party_counts,
            members_by_chamber=chamber_counts,
            trades_by_type=type_counts,
            avg_conviction_score=avg_conviction,
            high_conviction_trade_count=high_conviction,
            late_filing_count=late_filings,
            as_of=datetime.utcnow()
        )


class StoryService:
    """Service for story generation."""

    @staticmethod
    def generate_story(
        trade_id: Optional[int] = None,
        member_id: Optional[int] = None,
        format: StoryFormat = StoryFormat.NEWS_BRIEF
    ) -> Optional[StoryResponse]:
        """Generate a story about a trade or member."""
        import uuid

        if trade_id:
            trade = next((t for t in SAMPLE_TRADES if t["id"] == trade_id), None)
            if not trade:
                return None
            member = next((m for m in SAMPLE_MEMBERS if m["id"] == trade["member_id"]), None)
        elif member_id:
            member = next((m for m in SAMPLE_MEMBERS if m["id"] == member_id), None)
            if not member:
                return None
            trade = None
        else:
            return None

        member_name = member["full_name"] if member else "Unknown Member"

        # Generate content based on format
        if format == StoryFormat.TWEET:
            if trade:
                content = (
                    f"🚨 NEW: {member_name} ({member['party']}-{member['state']}) "
                    f"disclosed a ${trade['amount_mid']:,.0f} {trade['transaction_type']} "
                    f"of ${trade['symbol']}. Conviction score: {trade.get('conviction_score', 'N/A')}/100 "
                    f"#CongressTrading #StockAct"
                )
            else:
                content = (
                    f"📊 PROFILE: {member_name} ({member['party']}-{member['state']}) "
                    f"has {member.get('trade_count', 0)} trades worth ${member.get('total_trade_volume', 0):,.0f} "
                    f"on record. #CongressTrading"
                )
            title = "Tweet"

        elif format == StoryFormat.NEWS_BRIEF:
            if trade:
                title = f"{member_name} Discloses {trade['transaction_type'].title()} of {trade['symbol']}"
                content = (
                    f"{member_name}, the {member.get('leadership_role', 'Member')} from {member['state']}, "
                    f"has disclosed a {trade['transaction_type']} of {trade['symbol']} stock "
                    f"valued between ${trade['amount_min']:,.0f} and ${trade['amount_max']:,.0f}.\n\n"
                    f"The transaction occurred on {trade['transaction_date']} and was filed "
                    f"{trade.get('filing_delay_days', 'N/A')} days later. "
                    f"Our analysis gives this trade a conviction score of {trade.get('conviction_score', 'N/A')}/100, "
                    f"indicating {'high suspicion' if trade.get('conviction_score', 0) > 70 else 'moderate' if trade.get('conviction_score', 0) > 50 else 'low suspicion'}.\n\n"
                    f"{member_name} serves on the {', '.join(member.get('committees', ['Unknown']))} committee(s), "
                    f"which may have oversight relevant to this trade."
                )
            else:
                title = f"Trading Profile: {member_name}"
                content = (
                    f"{member_name} ({member['party']}-{member['state']}) has disclosed "
                    f"{member.get('trade_count', 0)} stock trades with a total estimated value of "
                    f"${member.get('total_trade_volume', 0):,.0f}.\n\n"
                    f"As a member of the {', '.join(member.get('committees', ['Unknown']))} committee(s), "
                    f"these trades warrant scrutiny for potential conflicts of interest."
                )
        else:
            title = "Analysis"
            content = f"Detailed analysis for {member_name}"

        facts = [
            f"Member: {member_name}",
            f"Party: {member['party']}",
            f"State: {member['state']}",
            f"Chamber: {member['chamber'].title()}",
        ]
        if trade:
            facts.extend([
                f"Symbol: {trade['symbol']}",
                f"Transaction: {trade['transaction_type'].title()}",
                f"Amount: ${trade['amount_mid']:,.0f}",
                f"Conviction Score: {trade.get('conviction_score', 'N/A')}/100",
            ])

        return StoryResponse(
            id=str(uuid.uuid4()),
            format=format,
            title=title,
            content=content,
            facts=facts,
            disclaimer="This analysis is for educational purposes only. High scores indicate patterns worth investigating, not proof of wrongdoing.",
            generated_at=datetime.utcnow(),
            sources=["STOCK Act Disclosures", "Congressional Records"]
        )
