# Congressional Trading Intelligence System - Intelligence Package
# Core intelligence analysis engines for congressional trading patterns

"""
Intelligence Package - Track A Components

This package provides the core intelligence analysis capabilities:

- ConvictionScorer: Scores trades 0-100 based on likelihood of informed trading
- TimingAnalyzer: Identifies trades close to market-moving events
- CommitteeCorrelator: Maps committee assignments to traded sectors

Usage:
    from src.intelligence import ConvictionScorer, TimingAnalyzer, CommitteeCorrelator

    # Score a trade
    scorer = ConvictionScorer()
    result = scorer.score_trade(trade, member)

    # Analyze timing
    analyzer = TimingAnalyzer()
    timing = analyzer.analyze_trade_timing(trade, member)

    # Check committee correlation
    correlator = CommitteeCorrelator()
    correlation = correlator.is_oversight_trade(trade, member)
"""

from .conviction_scorer import (
    ConvictionScorer,
    ConvictionResult,
    ConvictionFactor,
    RiskLevel,
    COMMITTEE_SECTOR_MAP,
    STOCK_SECTORS,
)

from .timing_analyzer import (
    TimingAnalyzer,
    TimingAnalysis,
    SuspiciousEvent,
    EventType,
    AccessLevel,
)

from .committee_correlator import (
    CommitteeCorrelator,
    CorrelationResult,
    MemberOversightProfile,
    COMMITTEE_SECTORS,
)

# For backwards compatibility, also import existing modules
try:
    from .suspicious_trading_detector import SuspiciousTradingDetector
except ImportError:
    pass

try:
    from .network_analyzer import NetworkAnalyzer
except ImportError:
    pass

try:
    from .news_monitor import NewsMonitor
except ImportError:
    pass

try:
    from .real_time_monitor import RealTimeMonitor
except ImportError:
    pass

__all__ = [
    # Track A Core Components
    "ConvictionScorer",
    "ConvictionResult",
    "ConvictionFactor",
    "RiskLevel",
    "TimingAnalyzer",
    "TimingAnalysis",
    "SuspiciousEvent",
    "EventType",
    "AccessLevel",
    "CommitteeCorrelator",
    "CorrelationResult",
    "MemberOversightProfile",
    # Data mappings
    "COMMITTEE_SECTOR_MAP",
    "COMMITTEE_SECTORS",
    "STOCK_SECTORS",
    # Legacy components
    "SuspiciousTradingDetector",
    "NetworkAnalyzer",
    "NewsMonitor",
    "RealTimeMonitor",
]
