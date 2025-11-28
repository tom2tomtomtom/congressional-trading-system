"""
Congressional Trading Intelligence API

A FastAPI-based REST API for analyzing congressional trading patterns.

This module provides:
- RESTful endpoints for members, trades, and analysis
- API key authentication with tiered rate limiting
- OpenAPI documentation
- AI-powered conviction scoring and timing analysis
"""

from .app import app
from .config import get_settings, Settings
from .auth import (
    create_api_key,
    validate_api_key,
    require_api_key,
    get_api_key,
    APIKeyData,
)
from .schemas import (
    MemberSummary,
    MemberDetail,
    TradeSummary,
    TradeDetail,
    ConvictionAnalysis,
    TimingAnalysis,
    SwampScore,
    Leaderboard,
    AggregateStats,
    AccessTierEnum,
)

__all__ = [
    # App
    "app",
    # Config
    "get_settings",
    "Settings",
    # Auth
    "create_api_key",
    "validate_api_key",
    "require_api_key",
    "get_api_key",
    "APIKeyData",
    # Schemas
    "MemberSummary",
    "MemberDetail",
    "TradeSummary",
    "TradeDetail",
    "ConvictionAnalysis",
    "TimingAnalysis",
    "SwampScore",
    "Leaderboard",
    "AggregateStats",
    "AccessTierEnum",
]

__version__ = "2.0.0"
