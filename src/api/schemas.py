"""
Pydantic schemas for the Congressional Trading Intelligence API.
These models are used for request/response validation and OpenAPI documentation.
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# Enums matching the database models
class ChamberEnum(str, Enum):
    """Congressional chamber"""
    HOUSE = "house"
    SENATE = "senate"


class PartyEnum(str, Enum):
    """Political party"""
    DEMOCRAT = "D"
    REPUBLICAN = "R"
    INDEPENDENT = "I"
    OTHER = "O"


class TransactionTypeEnum(str, Enum):
    """Transaction type"""
    BUY = "purchase"
    SELL = "sale"
    EXCHANGE = "exchange"


class AssetTypeEnum(str, Enum):
    """Asset type"""
    STOCK = "stock"
    BOND = "bond"
    MUTUAL_FUND = "mutual_fund"
    ETF = "etf"
    OPTION = "option"
    COMMODITY = "commodity"
    CRYPTOCURRENCY = "cryptocurrency"
    OTHER = "other"


class AlertLevelEnum(str, Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EXTREME = "extreme"


class AccessTierEnum(str, Enum):
    """API access tiers"""
    PUBLIC = "public"
    JOURNALIST = "journalist"
    ACADEMIC = "academic"
    PREMIUM = "premium"


# Base schemas
class TimestampMixin(BaseModel):
    """Mixin for timestamp fields"""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# Member schemas
class CommitteeBase(BaseModel):
    """Base committee schema"""
    name: str
    abbreviation: Optional[str] = None
    chamber: Optional[ChamberEnum] = None


class CommitteeResponse(CommitteeBase):
    """Committee response schema"""
    id: int
    is_active: bool = True

    model_config = ConfigDict(from_attributes=True)


class MemberBase(BaseModel):
    """Base member schema"""
    bioguide_id: str
    full_name: str
    first_name: str
    last_name: str
    party: PartyEnum
    state: str = Field(..., min_length=2, max_length=2)
    chamber: ChamberEnum
    district: Optional[str] = None


class MemberSummary(BaseModel):
    """Condensed member information for list views"""
    id: int
    bioguide_id: str
    full_name: str
    party: PartyEnum
    state: str
    chamber: ChamberEnum
    district: Optional[str] = None
    is_active: bool = True
    trade_count: Optional[int] = 0

    model_config = ConfigDict(from_attributes=True)


class MemberDetail(MemberBase):
    """Detailed member information"""
    id: int
    is_active: bool = True
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    website: Optional[str] = None
    twitter_handle: Optional[str] = None
    date_of_birth: Optional[date] = None
    leadership_role: Optional[str] = None
    committees: List[CommitteeResponse] = []
    trade_count: Optional[int] = 0
    total_trade_volume: Optional[float] = 0

    model_config = ConfigDict(from_attributes=True)


# Trade schemas
class TradeBase(BaseModel):
    """Base trade schema"""
    symbol: str
    asset_type: AssetTypeEnum = AssetTypeEnum.STOCK
    transaction_type: TransactionTypeEnum
    transaction_date: date


class TradeSummary(TradeBase):
    """Condensed trade information for list views"""
    id: int
    member_id: int
    member_name: Optional[str] = None
    amount_min: Optional[float] = None
    amount_max: Optional[float] = None
    amount_mid: Optional[float] = None
    filing_date: Optional[date] = None
    filing_delay_days: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)


class TradeDetail(TradeSummary):
    """Detailed trade information"""
    shares: Optional[float] = None
    price_per_share: Optional[float] = None
    source_document: Optional[str] = None
    source_url: Optional[str] = None
    description: Optional[str] = None
    comment: Optional[str] = None
    is_late_filing: bool = False
    is_large_trade: bool = False
    conviction_score: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


# Conviction/Analysis schemas
class ConvictionFactor(BaseModel):
    """Individual factor in conviction scoring"""
    name: str
    weight: float
    score: float
    max_score: float
    explanation: str


class ConvictionAnalysis(BaseModel):
    """Conviction score analysis for a trade"""
    trade_id: int
    score: float = Field(..., ge=0, le=100, description="Conviction score 0-100")
    factors: List[ConvictionFactor]
    explanation: str
    risk_level: AlertLevelEnum
    analyzed_at: datetime


class SuspiciousEvent(BaseModel):
    """Suspicious event linked to a trade"""
    event_type: str
    event_date: date
    days_before_trade: Optional[int] = None
    days_after_trade: Optional[int] = None
    member_access_level: str  # "direct", "committee", "public"
    description: str
    relevance_score: float = Field(..., ge=0, le=1)


class TimingAnalysis(BaseModel):
    """Timing analysis for a trade"""
    trade_id: int
    suspicious_events: List[SuspiciousEvent] = []
    timing_score: float = Field(..., ge=0, le=1)
    summary: str
    analyzed_at: datetime


# Member Score schemas
class SwampScoreComponents(BaseModel):
    """Components of the swamp score"""
    avg_conviction_score: float
    filing_compliance_rate: float
    oversight_trading_pct: float
    timing_suspicion: float
    volume_anomaly: float


class SwampScore(BaseModel):
    """Member swamp score (ethics indicator)"""
    member_id: int
    total_score: int = Field(..., ge=0, le=100)
    rank: int
    percentile: int
    components: SwampScoreComponents
    trend: str  # "improving", "worsening", "stable"
    explanation: str
    calculated_at: datetime


# Leaderboard schemas
class LeaderboardEntry(BaseModel):
    """Entry in leaderboard rankings"""
    rank: int
    member_id: int
    full_name: str
    party: PartyEnum
    state: str
    chamber: ChamberEnum
    score: float
    metric_value: Optional[float] = None


class Leaderboard(BaseModel):
    """Leaderboard response"""
    category: str
    description: str
    entries: List[LeaderboardEntry]
    total_count: int
    as_of: datetime


# Statistics schemas
class AggregateStats(BaseModel):
    """Aggregate statistics"""
    total_members: int
    total_trades: int
    total_trade_volume: float
    members_by_party: Dict[str, int]
    members_by_chamber: Dict[str, int]
    trades_by_type: Dict[str, int]
    avg_conviction_score: float
    high_conviction_trade_count: int
    late_filing_count: int
    as_of: datetime


# Story generation schemas
class StoryFormat(str, Enum):
    """Story output formats"""
    TWEET = "tweet"
    THREAD = "thread"
    NEWS_BRIEF = "news_brief"
    DEEP_DIVE = "deep_dive"
    DATA_CARD = "data_card"


class StoryGenerateRequest(BaseModel):
    """Request to generate a story"""
    trade_id: Optional[int] = None
    member_id: Optional[int] = None
    format: StoryFormat = StoryFormat.NEWS_BRIEF
    include_charts: bool = False


class StoryResponse(BaseModel):
    """Generated story response"""
    id: str
    format: StoryFormat
    title: str
    content: str
    facts: List[str]
    disclaimer: str
    generated_at: datetime
    sources: List[str] = []


# Pagination schemas
class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Generic paginated response"""
    items: List[Any]
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool


class MemberListResponse(BaseModel):
    """Paginated member list response"""
    items: List[MemberSummary]
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool


class TradeListResponse(BaseModel):
    """Paginated trade list response"""
    items: List[TradeSummary]
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool


# API Key schemas
class APIKeyCreate(BaseModel):
    """Request to create API key"""
    name: str = Field(..., min_length=1, max_length=100)
    tier: AccessTierEnum = AccessTierEnum.PUBLIC
    expires_days: Optional[int] = Field(default=365, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response (only shown once on creation)"""
    id: int
    name: str
    key: str  # Only returned on creation
    tier: AccessTierEnum
    rate_limit: int
    expires_at: Optional[datetime]
    created_at: datetime


class APIKeyInfo(BaseModel):
    """API key info (without the actual key)"""
    id: int
    name: str
    key_prefix: str
    tier: AccessTierEnum
    rate_limit: int
    requests_today: int
    requests_this_month: int
    last_used: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime
    is_active: bool


class UsageStats(BaseModel):
    """API usage statistics"""
    api_key_id: int
    requests_today: int
    requests_this_week: int
    requests_this_month: int
    rate_limit: int
    tier: AccessTierEnum
    endpoints_used: Dict[str, int]


# Error schemas
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class ValidationErrorDetail(BaseModel):
    """Validation error detail"""
    loc: List[str]
    msg: str
    type: str


class ValidationErrorResponse(BaseModel):
    """Validation error response"""
    error: str = "validation_error"
    message: str = "Request validation failed"
    details: List[ValidationErrorDetail]


# Health check schemas
class HealthCheck(BaseModel):
    """Health check response"""
    status: str = "healthy"
    service: str = "congressional-trading-api"
    version: str
    timestamp: datetime


class APIInfo(BaseModel):
    """API information response"""
    name: str
    version: str
    description: str
    documentation_url: str
    endpoints: Dict[str, str]
