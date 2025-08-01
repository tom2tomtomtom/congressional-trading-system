"""
Trading and transaction models
"""

from datetime import date, datetime
from decimal import Decimal
from typing import List, Optional

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Enum, ForeignKey,
    Integer, Numeric, String, Text, Index, CheckConstraint
)
from sqlalchemy.orm import relationship, Mapped
import enum

from .base import Base, TimestampMixin, IDMixin


class TransactionType(enum.Enum):
    """Transaction type enumeration"""
    BUY = "purchase"
    SELL = "sale"
    EXCHANGE = "exchange"


class AssetType(enum.Enum):
    """Asset type enumeration"""
    STOCK = "stock"
    BOND = "bond"
    MUTUAL_FUND = "mutual_fund"
    ETF = "etf"
    OPTION = "option"
    COMMODITY = "commodity"
    CRYPTOCURRENCY = "cryptocurrency"
    OTHER = "other"


class AlertLevel(enum.Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high" 
    CRITICAL = "critical"
    EXTREME = "extreme"


class AlertStatus(enum.Enum):
    """Alert processing status"""
    PENDING = "pending"
    REVIEWED = "reviewed"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class Trade(Base, TimestampMixin, IDMixin):
    """Individual trade/transaction model"""
    
    __tablename__ = "trades"
    
    # Member Information
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False, index=True)
    
    # Trade Details
    symbol = Column(String(20), nullable=False, index=True)
    asset_type = Column(Enum(AssetType), nullable=False, default=AssetType.STOCK)
    transaction_type = Column(Enum(TransactionType), nullable=False, index=True)
    
    # Amounts (using NUMERIC for precision)
    amount_min = Column(Numeric(15, 2), nullable=True)
    amount_max = Column(Numeric(15, 2), nullable=True)
    amount_mid = Column(Numeric(15, 2), nullable=True, index=True)
    shares = Column(Numeric(15, 4), nullable=True)
    price_per_share = Column(Numeric(10, 4), nullable=True)
    
    # Dates
    transaction_date = Column(Date, nullable=False, index=True)
    filing_date = Column(Date, nullable=True, index=True)
    filing_delay_days = Column(Integer, nullable=True, index=True)
    
    # Source Information
    source_document = Column(String(500), nullable=True)
    source_url = Column(String(1000), nullable=True)
    data_source = Column(String(100), nullable=True)
    
    # Additional Information
    description = Column(Text, nullable=True)
    comment = Column(Text, nullable=True)
    
    # Processing Status
    is_processed = Column(Boolean, default=False, nullable=False)
    processing_date = Column(DateTime, nullable=True)
    
    # Relationships
    member: Mapped["Member"] = relationship("Member", back_populates="trades")
    alerts: Mapped[List["TradeAlert"]] = relationship(
        "TradeAlert", back_populates="trade", cascade="all, delete-orphan"
    )
    patterns: Mapped[List["TradingPattern"]] = relationship(
        "TradingPattern", back_populates="trade", cascade="all, delete-orphan"
    )
    
    # Constraints and Indexes
    __table_args__ = (
        CheckConstraint("amount_min >= 0", name="ck_trade_amount_min_positive"),
        CheckConstraint("amount_max >= amount_min", name="ck_trade_amount_max_gte_min"),
        CheckConstraint("shares >= 0", name="ck_trade_shares_positive"),
        CheckConstraint("price_per_share >= 0", name="ck_trade_price_positive"),
        CheckConstraint("filing_delay_days >= 0", name="ck_filing_delay_positive"),
        Index("ix_trade_member_date", "member_id", "transaction_date"),
        Index("ix_trade_symbol_date", "symbol", "transaction_date"),
        Index("ix_trade_amount_date", "amount_mid", "transaction_date"),
        Index("ix_trade_filing", "filing_date", "filing_delay_days"),
    )
    
    def __repr__(self) -> str:
        return (f"<Trade {self.member.full_name} {self.transaction_type.value} "
                f"{self.symbol} ${self.amount_mid} on {self.transaction_date}>")
    
    @property
    def estimated_amount(self) -> Optional[Decimal]:
        """Get estimated trade amount (uses mid if available, otherwise average of min/max)"""
        if self.amount_mid:
            return self.amount_mid
        elif self.amount_min and self.amount_max:
            return (self.amount_min + self.amount_max) / 2
        elif self.shares and self.price_per_share:
            return self.shares * self.price_per_share
        return None
    
    @property
    def is_large_trade(self) -> bool:
        """Check if this is considered a large trade (>=50k)"""
        amount = self.estimated_amount
        return amount is not None and amount >= 50000
    
    @property
    def is_late_filing(self) -> bool:
        """Check if filing was late (>45 days)"""
        return self.filing_delay_days is not None and self.filing_delay_days > 45


class TradeAlert(Base, TimestampMixin, IDMixin):
    """Suspicious trading alerts"""
    
    __tablename__ = "trade_alerts"
    
    # Related Trade
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False, index=True)
    
    # Alert Information
    alert_type = Column(String(100), nullable=False, index=True)
    level = Column(Enum(AlertLevel), nullable=False, index=True)
    status = Column(Enum(AlertStatus), default=AlertStatus.PENDING, nullable=False)
    
    # Scoring
    suspicion_score = Column(Numeric(4, 2), nullable=False, index=True)
    confidence_score = Column(Numeric(4, 2), nullable=True)
    
    # Description
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    reason = Column(Text, nullable=False)
    
    # Processing Information
    generated_by = Column(String(100), nullable=False)  # ML model, rule engine, etc.
    reviewed_by_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    # Relationships
    trade: Mapped["Trade"] = relationship("Trade", back_populates="alerts")
    reviewed_by: Mapped[Optional["User"]] = relationship("User")
    
    # Indexes
    __table_args__ = (
        CheckConstraint("suspicion_score >= 0 AND suspicion_score <= 10", 
                       name="ck_suspicion_score_range"),
        CheckConstraint("confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)",
                       name="ck_confidence_score_range"),
        Index("ix_alert_level_status", "level", "status"),
        Index("ix_alert_score", "suspicion_score"),
        Index("ix_alert_generated", "generated_by", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<TradeAlert {self.alert_type} {self.level.value} Score:{self.suspicion_score}>"
    
    @property
    def is_high_priority(self) -> bool:
        """Check if this is a high priority alert"""
        return self.level in [AlertLevel.HIGH, AlertLevel.CRITICAL, AlertLevel.EXTREME]
    
    @property
    def needs_review(self) -> bool:
        """Check if alert needs human review"""
        return self.status == AlertStatus.PENDING and self.is_high_priority


class TradingPattern(Base, TimestampMixin, IDMixin):
    """Detected trading patterns"""
    
    __tablename__ = "trading_patterns"
    
    # Related Trade
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False, index=True)
    
    # Pattern Information
    pattern_type = Column(String(100), nullable=False, index=True)
    pattern_name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    
    # Metrics
    frequency = Column(Integer, nullable=True)
    duration_days = Column(Integer, nullable=True)
    total_amount = Column(Numeric(15, 2), nullable=True)
    
    # Significance
    statistical_significance = Column(Numeric(5, 4), nullable=True)
    correlation_coefficient = Column(Numeric(5, 4), nullable=True)
    
    # Detection Information
    detected_by = Column(String(100), nullable=False)
    detection_algorithm = Column(String(200), nullable=True)
    
    # Relationships
    trade: Mapped["Trade"] = relationship("Trade", back_populates="patterns")
    
    # Indexes
    __table_args__ = (
        Index("ix_pattern_type", "pattern_type"),
        Index("ix_pattern_significance", "statistical_significance"),
        Index("ix_pattern_detection", "detected_by", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<TradingPattern {self.pattern_name} for {self.trade.symbol}>"