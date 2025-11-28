"""
Social Media Data Models
Tracks social media posts, Reddit activity, and correlations with trading
"""

from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, Date, ForeignKey, ARRAY, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
from datetime import datetime

from src.models.base import Base, TimestampMixin, IDMixin


class SocialMediaPost(Base, TimestampMixin, IDMixin):
    """
    Social media posts from congressional members
    (Twitter/X, Truth Social, etc.)
    """
    __tablename__ = "social_media_posts"

    # Member who posted
    member_id = Column(Integer, ForeignKey("members.id"), nullable=False, index=True)

    # Post metadata
    platform = Column(String(50), nullable=False, index=True)  # twitter, truth_social
    post_id = Column(String(100), unique=True, nullable=False)
    content = Column(Text)
    posted_at = Column(DateTime, nullable=False, index=True)
    url = Column(String(500))

    # Engagement metrics
    likes = Column(Integer, default=0)
    retweets = Column(Integer, default=0)
    replies = Column(Integer, default=0)
    engagement_score = Column(Float)  # Weighted engagement metric

    # Extracted metadata (using NLP)
    mentioned_companies = Column(PG_ARRAY(String), default=list)  # ['GOOGL', 'TSLA']
    mentioned_sectors = Column(PG_ARRAY(String), default=list)  # ['tech', 'energy']
    mentioned_topics = Column(PG_ARRAY(String), default=list)  # ['regulation', 'china']

    # Sentiment analysis
    sentiment_score = Column(Float)  # -1 (negative) to +1 (positive)
    sentiment_magnitude = Column(Float)  # 0-1 (strength of sentiment)
    sentiment_explanation = Column(Text)

    # Content classification
    is_about_stocks = Column(Boolean, default=False)
    is_about_policy = Column(Boolean, default=False)
    is_controversial = Column(Boolean, default=False)

    # Relationships
    member = relationship("Member", back_populates="social_media_posts")
    related_trades = relationship(
        "Trade",
        secondary="social_trade_correlations",
        back_populates="related_social_posts"
    )

    def __repr__(self):
        return f"<SocialMediaPost(id={self.id}, member={self.member_id}, platform={self.platform}, posted_at={self.posted_at})>"


class SocialTradeCorrelation(Base, TimestampMixin):
    """
    Junction table linking social media posts to trades
    Stores correlation analysis results
    """
    __tablename__ = "social_trade_correlations"

    id = Column(Integer, primary_key=True)

    # The correlated entities
    post_id = Column(Integer, ForeignKey("social_media_posts.id"), nullable=False, index=True)
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False, index=True)

    # Correlation metadata
    correlation_type = Column(String(50), index=True)  # pre-trade, post-trade, divergence, coordination
    time_delta_hours = Column(Float)  # Hours between post and trade (negative = post before trade)

    # Analysis results
    suspicion_score = Column(Float, index=True)  # 0-100
    confidence_score = Column(Float)  # 0-1 (how confident in this correlation)
    explanation = Column(Text)

    # Pattern detected
    pattern_name = Column(String(100))  # "Pre-trade hype", "Post-trade justification", "Sentiment divergence"
    detected_by = Column(String(50))  # Which analyzer detected this

    # Relationships
    post = relationship("SocialMediaPost")
    trade = relationship("Trade")

    def __repr__(self):
        return f"<SocialTradeCorrelation(post={self.post_id}, trade={self.trade_id}, type={self.correlation_type}, suspicion={self.suspicion_score})>"


class RedditPost(Base, TimestampMixin, IDMixin):
    """
    Reddit posts mentioning stock tickers
    Tracks retail investor sentiment
    """
    __tablename__ = "reddit_posts"

    # Reddit metadata
    post_id = Column(String(20), unique=True, nullable=False)  # Reddit's unique ID
    subreddit = Column(String(100), nullable=False, index=True)
    title = Column(Text)
    content = Column(Text)
    author = Column(String(100))
    posted_at = Column(DateTime, nullable=False, index=True)
    url = Column(String(500))

    # Engagement metrics
    score = Column(Integer, default=0)  # Upvotes - Downvotes
    upvote_ratio = Column(Float)  # 0-1
    num_comments = Column(Integer, default=0)

    # Extracted stock mentions
    mentioned_tickers = Column(PG_ARRAY(String), default=list, index=True)

    # Sentiment analysis
    sentiment_score = Column(Float)  # -1 (bearish) to +1 (bullish)
    sentiment_magnitude = Column(Float)  # Strength of sentiment

    # Content classification
    is_dd_post = Column(Boolean, default=False)  # Due Diligence post (higher quality)
    is_meme = Column(Boolean, default=False)
    is_yolo = Column(Boolean, default=False)  # High-risk trade announcement

    # Hype scoring
    hype_score = Column(Float)  # 0-100 based on engagement + sentiment

    def __repr__(self):
        return f"<RedditPost(id={self.id}, subreddit=r/{self.subreddit}, posted_at={self.posted_at})>"


class RedditTickerSentiment(Base, TimestampMixin, IDMixin):
    """
    Aggregated daily sentiment per ticker from Reddit
    Pre-calculated for performance
    """
    __tablename__ = "reddit_ticker_sentiment"

    # Ticker and date (unique combination)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)

    # Aggregated metrics
    post_count = Column(Integer, default=0)
    total_score = Column(Integer, default=0)  # Sum of all post scores
    avg_sentiment = Column(Float)  # Average sentiment across all posts
    sentiment_stddev = Column(Float)  # Sentiment consistency

    # Hype classification
    hype_level = Column(String(20), index=True)  # low, medium, high, extreme

    # Subreddit breakdown
    wsb_mentions = Column(Integer, default=0)  # r/wallstreetbets specific
    stocks_mentions = Column(Integer, default=0)  # r/stocks specific
    investing_mentions = Column(Integer, default=0)  # r/investing specific

    # Top posts (for reference)
    top_posts = Column(JSON)  # List of top 5 post IDs

    # Trending detection
    is_trending = Column(Boolean, default=False, index=True)
    trend_score = Column(Float)  # Rate of mention increase

    def __repr__(self):
        return f"<RedditTickerSentiment(ticker={self.ticker}, date={self.date}, hype={self.hype_level})>"


class CongressRedditCorrelation(Base, TimestampMixin, IDMixin):
    """
    Links congressional trades to Reddit activity
    Detects predatory behavior and information advantage
    """
    __tablename__ = "congress_reddit_correlations"

    # The correlated entities
    trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False, index=True)
    ticker = Column(String(10), nullable=False, index=True)

    # Reddit metrics at time of trade
    reddit_sentiment_at_trade = Column(Float)  # -1 to +1
    reddit_hype_level = Column(String(20))  # low, medium, high, extreme
    reddit_post_count_7d = Column(Integer)  # Posts in 7 days before trade
    reddit_trend_direction = Column(String(20))  # rising, falling, stable

    # Timing analysis
    time_delta_vs_reddit_peak_days = Column(Integer)  # Days before/after Reddit sentiment peak
    traded_before_reddit_hype = Column(Boolean, index=True)  # True = suspicious
    traded_against_sentiment = Column(Boolean, index=True)  # True = possibly predatory

    # Behavior classification
    pattern_type = Column(String(50), index=True)  # dump_on_retail, information_advantage, contrarian, coordinated
    pattern_explanation = Column(Text)

    # Scoring
    information_advantage_score = Column(Float, index=True)  # 0-100
    predatory_behavior_score = Column(Float, index=True)  # 0-100
    overall_suspicion_score = Column(Float, index=True)  # 0-100

    # Outcome tracking (did the pattern pay off?)
    price_change_7d = Column(Float)  # % price change 7 days after trade
    member_profit_estimate = Column(Float)  # Estimated profit if sold at peak
    retail_impact = Column(String(50))  # positive, negative, neutral

    # Relationships
    trade = relationship("Trade", back_populates="reddit_correlations")

    def __repr__(self):
        return f"<CongressRedditCorrelation(trade={self.trade_id}, ticker={self.ticker}, pattern={self.pattern_type}, suspicion={self.overall_suspicion_score})>"


class CorporateInsider(Base, TimestampMixin, IDMixin):
    """
    Corporate insiders (CEOs, CFOs, Directors, 10%+ owners)
    """
    __tablename__ = "corporate_insiders"

    # Insider identity
    name = Column(String(200), nullable=False)
    company_ticker = Column(String(10), index=True)
    company_name = Column(String(200))

    # Position
    position = Column(String(100))  # CEO, CFO, Director, etc.
    is_director = Column(Boolean, default=False)
    is_officer = Column(Boolean, default=False)
    is_ten_percent_owner = Column(Boolean, default=False)

    # CIK (SEC identifier)
    cik = Column(String(10), unique=True)

    # Relationships
    trades = relationship("InsiderTrade", back_populates="insider")

    def __repr__(self):
        return f"<CorporateInsider(name={self.name}, company={self.company_ticker}, position={self.position})>"


class InsiderTrade(Base, TimestampMixin, IDMixin):
    """
    Corporate insider trades from SEC Form 4
    """
    __tablename__ = "insider_trades"

    # Insider who made the trade
    insider_id = Column(Integer, ForeignKey("corporate_insiders.id"), nullable=False, index=True)

    # Trade details
    ticker = Column(String(10), nullable=False, index=True)
    transaction_type = Column(String(50))  # Purchase, Sale, Option Exercise, etc.
    shares = Column(Integer)
    price_per_share = Column(Float)
    total_value = Column(Float, index=True)

    # Dates
    transaction_date = Column(Date, nullable=False, index=True)
    filing_date = Column(Date, index=True)
    filing_delay_days = Column(Integer)  # Days between transaction and filing

    # SEC document
    sec_form_type = Column(String(10))  # Form 4, Form 5, Form 144
    sec_form_url = Column(String(500))

    # Ownership after transaction
    shares_owned_after = Column(Integer)
    ownership_percentage = Column(Float)

    # Analysis flags
    is_planned_sale = Column(Boolean, default=False)  # 10b5-1 plan
    is_unusual_size = Column(Boolean, default=False)
    is_before_earnings = Column(Boolean, default=False)

    # Relationships
    insider = relationship("CorporateInsider", back_populates="trades")
    congressional_correlations = relationship(
        "CongressInsiderCorrelation",
        back_populates="insider_trade"
    )

    def __repr__(self):
        return f"<InsiderTrade(insider={self.insider_id}, ticker={self.ticker}, type={self.transaction_type}, value=${self.total_value})>"


class CongressInsiderCorrelation(Base, TimestampMixin, IDMixin):
    """
    Correlates congressional trades with corporate insider trades
    Detects coordination and information sharing
    """
    __tablename__ = "congress_insider_correlations"

    # The correlated trades
    congress_trade_id = Column(Integer, ForeignKey("trades.id"), nullable=False, index=True)
    insider_trade_id = Column(Integer, ForeignKey("insider_trades.id"), nullable=False, index=True)

    # Timing analysis
    time_delta_days = Column(Integer)  # Days between trades (negative = insider first)
    same_direction = Column(Boolean)  # Both buy or both sell
    same_ticker = Column(Boolean)

    # Pattern detection
    pattern_type = Column(String(50), index=True)  # coordination, front_running, parallel_thinking
    coordination_score = Column(Float, index=True)  # 0-100
    explanation = Column(Text)

    # Context
    news_between_trades = Column(JSON)  # Any announcements between the two trades
    price_movement = Column(Float)  # % price change between trades

    # Relationships
    congress_trade = relationship("Trade")
    insider_trade = relationship("InsiderTrade", back_populates="congressional_correlations")

    def __repr__(self):
        return f"<CongressInsiderCorrelation(congress={self.congress_trade_id}, insider={self.insider_trade_id}, pattern={self.pattern_type})>"


# Add relationships to existing Trade model
# This would be added to src/models/trading.py:
"""
# In Trade class:
related_social_posts = relationship(
    "SocialMediaPost",
    secondary="social_trade_correlations",
    back_populates="related_trades"
)

reddit_correlations = relationship(
    "CongressRedditCorrelation",
    back_populates="trade"
)
"""

# Add relationships to existing Member model
# This would be added to src/models/member.py:
"""
# In Member class:
social_media_posts = relationship(
    "SocialMediaPost",
    back_populates="member"
)
"""
