"""
Base infrastructure for distribution bots.

Provides shared utilities for Twitter, Discord, and other distribution channels.
"""

import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json

logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of alerts that can be distributed."""
    NEW_FILING = "new_filing"
    HIGH_CONVICTION = "high_conviction"
    PATTERN_ALERT = "pattern_alert"
    WEEKLY_ROUNDUP = "weekly_roundup"
    DAILY_DIGEST = "daily_digest"
    BREAKING = "breaking"
    LEADERBOARD_UPDATE = "leaderboard_update"


class AlertPriority(Enum):
    """Priority levels for alerts."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    BREAKING = 5


@dataclass
class TradeAlertData:
    """Standardized trade alert data for distribution."""
    trade_id: str
    member_name: str
    member_party: str  # "D", "R", "I"
    member_state: str
    member_chamber: str  # "House" or "Senate"
    symbol: str
    company_name: Optional[str]
    transaction_type: str  # "Purchase" or "Sale"
    amount_min: float
    amount_max: float
    transaction_date: date
    filing_date: date
    conviction_score: Optional[float] = None
    risk_level: Optional[str] = None
    top_factors: List[str] = field(default_factory=list)
    explanation: Optional[str] = None
    committee_overlap: Optional[str] = None
    filing_delay_days: Optional[int] = None

    @property
    def amount_range_str(self) -> str:
        """Format amount range as string."""
        if self.amount_min == self.amount_max:
            return f"${self.amount_min:,.0f}"
        return f"${self.amount_min:,.0f} - ${self.amount_max:,.0f}"

    @property
    def member_display(self) -> str:
        """Format member name with party and state."""
        title = "Sen." if self.member_chamber == "Senate" else "Rep."
        return f"{title} {self.member_name} ({self.member_party}-{self.member_state})"

    @property
    def is_high_conviction(self) -> bool:
        """Check if this is a high conviction trade (score >= 70)."""
        return self.conviction_score is not None and self.conviction_score >= 70

    @property
    def is_suspicious(self) -> bool:
        """Check if trade has suspicious indicators."""
        return (
            self.is_high_conviction or
            (self.filing_delay_days is not None and self.filing_delay_days > 45) or
            bool(self.committee_overlap)
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "member_name": self.member_name,
            "member_party": self.member_party,
            "member_state": self.member_state,
            "member_chamber": self.member_chamber,
            "symbol": self.symbol,
            "company_name": self.company_name,
            "transaction_type": self.transaction_type,
            "amount_min": self.amount_min,
            "amount_max": self.amount_max,
            "transaction_date": str(self.transaction_date),
            "filing_date": str(self.filing_date),
            "conviction_score": self.conviction_score,
            "risk_level": self.risk_level,
            "top_factors": self.top_factors,
            "explanation": self.explanation,
            "committee_overlap": self.committee_overlap,
            "filing_delay_days": self.filing_delay_days,
        }


@dataclass
class BotConfig:
    """Configuration for distribution bots."""
    # General settings
    environment: str = "development"  # "development", "staging", "production"
    dry_run: bool = True  # Don't actually post, just log
    rate_limit_per_hour: int = 30

    # Content settings
    include_conviction_scores: bool = True
    include_charts: bool = False
    hashtags: List[str] = field(default_factory=lambda: [
        "#CongressTrading", "#STOCKAct", "#CongressionalTrading"
    ])

    # Filtering
    min_conviction_score: float = 0.0  # Minimum score to alert
    min_trade_amount: float = 15000.0  # Minimum amount to alert
    alert_on_high_conviction_only: bool = False

    # Timing
    quiet_hours_start: int = 23  # 11 PM
    quiet_hours_end: int = 7  # 7 AM

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load configuration from environment variables."""
        return cls(
            environment=os.getenv("BOT_ENVIRONMENT", "development"),
            dry_run=os.getenv("BOT_DRY_RUN", "true").lower() == "true",
            rate_limit_per_hour=int(os.getenv("BOT_RATE_LIMIT", "30")),
            include_conviction_scores=os.getenv("BOT_INCLUDE_SCORES", "true").lower() == "true",
            min_conviction_score=float(os.getenv("BOT_MIN_CONVICTION", "0")),
            min_trade_amount=float(os.getenv("BOT_MIN_AMOUNT", "15000")),
        )


class AlertFormatter:
    """
    Formats trade alerts for various platforms.

    Handles character limits, formatting differences, and platform-specific
    requirements for Twitter, Discord, Email, etc.
    """

    # Risk level emojis
    RISK_EMOJIS = {
        "low": "",
        "moderate": "",
        "elevated": "",
        "high": "",
        "critical": "",
    }

    # Transaction type emojis
    TRANSACTION_EMOJIS = {
        "Purchase": "",
        "Sale": "",
        "purchase": "",
        "sale": "",
    }

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()

    def format_tweet(self, alert: TradeAlertData, max_length: int = 280) -> str:
        """
        Format a trade alert as a single tweet.

        Args:
            alert: Trade alert data
            max_length: Maximum character length (Twitter default: 280)

        Returns:
            Formatted tweet string
        """
        txn_emoji = self.TRANSACTION_EMOJIS.get(alert.transaction_type, "")
        risk_emoji = self.RISK_EMOJIS.get(alert.risk_level or "", "")

        # Build tweet components
        header = f"NEW DISCLOSURE {risk_emoji}".strip()

        member_line = alert.member_display
        trade_line = f"{txn_emoji} {alert.transaction_type}: ${alert.symbol}"
        amount_line = f"Amount: {alert.amount_range_str}"

        # Add conviction score if available and significant
        score_line = ""
        if alert.conviction_score and alert.conviction_score >= 50:
            score_line = f"Suspicion Score: {alert.conviction_score:.0f}/100"

        # Add committee overlap if present
        overlap_line = ""
        if alert.committee_overlap:
            overlap_line = f"Committee Overlap: {alert.committee_overlap}"

        # Build hashtags
        hashtags = " ".join(self.config.hashtags[:2])  # Limit hashtags for space

        # Assemble tweet
        lines = [header, member_line, trade_line, amount_line]
        if score_line:
            lines.append(score_line)
        if overlap_line:
            lines.append(overlap_line)
        lines.append(hashtags)

        tweet = "\n".join(lines)

        # Truncate if too long
        if len(tweet) > max_length:
            # Remove hashtags first
            tweet = "\n".join(lines[:-1])
            if len(tweet) > max_length:
                # Remove overlap line
                if overlap_line in tweet:
                    tweet = tweet.replace(f"\n{overlap_line}", "")
                if len(tweet) > max_length:
                    tweet = tweet[:max_length-3] + "..."

        return tweet

    def format_tweet_thread(self, alert: TradeAlertData) -> List[str]:
        """
        Format a trade alert as a multi-tweet thread.

        Used for high conviction trades that need more context.

        Args:
            alert: Trade alert data

        Returns:
            List of tweet strings (3-5 tweets)
        """
        tweets = []

        # Tweet 1: Breaking alert
        risk_emoji = self.RISK_EMOJIS.get(alert.risk_level or "", "")
        tweet1 = f"""THREAD: Suspicious Trade Alert {risk_emoji}

{alert.member_display} just disclosed a {alert.transaction_type.lower()} of ${alert.symbol}

Amount: {alert.amount_range_str}
Trade Date: {alert.transaction_date.strftime('%b %d, %Y')}

This trade has a {alert.conviction_score:.0f}/100 suspicion score.

1/"""
        tweets.append(tweet1[:280])

        # Tweet 2: Why it's suspicious
        factors_text = "\n".join(f"- {factor}" for factor in alert.top_factors[:3]) if alert.top_factors else "Multiple concerning patterns detected"
        tweet2 = f"""Why this trade is suspicious:

{factors_text}

2/"""
        tweets.append(tweet2[:280])

        # Tweet 3: Explanation
        if alert.explanation:
            explanation = alert.explanation[:200] + "..." if len(alert.explanation) > 200 else alert.explanation
            tweet3 = f"""Analysis:

{explanation}

3/"""
            tweets.append(tweet3[:280])

        # Tweet 4: Filing delay if applicable
        if alert.filing_delay_days and alert.filing_delay_days > 30:
            delay_tweet = f"""Filing Delay Alert:

This trade was filed {alert.filing_delay_days} days after the transaction.

The STOCK Act requires disclosure within 45 days. {"This filing was LATE." if alert.filing_delay_days > 45 else "Filed near the deadline."}

4/"""
            tweets.append(delay_tweet[:280])

        # Final tweet: Call to action
        final_tweet = f"""All data from official STOCK Act disclosures.

This is educational analysis - high scores indicate patterns worth investigating, not proof of wrongdoing.

Follow for more congressional trading alerts.

{' '.join(self.config.hashtags)}"""
        tweets.append(final_tweet[:280])

        return tweets

    def format_discord_embed(self, alert: TradeAlertData) -> Dict:
        """
        Format a trade alert as a Discord embed.

        Args:
            alert: Trade alert data

        Returns:
            Discord embed dictionary
        """
        # Color based on conviction score
        if alert.conviction_score and alert.conviction_score >= 80:
            color = 0xFF0000  # Red - Critical
        elif alert.conviction_score and alert.conviction_score >= 65:
            color = 0xFFA500  # Orange - High
        elif alert.conviction_score and alert.conviction_score >= 50:
            color = 0xFFFF00  # Yellow - Elevated
        else:
            color = 0x00FF00  # Green - Low

        txn_emoji = self.TRANSACTION_EMOJIS.get(alert.transaction_type, "")
        risk_emoji = self.RISK_EMOJIS.get(alert.risk_level or "", "")

        embed = {
            "title": f"{risk_emoji} New Trade Disclosure: {alert.symbol}",
            "description": f"**{alert.member_display}**\n{txn_emoji} {alert.transaction_type}",
            "color": color,
            "fields": [
                {
                    "name": "Symbol",
                    "value": f"${alert.symbol}" + (f" ({alert.company_name})" if alert.company_name else ""),
                    "inline": True
                },
                {
                    "name": "Amount",
                    "value": alert.amount_range_str,
                    "inline": True
                },
                {
                    "name": "Trade Date",
                    "value": alert.transaction_date.strftime("%Y-%m-%d"),
                    "inline": True
                },
            ],
            "footer": {
                "text": "Congressional Trading Intelligence | Educational analysis only"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

        # Add conviction score if available
        if alert.conviction_score:
            embed["fields"].append({
                "name": "Suspicion Score",
                "value": f"{alert.conviction_score:.0f}/100 ({alert.risk_level or 'Unknown'})",
                "inline": True
            })

        # Add filing delay if significant
        if alert.filing_delay_days:
            delay_status = " (LATE)" if alert.filing_delay_days > 45 else ""
            embed["fields"].append({
                "name": "Filing Delay",
                "value": f"{alert.filing_delay_days} days{delay_status}",
                "inline": True
            })

        # Add committee overlap if present
        if alert.committee_overlap:
            embed["fields"].append({
                "name": "Committee Overlap",
                "value": alert.committee_overlap,
                "inline": False
            })

        # Add top factors if available
        if alert.top_factors:
            embed["fields"].append({
                "name": "Key Factors",
                "value": "\n".join(f" {f}" for f in alert.top_factors[:3]),
                "inline": False
            })

        return embed

    def format_email_html(self, alert: TradeAlertData) -> str:
        """
        Format a trade alert as HTML email content.

        Args:
            alert: Trade alert data

        Returns:
            HTML string for email body
        """
        risk_color = {
            "low": "#28a745",
            "moderate": "#17a2b8",
            "elevated": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545",
        }.get(alert.risk_level or "", "#6c757d")

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 20px; border-radius: 10px 10px 0 0;">
                <h2 style="margin: 0; font-size: 18px;">New Trade Disclosure</h2>
                <p style="margin: 10px 0 0; opacity: 0.9;">{alert.member_display}</p>
            </div>

            <div style="background: #f8f9fa; padding: 20px; border: 1px solid #dee2e6;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;">
                            <strong>Symbol:</strong>
                        </td>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6; text-align: right;">
                            <span style="font-size: 18px; font-weight: bold;">${alert.symbol}</span>
                            {f'<br><span style="color: #6c757d; font-size: 12px;">{alert.company_name}</span>' if alert.company_name else ''}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;">
                            <strong>Transaction:</strong>
                        </td>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6; text-align: right;">
                            {alert.transaction_type}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;">
                            <strong>Amount:</strong>
                        </td>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6; text-align: right;">
                            {alert.amount_range_str}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;">
                            <strong>Trade Date:</strong>
                        </td>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6; text-align: right;">
                            {alert.transaction_date.strftime('%B %d, %Y')}
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6;">
                            <strong>Filed:</strong>
                        </td>
                        <td style="padding: 10px 0; border-bottom: 1px solid #dee2e6; text-align: right;">
                            {alert.filing_date.strftime('%B %d, %Y')}
                            {f' <span style="color: #dc3545;">({alert.filing_delay_days} days - LATE)</span>' if alert.filing_delay_days and alert.filing_delay_days > 45 else f' ({alert.filing_delay_days} days)' if alert.filing_delay_days else ''}
                        </td>
                    </tr>
                </table>
            </div>

            {"" if not alert.conviction_score else f'''
            <div style="background: white; padding: 20px; border: 1px solid #dee2e6; border-top: none;">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <div style="flex: 1;">
                        <h3 style="margin: 0 0 5px; font-size: 14px; color: #6c757d;">SUSPICION SCORE</h3>
                        <span style="font-size: 36px; font-weight: bold; color: {risk_color};">{alert.conviction_score:.0f}</span>
                        <span style="font-size: 18px; color: #6c757d;">/100</span>
                    </div>
                    <div style="background: {risk_color}; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; text-transform: uppercase; font-size: 12px;">
                        {alert.risk_level or 'Unknown'}
                    </div>
                </div>

                {"" if not alert.top_factors else f'''
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <h4 style="margin: 0 0 10px; font-size: 12px; color: #6c757d; text-transform: uppercase;">Key Factors</h4>
                    <ul style="margin: 0; padding-left: 20px; color: #333;">
                        {"".join(f"<li style='margin: 5px 0;'>{factor}</li>" for factor in alert.top_factors[:3])}
                    </ul>
                </div>
                '''}
            </div>
            '''}

            {"" if not alert.committee_overlap else f'''
            <div style="background: #fff3cd; padding: 15px; border: 1px solid #ffc107; margin-top: -1px;">
                <strong style="color: #856404;">Committee Overlap Detected</strong>
                <p style="margin: 5px 0 0; color: #856404;">{alert.committee_overlap}</p>
            </div>
            '''}

            <div style="background: #e9ecef; padding: 15px; border-radius: 0 0 10px 10px; text-align: center; font-size: 12px; color: #6c757d;">
                <p style="margin: 0;">
                    This is educational analysis based on official STOCK Act disclosures.<br>
                    High scores indicate patterns worth investigating, not proof of wrongdoing.
                </p>
            </div>
        </div>
        """
        return html


class ContentGenerator:
    """
    Generates content for weekly roundups, digests, and special reports.
    """

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig()
        self.formatter = AlertFormatter(config)

    def generate_weekly_roundup(self, trades: List[TradeAlertData]) -> Dict[str, Any]:
        """
        Generate weekly roundup content.

        Args:
            trades: List of trade alerts from the past week

        Returns:
            Dictionary with content for various platforms
        """
        if not trades:
            return {"error": "No trades to summarize"}

        # Sort by conviction score
        sorted_trades = sorted(
            trades,
            key=lambda x: x.conviction_score or 0,
            reverse=True
        )

        # Calculate statistics
        total_volume = sum((t.amount_min + t.amount_max) / 2 for t in trades)
        high_conviction = [t for t in trades if t.conviction_score and t.conviction_score >= 70]
        purchases = [t for t in trades if t.transaction_type.lower() == "purchase"]
        sales = [t for t in trades if t.transaction_type.lower() == "sale"]

        # Party breakdown
        dem_trades = [t for t in trades if t.member_party == "D"]
        rep_trades = [t for t in trades if t.member_party == "R"]

        # Top symbols
        symbol_counts = {}
        for t in trades:
            symbol_counts[t.symbol] = symbol_counts.get(t.symbol, 0) + 1
        top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "summary": {
                "total_trades": len(trades),
                "total_volume": total_volume,
                "high_conviction_count": len(high_conviction),
                "purchases": len(purchases),
                "sales": len(sales),
                "democrat_trades": len(dem_trades),
                "republican_trades": len(rep_trades),
            },
            "top_suspicious": sorted_trades[:5],
            "top_symbols": top_symbols,
            "twitter_thread": self._generate_roundup_thread(trades, sorted_trades),
            "discord_embed": self._generate_roundup_embed(trades, sorted_trades, total_volume),
        }

    def _generate_roundup_thread(
        self,
        trades: List[TradeAlertData],
        sorted_trades: List[TradeAlertData]
    ) -> List[str]:
        """Generate Twitter thread for weekly roundup."""
        tweets = []

        high_conviction = [t for t in trades if t.conviction_score and t.conviction_score >= 70]
        total_volume = sum((t.amount_min + t.amount_max) / 2 for t in trades)

        # Opening tweet
        tweet1 = f"""WEEKLY CONGRESSIONAL TRADING ROUNDUP

{len(trades)} trades disclosed this week
Total volume: ${total_volume:,.0f}
High suspicion trades: {len(high_conviction)}

Here are the most concerning patterns we found...

1/"""
        tweets.append(tweet1[:280])

        # Top 3 suspicious trades
        for i, trade in enumerate(sorted_trades[:3], 2):
            tweet = f"""{trade.member_display}
{trade.transaction_type}: ${trade.symbol}
Amount: {trade.amount_range_str}
Score: {trade.conviction_score:.0f}/100

{i}/"""
            tweets.append(tweet[:280])

        # Closing tweet
        final = f"""Follow for real-time alerts on congressional trading.

All data from official STOCK Act disclosures.

#CongressTrading #STOCKAct #Transparency"""
        tweets.append(final[:280])

        return tweets

    def _generate_roundup_embed(
        self,
        trades: List[TradeAlertData],
        sorted_trades: List[TradeAlertData],
        total_volume: float
    ) -> Dict:
        """Generate Discord embed for weekly roundup."""
        high_conviction = [t for t in trades if t.conviction_score and t.conviction_score >= 70]
        purchases = [t for t in trades if t.transaction_type.lower() == "purchase"]

        # Top 3 as text
        top_trades_text = "\n".join(
            f"{i}. {t.member_display}: ${t.symbol} ({t.conviction_score:.0f}/100)"
            for i, t in enumerate(sorted_trades[:5], 1)
        )

        return {
            "title": "Weekly Congressional Trading Roundup",
            "description": f"Summary of congressional trading disclosures this week",
            "color": 0x5865F2,  # Discord blurple
            "fields": [
                {"name": "Total Trades", "value": str(len(trades)), "inline": True},
                {"name": "Total Volume", "value": f"${total_volume:,.0f}", "inline": True},
                {"name": "High Suspicion", "value": str(len(high_conviction)), "inline": True},
                {"name": "Purchases", "value": str(len(purchases)), "inline": True},
                {"name": "Sales", "value": str(len(trades) - len(purchases)), "inline": True},
                {"name": "Avg Score", "value": f"{sum(t.conviction_score or 0 for t in trades) / len(trades):.0f}", "inline": True},
                {"name": "Top Suspicious Trades", "value": top_trades_text, "inline": False},
            ],
            "footer": {"text": "Congressional Trading Intelligence"},
            "timestamp": datetime.utcnow().isoformat()
        }

    def generate_daily_digest(self, trades: List[TradeAlertData]) -> Dict[str, str]:
        """
        Generate daily digest email content.

        Args:
            trades: List of trade alerts from today

        Returns:
            Dictionary with subject, html, and text content
        """
        if not trades:
            return {
                "subject": "Daily Congressional Trading Digest - No New Filings",
                "html": "<p>No new congressional trading disclosures were filed today.</p>",
                "text": "No new congressional trading disclosures were filed today."
            }

        sorted_trades = sorted(
            trades,
            key=lambda x: x.conviction_score or 0,
            reverse=True
        )

        high_conviction = [t for t in trades if t.conviction_score and t.conviction_score >= 70]
        total_volume = sum((t.amount_min + t.amount_max) / 2 for t in trades)

        # Generate HTML
        trades_html = "\n".join(self.formatter.format_email_html(t) for t in sorted_trades[:10])

        subject = f"Daily Digest: {len(trades)} Congressional Trades"
        if high_conviction:
            subject += f" ({len(high_conviction)} High Suspicion)"

        html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f5f5f5;">
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="color: #1a1a2e; margin: 0;">Congressional Trading Digest</h1>
                <p style="color: #6c757d; margin: 10px 0 0;">{datetime.now().strftime('%B %d, %Y')}</p>
            </div>

            <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="margin: 0 0 15px; font-size: 16px; color: #333;">Today's Summary</h2>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; text-align: center;">
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                        <div style="font-size: 24px; font-weight: bold; color: #1a1a2e;">{len(trades)}</div>
                        <div style="font-size: 12px; color: #6c757d;">Trades Filed</div>
                    </div>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                        <div style="font-size: 24px; font-weight: bold; color: #1a1a2e;">${total_volume:,.0f}</div>
                        <div style="font-size: 12px; color: #6c757d;">Total Volume</div>
                    </div>
                    <div style="background: {'#ffebee' if high_conviction else '#f8f9fa'}; padding: 15px; border-radius: 5px;">
                        <div style="font-size: 24px; font-weight: bold; color: {'#dc3545' if high_conviction else '#1a1a2e'};">{len(high_conviction)}</div>
                        <div style="font-size: 12px; color: #6c757d;">High Suspicion</div>
                    </div>
                </div>
            </div>

            <h2 style="color: #333; font-size: 18px; margin: 30px 0 20px;">Today's Disclosures</h2>

            {trades_html}

            <div style="text-align: center; margin-top: 30px; padding: 20px; background: #1a1a2e; color: white; border-radius: 10px;">
                <p style="margin: 0; font-size: 12px; opacity: 0.8;">
                    This is educational analysis based on official STOCK Act disclosures.<br>
                    High scores indicate patterns worth investigating, not proof of wrongdoing.
                </p>
            </div>
        </div>
        """

        # Generate plain text version
        text_lines = [
            f"Congressional Trading Digest - {datetime.now().strftime('%B %d, %Y')}",
            "",
            f"Total Trades: {len(trades)}",
            f"Total Volume: ${total_volume:,.0f}",
            f"High Suspicion Trades: {len(high_conviction)}",
            "",
            "Top Trades:",
        ]

        for t in sorted_trades[:10]:
            text_lines.append(f"- {t.member_display}: {t.transaction_type} ${t.symbol} ({t.amount_range_str})")
            if t.conviction_score:
                text_lines.append(f"  Suspicion Score: {t.conviction_score:.0f}/100")

        return {
            "subject": subject,
            "html": html,
            "text": "\n".join(text_lines)
        }


class BaseBot(ABC):
    """Abstract base class for distribution bots."""

    def __init__(self, config: Optional[BotConfig] = None):
        self.config = config or BotConfig.from_env()
        self.formatter = AlertFormatter(self.config)
        self.content_generator = ContentGenerator(self.config)
        self._rate_limit_count = 0
        self._rate_limit_reset = datetime.now()

    @abstractmethod
    async def send_alert(self, alert: TradeAlertData) -> bool:
        """Send a trade alert. Returns True if successful."""
        pass

    @abstractmethod
    async def send_thread(self, alert: TradeAlertData) -> bool:
        """Send a multi-part thread/message for high conviction trades."""
        pass

    @abstractmethod
    async def send_digest(self, trades: List[TradeAlertData]) -> bool:
        """Send a daily/weekly digest."""
        pass

    def should_alert(self, alert: TradeAlertData) -> bool:
        """Check if alert meets criteria for distribution."""
        # Check minimum amount
        avg_amount = (alert.amount_min + alert.amount_max) / 2
        if avg_amount < self.config.min_trade_amount:
            logger.debug(f"Trade {alert.trade_id} below minimum amount threshold")
            return False

        # Check minimum conviction score
        if alert.conviction_score and alert.conviction_score < self.config.min_conviction_score:
            logger.debug(f"Trade {alert.trade_id} below minimum conviction score")
            return False

        # Check high conviction only mode
        if self.config.alert_on_high_conviction_only and not alert.is_high_conviction:
            logger.debug(f"Trade {alert.trade_id} not high conviction in high-conviction-only mode")
            return False

        return True

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now()

        # Reset counter every hour
        if (now - self._rate_limit_reset).total_seconds() > 3600:
            self._rate_limit_count = 0
            self._rate_limit_reset = now

        if self._rate_limit_count >= self.config.rate_limit_per_hour:
            logger.warning("Rate limit reached, skipping alert")
            return False

        return True

    def _increment_rate_limit(self):
        """Increment rate limit counter."""
        self._rate_limit_count += 1
