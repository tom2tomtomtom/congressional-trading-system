"""
Congressional Trading Intelligence System
Track D - Task D3: Email Alert System

Email subscription and notification system for trade alerts.

Features:
- Instant alerts for high conviction trades
- Daily digest emails
- Weekly roundup emails
- Member-specific watchlist alerts
- Subscription management

Environment Variables:
- EMAIL_PROVIDER: "sendgrid" or "ses" (default: sendgrid)
- SENDGRID_API_KEY: SendGrid API key
- AWS_ACCESS_KEY_ID: AWS access key (for SES)
- AWS_SECRET_ACCESS_KEY: AWS secret key (for SES)
- AWS_REGION: AWS region (for SES, default: us-east-1)
- EMAIL_FROM_ADDRESS: Sender email address
- EMAIL_FROM_NAME: Sender name
- EMAIL_REPLY_TO: Reply-to address
"""

import os
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import json
import hashlib
import secrets
import re

# Add parent directory for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.base import (
    BotConfig, AlertFormatter, ContentGenerator,
    TradeAlertData, AlertType, AlertPriority
)

logger = logging.getLogger(__name__)


class EmailProvider(Enum):
    """Supported email providers."""
    SENDGRID = "sendgrid"
    SES = "ses"
    SMTP = "smtp"


class SubscriptionFrequency(Enum):
    """Email subscription frequency options."""
    INSTANT = "instant"      # Real-time alerts for high priority
    DAILY = "daily"          # Daily digest
    WEEKLY = "weekly"        # Weekly roundup
    NONE = "none"            # Unsubscribed


@dataclass
class EmailCredentials:
    """Email service credentials."""
    provider: EmailProvider
    api_key: Optional[str] = None  # For SendGrid
    aws_access_key: Optional[str] = None  # For SES
    aws_secret_key: Optional[str] = None
    aws_region: str = "us-east-1"
    smtp_host: Optional[str] = None  # For SMTP
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    from_address: str = "alerts@congresstrading.io"
    from_name: str = "Congressional Trading Intelligence"
    reply_to: Optional[str] = None

    @classmethod
    def from_env(cls) -> "EmailCredentials":
        """Load credentials from environment variables."""
        provider_str = os.getenv("EMAIL_PROVIDER", "sendgrid").lower()
        provider = EmailProvider(provider_str) if provider_str in [e.value for e in EmailProvider] else EmailProvider.SENDGRID

        return cls(
            provider=provider,
            api_key=os.getenv("SENDGRID_API_KEY"),
            aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            smtp_host=os.getenv("SMTP_HOST"),
            smtp_port=int(os.getenv("SMTP_PORT", "587")),
            smtp_username=os.getenv("SMTP_USERNAME"),
            smtp_password=os.getenv("SMTP_PASSWORD"),
            from_address=os.getenv("EMAIL_FROM_ADDRESS", "alerts@congresstrading.io"),
            from_name=os.getenv("EMAIL_FROM_NAME", "Congressional Trading Intelligence"),
            reply_to=os.getenv("EMAIL_REPLY_TO"),
        )

    @property
    def is_configured(self) -> bool:
        """Check if credentials are properly configured."""
        if self.provider == EmailProvider.SENDGRID:
            return bool(self.api_key)
        elif self.provider == EmailProvider.SES:
            return bool(self.aws_access_key and self.aws_secret_key)
        elif self.provider == EmailProvider.SMTP:
            return bool(self.smtp_host and self.smtp_username)
        return False


@dataclass
class Subscriber:
    """Email subscriber information."""
    id: str
    email: str
    created_at: datetime = field(default_factory=datetime.now)
    confirmed: bool = False
    confirmation_token: Optional[str] = None
    unsubscribe_token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    frequency: SubscriptionFrequency = SubscriptionFrequency.DAILY
    watchlist: Set[str] = field(default_factory=set)  # Member IDs or symbols
    preferences: Dict[str, Any] = field(default_factory=dict)
    last_email_sent: Optional[datetime] = None

    def __post_init__(self):
        if not self.confirmation_token:
            self.confirmation_token = secrets.token_urlsafe(32)

    @property
    def is_active(self) -> bool:
        """Check if subscriber is active (confirmed and not unsubscribed)."""
        return self.confirmed and self.frequency != SubscriptionFrequency.NONE

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
            "confirmed": self.confirmed,
            "frequency": self.frequency.value,
            "watchlist": list(self.watchlist),
            "preferences": self.preferences,
            "last_email_sent": self.last_email_sent.isoformat() if self.last_email_sent else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Subscriber":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            email=data["email"],
            created_at=datetime.fromisoformat(data["created_at"]),
            confirmed=data.get("confirmed", False),
            frequency=SubscriptionFrequency(data.get("frequency", "daily")),
            watchlist=set(data.get("watchlist", [])),
            preferences=data.get("preferences", {}),
            last_email_sent=datetime.fromisoformat(data["last_email_sent"]) if data.get("last_email_sent") else None,
        )


class EmailTemplate:
    """Email template management."""

    @staticmethod
    def get_base_template() -> str:
        """Get the base HTML email template."""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{subject}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 10px 10px 0 0;
        }}
        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}
        .header p {{
            margin: 10px 0 0;
            opacity: 0.9;
        }}
        .content {{
            background: white;
            padding: 30px;
            border: 1px solid #dee2e6;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px;
            text-align: center;
            font-size: 12px;
            color: #6c757d;
            border-radius: 0 0 10px 10px;
            border: 1px solid #dee2e6;
            border-top: none;
        }}
        .button {{
            display: inline-block;
            padding: 12px 24px;
            background-color: #16213e;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .alert-card {{
            border: 1px solid #dee2e6;
            border-radius: 8px;
            margin: 15px 0;
            overflow: hidden;
        }}
        .alert-header {{
            background: #f8f9fa;
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        .alert-body {{
            padding: 15px;
        }}
        .score-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }}
        .score-low {{ background: #d4edda; color: #155724; }}
        .score-moderate {{ background: #d1ecf1; color: #0c5460; }}
        .score-elevated {{ background: #fff3cd; color: #856404; }}
        .score-high {{ background: #f8d7da; color: #721c24; }}
        .score-critical {{ background: #dc3545; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        {content}
    </div>
</body>
</html>
"""

    @staticmethod
    def get_confirmation_email(subscriber: Subscriber, confirm_url: str) -> Dict[str, str]:
        """Generate confirmation email content."""
        subject = "Confirm Your Congressional Trading Alerts Subscription"

        html_content = f"""
        <div class="header">
            <h1>Congressional Trading Intelligence</h1>
            <p>Confirm Your Subscription</p>
        </div>
        <div class="content">
            <h2>Welcome!</h2>
            <p>Thank you for subscribing to congressional trading alerts.</p>
            <p>Please confirm your email address by clicking the button below:</p>
            <p style="text-align: center;">
                <a href="{confirm_url}" class="button">Confirm Subscription</a>
            </p>
            <p style="font-size: 12px; color: #6c757d;">
                If you didn't request this subscription, you can safely ignore this email.
            </p>
        </div>
        <div class="footer">
            <p>Congressional Trading Intelligence</p>
            <p>Educational analysis of public STOCK Act disclosures</p>
        </div>
        """

        text_content = f"""
Congressional Trading Intelligence - Confirm Your Subscription

Thank you for subscribing to congressional trading alerts.

Please confirm your email address by visiting:
{confirm_url}

If you didn't request this subscription, you can safely ignore this email.
"""

        return {
            "subject": subject,
            "html": EmailTemplate.get_base_template().format(subject=subject, content=html_content),
            "text": text_content,
        }

    @staticmethod
    def get_instant_alert(alert: TradeAlertData, unsubscribe_url: str) -> Dict[str, str]:
        """Generate instant alert email content."""
        formatter = AlertFormatter()
        alert_html = formatter.format_email_html(alert)

        subject = f"Trade Alert: {alert.member_display} - ${alert.symbol}"
        if alert.conviction_score and alert.conviction_score >= 80:
            subject = f"[HIGH] {subject}"

        html_content = f"""
        <div class="header">
            <h1>Trade Alert</h1>
            <p>{alert.member_display}</p>
        </div>
        <div class="content">
            {alert_html}
        </div>
        <div class="footer">
            <p>This is educational analysis based on official STOCK Act disclosures.</p>
            <p>High scores indicate patterns worth investigating, not proof of wrongdoing.</p>
            <p><a href="{unsubscribe_url}">Unsubscribe</a> from these alerts</p>
        </div>
        """

        text_content = f"""
Trade Alert: {alert.member_display}

{alert.transaction_type}: ${alert.symbol}
Amount: {alert.amount_range_str}
Trade Date: {alert.transaction_date}

{"Suspicion Score: " + str(alert.conviction_score) + "/100" if alert.conviction_score else ""}

This is educational analysis based on official STOCK Act disclosures.

Unsubscribe: {unsubscribe_url}
"""

        return {
            "subject": subject,
            "html": EmailTemplate.get_base_template().format(subject=subject, content=html_content),
            "text": text_content,
        }


class SubscriptionManager:
    """
    Manages email subscriptions.

    In production, this would use a database. For now, uses in-memory storage.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path
        self._subscribers: Dict[str, Subscriber] = {}
        self._email_index: Dict[str, str] = {}  # email -> subscriber_id

        if storage_path and os.path.exists(storage_path):
            self._load_subscribers()

    def _load_subscribers(self):
        """Load subscribers from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for sub_data in data.get("subscribers", []):
                    sub = Subscriber.from_dict(sub_data)
                    self._subscribers[sub.id] = sub
                    self._email_index[sub.email.lower()] = sub.id
        except Exception as e:
            logger.error(f"Failed to load subscribers: {e}")

    def _save_subscribers(self):
        """Save subscribers to storage."""
        if not self.storage_path:
            return

        try:
            data = {
                "subscribers": [s.to_dict() for s in self._subscribers.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save subscribers: {e}")

    def _generate_id(self, email: str) -> str:
        """Generate subscriber ID from email."""
        return hashlib.sha256(email.lower().encode()).hexdigest()[:16]

    def _validate_email(self, email: str) -> bool:
        """Validate email format."""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    def subscribe(
        self,
        email: str,
        frequency: SubscriptionFrequency = SubscriptionFrequency.DAILY,
        watchlist: Optional[Set[str]] = None
    ) -> Optional[Subscriber]:
        """
        Add a new subscriber.

        Args:
            email: Email address
            frequency: Subscription frequency
            watchlist: Optional set of member IDs or symbols to watch

        Returns:
            Subscriber object or None if invalid
        """
        if not self._validate_email(email):
            logger.warning(f"Invalid email format: {email}")
            return None

        email_lower = email.lower()

        # Check if already subscribed
        if email_lower in self._email_index:
            existing_id = self._email_index[email_lower]
            existing = self._subscribers.get(existing_id)
            if existing and existing.is_active:
                logger.info(f"Email already subscribed: {email}")
                return existing

        # Create new subscriber
        subscriber = Subscriber(
            id=self._generate_id(email),
            email=email_lower,
            frequency=frequency,
            watchlist=watchlist or set(),
        )

        self._subscribers[subscriber.id] = subscriber
        self._email_index[email_lower] = subscriber.id
        self._save_subscribers()

        logger.info(f"New subscriber: {email}")
        return subscriber

    def confirm(self, token: str) -> Optional[Subscriber]:
        """
        Confirm a subscription.

        Args:
            token: Confirmation token

        Returns:
            Confirmed subscriber or None
        """
        for subscriber in self._subscribers.values():
            if subscriber.confirmation_token == token:
                subscriber.confirmed = True
                subscriber.confirmation_token = None  # Invalidate token
                self._save_subscribers()
                logger.info(f"Subscription confirmed: {subscriber.email}")
                return subscriber

        logger.warning(f"Invalid confirmation token: {token}")
        return None

    def unsubscribe(self, token: str) -> bool:
        """
        Unsubscribe using unsubscribe token.

        Args:
            token: Unsubscribe token

        Returns:
            True if unsubscribed successfully
        """
        for subscriber in self._subscribers.values():
            if subscriber.unsubscribe_token == token:
                subscriber.frequency = SubscriptionFrequency.NONE
                self._save_subscribers()
                logger.info(f"Unsubscribed: {subscriber.email}")
                return True

        logger.warning(f"Invalid unsubscribe token: {token}")
        return False

    def get_subscriber(self, email: str) -> Optional[Subscriber]:
        """Get subscriber by email."""
        email_lower = email.lower()
        if email_lower in self._email_index:
            return self._subscribers.get(self._email_index[email_lower])
        return None

    def get_subscribers_for_frequency(
        self,
        frequency: SubscriptionFrequency
    ) -> List[Subscriber]:
        """Get all active subscribers for a given frequency."""
        return [
            s for s in self._subscribers.values()
            if s.is_active and s.frequency == frequency
        ]

    def get_subscribers_watching(self, identifier: str) -> List[Subscriber]:
        """Get subscribers watching a specific member or symbol."""
        return [
            s for s in self._subscribers.values()
            if s.is_active and identifier in s.watchlist
        ]

    def update_preferences(
        self,
        email: str,
        frequency: Optional[SubscriptionFrequency] = None,
        watchlist: Optional[Set[str]] = None,
        preferences: Optional[Dict] = None
    ) -> Optional[Subscriber]:
        """Update subscriber preferences."""
        subscriber = self.get_subscriber(email)
        if not subscriber:
            return None

        if frequency is not None:
            subscriber.frequency = frequency
        if watchlist is not None:
            subscriber.watchlist = watchlist
        if preferences is not None:
            subscriber.preferences.update(preferences)

        self._save_subscribers()
        return subscriber


class EmailService:
    """
    Email service for sending trade alerts and digests.

    Supports SendGrid and AWS SES.
    """

    def __init__(
        self,
        credentials: Optional[EmailCredentials] = None,
        subscription_manager: Optional[SubscriptionManager] = None,
        config: Optional[BotConfig] = None
    ):
        self.credentials = credentials or EmailCredentials.from_env()
        self.subscription_manager = subscription_manager or SubscriptionManager()
        self.config = config or BotConfig.from_env()
        self.formatter = AlertFormatter(self.config)
        self.content_generator = ContentGenerator(self.config)
        self._client = None

        if not self.credentials.is_configured:
            logger.warning(
                "Email credentials not configured. Service will run in dry-run mode. "
                "Set SENDGRID_API_KEY or AWS_ACCESS_KEY_ID environment variables."
            )
            self.config.dry_run = True

    def _get_sendgrid_client(self):
        """Get SendGrid client."""
        try:
            from sendgrid import SendGridAPIClient
            return SendGridAPIClient(self.credentials.api_key)
        except ImportError:
            logger.warning("sendgrid library not installed. Run: pip install sendgrid")
            return None

    def _get_ses_client(self):
        """Get AWS SES client."""
        try:
            import boto3
            return boto3.client(
                'ses',
                region_name=self.credentials.aws_region,
                aws_access_key_id=self.credentials.aws_access_key,
                aws_secret_access_key=self.credentials.aws_secret_key
            )
        except ImportError:
            logger.warning("boto3 library not installed. Run: pip install boto3")
            return None

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Send an email.

        Args:
            to_email: Recipient email
            subject: Email subject
            html_content: HTML body
            text_content: Plain text body (optional)

        Returns:
            True if sent successfully
        """
        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would send email to {to_email}")
            logger.info(f"  Subject: {subject}")
            return True

        try:
            if self.credentials.provider == EmailProvider.SENDGRID:
                return await self._send_via_sendgrid(to_email, subject, html_content, text_content)
            elif self.credentials.provider == EmailProvider.SES:
                return await self._send_via_ses(to_email, subject, html_content, text_content)
            else:
                logger.error(f"Unsupported email provider: {self.credentials.provider}")
                return False

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    async def _send_via_sendgrid(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str]
    ) -> bool:
        """Send email via SendGrid."""
        try:
            from sendgrid.helpers.mail import Mail, Email, To, Content

            client = self._get_sendgrid_client()
            if not client:
                return False

            message = Mail(
                from_email=Email(self.credentials.from_address, self.credentials.from_name),
                to_emails=To(to_email),
                subject=subject
            )
            message.add_content(Content("text/html", html_content))
            if text_content:
                message.add_content(Content("text/plain", text_content))

            response = client.send(message)
            success = 200 <= response.status_code < 300

            if success:
                logger.info(f"Email sent to {to_email} via SendGrid")
            else:
                logger.error(f"SendGrid error: {response.status_code}")

            return success

        except Exception as e:
            logger.error(f"SendGrid send error: {e}")
            return False

    async def _send_via_ses(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str]
    ) -> bool:
        """Send email via AWS SES."""
        try:
            client = self._get_ses_client()
            if not client:
                return False

            body = {"Html": {"Data": html_content}}
            if text_content:
                body["Text"] = {"Data": text_content}

            response = client.send_email(
                Source=f"{self.credentials.from_name} <{self.credentials.from_address}>",
                Destination={"ToAddresses": [to_email]},
                Message={
                    "Subject": {"Data": subject},
                    "Body": body
                }
            )

            message_id = response.get("MessageId")
            if message_id:
                logger.info(f"Email sent to {to_email} via SES: {message_id}")
                return True

            return False

        except Exception as e:
            logger.error(f"SES send error: {e}")
            return False

    async def send_instant_alert(
        self,
        alert: TradeAlertData,
        subscribers: Optional[List[Subscriber]] = None
    ) -> int:
        """
        Send instant alert to subscribers.

        Args:
            alert: Trade alert data
            subscribers: List of subscribers (or None to use all instant subscribers)

        Returns:
            Number of emails sent successfully
        """
        if subscribers is None:
            subscribers = self.subscription_manager.get_subscribers_for_frequency(
                SubscriptionFrequency.INSTANT
            )

        # Also include watchlist subscribers
        watchlist_subs = (
            self.subscription_manager.get_subscribers_watching(alert.symbol) +
            self.subscription_manager.get_subscribers_watching(alert.member_name)
        )
        subscribers = list(set(subscribers + watchlist_subs))

        sent = 0
        for subscriber in subscribers:
            if not subscriber.is_active:
                continue

            unsubscribe_url = f"https://congresstrading.io/unsubscribe?token={subscriber.unsubscribe_token}"
            email_content = EmailTemplate.get_instant_alert(alert, unsubscribe_url)

            success = await self.send_email(
                subscriber.email,
                email_content["subject"],
                email_content["html"],
                email_content["text"]
            )

            if success:
                subscriber.last_email_sent = datetime.now()
                sent += 1

        logger.info(f"Sent instant alert to {sent} subscribers")
        return sent

    async def send_daily_digest(
        self,
        trades: List[TradeAlertData],
        subscribers: Optional[List[Subscriber]] = None
    ) -> int:
        """
        Send daily digest to subscribers.

        Args:
            trades: List of trades from today
            subscribers: List of subscribers (or None to use all daily subscribers)

        Returns:
            Number of emails sent successfully
        """
        if subscribers is None:
            subscribers = self.subscription_manager.get_subscribers_for_frequency(
                SubscriptionFrequency.DAILY
            )

        digest_content = self.content_generator.generate_daily_digest(trades)

        sent = 0
        for subscriber in subscribers:
            if not subscriber.is_active:
                continue

            # Add unsubscribe link to footer
            unsubscribe_url = f"https://congresstrading.io/unsubscribe?token={subscriber.unsubscribe_token}"
            html_with_unsub = digest_content["html"].replace(
                "</div>\n        </div>",
                f'<p style="margin-top: 20px;"><a href="{unsubscribe_url}">Unsubscribe</a></p></div>\n        </div>'
            )

            success = await self.send_email(
                subscriber.email,
                digest_content["subject"],
                html_with_unsub,
                digest_content["text"] + f"\n\nUnsubscribe: {unsubscribe_url}"
            )

            if success:
                subscriber.last_email_sent = datetime.now()
                sent += 1

        logger.info(f"Sent daily digest to {sent} subscribers")
        return sent

    async def send_weekly_roundup(
        self,
        trades: List[TradeAlertData],
        subscribers: Optional[List[Subscriber]] = None
    ) -> int:
        """
        Send weekly roundup to subscribers.

        Args:
            trades: List of trades from the past week
            subscribers: List of subscribers (or None to use all weekly subscribers)

        Returns:
            Number of emails sent successfully
        """
        if subscribers is None:
            subscribers = self.subscription_manager.get_subscribers_for_frequency(
                SubscriptionFrequency.WEEKLY
            )

        # Also include daily subscribers in weekly roundup
            daily_subs = self.subscription_manager.get_subscribers_for_frequency(
                SubscriptionFrequency.DAILY
            )
            subscribers = list(set(subscribers + daily_subs))

        roundup = self.content_generator.generate_weekly_roundup(trades)

        # Create email content
        summary = roundup.get("summary", {})
        top_trades = roundup.get("top_suspicious", [])

        # Build HTML for top trades
        trades_html = ""
        for trade in top_trades[:5]:
            trades_html += self.formatter.format_email_html(trade)

        subject = f"Weekly Congressional Trading Roundup - {summary.get('total_trades', 0)} Trades"

        html_content = f"""
        <div class="header">
            <h1>Weekly Congressional Trading Roundup</h1>
            <p>{datetime.now().strftime('%B %d, %Y')}</p>
        </div>
        <div class="content">
            <h2>This Week's Summary</h2>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; text-align: center; margin-bottom: 30px;">
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 24px; font-weight: bold;">{summary.get('total_trades', 0)}</div>
                    <div style="font-size: 12px; color: #6c757d;">Total Trades</div>
                </div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 24px; font-weight: bold;">${summary.get('total_volume', 0):,.0f}</div>
                    <div style="font-size: 12px; color: #6c757d;">Total Volume</div>
                </div>
                <div style="background: #ffebee; padding: 15px; border-radius: 5px;">
                    <div style="font-size: 24px; font-weight: bold; color: #dc3545;">{summary.get('high_conviction_count', 0)}</div>
                    <div style="font-size: 12px; color: #6c757d;">High Suspicion</div>
                </div>
            </div>

            <h2>Most Suspicious Trades</h2>
            {trades_html}
        </div>
        <div class="footer">
            <p>This is educational analysis based on official STOCK Act disclosures.</p>
            <p>High scores indicate patterns worth investigating, not proof of wrongdoing.</p>
            <p><a href="{{{{unsubscribe_url}}}}">Unsubscribe</a> from these emails</p>
        </div>
        """

        sent = 0
        for subscriber in subscribers:
            if not subscriber.is_active:
                continue

            unsubscribe_url = f"https://congresstrading.io/unsubscribe?token={subscriber.unsubscribe_token}"
            html_with_unsub = html_content.replace("{{unsubscribe_url}}", unsubscribe_url)
            full_html = EmailTemplate.get_base_template().format(
                subject=subject,
                content=html_with_unsub
            )

            success = await self.send_email(
                subscriber.email,
                subject,
                full_html
            )

            if success:
                subscriber.last_email_sent = datetime.now()
                sent += 1

        logger.info(f"Sent weekly roundup to {sent} subscribers")
        return sent

    async def send_confirmation_email(self, subscriber: Subscriber) -> bool:
        """
        Send subscription confirmation email.

        Args:
            subscriber: Subscriber to confirm

        Returns:
            True if sent successfully
        """
        confirm_url = f"https://congresstrading.io/confirm?token={subscriber.confirmation_token}"
        email_content = EmailTemplate.get_confirmation_email(subscriber, confirm_url)

        return await self.send_email(
            subscriber.email,
            email_content["subject"],
            email_content["html"],
            email_content["text"]
        )

    def get_status(self) -> Dict[str, Any]:
        """Get service status and statistics."""
        return {
            "configured": self.credentials.is_configured,
            "provider": self.credentials.provider.value,
            "dry_run": self.config.dry_run,
            "total_subscribers": len(self.subscription_manager._subscribers),
            "active_subscribers": sum(
                1 for s in self.subscription_manager._subscribers.values()
                if s.is_active
            ),
            "frequency_breakdown": {
                freq.value: len(self.subscription_manager.get_subscribers_for_frequency(freq))
                for freq in SubscriptionFrequency
                if freq != SubscriptionFrequency.NONE
            }
        }


# Scheduler for automated emails
class EmailScheduler:
    """
    Scheduler for automated email dispatching.

    Handles:
    - Daily digest emails (sent at configured time)
    - Weekly roundup emails (sent on configured day)
    """

    def __init__(
        self,
        email_service: EmailService,
        daily_hour: int = 18,  # 6 PM
        weekly_day: int = 0,   # Monday
        weekly_hour: int = 9   # 9 AM
    ):
        self.email_service = email_service
        self.daily_hour = daily_hour
        self.weekly_day = weekly_day
        self.weekly_hour = weekly_hour
        self._running = False
        self._last_daily: Optional[date] = None
        self._last_weekly: Optional[date] = None

    async def start(self, get_trades_func):
        """
        Start the scheduler.

        Args:
            get_trades_func: Async function that returns list of TradeAlertData
        """
        self._running = True
        logger.info("Email scheduler started")

        while self._running:
            now = datetime.now()

            # Check for daily digest
            if (
                now.hour == self.daily_hour and
                (self._last_daily is None or self._last_daily < now.date())
            ):
                try:
                    trades = await get_trades_func(days=1)
                    await self.email_service.send_daily_digest(trades)
                    self._last_daily = now.date()
                except Exception as e:
                    logger.error(f"Daily digest error: {e}")

            # Check for weekly roundup
            if (
                now.weekday() == self.weekly_day and
                now.hour == self.weekly_hour and
                (self._last_weekly is None or self._last_weekly < now.date())
            ):
                try:
                    trades = await get_trades_func(days=7)
                    await self.email_service.send_weekly_roundup(trades)
                    self._last_weekly = now.date()
                except Exception as e:
                    logger.error(f"Weekly roundup error: {e}")

            await asyncio.sleep(60)  # Check every minute

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        logger.info("Email scheduler stopped")


# Command-line interface
async def main():
    """Test the email service functionality."""
    import argparse

    parser = argparse.ArgumentParser(description="Email Service for Congressional Trading")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Don't actually send")
    parser.add_argument("--test-alert", action="store_true", help="Send a test instant alert")
    parser.add_argument("--test-digest", action="store_true", help="Send a test daily digest")
    parser.add_argument("--subscribe", type=str, help="Subscribe an email address")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = BotConfig(dry_run=args.dry_run)
    service = EmailService(config=config)

    print("Email Service Status:")
    print(json.dumps(service.get_status(), indent=2))

    if args.subscribe:
        subscriber = service.subscription_manager.subscribe(args.subscribe)
        if subscriber:
            print(f"\nSubscribed: {subscriber.email}")
            print(f"Confirmation token: {subscriber.confirmation_token}")
            await service.send_confirmation_email(subscriber)

    if args.test_alert or args.test_digest:
        test_alert = TradeAlertData(
            trade_id="TEST001",
            member_name="Nancy Pelosi",
            member_party="D",
            member_state="CA",
            member_chamber="House",
            symbol="NVDA",
            company_name="NVIDIA Corporation",
            transaction_type="Purchase",
            amount_min=1000000,
            amount_max=5000000,
            transaction_date=date(2024, 1, 15),
            filing_date=date(2024, 2, 20),
            conviction_score=85.5,
            risk_level="critical",
            top_factors=[
                "Trade in sector overseen by committee",
                "Large trade relative to portfolio",
                "Filed near deadline"
            ],
            explanation="This trade has a high conviction score due to multiple risk factors.",
            committee_overlap="Intelligence Committee - Technology sector",
            filing_delay_days=36
        )

        if args.test_alert:
            print("\nSending test instant alert...")
            # Would need subscribers configured
            # For dry run, just show what would be sent

        if args.test_digest:
            print("\nSending test daily digest...")
            await service.send_daily_digest([test_alert])


if __name__ == "__main__":
    asyncio.run(main())
