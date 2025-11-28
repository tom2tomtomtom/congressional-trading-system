"""
Congressional Trading Intelligence System
Track D - Task D1: Twitter/X Bot

Automated Twitter bot that posts about new filings and suspicious patterns.

Features:
- New filing alerts
- High conviction trade threads
- Weekly roundups
- Leaderboard updates

Environment Variables:
- TWITTER_API_KEY: Twitter API key (consumer key)
- TWITTER_API_SECRET: Twitter API secret (consumer secret)
- TWITTER_ACCESS_TOKEN: OAuth access token
- TWITTER_ACCESS_SECRET: OAuth access token secret
- TWITTER_BEARER_TOKEN: Bearer token for API v2
"""

import os
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import time

from .base import (
    BaseBot, BotConfig, AlertFormatter, ContentGenerator,
    TradeAlertData, AlertType, AlertPriority
)

logger = logging.getLogger(__name__)


@dataclass
class TweetThread:
    """Represents a Twitter thread."""
    tweets: List[str]
    trade_id: str
    created_at: datetime = field(default_factory=datetime.now)
    tweet_ids: List[str] = field(default_factory=list)
    posted: bool = False

    def __len__(self) -> int:
        return len(self.tweets)


@dataclass
class TwitterCredentials:
    """Twitter API credentials."""
    api_key: str
    api_secret: str
    access_token: str
    access_secret: str
    bearer_token: str

    @classmethod
    def from_env(cls) -> "TwitterCredentials":
        """Load credentials from environment variables."""
        return cls(
            api_key=os.getenv("TWITTER_API_KEY", ""),
            api_secret=os.getenv("TWITTER_API_SECRET", ""),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN", ""),
            access_secret=os.getenv("TWITTER_ACCESS_SECRET", ""),
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN", ""),
        )

    @property
    def is_configured(self) -> bool:
        """Check if all credentials are set."""
        return all([
            self.api_key,
            self.api_secret,
            self.access_token,
            self.access_secret,
        ])


class TwitterBot(BaseBot):
    """
    Twitter/X bot for distributing congressional trading alerts.

    Handles:
    - Posting single tweets for new filings
    - Creating threads for high conviction trades
    - Weekly roundup threads
    - Leaderboard updates

    Uses Twitter API v2 with OAuth 1.0a authentication.
    """

    # Rate limits per Twitter API v2
    TWEETS_PER_15_MIN = 50
    TWEETS_PER_3_HOURS = 300

    def __init__(
        self,
        config: Optional[BotConfig] = None,
        credentials: Optional[TwitterCredentials] = None
    ):
        super().__init__(config)
        self.credentials = credentials or TwitterCredentials.from_env()
        self._client = None
        self._last_tweet_times: List[datetime] = []
        self._pending_queue: List[Tuple[TradeAlertData, AlertPriority]] = []

        if not self.credentials.is_configured:
            logger.warning(
                "Twitter credentials not configured. Bot will run in dry-run mode. "
                "Set TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, "
                "and TWITTER_ACCESS_SECRET environment variables."
            )
            self.config.dry_run = True

    def _get_client(self):
        """
        Get or create Twitter API client.

        Uses tweepy library if available, otherwise operates in simulation mode.
        """
        if self._client is not None:
            return self._client

        if self.config.dry_run:
            return None

        try:
            import tweepy

            self._client = tweepy.Client(
                bearer_token=self.credentials.bearer_token,
                consumer_key=self.credentials.api_key,
                consumer_secret=self.credentials.api_secret,
                access_token=self.credentials.access_token,
                access_token_secret=self.credentials.access_secret,
                wait_on_rate_limit=True
            )
            logger.info("Twitter client initialized successfully")
            return self._client

        except ImportError:
            logger.warning(
                "tweepy library not installed. Install with: pip install tweepy"
            )
            self.config.dry_run = True
            return None

        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")
            self.config.dry_run = True
            return None

    async def send_alert(self, alert: TradeAlertData) -> bool:
        """
        Send a single tweet for a trade alert.

        Args:
            alert: Trade alert data

        Returns:
            True if tweet was posted successfully
        """
        if not self.should_alert(alert):
            return False

        if not self._check_rate_limit():
            self._pending_queue.append((alert, AlertPriority.MEDIUM))
            return False

        tweet_text = self.formatter.format_tweet(alert)

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would post tweet:\n{tweet_text}")
            return True

        try:
            client = self._get_client()
            if client is None:
                logger.warning("No Twitter client available")
                return False

            response = client.create_tweet(text=tweet_text)
            self._increment_rate_limit()
            self._last_tweet_times.append(datetime.now())

            logger.info(f"Posted tweet for trade {alert.trade_id}: {response.data['id']}")
            return True

        except Exception as e:
            logger.error(f"Failed to post tweet: {e}")
            return False

    async def send_thread(self, alert: TradeAlertData) -> bool:
        """
        Send a multi-tweet thread for high conviction trades.

        Args:
            alert: Trade alert data

        Returns:
            True if thread was posted successfully
        """
        if not alert.is_high_conviction:
            # For non-high-conviction trades, just send a single tweet
            return await self.send_alert(alert)

        tweets = self.formatter.format_tweet_thread(alert)
        thread = TweetThread(tweets=tweets, trade_id=alert.trade_id)

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would post thread ({len(tweets)} tweets):")
            for i, tweet in enumerate(tweets, 1):
                logger.info(f"  Tweet {i}:\n{tweet}")
            return True

        try:
            client = self._get_client()
            if client is None:
                logger.warning("No Twitter client available")
                return False

            previous_tweet_id = None

            for i, tweet_text in enumerate(tweets):
                # Check rate limit before each tweet
                if not self._check_rate_limit():
                    logger.warning(f"Rate limit hit during thread at tweet {i+1}")
                    return False

                # Post tweet (as reply if not first)
                if previous_tweet_id:
                    response = client.create_tweet(
                        text=tweet_text,
                        in_reply_to_tweet_id=previous_tweet_id
                    )
                else:
                    response = client.create_tweet(text=tweet_text)

                previous_tweet_id = response.data['id']
                thread.tweet_ids.append(previous_tweet_id)
                self._increment_rate_limit()
                self._last_tweet_times.append(datetime.now())

                # Small delay between tweets in thread
                await asyncio.sleep(1)

            thread.posted = True
            logger.info(
                f"Posted thread for trade {alert.trade_id}: "
                f"{len(thread.tweet_ids)} tweets, first ID: {thread.tweet_ids[0]}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to post thread: {e}")
            return False

    async def send_digest(self, trades: List[TradeAlertData]) -> bool:
        """
        Send a weekly roundup thread.

        Args:
            trades: List of trade alerts from the past week

        Returns:
            True if digest was posted successfully
        """
        if not trades:
            logger.info("No trades for digest")
            return False

        roundup = self.content_generator.generate_weekly_roundup(trades)
        tweets = roundup.get("twitter_thread", [])

        if not tweets:
            logger.warning("Failed to generate roundup thread")
            return False

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would post weekly roundup ({len(tweets)} tweets):")
            for i, tweet in enumerate(tweets, 1):
                logger.info(f"  Tweet {i}:\n{tweet}")
            return True

        try:
            client = self._get_client()
            if client is None:
                return False

            previous_tweet_id = None

            for tweet_text in tweets:
                if not self._check_rate_limit():
                    logger.warning("Rate limit hit during roundup thread")
                    return False

                if previous_tweet_id:
                    response = client.create_tweet(
                        text=tweet_text,
                        in_reply_to_tweet_id=previous_tweet_id
                    )
                else:
                    response = client.create_tweet(text=tweet_text)

                previous_tweet_id = response.data['id']
                self._increment_rate_limit()
                await asyncio.sleep(1)

            logger.info(f"Posted weekly roundup: {len(tweets)} tweets")
            return True

        except Exception as e:
            logger.error(f"Failed to post digest: {e}")
            return False

    async def post_breaking_alert(self, alert: TradeAlertData) -> bool:
        """
        Post a breaking/urgent alert for critical trades.

        Uses different formatting with emphasis on urgency.

        Args:
            alert: Trade alert data

        Returns:
            True if posted successfully
        """
        if not alert.conviction_score or alert.conviction_score < 80:
            return await self.send_alert(alert)

        risk_emoji = ""
        tweet_text = f"""{risk_emoji} BREAKING: Critical Trade Alert {risk_emoji}

{alert.member_display}

{alert.transaction_type}: ${alert.symbol}
Amount: {alert.amount_range_str}

Suspicion Score: {alert.conviction_score:.0f}/100 (CRITICAL)

Thread with full analysis below...

#CongressTrading #STOCKAct"""

        if len(tweet_text) > 280:
            tweet_text = tweet_text[:277] + "..."

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would post breaking alert:\n{tweet_text}")
            # Then post the full thread
            return await self.send_thread(alert)

        try:
            client = self._get_client()
            if client is None:
                return False

            # Post breaking alert
            response = client.create_tweet(text=tweet_text)
            first_tweet_id = response.data['id']
            self._increment_rate_limit()

            # Then post the analysis thread as replies
            thread_tweets = self.formatter.format_tweet_thread(alert)

            for tweet in thread_tweets:
                if not self._check_rate_limit():
                    break

                response = client.create_tweet(
                    text=tweet,
                    in_reply_to_tweet_id=first_tweet_id
                )
                first_tweet_id = response.data['id']
                self._increment_rate_limit()
                await asyncio.sleep(1)

            logger.info(f"Posted breaking alert for trade {alert.trade_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to post breaking alert: {e}")
            return False

    async def post_leaderboard_update(
        self,
        leaderboard: List[Dict],
        leaderboard_type: str = "weekly"
    ) -> bool:
        """
        Post leaderboard update (best/worst traders).

        Args:
            leaderboard: List of member performance data
            leaderboard_type: "weekly", "monthly", or "all_time"

        Returns:
            True if posted successfully
        """
        if not leaderboard:
            return False

        type_display = {
            "weekly": "This Week's",
            "monthly": "This Month's",
            "all_time": "All-Time"
        }.get(leaderboard_type, "Latest")

        # Format leaderboard entries
        entries = []
        for i, entry in enumerate(leaderboard[:5], 1):
            name = entry.get("member_name", "Unknown")
            party = entry.get("party", "?")
            score = entry.get("average_conviction", 0)
            entries.append(f"{i}. {name} ({party}): {score:.0f}/100")

        tweet_text = f"""CONGRESS TRADING LEADERBOARD

{type_display} Most Suspicious Traders:

{chr(10).join(entries)}

Based on average conviction scores across all trades.

#CongressTrading #Accountability"""

        if len(tweet_text) > 280:
            # Truncate entries if needed
            entries = entries[:3]
            tweet_text = f"""LEADERBOARD UPDATE

{type_display} Top Suspicious:

{chr(10).join(entries)}

#CongressTrading"""

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would post leaderboard:\n{tweet_text}")
            return True

        try:
            client = self._get_client()
            if client is None:
                return False

            response = client.create_tweet(text=tweet_text)
            self._increment_rate_limit()

            logger.info(f"Posted leaderboard update: {response.data['id']}")
            return True

        except Exception as e:
            logger.error(f"Failed to post leaderboard: {e}")
            return False

    async def process_queue(self) -> int:
        """
        Process pending tweets from the queue.

        Returns:
            Number of tweets processed
        """
        if not self._pending_queue:
            return 0

        # Sort by priority
        self._pending_queue.sort(key=lambda x: x[1].value, reverse=True)

        processed = 0
        while self._pending_queue and self._check_rate_limit():
            alert, priority = self._pending_queue.pop(0)

            if priority.value >= AlertPriority.HIGH.value:
                success = await self.send_thread(alert)
            else:
                success = await self.send_alert(alert)

            if success:
                processed += 1

            # Small delay between posts
            await asyncio.sleep(2)

        return processed

    def get_status(self) -> Dict[str, Any]:
        """Get bot status and statistics."""
        return {
            "configured": self.credentials.is_configured,
            "dry_run": self.config.dry_run,
            "rate_limit_count": self._rate_limit_count,
            "rate_limit_max": self.config.rate_limit_per_hour,
            "pending_queue_size": len(self._pending_queue),
            "last_tweet_times": [t.isoformat() for t in self._last_tweet_times[-10:]],
            "environment": self.config.environment,
        }


class TwitterBotScheduler:
    """
    Scheduler for Twitter bot operations.

    Handles:
    - Periodic polling for new trades
    - Weekly digest scheduling
    - Queue processing
    """

    def __init__(self, bot: TwitterBot):
        self.bot = bot
        self._running = False
        self._last_check = datetime.now()

    async def start(self):
        """Start the scheduler."""
        self._running = True
        logger.info("Twitter bot scheduler started")

        while self._running:
            try:
                # Process any pending queue items
                processed = await self.bot.process_queue()
                if processed > 0:
                    logger.info(f"Processed {processed} queued tweets")

                # Sleep for a minute before next check
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        logger.info("Twitter bot scheduler stopped")


# Command-line interface for testing
async def main():
    """Test the Twitter bot functionality."""
    import argparse

    parser = argparse.ArgumentParser(description="Twitter Bot for Congressional Trading")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Don't actually post")
    parser.add_argument("--test-alert", action="store_true", help="Send a test alert")
    parser.add_argument("--test-thread", action="store_true", help="Send a test thread")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = BotConfig(dry_run=args.dry_run)
    bot = TwitterBot(config=config)

    print("Twitter Bot Status:")
    print(json.dumps(bot.get_status(), indent=2))

    if args.test_alert or args.test_thread:
        # Create test alert
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
                "Trade in sector overseen by member's committee",
                "Large trade relative to portfolio",
                "Filed 36 days after transaction"
            ],
            explanation="This trade has a high conviction score due to the member's position on the Intelligence Committee which has oversight of technology and defense sectors.",
            committee_overlap="Intelligence Committee - Technology sector",
            filing_delay_days=36
        )

        if args.test_thread:
            print("\nSending test thread...")
            success = await bot.send_thread(test_alert)
            print(f"Thread sent: {success}")
        else:
            print("\nSending test alert...")
            success = await bot.send_alert(test_alert)
            print(f"Alert sent: {success}")


if __name__ == "__main__":
    asyncio.run(main())
