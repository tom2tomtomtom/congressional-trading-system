"""
Congressional Trading Intelligence System - Distribution Bots

Track D: Distribution & Engagement
- D1: Twitter/X Bot for automated trade alerts and thread generation
- D2: Discord Bot for real-time alerts and slash commands
- D3: Email Alert System (see notifications package)

All bots use environment variables for API credentials:
- TWITTER_API_KEY, TWITTER_API_SECRET, etc.
- DISCORD_BOT_TOKEN
- SENDGRID_API_KEY or AWS_SES credentials
"""

from .base import BotConfig, AlertFormatter, ContentGenerator
from .twitter_bot import TwitterBot, TweetThread
from .discord_bot import DiscordBot, DiscordEmbed

__all__ = [
    "BotConfig",
    "AlertFormatter",
    "ContentGenerator",
    "TwitterBot",
    "TweetThread",
    "DiscordBot",
    "DiscordEmbed",
]
