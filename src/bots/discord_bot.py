"""
Congressional Trading Intelligence System
Track D - Task D2: Discord Bot

Discord bot for real-time alerts and community engagement.

Features:
- Slash commands for querying trades and scores
- Auto-alerts for new filings
- High conviction trade notifications
- Weekly roundups
- Leaderboard commands

Environment Variables:
- DISCORD_BOT_TOKEN: Bot authentication token
- DISCORD_GUILD_ID: Server ID for guild-specific commands (optional)
- DISCORD_ALERT_CHANNEL_ID: Channel ID for automated alerts
- DISCORD_ADMIN_ROLE_ID: Role ID for admin commands
"""

import os
import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json

from .base import (
    BaseBot, BotConfig, AlertFormatter, ContentGenerator,
    TradeAlertData, AlertType, AlertPriority
)

logger = logging.getLogger(__name__)


@dataclass
class DiscordCredentials:
    """Discord bot credentials."""
    bot_token: str
    guild_id: Optional[str] = None
    alert_channel_id: Optional[str] = None
    admin_role_id: Optional[str] = None

    @classmethod
    def from_env(cls) -> "DiscordCredentials":
        """Load credentials from environment variables."""
        return cls(
            bot_token=os.getenv("DISCORD_BOT_TOKEN", ""),
            guild_id=os.getenv("DISCORD_GUILD_ID"),
            alert_channel_id=os.getenv("DISCORD_ALERT_CHANNEL_ID"),
            admin_role_id=os.getenv("DISCORD_ADMIN_ROLE_ID"),
        )

    @property
    def is_configured(self) -> bool:
        """Check if bot token is set."""
        return bool(self.bot_token)


@dataclass
class DiscordEmbed:
    """Discord embed message structure."""
    title: str
    description: str = ""
    color: int = 0x5865F2  # Discord blurple
    fields: List[Dict] = field(default_factory=list)
    footer: Optional[Dict] = None
    timestamp: Optional[str] = None
    thumbnail: Optional[Dict] = None
    image: Optional[Dict] = None
    author: Optional[Dict] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to Discord API format."""
        embed = {
            "title": self.title,
            "description": self.description,
            "color": self.color,
        }

        if self.fields:
            embed["fields"] = self.fields
        if self.footer:
            embed["footer"] = self.footer
        if self.timestamp:
            embed["timestamp"] = self.timestamp
        if self.thumbnail:
            embed["thumbnail"] = self.thumbnail
        if self.image:
            embed["image"] = self.image
        if self.author:
            embed["author"] = self.author
        if self.url:
            embed["url"] = self.url

        return embed


class SlashCommand:
    """Represents a Discord slash command."""

    def __init__(
        self,
        name: str,
        description: str,
        options: Optional[List[Dict]] = None,
        handler: Optional[Callable] = None
    ):
        self.name = name
        self.description = description
        self.options = options or []
        self.handler = handler

    def to_dict(self) -> Dict:
        """Convert to Discord API format for registration."""
        command = {
            "name": self.name,
            "description": self.description,
        }
        if self.options:
            command["options"] = self.options
        return command


class DiscordBot(BaseBot):
    """
    Discord bot for congressional trading alerts and queries.

    Provides:
    - Real-time trade alerts in configured channels
    - Slash commands for querying member trades and scores
    - Weekly roundup summaries
    - Interactive leaderboards

    Uses discord.py library with application commands (slash commands).
    """

    # Discord embed color codes
    COLORS = {
        "low": 0x28A745,      # Green
        "moderate": 0x17A2B8,  # Cyan
        "elevated": 0xFFC107,  # Yellow
        "high": 0xFD7E14,      # Orange
        "critical": 0xDC3545,  # Red
        "info": 0x5865F2,      # Discord blurple
        "success": 0x57F287,   # Discord green
        "warning": 0xFEE75C,   # Discord yellow
        "error": 0xED4245,     # Discord red
    }

    def __init__(
        self,
        config: Optional[BotConfig] = None,
        credentials: Optional[DiscordCredentials] = None
    ):
        super().__init__(config)
        self.credentials = credentials or DiscordCredentials.from_env()
        self._client = None
        self._commands: Dict[str, SlashCommand] = {}
        self._subscribed_channels: List[str] = []

        # Register default commands
        self._register_default_commands()

        if not self.credentials.is_configured:
            logger.warning(
                "Discord credentials not configured. Bot will run in dry-run mode. "
                "Set DISCORD_BOT_TOKEN environment variable."
            )
            self.config.dry_run = True

    def _register_default_commands(self):
        """Register the default slash commands."""

        # /trade command - get latest trades for a member
        self._commands["trade"] = SlashCommand(
            name="trade",
            description="Get latest trades for a congressional member",
            options=[
                {
                    "name": "member",
                    "description": "Member name to search for",
                    "type": 3,  # STRING
                    "required": True
                },
                {
                    "name": "limit",
                    "description": "Number of trades to show (default: 5)",
                    "type": 4,  # INTEGER
                    "required": False
                }
            ]
        )

        # /score command - get swamp/conviction score
        self._commands["score"] = SlashCommand(
            name="score",
            description="Get the suspicion score for a member or trade",
            options=[
                {
                    "name": "member",
                    "description": "Member name to get score for",
                    "type": 3,  # STRING
                    "required": True
                }
            ]
        )

        # /leaderboard command
        self._commands["leaderboard"] = SlashCommand(
            name="leaderboard",
            description="Show the congressional trading leaderboard",
            options=[
                {
                    "name": "type",
                    "description": "Leaderboard type",
                    "type": 3,  # STRING
                    "required": False,
                    "choices": [
                        {"name": "Most Suspicious", "value": "suspicious"},
                        {"name": "Most Trades", "value": "volume"},
                        {"name": "Best Performers", "value": "performance"},
                        {"name": "Worst Performers", "value": "worst"},
                    ]
                }
            ]
        )

        # /alerts command - manage channel subscriptions
        self._commands["alerts"] = SlashCommand(
            name="alerts",
            description="Manage trade alert subscriptions for this channel",
            options=[
                {
                    "name": "action",
                    "description": "Subscribe or unsubscribe from alerts",
                    "type": 3,  # STRING
                    "required": True,
                    "choices": [
                        {"name": "Subscribe", "value": "subscribe"},
                        {"name": "Unsubscribe", "value": "unsubscribe"},
                        {"name": "Status", "value": "status"},
                    ]
                }
            ]
        )

        # /lookup command - look up a specific stock
        self._commands["lookup"] = SlashCommand(
            name="lookup",
            description="Look up congressional trades for a specific stock",
            options=[
                {
                    "name": "symbol",
                    "description": "Stock symbol (e.g., NVDA, AAPL)",
                    "type": 3,  # STRING
                    "required": True
                },
                {
                    "name": "days",
                    "description": "Look back period in days (default: 30)",
                    "type": 4,  # INTEGER
                    "required": False
                }
            ]
        )

        # /help command
        self._commands["help"] = SlashCommand(
            name="help",
            description="Show help information for the Congressional Trading Bot"
        )

    def _get_client(self):
        """
        Get or create Discord client.

        Uses discord.py library if available.
        """
        if self._client is not None:
            return self._client

        if self.config.dry_run:
            return None

        try:
            import discord
            from discord import app_commands

            # Create intents
            intents = discord.Intents.default()
            intents.message_content = True

            # Create client
            self._client = discord.Client(intents=intents)

            # Attach command tree for slash commands
            self._client.tree = app_commands.CommandTree(self._client)

            logger.info("Discord client initialized")
            return self._client

        except ImportError:
            logger.warning(
                "discord.py library not installed. Install with: pip install discord.py"
            )
            self.config.dry_run = True
            return None

        except Exception as e:
            logger.error(f"Failed to initialize Discord client: {e}")
            self.config.dry_run = True
            return None

    def create_trade_embed(self, alert: TradeAlertData) -> DiscordEmbed:
        """
        Create a Discord embed for a trade alert.

        Args:
            alert: Trade alert data

        Returns:
            DiscordEmbed object
        """
        embed_dict = self.formatter.format_discord_embed(alert)
        return DiscordEmbed(**embed_dict)

    async def send_alert(self, alert: TradeAlertData) -> bool:
        """
        Send a trade alert to subscribed channels.

        Args:
            alert: Trade alert data

        Returns:
            True if sent successfully
        """
        if not self.should_alert(alert):
            return False

        if not self._check_rate_limit():
            return False

        embed = self.create_trade_embed(alert)

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would send Discord embed:\n{json.dumps(embed.to_dict(), indent=2)}")
            return True

        try:
            client = self._get_client()
            if client is None:
                return False

            # Send to alert channel if configured
            if self.credentials.alert_channel_id:
                channel = client.get_channel(int(self.credentials.alert_channel_id))
                if channel:
                    import discord
                    discord_embed = discord.Embed.from_dict(embed.to_dict())
                    await channel.send(embed=discord_embed)
                    self._increment_rate_limit()
                    logger.info(f"Sent alert to channel {self.credentials.alert_channel_id}")
                    return True

            # Send to all subscribed channels
            sent = 0
            for channel_id in self._subscribed_channels:
                try:
                    channel = client.get_channel(int(channel_id))
                    if channel:
                        import discord
                        discord_embed = discord.Embed.from_dict(embed.to_dict())
                        await channel.send(embed=discord_embed)
                        sent += 1
                except Exception as e:
                    logger.error(f"Failed to send to channel {channel_id}: {e}")

            self._increment_rate_limit()
            return sent > 0

        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    async def send_thread(self, alert: TradeAlertData) -> bool:
        """
        Send a detailed analysis for high conviction trades.

        Discord doesn't have threads like Twitter, so we send multiple embeds.

        Args:
            alert: Trade alert data

        Returns:
            True if sent successfully
        """
        if not alert.is_high_conviction:
            return await self.send_alert(alert)

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would send high conviction alert for {alert.trade_id}")
            return True

        try:
            # Send main alert embed
            main_embed = self.create_trade_embed(alert)

            # Create analysis embed
            analysis_embed = DiscordEmbed(
                title="Trade Analysis",
                description=alert.explanation or "No detailed analysis available.",
                color=self.COLORS.get(alert.risk_level or "info", self.COLORS["info"]),
                fields=[
                    {
                        "name": "Key Risk Factors",
                        "value": "\n".join(f" {f}" for f in alert.top_factors) if alert.top_factors else "None identified",
                        "inline": False
                    }
                ]
            )

            if self.config.dry_run:
                logger.info(f"[DRY RUN] Main embed:\n{json.dumps(main_embed.to_dict(), indent=2)}")
                logger.info(f"[DRY RUN] Analysis embed:\n{json.dumps(analysis_embed.to_dict(), indent=2)}")
                return True

            client = self._get_client()
            if client is None:
                return False

            if self.credentials.alert_channel_id:
                channel = client.get_channel(int(self.credentials.alert_channel_id))
                if channel:
                    import discord
                    await channel.send(embed=discord.Embed.from_dict(main_embed.to_dict()))
                    await channel.send(embed=discord.Embed.from_dict(analysis_embed.to_dict()))
                    self._increment_rate_limit()
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to send thread: {e}")
            return False

    async def send_digest(self, trades: List[TradeAlertData]) -> bool:
        """
        Send a weekly roundup to subscribed channels.

        Args:
            trades: List of trades from the past week

        Returns:
            True if sent successfully
        """
        if not trades:
            return False

        roundup = self.content_generator.generate_weekly_roundup(trades)
        embed_data = roundup.get("discord_embed", {})

        if not embed_data:
            return False

        embed = DiscordEmbed(**embed_data)

        if self.config.dry_run:
            logger.info(f"[DRY RUN] Would send weekly roundup:\n{json.dumps(embed.to_dict(), indent=2)}")
            return True

        try:
            client = self._get_client()
            if client is None:
                return False

            if self.credentials.alert_channel_id:
                channel = client.get_channel(int(self.credentials.alert_channel_id))
                if channel:
                    import discord
                    discord_embed = discord.Embed.from_dict(embed.to_dict())
                    await channel.send(
                        content="**Weekly Congressional Trading Roundup**",
                        embed=discord_embed
                    )
                    self._increment_rate_limit()
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to send digest: {e}")
            return False

    async def handle_command(
        self,
        command_name: str,
        options: Dict[str, Any],
        channel_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Handle a slash command.

        Args:
            command_name: Name of the command
            options: Command options/arguments
            channel_id: Channel where command was invoked
            user_id: User who invoked the command

        Returns:
            Response data for the command
        """
        if command_name == "help":
            return self._handle_help_command()

        elif command_name == "trade":
            return await self._handle_trade_command(options)

        elif command_name == "score":
            return await self._handle_score_command(options)

        elif command_name == "leaderboard":
            return await self._handle_leaderboard_command(options)

        elif command_name == "alerts":
            return await self._handle_alerts_command(options, channel_id)

        elif command_name == "lookup":
            return await self._handle_lookup_command(options)

        else:
            return {
                "type": "error",
                "message": f"Unknown command: {command_name}"
            }

    def _handle_help_command(self) -> Dict:
        """Handle the /help command."""
        embed = DiscordEmbed(
            title="Congressional Trading Bot - Help",
            description="Track and analyze congressional stock trading disclosures.",
            color=self.COLORS["info"],
            fields=[
                {
                    "name": "/trade <member>",
                    "value": "Get latest trades for a congressional member",
                    "inline": False
                },
                {
                    "name": "/score <member>",
                    "value": "Get the suspicion score for a member",
                    "inline": False
                },
                {
                    "name": "/leaderboard [type]",
                    "value": "Show trading leaderboards (suspicious, volume, performance)",
                    "inline": False
                },
                {
                    "name": "/lookup <symbol>",
                    "value": "Find congressional trades for a specific stock",
                    "inline": False
                },
                {
                    "name": "/alerts <subscribe|unsubscribe>",
                    "value": "Manage trade alert subscriptions for this channel",
                    "inline": False
                }
            ],
            footer={"text": "Data from official STOCK Act disclosures | Educational purposes only"}
        )

        return {
            "type": "embed",
            "embed": embed.to_dict()
        }

    async def _handle_trade_command(self, options: Dict) -> Dict:
        """Handle the /trade command."""
        member_name = options.get("member", "")
        limit = options.get("limit", 5)

        # This would connect to the actual database
        # For now, return a placeholder response
        embed = DiscordEmbed(
            title=f"Trades for {member_name}",
            description=f"Showing last {limit} trades",
            color=self.COLORS["info"],
            fields=[
                {
                    "name": "Note",
                    "value": "Connect to database to get actual trade data",
                    "inline": False
                }
            ]
        )

        return {
            "type": "embed",
            "embed": embed.to_dict()
        }

    async def _handle_score_command(self, options: Dict) -> Dict:
        """Handle the /score command."""
        member_name = options.get("member", "")

        embed = DiscordEmbed(
            title=f"Suspicion Score: {member_name}",
            description="Aggregate suspicion analysis",
            color=self.COLORS["warning"],
            fields=[
                {
                    "name": "Note",
                    "value": "Connect to intelligence engine to get actual scores",
                    "inline": False
                }
            ]
        )

        return {
            "type": "embed",
            "embed": embed.to_dict()
        }

    async def _handle_leaderboard_command(self, options: Dict) -> Dict:
        """Handle the /leaderboard command."""
        leaderboard_type = options.get("type", "suspicious")

        type_titles = {
            "suspicious": "Most Suspicious Traders",
            "volume": "Highest Trading Volume",
            "performance": "Best Trading Performance",
            "worst": "Worst Trading Performance"
        }

        embed = DiscordEmbed(
            title=f"Leaderboard: {type_titles.get(leaderboard_type, 'Unknown')}",
            description="Congressional trading rankings",
            color=self.COLORS["info"],
            fields=[
                {
                    "name": "Note",
                    "value": "Connect to database to get actual leaderboard data",
                    "inline": False
                }
            ]
        )

        return {
            "type": "embed",
            "embed": embed.to_dict()
        }

    async def _handle_alerts_command(self, options: Dict, channel_id: str) -> Dict:
        """Handle the /alerts command."""
        action = options.get("action", "status")

        if action == "subscribe":
            if channel_id not in self._subscribed_channels:
                self._subscribed_channels.append(channel_id)
            return {
                "type": "message",
                "message": "This channel is now subscribed to trade alerts!"
            }

        elif action == "unsubscribe":
            if channel_id in self._subscribed_channels:
                self._subscribed_channels.remove(channel_id)
            return {
                "type": "message",
                "message": "This channel has been unsubscribed from trade alerts."
            }

        else:  # status
            is_subscribed = channel_id in self._subscribed_channels
            return {
                "type": "message",
                "message": f"Alert status: {'Subscribed' if is_subscribed else 'Not subscribed'}"
            }

    async def _handle_lookup_command(self, options: Dict) -> Dict:
        """Handle the /lookup command."""
        symbol = options.get("symbol", "").upper()
        days = options.get("days", 30)

        embed = DiscordEmbed(
            title=f"Congressional Trades: ${symbol}",
            description=f"Trades in the last {days} days",
            color=self.COLORS["info"],
            fields=[
                {
                    "name": "Note",
                    "value": "Connect to database to get actual trade data",
                    "inline": False
                }
            ]
        )

        return {
            "type": "embed",
            "embed": embed.to_dict()
        }

    def get_commands(self) -> List[Dict]:
        """Get list of slash commands for registration."""
        return [cmd.to_dict() for cmd in self._commands.values()]

    def get_status(self) -> Dict[str, Any]:
        """Get bot status and statistics."""
        return {
            "configured": self.credentials.is_configured,
            "dry_run": self.config.dry_run,
            "rate_limit_count": self._rate_limit_count,
            "rate_limit_max": self.config.rate_limit_per_hour,
            "subscribed_channels": len(self._subscribed_channels),
            "registered_commands": list(self._commands.keys()),
            "environment": self.config.environment,
        }


def create_discord_bot_runner():
    """
    Create a full Discord bot runner with event handlers.

    Returns a function that can be used to run the bot.
    """
    config = BotConfig.from_env()
    credentials = DiscordCredentials.from_env()
    bot = DiscordBot(config=config, credentials=credentials)

    if not credentials.is_configured:
        logger.error("Discord credentials not configured")
        return None

    try:
        import discord
        from discord import app_commands

        intents = discord.Intents.default()
        intents.message_content = True

        client = discord.Client(intents=intents)
        tree = app_commands.CommandTree(client)

        @client.event
        async def on_ready():
            logger.info(f"Discord bot logged in as {client.user}")

            # Sync commands
            if credentials.guild_id:
                guild = discord.Object(id=int(credentials.guild_id))
                tree.copy_global_to(guild=guild)
                await tree.sync(guild=guild)
            else:
                await tree.sync()

            logger.info("Slash commands synced")

        # Register commands
        @tree.command(name="help", description="Show help information")
        async def help_command(interaction: discord.Interaction):
            result = bot._handle_help_command()
            embed = discord.Embed.from_dict(result["embed"])
            await interaction.response.send_message(embed=embed)

        @tree.command(name="trade", description="Get latest trades for a member")
        @app_commands.describe(member="Member name to search for", limit="Number of trades")
        async def trade_command(interaction: discord.Interaction, member: str, limit: int = 5):
            result = await bot._handle_trade_command({"member": member, "limit": limit})
            embed = discord.Embed.from_dict(result["embed"])
            await interaction.response.send_message(embed=embed)

        @tree.command(name="score", description="Get suspicion score for a member")
        @app_commands.describe(member="Member name")
        async def score_command(interaction: discord.Interaction, member: str):
            result = await bot._handle_score_command({"member": member})
            embed = discord.Embed.from_dict(result["embed"])
            await interaction.response.send_message(embed=embed)

        @tree.command(name="leaderboard", description="Show trading leaderboard")
        @app_commands.describe(type="Leaderboard type")
        @app_commands.choices(type=[
            app_commands.Choice(name="Most Suspicious", value="suspicious"),
            app_commands.Choice(name="Most Trades", value="volume"),
            app_commands.Choice(name="Best Performers", value="performance"),
            app_commands.Choice(name="Worst Performers", value="worst"),
        ])
        async def leaderboard_command(
            interaction: discord.Interaction,
            type: str = "suspicious"
        ):
            result = await bot._handle_leaderboard_command({"type": type})
            embed = discord.Embed.from_dict(result["embed"])
            await interaction.response.send_message(embed=embed)

        @tree.command(name="lookup", description="Find trades for a stock symbol")
        @app_commands.describe(symbol="Stock symbol (e.g., NVDA)", days="Lookback period")
        async def lookup_command(
            interaction: discord.Interaction,
            symbol: str,
            days: int = 30
        ):
            result = await bot._handle_lookup_command({"symbol": symbol, "days": days})
            embed = discord.Embed.from_dict(result["embed"])
            await interaction.response.send_message(embed=embed)

        @tree.command(name="alerts", description="Manage alert subscriptions")
        @app_commands.describe(action="Subscribe, unsubscribe, or check status")
        @app_commands.choices(action=[
            app_commands.Choice(name="Subscribe", value="subscribe"),
            app_commands.Choice(name="Unsubscribe", value="unsubscribe"),
            app_commands.Choice(name="Status", value="status"),
        ])
        async def alerts_command(interaction: discord.Interaction, action: str):
            result = await bot._handle_alerts_command(
                {"action": action},
                str(interaction.channel_id)
            )
            await interaction.response.send_message(result["message"])

        return lambda: client.run(credentials.bot_token)

    except ImportError:
        logger.error("discord.py not installed. Run: pip install discord.py")
        return None


# Command-line interface
async def main():
    """Test the Discord bot functionality."""
    import argparse

    parser = argparse.ArgumentParser(description="Discord Bot for Congressional Trading")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Don't actually connect")
    parser.add_argument("--test-alert", action="store_true", help="Test alert formatting")
    parser.add_argument("--run", action="store_true", help="Actually run the bot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.run:
        runner = create_discord_bot_runner()
        if runner:
            runner()
        return

    config = BotConfig(dry_run=args.dry_run)
    bot = DiscordBot(config=config)

    print("Discord Bot Status:")
    print(json.dumps(bot.get_status(), indent=2))

    print("\nRegistered Commands:")
    for cmd in bot.get_commands():
        print(f"  /{cmd['name']}: {cmd['description']}")

    if args.test_alert:
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
                "Trade in oversight sector",
                "Large trade size",
                "Near filing deadline"
            ],
            explanation="High conviction due to committee position.",
            committee_overlap="Intelligence - Tech",
            filing_delay_days=36
        )

        print("\nTest Alert Embed:")
        embed = bot.create_trade_embed(test_alert)
        print(json.dumps(embed.to_dict(), indent=2))

        print("\nSending test alert...")
        success = await bot.send_alert(test_alert)
        print(f"Result: {success}")


if __name__ == "__main__":
    asyncio.run(main())
