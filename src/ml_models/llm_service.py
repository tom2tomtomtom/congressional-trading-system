#!/usr/bin/env python3
"""
Congressional Trading Intelligence System
Track F - Task F1: LLM Story Generation Service

This module integrates with the Claude API to generate journalist-ready stories
about congressional trading patterns. It supports multiple output formats
including tweets, news briefs, and deep-dive investigations.

Formats:
- Tweet Thread: 280 char segments, 3-5 tweets
- News Brief: 200 words, inverted pyramid style
- Deep Dive: 1000 words, full investigation format
- Data Card: Key stats for social sharing
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


class StoryFormat(Enum):
    """Available story output formats."""
    TWEET_THREAD = "tweet_thread"
    NEWS_BRIEF = "news_brief"
    DEEP_DIVE = "deep_dive"
    DATA_CARD = "data_card"


class StoryType(Enum):
    """Types of stories that can be generated."""
    TRADE_STORY = "trade_story"
    MEMBER_PROFILE = "member_profile"
    PATTERN_ALERT = "pattern_alert"
    WEEKLY_ROUNDUP = "weekly_roundup"
    COMMITTEE_ANALYSIS = "committee_analysis"


@dataclass
class Story:
    """Generated story output."""
    story_id: str
    story_type: StoryType
    format: StoryFormat
    content: str
    headline: Optional[str] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    disclaimer: str = field(default="")
    word_count: int = 0

    def __post_init__(self):
        """Calculate word count and add disclaimer if missing."""
        self.word_count = len(self.content.split())
        if not self.disclaimer:
            self.disclaimer = (
                "DISCLAIMER: This analysis is for educational purposes only, based on "
                "publicly disclosed STOCK Act filings. High conviction scores indicate "
                "patterns worth investigating, not proof of wrongdoing. This is not "
                "financial or legal advice."
            )

    def to_dict(self) -> Dict:
        """Convert story to dictionary for serialization."""
        return {
            "story_id": self.story_id,
            "story_type": self.story_type.value,
            "format": self.format.value,
            "headline": self.headline,
            "content": self.content,
            "summary": self.summary,
            "metadata": self.metadata,
            "generated_at": self.generated_at.isoformat(),
            "disclaimer": self.disclaimer,
            "word_count": self.word_count
        }

    def to_markdown(self) -> str:
        """Convert story to markdown format."""
        md = ""
        if self.headline:
            md += f"# {self.headline}\n\n"
        if self.summary:
            md += f"*{self.summary}*\n\n"
        md += self.content
        md += f"\n\n---\n\n*{self.disclaimer}*"
        return md

    def to_html(self) -> str:
        """Convert story to HTML format."""
        html = "<article class='congressional-trading-story'>\n"
        if self.headline:
            html += f"  <h1>{self.headline}</h1>\n"
        if self.summary:
            html += f"  <p class='summary'><em>{self.summary}</em></p>\n"
        html += f"  <div class='content'>{self.content}</div>\n"
        html += f"  <footer class='disclaimer'>{self.disclaimer}</footer>\n"
        html += "</article>"
        return html


class PromptTemplate:
    """Manages prompt templates for different story types and formats."""

    # Base system prompt establishing the AI's role and guidelines
    SYSTEM_PROMPT = """You are an investigative financial journalist analyzing congressional trading patterns.
Your goal is to write factual, compelling stories about potential conflicts of interest based on
publicly disclosed STOCK Act filings.

Guidelines:
1. Be factual and precise - cite specific dates, amounts, and percentages
2. Lead with the most newsworthy element
3. Explain why this matters to ordinary citizens
4. Never make accusations - describe patterns and let readers draw conclusions
5. Include relevant context about committee assignments and oversight areas
6. Use plain language accessible to general audiences
7. End with what this pattern suggests and any patterns in similar cases

Important: Always refer to trades as "disclosed trades" and scores as "pattern analysis scores"
to maintain objectivity."""

    # Tweet thread template (280 chars per tweet, 3-5 tweets)
    TWEET_THREAD = """Generate a Twitter/X thread (3-5 tweets, max 280 chars each) about this congressional trade:

Trade Data:
- Member: {member_name} ({party}-{state})
- Stock: {symbol} ({company_name})
- Transaction: {transaction_type} of ${amount_display}
- Trade Date: {transaction_date}
- Filing Date: {filing_date}
- Filing Delay: {filing_delay} days
- Committee: {committee}
- Conviction Score: {conviction_score}/100

Key Factors:
{conviction_factors}

Format each tweet on its own line, numbered (1/, 2/, etc). Include:
- Tweet 1: Hook with the most surprising fact
- Tweet 2-3: Key details and context
- Tweet 4: Why this matters
- Tweet 5 (optional): Call to action or broader pattern

Use hashtags sparingly: #CongressTrading #StockAct #Accountability"""

    # News brief template (200 words, inverted pyramid)
    NEWS_BRIEF = """Write a 200-word news brief about this congressional trade in inverted pyramid style:

Trade Data:
- Member: {member_name} ({party}-{state})
- Chamber: {chamber}
- Stock: {symbol} ({company_name})
- Transaction: {transaction_type} of ${amount_display}
- Trade Date: {transaction_date}
- Filing Date: {filing_date}
- Filing Delay: {filing_delay} days
- Committee: {committee}
- Conviction Score: {conviction_score}/100 ({risk_level} risk)

Key Factors:
{conviction_factors}

Additional Context:
{additional_context}

Structure:
1. Lead paragraph: Who, What, When, with the most newsworthy angle
2. Second paragraph: Key details about timing and amounts
3. Third paragraph: Committee context and why this matters
4. Fourth paragraph: Historical pattern or broader context
5. Final paragraph: What happens next or pattern implications

Write in objective journalistic style. Start with the most important information."""

    # Deep dive template (1000 words, full investigation)
    DEEP_DIVE = """Write a 1000-word investigative piece about this congressional trading pattern:

Member Profile:
- Name: {member_name}
- Party: {party}
- State: {state}
- Chamber: {chamber}
- Committees: {committees}
- Net Worth: ${net_worth_display}
- Leadership Role: {leadership_role}

Trade Under Analysis:
- Stock: {symbol} ({company_name})
- Transaction: {transaction_type} of ${amount_display}
- Trade Date: {transaction_date}
- Filing Date: {filing_date}
- Filing Delay: {filing_delay} days
- Conviction Score: {conviction_score}/100 ({risk_level} risk)

Detailed Factor Analysis:
{detailed_factors}

Trading History:
{trading_history}

Committee Oversight Context:
{committee_context}

Market Context:
{market_context}

Structure your article as:
1. HEADLINE: Attention-grabbing but factual
2. SUBHEAD: Key takeaway in one sentence
3. OPENING (150 words): Scene-setting and the central tension
4. THE TRADE (200 words): Detailed breakdown of what happened
5. THE PATTERN (200 words): Historical context and similar trades
6. THE COMMITTEE CONNECTION (200 words): Oversight and access
7. WHAT IT MEANS (150 words): Implications and accountability
8. CONCLUSION (100 words): What readers should watch for

Write with authority but remain objective. Let the data tell the story."""

    # Data card template (social sharing)
    DATA_CARD = """Generate a data card summary for social sharing about this trade:

Trade Data:
- Member: {member_name} ({party}-{state})
- Stock: {symbol}
- Transaction: {transaction_type} of ${amount_display}
- Trade Date: {transaction_date}
- Conviction Score: {conviction_score}/100
- Top Factor: {top_factor}

Create a concise data card with:
1. ONE-LINE HOOK: Most surprising fact (max 100 chars)
2. KEY STATS (bullet points):
   - Amount
   - Timing
   - Committee connection
   - Score
3. BOTTOM LINE: One sentence takeaway

Format for easy reading on social media. Use numbers and percentages for impact."""

    # Member profile template
    MEMBER_PROFILE = """Write a comprehensive trading profile for this congressional member:

Member Information:
- Name: {member_name}
- Party: {party}
- State: {state}
- Chamber: {chamber}
- District: {district}
- Committees: {committees}
- Leadership Role: {leadership_role}
- Tenure: {tenure_years} years
- Net Worth: ${net_worth_display}

Trading Summary:
- Total Trades: {total_trades}
- Estimated Volume: ${total_volume_display}
- Average Trade Size: ${avg_trade_display}
- Most Traded Sector: {top_sector}
- Highest Conviction Trade: {highest_conviction_trade}
- Average Conviction Score: {avg_conviction_score}/100

Notable Trades:
{notable_trades}

Committee-Trading Overlap:
{committee_overlap}

Create a profile with format {format_type} including:
1. Opening hook about their trading pattern
2. Key statistics and rankings
3. Most concerning patterns
4. Historical context
5. Accountability implications"""

    # Pattern alert template
    PATTERN_ALERT = """Write an alert about this detected trading pattern:

Pattern Type: {pattern_type}
Pattern Description: {pattern_description}

Affected Trades:
{trades_summary}

Members Involved:
{members_summary}

Statistical Significance: {significance}
Time Period: {time_period}

Key Findings:
{key_findings}

Create a {format_type} alert that:
1. Leads with the pattern discovery
2. Shows specific examples
3. Explains statistical significance
4. Puts it in historical context
5. Suggests what this pattern indicates"""

    # Weekly roundup template
    WEEKLY_ROUNDUP = """Write a weekly roundup of congressional trading activity:

Period: {start_date} to {end_date}

Summary Statistics:
- Total New Disclosures: {total_disclosures}
- Total Transaction Volume: ${total_volume_display}
- Average Conviction Score: {avg_conviction_score}/100
- High-Risk Trades (score > 70): {high_risk_count}

Top 5 Most Suspicious Trades:
{top_suspicious_trades}

Notable Members This Week:
{notable_members}

Sector Breakdown:
{sector_breakdown}

Timing Patterns:
{timing_patterns}

Create a {format_type} roundup that:
1. Opens with the week's biggest story
2. Ranks the most concerning disclosures
3. Shows sector trends
4. Highlights unusual patterns
5. Compares to historical averages
6. Ends with what to watch next week"""

    @classmethod
    def get_template(cls, story_type: StoryType, format: StoryFormat) -> str:
        """Get the appropriate template for the story type and format."""
        templates = {
            StoryType.TRADE_STORY: {
                StoryFormat.TWEET_THREAD: cls.TWEET_THREAD,
                StoryFormat.NEWS_BRIEF: cls.NEWS_BRIEF,
                StoryFormat.DEEP_DIVE: cls.DEEP_DIVE,
                StoryFormat.DATA_CARD: cls.DATA_CARD,
            },
            StoryType.MEMBER_PROFILE: {
                StoryFormat.TWEET_THREAD: cls.MEMBER_PROFILE,
                StoryFormat.NEWS_BRIEF: cls.MEMBER_PROFILE,
                StoryFormat.DEEP_DIVE: cls.MEMBER_PROFILE,
                StoryFormat.DATA_CARD: cls.MEMBER_PROFILE,
            },
            StoryType.PATTERN_ALERT: {
                StoryFormat.TWEET_THREAD: cls.PATTERN_ALERT,
                StoryFormat.NEWS_BRIEF: cls.PATTERN_ALERT,
                StoryFormat.DEEP_DIVE: cls.PATTERN_ALERT,
                StoryFormat.DATA_CARD: cls.PATTERN_ALERT,
            },
            StoryType.WEEKLY_ROUNDUP: {
                StoryFormat.TWEET_THREAD: cls.WEEKLY_ROUNDUP,
                StoryFormat.NEWS_BRIEF: cls.WEEKLY_ROUNDUP,
                StoryFormat.DEEP_DIVE: cls.WEEKLY_ROUNDUP,
                StoryFormat.DATA_CARD: cls.WEEKLY_ROUNDUP,
            },
        }
        return templates.get(story_type, {}).get(format, cls.NEWS_BRIEF)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, max_tokens: int = 2000) -> str:
        """Generate text from the LLM."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class ClaudeProvider(LLMProvider):
    """Claude API provider for story generation."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
            model: Model to use for generation.
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy initialization of the Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                raise ImportError("Please install the anthropic package: pip install anthropic")
        return self._client

    def is_available(self) -> bool:
        """Check if Claude API is available."""
        return bool(self.api_key)

    def generate(self, prompt: str, system_prompt: str, max_tokens: int = 2000) -> str:
        """
        Generate text using Claude API.

        Args:
            prompt: User prompt with data and instructions
            system_prompt: System prompt establishing context
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text content
        """
        if not self.is_available():
            raise ValueError("Claude API key not configured. Set ANTHROPIC_API_KEY environment variable.")

        try:
            client = self._get_client()
            message = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing without API access."""

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, system_prompt: str, max_tokens: int = 2000) -> str:
        """Generate mock story content for testing."""
        # Extract key information from prompt for mock response
        if "TWEET" in prompt.upper() or "thread" in prompt.lower():
            return self._mock_tweet_thread(prompt)
        elif "1000-word" in prompt.lower() or "deep dive" in prompt.lower():
            return self._mock_deep_dive(prompt)
        elif "data card" in prompt.lower():
            return self._mock_data_card(prompt)
        else:
            return self._mock_news_brief(prompt)

    def _mock_tweet_thread(self, prompt: str) -> str:
        return """1/ BREAKING: Congressional trading pattern detected that raises significant questions about timing and oversight.

2/ The disclosed trade occurred just days before committee action on related legislation, representing a substantial position in the affected sector.

3/ Our pattern analysis found this trade scores in the elevated range based on committee access, timing proximity, and historical patterns.

4/ Why this matters: Members of Congress have access to non-public information through their oversight roles. These patterns deserve scrutiny.

5/ Follow along as we track congressional trading patterns. All data from public STOCK Act disclosures. #CongressTrading #Accountability"""

    def _mock_news_brief(self, prompt: str) -> str:
        return """A recently disclosed congressional trade has raised questions about the timing of stock transactions by lawmakers with oversight of related sectors.

The trade, filed with the House or Senate as required by the STOCK Act, shows a significant position taken in a sector under the member's committee jurisdiction. The timing of the transaction coincides with legislative and regulatory developments that would not have been public knowledge at the time.

Our analysis assigns this trade an elevated pattern score based on factors including committee access, timing proximity to relevant events, and deviation from the member's historical trading patterns.

The STOCK Act, passed in 2012, requires members of Congress to disclose securities transactions within 45 days. While trading by lawmakers is legal, patterns that suggest use of non-public information raise ethical concerns.

This trade adds to ongoing scrutiny of congressional trading practices and calls for stricter oversight of lawmakers' financial activities."""

    def _mock_deep_dive(self, prompt: str) -> str:
        return """# Congressional Trading Pattern: An Investigation

## Summary
A pattern of trading activity by a congressional member warrants closer examination based on timing, sector alignment, and historical behavior.

## The Trade
The disclosed transaction represents a significant position in a sector directly under the member's committee oversight. The timing of this trade, occurring before material developments in the sector, raises questions about the information available to the member at the time of the transaction.

## The Pattern
This is not an isolated incident. Historical analysis reveals a pattern of trades that correlate with committee activities and legislative developments. The consistency of this pattern over time suggests more than coincidence.

## Committee Connection
The member serves on a committee with direct oversight of this sector. This position provides access to non-public information through briefings, hearings, and regulatory discussions that could inform trading decisions.

## The Data
Our analysis incorporates multiple factors:
- Committee membership and oversight areas
- Timing relative to known events
- Filing delays and disclosure patterns
- Trade size relative to portfolio and historical behavior
- Sector concentration in oversight areas

## What It Means
While no accusation can be made based on trading patterns alone, these patterns are exactly what investigators look for when examining potential misuse of material non-public information. The concentration of profitable trades in oversight sectors deserves continued scrutiny.

## Conclusion
Congressional trading remains an area of significant public interest. As citizens, we have the right to monitor how our elected officials' financial interests may intersect with their public duties. This analysis is based entirely on public STOCK Act disclosures and represents patterns, not accusations."""

    def _mock_data_card(self, prompt: str) -> str:
        return """HOOK: Member's trade timed perfectly with committee action on related sector

KEY STATS:
- Transaction: Significant position in oversight sector
- Timing: Days before material developments
- Committee: Direct oversight of affected industry
- Pattern Score: Elevated based on multiple factors

BOTTOM LINE: This trade pattern warrants scrutiny based on timing and committee access."""


class LLMService:
    """
    Main service for generating stories about congressional trading patterns.

    Integrates with Claude API for high-quality story generation, with fallback
    to mock generation for testing.
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        use_mock: bool = False
    ):
        """
        Initialize the LLM service.

        Args:
            provider: LLM provider to use. Defaults to Claude.
            use_mock: If True, use mock provider for testing.
        """
        if use_mock:
            self.provider = MockLLMProvider()
        elif provider:
            self.provider = provider
        else:
            self.provider = ClaudeProvider()

        self._story_counter = 0
        self._cost_tracker = {"total_tokens": 0, "estimated_cost": 0.0}

    def _generate_story_id(self) -> str:
        """Generate unique story ID."""
        self._story_counter += 1
        return f"story_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._story_counter}"

    def _format_amount(self, amount: Union[int, float]) -> str:
        """Format monetary amounts for display."""
        if amount >= 1_000_000:
            return f"{amount/1_000_000:.1f}M"
        elif amount >= 1_000:
            return f"{amount/1_000:.0f}K"
        else:
            return f"{amount:,.0f}"

    def _format_conviction_factors(self, factors: Dict) -> str:
        """Format conviction factors for prompt."""
        lines = []
        for name, factor in sorted(factors.items(), key=lambda x: x[1].get('weighted_score', 0), reverse=True):
            score = factor.get('score', 0)
            explanation = factor.get('explanation', '')
            lines.append(f"- {name}: {score}/100 - {explanation}")
        return "\n".join(lines)

    def generate_trade_story(
        self,
        trade: Dict,
        member: Dict,
        conviction_result: Optional[Dict] = None,
        format: StoryFormat = StoryFormat.NEWS_BRIEF
    ) -> Story:
        """
        Generate a story about a specific trade.

        Args:
            trade: Trade data dictionary
            member: Member data dictionary
            conviction_result: Optional conviction analysis result
            format: Output format for the story

        Returns:
            Generated Story object
        """
        # Prepare template variables
        amount_from = trade.get('amount_from', trade.get('amountFrom', 0))
        amount_to = trade.get('amount_to', trade.get('amountTo', amount_from))
        avg_amount = (amount_from + amount_to) / 2 if amount_to else amount_from

        # Calculate filing delay
        from datetime import datetime as dt
        trans_date = trade.get('transaction_date', '')
        filing_date = trade.get('filing_date', '')
        filing_delay = 0
        if trans_date and filing_date:
            try:
                td = dt.strptime(trans_date, '%Y-%m-%d')
                fd = dt.strptime(filing_date, '%Y-%m-%d')
                filing_delay = (fd - td).days
            except ValueError:
                pass

        # Get conviction data
        conviction_score = 0
        risk_level = "unknown"
        conviction_factors = ""
        top_factor = "Unable to determine"

        if conviction_result:
            conviction_score = conviction_result.get('score', 0)
            risk_level = conviction_result.get('risk_level', 'unknown')
            if isinstance(risk_level, Enum):
                risk_level = risk_level.value
            factors = conviction_result.get('factors', {})
            conviction_factors = self._format_conviction_factors(factors)
            top_factors = conviction_result.get('top_factors', [])
            if top_factors:
                top_factor = top_factors[0]

        # Get committee info
        committees = member.get('committees', [])
        if isinstance(committees, str):
            committees = [committees]
        committee = member.get('committee', '')
        if committee and committee not in committees:
            committees.append(committee)
        committee_str = ", ".join(committees) if committees else "Unknown"

        # Build template variables
        template_vars = {
            'member_name': member.get('name', member.get('full_name', 'Unknown')),
            'party': member.get('party', 'Unknown'),
            'state': member.get('state', 'Unknown'),
            'chamber': member.get('chamber', 'Unknown'),
            'symbol': trade.get('symbol', 'Unknown'),
            'company_name': trade.get('asset_name', trade.get('company_name', f"{trade.get('symbol', 'Unknown')} Corporation")),
            'transaction_type': trade.get('transaction_type', 'Unknown'),
            'amount_display': self._format_amount(avg_amount),
            'transaction_date': trans_date,
            'filing_date': filing_date,
            'filing_delay': filing_delay,
            'committee': committee_str,
            'conviction_score': round(conviction_score),
            'risk_level': risk_level,
            'conviction_factors': conviction_factors,
            'top_factor': top_factor,
            'additional_context': f"Member serves on {committee_str} with potential oversight of this sector.",
            'format_type': format.value
        }

        # Get template and format prompt
        template = PromptTemplate.get_template(StoryType.TRADE_STORY, format)
        prompt = template.format(**template_vars)

        # Generate story
        content = self.provider.generate(
            prompt=prompt,
            system_prompt=PromptTemplate.SYSTEM_PROMPT,
            max_tokens=self._get_max_tokens(format)
        )

        # Create headline based on format
        headline = self._generate_headline(trade, member, conviction_score, format)

        return Story(
            story_id=self._generate_story_id(),
            story_type=StoryType.TRADE_STORY,
            format=format,
            content=content,
            headline=headline,
            summary=f"{member.get('name', 'Member')}'s {trade.get('symbol', 'stock')} trade scores {conviction_score}/100",
            metadata={
                'trade_id': trade.get('id'),
                'member_id': member.get('id'),
                'symbol': trade.get('symbol'),
                'conviction_score': conviction_score
            }
        )

    def generate_member_profile(
        self,
        member: Dict,
        trades: List[Dict],
        conviction_results: Optional[List[Dict]] = None,
        format: StoryFormat = StoryFormat.NEWS_BRIEF
    ) -> Story:
        """
        Generate a profile story about a member's trading activity.

        Args:
            member: Member data dictionary
            trades: List of member's trades
            conviction_results: Optional list of conviction results
            format: Output format

        Returns:
            Generated Story object
        """
        # Calculate trading statistics
        total_volume = sum(
            ((t.get('amount_from', 0) or 0) + (t.get('amount_to', 0) or 0)) / 2
            for t in trades
        )
        avg_trade = total_volume / len(trades) if trades else 0

        # Find most traded sector
        sectors = {}
        for t in trades:
            symbol = t.get('symbol', '')
            sector = self._get_sector(symbol)
            sectors[sector] = sectors.get(sector, 0) + 1
        top_sector = max(sectors.items(), key=lambda x: x[1])[0] if sectors else "Unknown"

        # Get conviction scores
        avg_conviction = 0
        highest_conviction = None
        if conviction_results:
            scores = [r.get('score', 0) for r in conviction_results]
            avg_conviction = sum(scores) / len(scores) if scores else 0
            if scores:
                max_idx = scores.index(max(scores))
                highest_conviction = conviction_results[max_idx]

        # Format notable trades
        notable_trades = self._format_notable_trades(trades[:5], conviction_results)

        # Get committee overlap analysis
        committee_overlap = self._analyze_committee_overlap(member, trades)

        # Build template variables
        committees = member.get('committees', [])
        if isinstance(committees, str):
            committees = [committees]

        template_vars = {
            'member_name': member.get('name', 'Unknown'),
            'party': member.get('party', 'Unknown'),
            'state': member.get('state', 'Unknown'),
            'chamber': member.get('chamber', 'Unknown'),
            'district': member.get('district', 'N/A'),
            'committees': ", ".join(committees) if committees else member.get('committee', 'Unknown'),
            'leadership_role': member.get('leadership_role', 'None'),
            'tenure_years': member.get('tenure_years', 'Unknown'),
            'net_worth_display': self._format_amount(member.get('net_worth', 0)),
            'total_trades': len(trades),
            'total_volume_display': self._format_amount(total_volume),
            'avg_trade_display': self._format_amount(avg_trade),
            'top_sector': top_sector,
            'highest_conviction_trade': str(highest_conviction) if highest_conviction else "N/A",
            'avg_conviction_score': round(avg_conviction),
            'notable_trades': notable_trades,
            'committee_overlap': committee_overlap,
            'format_type': format.value
        }

        template = PromptTemplate.get_template(StoryType.MEMBER_PROFILE, format)
        prompt = template.format(**template_vars)

        content = self.provider.generate(
            prompt=prompt,
            system_prompt=PromptTemplate.SYSTEM_PROMPT,
            max_tokens=self._get_max_tokens(format)
        )

        return Story(
            story_id=self._generate_story_id(),
            story_type=StoryType.MEMBER_PROFILE,
            format=format,
            content=content,
            headline=f"Trading Profile: {member.get('name', 'Member')}",
            summary=f"{len(trades)} trades, ${self._format_amount(total_volume)} volume, {round(avg_conviction)}/100 avg score",
            metadata={
                'member_id': member.get('id'),
                'total_trades': len(trades),
                'avg_conviction': avg_conviction
            }
        )

    def generate_pattern_alert(
        self,
        pattern_type: str,
        trades: List[Dict],
        members: List[Dict],
        format: StoryFormat = StoryFormat.NEWS_BRIEF
    ) -> Story:
        """
        Generate an alert about a detected pattern.

        Args:
            pattern_type: Type of pattern detected
            trades: List of trades involved
            members: List of members involved
            format: Output format

        Returns:
            Generated Story object
        """
        trades_summary = "\n".join([
            f"- {t.get('member_name', 'Unknown')}: {t.get('symbol')} ({t.get('transaction_type')}) ${self._format_amount((t.get('amount_from', 0) + t.get('amount_to', 0)) / 2)}"
            for t in trades[:10]
        ])

        members_summary = "\n".join([
            f"- {m.get('name', 'Unknown')} ({m.get('party')}-{m.get('state')})"
            for m in members[:10]
        ])

        template_vars = {
            'pattern_type': pattern_type,
            'pattern_description': f"Detected {pattern_type} pattern across {len(trades)} trades",
            'trades_summary': trades_summary,
            'members_summary': members_summary,
            'significance': "High" if len(trades) > 5 else "Moderate",
            'time_period': "Last 30 days",
            'key_findings': f"- {len(trades)} trades identified\n- {len(members)} members involved\n- Pattern suggests coordinated timing",
            'format_type': format.value
        }

        template = PromptTemplate.get_template(StoryType.PATTERN_ALERT, format)
        prompt = template.format(**template_vars)

        content = self.provider.generate(
            prompt=prompt,
            system_prompt=PromptTemplate.SYSTEM_PROMPT,
            max_tokens=self._get_max_tokens(format)
        )

        return Story(
            story_id=self._generate_story_id(),
            story_type=StoryType.PATTERN_ALERT,
            format=format,
            content=content,
            headline=f"Pattern Alert: {pattern_type}",
            summary=f"{len(trades)} trades across {len(members)} members",
            metadata={
                'pattern_type': pattern_type,
                'trade_count': len(trades),
                'member_count': len(members)
            }
        )

    def generate_weekly_roundup(
        self,
        trades: List[Dict],
        start_date: str,
        end_date: str,
        conviction_results: Optional[List[Dict]] = None,
        format: StoryFormat = StoryFormat.NEWS_BRIEF
    ) -> Story:
        """
        Generate a weekly roundup of trading activity.

        Args:
            trades: List of trades in the period
            start_date: Start date of the period
            end_date: End date of the period
            conviction_results: Optional conviction results
            format: Output format

        Returns:
            Generated Story object
        """
        # Calculate statistics
        total_volume = sum(
            ((t.get('amount_from', 0) or 0) + (t.get('amount_to', 0) or 0)) / 2
            for t in trades
        )

        avg_conviction = 0
        high_risk_count = 0
        top_suspicious = []

        if conviction_results:
            scores = [r.get('score', 0) for r in conviction_results]
            avg_conviction = sum(scores) / len(scores) if scores else 0
            high_risk_count = sum(1 for s in scores if s > 70)

            # Get top suspicious trades
            sorted_results = sorted(conviction_results, key=lambda x: x.get('score', 0), reverse=True)
            top_suspicious = sorted_results[:5]

        top_suspicious_str = "\n".join([
            f"- Score {r.get('score', 0)}: {r.get('member_name', 'Unknown')} - {r.get('top_factors', ['Unknown'])[0] if r.get('top_factors') else 'Unknown'}"
            for r in top_suspicious
        ])

        # Get sector breakdown
        sectors = {}
        for t in trades:
            symbol = t.get('symbol', '')
            sector = self._get_sector(symbol)
            sectors[sector] = sectors.get(sector, 0) + 1
        sector_breakdown = "\n".join([f"- {k}: {v} trades" for k, v in sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:5]])

        template_vars = {
            'start_date': start_date,
            'end_date': end_date,
            'total_disclosures': len(trades),
            'total_volume_display': self._format_amount(total_volume),
            'avg_conviction_score': round(avg_conviction),
            'high_risk_count': high_risk_count,
            'top_suspicious_trades': top_suspicious_str or "None identified",
            'notable_members': "See individual trade analysis",
            'sector_breakdown': sector_breakdown or "No sector data",
            'timing_patterns': "Detailed timing analysis available in individual reports",
            'format_type': format.value
        }

        template = PromptTemplate.get_template(StoryType.WEEKLY_ROUNDUP, format)
        prompt = template.format(**template_vars)

        content = self.provider.generate(
            prompt=prompt,
            system_prompt=PromptTemplate.SYSTEM_PROMPT,
            max_tokens=self._get_max_tokens(format)
        )

        return Story(
            story_id=self._generate_story_id(),
            story_type=StoryType.WEEKLY_ROUNDUP,
            format=format,
            content=content,
            headline=f"Weekly Congressional Trading Roundup: {start_date} to {end_date}",
            summary=f"{len(trades)} disclosures, ${self._format_amount(total_volume)} volume, {high_risk_count} high-risk trades",
            metadata={
                'start_date': start_date,
                'end_date': end_date,
                'total_trades': len(trades),
                'high_risk_count': high_risk_count
            }
        )

    def _get_max_tokens(self, format: StoryFormat) -> int:
        """Get appropriate max tokens for format."""
        token_limits = {
            StoryFormat.TWEET_THREAD: 500,
            StoryFormat.NEWS_BRIEF: 800,
            StoryFormat.DEEP_DIVE: 2500,
            StoryFormat.DATA_CARD: 300
        }
        return token_limits.get(format, 1000)

    def _generate_headline(
        self,
        trade: Dict,
        member: Dict,
        conviction_score: float,
        format: StoryFormat
    ) -> str:
        """Generate appropriate headline for the story."""
        name = member.get('name', 'Member')
        symbol = trade.get('symbol', 'Stock')
        trans_type = trade.get('transaction_type', 'Trade')

        if conviction_score >= 80:
            return f"High Alert: {name}'s {symbol} {trans_type} Scores {conviction_score}/100"
        elif conviction_score >= 60:
            return f"Pattern Alert: {name}'s {symbol} Trade Under Scrutiny"
        else:
            return f"Congressional Trade: {name} Discloses {symbol} {trans_type}"

    def _get_sector(self, symbol: str) -> str:
        """Get sector for a stock symbol."""
        sectors = {
            'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
            'AMZN': 'Technology', 'META': 'Technology', 'NVDA': 'Technology',
            'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials',
            'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
            'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'PFE': 'Healthcare',
            'BA': 'Industrial', 'LMT': 'Defense', 'RTX': 'Defense',
        }
        return sectors.get(symbol.upper(), 'Other')

    def _format_notable_trades(
        self,
        trades: List[Dict],
        conviction_results: Optional[List[Dict]]
    ) -> str:
        """Format notable trades for display."""
        lines = []
        for i, trade in enumerate(trades):
            symbol = trade.get('symbol', 'Unknown')
            amount = (trade.get('amount_from', 0) + trade.get('amount_to', 0)) / 2
            trans_type = trade.get('transaction_type', 'Trade')
            score = "N/A"
            if conviction_results and i < len(conviction_results):
                score = f"{conviction_results[i].get('score', 0)}/100"
            lines.append(f"- {symbol}: {trans_type} ${self._format_amount(amount)} (Score: {score})")
        return "\n".join(lines) if lines else "No notable trades"

    def _analyze_committee_overlap(self, member: Dict, trades: List[Dict]) -> str:
        """Analyze committee-trading sector overlap."""
        committees = member.get('committees', [])
        if isinstance(committees, str):
            committees = [committees]
        committee = member.get('committee', '')
        if committee:
            committees.append(committee)

        if not committees:
            return "No committee data available"

        # Count trades by sector
        sectors = {}
        for t in trades:
            symbol = t.get('symbol', '')
            sector = self._get_sector(symbol)
            sectors[sector] = sectors.get(sector, 0) + 1

        # Simple overlap analysis
        overlap_found = []
        for comm in committees:
            comm_lower = comm.lower()
            if 'financial' in comm_lower or 'banking' in comm_lower:
                if sectors.get('Financials', 0) > 0:
                    overlap_found.append(f"Financial Services committee + {sectors['Financials']} financial sector trades")
            if 'energy' in comm_lower:
                if sectors.get('Energy', 0) > 0:
                    overlap_found.append(f"Energy committee + {sectors['Energy']} energy sector trades")
            if 'armed' in comm_lower or 'defense' in comm_lower:
                if sectors.get('Defense', 0) > 0:
                    overlap_found.append(f"Defense committee + {sectors['Defense']} defense sector trades")

        if overlap_found:
            return "\n".join([f"- {o}" for o in overlap_found])
        return "No significant committee-sector overlap detected"


def main():
    """Example usage of the LLM service."""
    print("=" * 60)
    print("Congressional Trading LLM Story Generator - Demo")
    print("=" * 60)

    # Initialize service with mock provider for demo
    service = LLMService(use_mock=True)

    # Sample trade data
    sample_trade = {
        'id': 'T001',
        'symbol': 'NVDA',
        'asset_name': 'NVIDIA Corporation',
        'transaction_type': 'Purchase',
        'transaction_date': '2024-01-15',
        'filing_date': '2024-02-28',
        'amount_from': 500000,
        'amount_to': 1000000
    }

    # Sample member data
    sample_member = {
        'id': 'M001',
        'name': 'Sample Representative',
        'party': 'D',
        'state': 'CA',
        'chamber': 'House',
        'committee': 'Science & Technology',
        'committees': ['Science & Technology', 'Oversight'],
        'net_worth': 5000000
    }

    # Sample conviction result
    sample_conviction = {
        'score': 72,
        'risk_level': 'elevated',
        'factors': {
            'committee_access': {'score': 80, 'weighted_score': 20, 'explanation': 'Direct oversight of tech sector'},
            'timing_proximity': {'score': 65, 'weighted_score': 16.25, 'explanation': 'Trade before earnings'},
            'filing_delay': {'score': 60, 'weighted_score': 9, 'explanation': 'Filed 44 days after trade'}
        },
        'top_factors': ['committee_access', 'timing_proximity', 'filing_delay']
    }

    # Generate stories in different formats
    print("\n--- Tweet Thread ---")
    story = service.generate_trade_story(
        sample_trade, sample_member, sample_conviction,
        format=StoryFormat.TWEET_THREAD
    )
    print(story.content)

    print("\n--- News Brief ---")
    story = service.generate_trade_story(
        sample_trade, sample_member, sample_conviction,
        format=StoryFormat.NEWS_BRIEF
    )
    print(story.content)

    print("\n--- Data Card ---")
    story = service.generate_trade_story(
        sample_trade, sample_member, sample_conviction,
        format=StoryFormat.DATA_CARD
    )
    print(story.content)

    print("\n" + "=" * 60)
    print("Demo complete! In production, set ANTHROPIC_API_KEY for real generation.")
    print("=" * 60)


if __name__ == "__main__":
    main()
