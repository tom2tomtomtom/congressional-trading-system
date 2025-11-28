#!/usr/bin/env python3
"""
Congressional Trading Intelligence System
Track F - Task F2: Natural Language Query Parser

This module parses natural language questions about congressional trading
and converts them into database queries/filters.

Example queries:
- "Show me Pelosi's tech trades"
- "Who bought NVDA before the earnings?"
- "Which Republicans trade the most?"
- "Most suspicious trade this month"
"""

import os
import re
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of query intents."""
    MEMBER_TRADES = "member_trades"
    STOCK_TRADES = "stock_trades"
    SECTOR_TRADES = "sector_trades"
    TIME_FILTERED = "time_filtered"
    TOP_TRADERS = "top_traders"
    SUSPICIOUS_TRADES = "suspicious_trades"
    COMMITTEE_ANALYSIS = "committee_analysis"
    PARTY_COMPARISON = "party_comparison"
    GENERAL_STATS = "general_stats"
    UNKNOWN = "unknown"


class FilterOperator(Enum):
    """SQL filter operators."""
    EQUALS = "="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    IN = "IN"
    LIKE = "LIKE"
    BETWEEN = "BETWEEN"
    IS_NULL = "IS NULL"
    IS_NOT_NULL = "IS NOT NULL"


@dataclass
class QueryFilter:
    """Individual filter condition."""
    field: str
    operator: FilterOperator
    value: Any
    table: Optional[str] = None

    def to_sql(self) -> Tuple[str, List[Any]]:
        """Convert filter to SQL WHERE clause component."""
        table_prefix = f"{self.table}." if self.table else ""
        field_ref = f"{table_prefix}{self.field}"

        if self.operator == FilterOperator.IN:
            placeholders = ", ".join(["?" for _ in self.value])
            return f"{field_ref} IN ({placeholders})", list(self.value)
        elif self.operator == FilterOperator.BETWEEN:
            return f"{field_ref} BETWEEN ? AND ?", [self.value[0], self.value[1]]
        elif self.operator == FilterOperator.LIKE:
            return f"{field_ref} LIKE ?", [f"%{self.value}%"]
        elif self.operator in [FilterOperator.IS_NULL, FilterOperator.IS_NOT_NULL]:
            return f"{field_ref} {self.operator.value}", []
        else:
            return f"{field_ref} {self.operator.value} ?", [self.value]

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "field": self.field,
            "operator": self.operator.value,
            "value": self.value,
            "table": self.table
        }


@dataclass
class ParsedQuery:
    """Parsed natural language query result."""
    original_query: str
    intent: QueryIntent
    filters: List[QueryFilter] = field(default_factory=list)
    order_by: Optional[str] = None
    order_direction: str = "DESC"
    limit: int = 50
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    suggestions: List[str] = field(default_factory=list)
    explanation: str = ""

    def to_sql_where(self) -> Tuple[str, List[Any]]:
        """Convert filters to SQL WHERE clause."""
        if not self.filters:
            return "", []

        conditions = []
        params = []
        for f in self.filters:
            sql, p = f.to_sql()
            conditions.append(sql)
            params.extend(p)

        return " AND ".join(conditions), params

    def to_dict(self) -> Dict:
        """Convert parsed query to dictionary."""
        return {
            "original_query": self.original_query,
            "intent": self.intent.value,
            "filters": [f.to_dict() for f in self.filters],
            "order_by": self.order_by,
            "order_direction": self.order_direction,
            "limit": self.limit,
            "entities": self.entities,
            "confidence": self.confidence,
            "suggestions": self.suggestions,
            "explanation": self.explanation
        }


# Member name mappings (common nicknames and variations)
MEMBER_ALIASES = {
    "pelosi": "Nancy Pelosi",
    "nancy pelosi": "Nancy Pelosi",
    "mccarthy": "Kevin McCarthy",
    "kevin mccarthy": "Kevin McCarthy",
    "aoc": "Alexandria Ocasio-Cortez",
    "ocasio-cortez": "Alexandria Ocasio-Cortez",
    "manchin": "Joe Manchin",
    "joe manchin": "Joe Manchin",
    "warren": "Elizabeth Warren",
    "elizabeth warren": "Elizabeth Warren",
    "cruz": "Ted Cruz",
    "ted cruz": "Ted Cruz",
    "schumer": "Chuck Schumer",
    "chuck schumer": "Chuck Schumer",
    "mcconnell": "Mitch McConnell",
    "mitch mcconnell": "Mitch McConnell",
    "sinema": "Kyrsten Sinema",
    "kyrsten sinema": "Kyrsten Sinema",
    "greene": "Marjorie Taylor Greene",
    "mtg": "Marjorie Taylor Greene",
    "gaetz": "Matt Gaetz",
    "matt gaetz": "Matt Gaetz",
    "porter": "Katie Porter",
    "katie porter": "Katie Porter",
    "crenshaw": "Dan Crenshaw",
    "dan crenshaw": "Dan Crenshaw",
    "hawley": "Josh Hawley",
    "josh hawley": "Josh Hawley",
    "burr": "Richard Burr",
    "richard burr": "Richard Burr",
    "toomey": "Pat Toomey",
    "pat toomey": "Pat Toomey",
    "warner": "Mark Warner",
    "mark warner": "Mark Warner",
    "gottheimer": "Josh Gottheimer",
    "jeffries": "Hakeem Jeffries",
    "scalise": "Steve Scalise",
}

# Stock ticker to company name mappings
STOCK_COMPANIES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet/Google",
    "GOOG": "Alphabet/Google",
    "AMZN": "Amazon",
    "META": "Meta/Facebook",
    "FB": "Meta/Facebook",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "WFC": "Wells Fargo",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "UNH": "UnitedHealth",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer",
    "MRNA": "Moderna",
    "BA": "Boeing",
    "LMT": "Lockheed Martin",
    "RTX": "Raytheon",
    "NOC": "Northrop Grumman",
    "DIS": "Disney",
    "NFLX": "Netflix",
}

# Sector keywords and associated stocks
SECTOR_MAPPING = {
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ORCL", "CRM", "ADBE", "INTC", "AMD"],
    "technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ORCL", "CRM", "ADBE", "INTC", "AMD"],
    "semiconductor": ["NVDA", "AMD", "INTC", "TSM", "AVGO", "QCOM", "TXN"],
    "ai": ["NVDA", "MSFT", "GOOGL", "AMZN", "META", "AMD", "ORCL"],
    "bank": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC"],
    "banks": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC"],
    "banking": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC"],
    "financial": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "V", "MA", "AXP"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "V", "MA", "AXP"],
    "finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "V", "MA", "AXP"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "KMI", "OKE"],
    "oil": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX"],
    "pharma": ["PFE", "JNJ", "MRK", "ABBV", "BMY", "LLY", "AMGN", "GILD"],
    "pharmaceutical": ["PFE", "JNJ", "MRK", "ABBV", "BMY", "LLY", "AMGN", "GILD"],
    "healthcare": ["UNH", "JNJ", "PFE", "ABT", "TMO", "DHR", "BMY", "AMGN", "GILD"],
    "health": ["UNH", "JNJ", "PFE", "ABT", "TMO", "DHR", "BMY", "AMGN", "GILD"],
    "defense": ["LMT", "RTX", "NOC", "GD", "BA", "HII", "LHX"],
    "military": ["LMT", "RTX", "NOC", "GD", "BA", "HII", "LHX"],
    "aerospace": ["BA", "LMT", "RTX", "NOC", "GD", "HII"],
    "retail": ["WMT", "AMZN", "HD", "TGT", "COST", "LOW"],
    "consumer": ["WMT", "HD", "PG", "KO", "PEP", "COST", "LOW", "TGT", "SBUX", "MCD"],
    "crypto": ["COIN", "MSTR"],
    "cryptocurrency": ["COIN", "MSTR"],
}

# Time period patterns
TIME_PATTERNS = {
    r"today": (0, "day"),
    r"yesterday": (1, "day"),
    r"this week": (7, "day"),
    r"last week": (14, "day"),
    r"this month": (30, "day"),
    r"last month": (60, "day"),
    r"this quarter": (90, "day"),
    r"last quarter": (180, "day"),
    r"this year": (365, "day"),
    r"last year": (730, "day"),
    r"(\d+)\s*days?\s*(?:ago)?": (None, "day"),
    r"(\d+)\s*weeks?\s*(?:ago)?": (None, "week"),
    r"(\d+)\s*months?\s*(?:ago)?": (None, "month"),
    r"past\s*(\d+)\s*days?": (None, "day"),
    r"past\s*(\d+)\s*weeks?": (None, "week"),
    r"past\s*(\d+)\s*months?": (None, "month"),
    r"before\s*(\w+\s*\d+,?\s*\d{4})": (None, "before_date"),
    r"after\s*(\w+\s*\d+,?\s*\d{4})": (None, "after_date"),
    r"in\s*(\d{4})": (None, "year"),
    r"2020": (None, "year_2020"),
    r"2021": (None, "year_2021"),
    r"2022": (None, "year_2022"),
    r"2023": (None, "year_2023"),
    r"2024": (None, "year_2024"),
    r"covid": (None, "covid_period"),
    r"pandemic": (None, "covid_period"),
}

# Committee keywords
COMMITTEE_KEYWORDS = {
    "financial services": "Financial Services",
    "finance": "Financial Services",
    "banking": "Banking",
    "armed services": "Armed Services",
    "defense": "Armed Services",
    "military": "Armed Services",
    "energy": "Energy and Commerce",
    "commerce": "Energy and Commerce",
    "intelligence": "Intelligence",
    "intel": "Intelligence",
    "health": "Health, Education, Labor and Pensions",
    "healthcare": "Health, Education, Labor and Pensions",
    "judiciary": "Judiciary",
    "oversight": "Oversight",
    "agriculture": "Agriculture",
    "transportation": "Transportation",
    "ways and means": "Ways and Means",
    "tax": "Ways and Means",
}


class NLQueryParser:
    """
    Parses natural language queries into structured database filters.

    This parser uses pattern matching and entity extraction to convert
    questions like "Show me Pelosi's tech trades" into database queries.
    """

    def __init__(self, use_llm: bool = False, llm_api_key: Optional[str] = None):
        """
        Initialize the parser.

        Args:
            use_llm: Whether to use LLM for complex query understanding
            llm_api_key: API key for LLM provider
        """
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key or os.getenv("ANTHROPIC_API_KEY")
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for entity extraction."""
        # Transaction type patterns
        self.buy_pattern = re.compile(
            r'\b(bought?|purchase[ds]?|buys?|buying|acquired?)\b',
            re.IGNORECASE
        )
        self.sell_pattern = re.compile(
            r'\b(sold?|sells?|selling|sale[s]?|divested?)\b',
            re.IGNORECASE
        )

        # Quantity patterns
        self.amount_pattern = re.compile(
            r'\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:k|K|m|M|million|thousand)?',
            re.IGNORECASE
        )

        # Stock ticker pattern (uppercase letters, 1-5 chars)
        self.ticker_pattern = re.compile(r'\b([A-Z]{1,5})\b')

        # Party patterns
        self.party_patterns = {
            'D': re.compile(r'\b(democrat[s]?|dem[s]?|democratic)\b', re.IGNORECASE),
            'R': re.compile(r'\b(republican[s]?|rep[s]?|gop)\b', re.IGNORECASE),
            'I': re.compile(r'\b(independent[s]?|indie[s]?)\b', re.IGNORECASE),
        }

        # Chamber patterns
        self.chamber_patterns = {
            'House': re.compile(r'\b(house|representative[s]?|rep[s]?\.?)\b', re.IGNORECASE),
            'Senate': re.compile(r'\b(senate|senator[s]?|sen\.?)\b', re.IGNORECASE),
        }

        # Superlative patterns for ranking queries
        self.superlative_patterns = {
            'most': re.compile(r'\b(most|top|highest|largest|biggest)\b', re.IGNORECASE),
            'least': re.compile(r'\b(least|bottom|lowest|smallest|fewest)\b', re.IGNORECASE),
        }

        # Question type patterns
        self.question_patterns = {
            'who': re.compile(r'^who\b', re.IGNORECASE),
            'what': re.compile(r'^what\b', re.IGNORECASE),
            'when': re.compile(r'^when\b', re.IGNORECASE),
            'where': re.compile(r'^where\b', re.IGNORECASE),
            'which': re.compile(r'^which\b', re.IGNORECASE),
            'how_much': re.compile(r'^how\s+much\b', re.IGNORECASE),
            'how_many': re.compile(r'^how\s+many\b', re.IGNORECASE),
            'show': re.compile(r'^show\b', re.IGNORECASE),
            'find': re.compile(r'^find\b', re.IGNORECASE),
            'list': re.compile(r'^list\b', re.IGNORECASE),
            'get': re.compile(r'^get\b', re.IGNORECASE),
        }

    def parse(self, query: str) -> ParsedQuery:
        """
        Parse a natural language query into structured filters.

        Args:
            query: Natural language query string

        Returns:
            ParsedQuery object with extracted information
        """
        query = query.strip()
        original_query = query

        # Initialize result
        result = ParsedQuery(
            original_query=original_query,
            intent=QueryIntent.UNKNOWN,
            confidence=0.0
        )

        # Extract entities
        entities = self._extract_entities(query)
        result.entities = entities

        # Determine intent
        result.intent = self._determine_intent(query, entities)

        # Build filters based on entities
        result.filters = self._build_filters(entities)

        # Set ordering
        result.order_by, result.order_direction = self._determine_ordering(query, entities)

        # Set limit
        result.limit = self._extract_limit(query)

        # Calculate confidence
        result.confidence = self._calculate_confidence(result)

        # Generate explanation
        result.explanation = self._generate_explanation(result)

        # Add suggestions if confidence is low
        if result.confidence < 0.7:
            result.suggestions = self._generate_suggestions(query, entities)

        return result

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract all entities from the query."""
        entities = {
            'members': [],
            'stocks': [],
            'sectors': [],
            'parties': [],
            'chambers': [],
            'committees': [],
            'transaction_types': [],
            'time_filters': [],
            'amounts': [],
            'rankings': [],
            'states': [],
        }

        query_lower = query.lower()

        # Extract member names
        for alias, full_name in MEMBER_ALIASES.items():
            if alias in query_lower:
                if full_name not in entities['members']:
                    entities['members'].append(full_name)

        # Extract stock tickers and company names
        for ticker, company in STOCK_COMPANIES.items():
            if ticker.lower() in query_lower or company.lower() in query_lower:
                if ticker not in entities['stocks']:
                    entities['stocks'].append(ticker)

        # Also check for uppercase tickers in original query
        potential_tickers = self.ticker_pattern.findall(query)
        for ticker in potential_tickers:
            if ticker in STOCK_COMPANIES and ticker not in entities['stocks']:
                entities['stocks'].append(ticker)

        # Extract sectors
        for sector_key, stocks in SECTOR_MAPPING.items():
            if sector_key in query_lower:
                entities['sectors'].append(sector_key)
                # Add associated stocks as hints (but don't filter by them directly)
                entities['sector_stocks'] = entities.get('sector_stocks', []) + stocks

        # Extract parties
        for party, pattern in self.party_patterns.items():
            if pattern.search(query):
                entities['parties'].append(party)

        # Extract chambers
        for chamber, pattern in self.chamber_patterns.items():
            if pattern.search(query):
                entities['chambers'].append(chamber)

        # Extract committees
        for keyword, committee in COMMITTEE_KEYWORDS.items():
            if keyword in query_lower:
                if committee not in entities['committees']:
                    entities['committees'].append(committee)

        # Extract transaction types
        if self.buy_pattern.search(query):
            entities['transaction_types'].append('Purchase')
        if self.sell_pattern.search(query):
            entities['transaction_types'].append('Sale')

        # Extract time filters
        entities['time_filters'] = self._extract_time_filters(query)

        # Extract amounts
        entities['amounts'] = self._extract_amounts(query)

        # Extract ranking modifiers
        for rank_type, pattern in self.superlative_patterns.items():
            if pattern.search(query):
                entities['rankings'].append(rank_type)

        # Extract state abbreviations
        states = re.findall(r'\b([A-Z]{2})\b', query)
        valid_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
        entities['states'] = [s for s in states if s in valid_states]

        # Check for suspicious/conviction related queries
        if any(word in query_lower for word in ['suspicious', 'conviction', 'score', 'risk', 'concerning', 'flagged']):
            entities['suspicious_filter'] = True

        return entities

    def _extract_time_filters(self, query: str) -> List[Dict]:
        """Extract time-related filters from the query."""
        filters = []
        query_lower = query.lower()
        today = date.today()

        for pattern_str, (default_val, period_type) in TIME_PATTERNS.items():
            pattern = re.compile(pattern_str, re.IGNORECASE)
            match = pattern.search(query_lower)
            if match:
                if period_type == "year_2020":
                    filters.append({
                        'start': date(2020, 1, 1),
                        'end': date(2020, 12, 31)
                    })
                elif period_type == "year_2021":
                    filters.append({
                        'start': date(2021, 1, 1),
                        'end': date(2021, 12, 31)
                    })
                elif period_type == "year_2022":
                    filters.append({
                        'start': date(2022, 1, 1),
                        'end': date(2022, 12, 31)
                    })
                elif period_type == "year_2023":
                    filters.append({
                        'start': date(2023, 1, 1),
                        'end': date(2023, 12, 31)
                    })
                elif period_type == "year_2024":
                    filters.append({
                        'start': date(2024, 1, 1),
                        'end': today
                    })
                elif period_type == "covid_period":
                    # COVID trading scrutiny period: Jan-Mar 2020
                    filters.append({
                        'start': date(2020, 1, 15),
                        'end': date(2020, 3, 31)
                    })
                elif default_val is not None:
                    # Fixed period (e.g., "this week" = 7 days)
                    start = today - timedelta(days=default_val)
                    filters.append({
                        'start': start,
                        'end': today
                    })
                elif match.groups():
                    # Dynamic period (e.g., "5 days ago")
                    num = int(match.group(1))
                    if period_type == "day":
                        delta = timedelta(days=num)
                    elif period_type == "week":
                        delta = timedelta(weeks=num)
                    elif period_type == "month":
                        delta = timedelta(days=num * 30)
                    else:
                        delta = timedelta(days=num)
                    start = today - delta
                    filters.append({
                        'start': start,
                        'end': today
                    })

        return filters

    def _extract_amounts(self, query: str) -> List[Dict]:
        """Extract amount filters from the query."""
        amounts = []
        query_lower = query.lower()

        # Check for relative amount terms
        if 'large' in query_lower or 'big' in query_lower:
            amounts.append({'type': 'min', 'value': 100000})
        if 'small' in query_lower:
            amounts.append({'type': 'max', 'value': 15000})
        if 'million' in query_lower:
            amounts.append({'type': 'min', 'value': 1000000})

        # Extract specific amounts
        matches = self.amount_pattern.findall(query)
        for match in matches:
            value = float(match.replace(',', ''))
            # Check for k/m suffixes in original context
            context = query_lower
            if 'k' in context or 'thousand' in context:
                value *= 1000
            elif 'm' in context or 'million' in context:
                value *= 1000000

            # Determine if this is a min/max based on context
            if 'over' in context or 'above' in context or 'more than' in context:
                amounts.append({'type': 'min', 'value': value})
            elif 'under' in context or 'below' in context or 'less than' in context:
                amounts.append({'type': 'max', 'value': value})
            else:
                amounts.append({'type': 'approx', 'value': value})

        return amounts

    def _determine_intent(self, query: str, entities: Dict) -> QueryIntent:
        """Determine the primary intent of the query."""
        query_lower = query.lower()

        # Check for suspicious/conviction queries
        if entities.get('suspicious_filter'):
            return QueryIntent.SUSPICIOUS_TRADES

        # Check for specific member queries
        if entities['members']:
            return QueryIntent.MEMBER_TRADES

        # Check for stock-specific queries
        if entities['stocks'] and not entities['sectors']:
            return QueryIntent.STOCK_TRADES

        # Check for sector queries
        if entities['sectors']:
            return QueryIntent.SECTOR_TRADES

        # Check for party comparison
        if len(entities['parties']) > 1 or 'compare' in query_lower or 'vs' in query_lower:
            return QueryIntent.PARTY_COMPARISON

        # Check for top traders query
        if entities['rankings'] and any(
            word in query_lower for word in ['trade', 'trader', 'trading', 'active']
        ):
            return QueryIntent.TOP_TRADERS

        # Check for committee analysis
        if entities['committees']:
            return QueryIntent.COMMITTEE_ANALYSIS

        # Check for time-filtered queries
        if entities['time_filters']:
            return QueryIntent.TIME_FILTERED

        # Check for general stats queries
        if any(word in query_lower for word in ['how many', 'total', 'count', 'average', 'stats', 'statistics']):
            return QueryIntent.GENERAL_STATS

        return QueryIntent.UNKNOWN

    def _build_filters(self, entities: Dict) -> List[QueryFilter]:
        """Build database filters from extracted entities."""
        filters = []

        # Member filters
        if entities['members']:
            if len(entities['members']) == 1:
                filters.append(QueryFilter(
                    field='member_name',
                    operator=FilterOperator.LIKE,
                    value=entities['members'][0],
                    table='trades'
                ))
            else:
                filters.append(QueryFilter(
                    field='member_name',
                    operator=FilterOperator.IN,
                    value=entities['members'],
                    table='trades'
                ))

        # Stock filters
        if entities['stocks']:
            if len(entities['stocks']) == 1:
                filters.append(QueryFilter(
                    field='symbol',
                    operator=FilterOperator.EQUALS,
                    value=entities['stocks'][0],
                    table='trades'
                ))
            else:
                filters.append(QueryFilter(
                    field='symbol',
                    operator=FilterOperator.IN,
                    value=entities['stocks'],
                    table='trades'
                ))

        # Sector filters (use IN with sector stocks)
        if entities.get('sector_stocks'):
            # Remove duplicates and limit
            sector_stocks = list(set(entities['sector_stocks']))
            filters.append(QueryFilter(
                field='symbol',
                operator=FilterOperator.IN,
                value=sector_stocks,
                table='trades'
            ))

        # Party filters
        if entities['parties']:
            if len(entities['parties']) == 1:
                filters.append(QueryFilter(
                    field='party',
                    operator=FilterOperator.EQUALS,
                    value=entities['parties'][0],
                    table='trades'
                ))
            else:
                filters.append(QueryFilter(
                    field='party',
                    operator=FilterOperator.IN,
                    value=entities['parties'],
                    table='trades'
                ))

        # Chamber filters
        if entities['chambers']:
            if len(entities['chambers']) == 1:
                filters.append(QueryFilter(
                    field='chamber',
                    operator=FilterOperator.EQUALS,
                    value=entities['chambers'][0],
                    table='trades'
                ))
            else:
                filters.append(QueryFilter(
                    field='chamber',
                    operator=FilterOperator.IN,
                    value=entities['chambers'],
                    table='trades'
                ))

        # Transaction type filters
        if entities['transaction_types']:
            if len(entities['transaction_types']) == 1:
                filters.append(QueryFilter(
                    field='transaction_type',
                    operator=FilterOperator.EQUALS,
                    value=entities['transaction_types'][0],
                    table='trades'
                ))
            else:
                filters.append(QueryFilter(
                    field='transaction_type',
                    operator=FilterOperator.IN,
                    value=entities['transaction_types'],
                    table='trades'
                ))

        # Time filters
        if entities['time_filters']:
            time_filter = entities['time_filters'][0]  # Use first time filter
            if 'start' in time_filter:
                filters.append(QueryFilter(
                    field='transaction_date',
                    operator=FilterOperator.BETWEEN,
                    value=[
                        time_filter['start'].isoformat(),
                        time_filter['end'].isoformat()
                    ],
                    table='trades'
                ))

        # Amount filters
        for amount in entities['amounts']:
            if amount['type'] == 'min':
                filters.append(QueryFilter(
                    field='amount_from',
                    operator=FilterOperator.GREATER_EQUAL,
                    value=amount['value'],
                    table='trades'
                ))
            elif amount['type'] == 'max':
                filters.append(QueryFilter(
                    field='amount_to',
                    operator=FilterOperator.LESS_EQUAL,
                    value=amount['value'],
                    table='trades'
                ))

        # State filters
        if entities['states']:
            if len(entities['states']) == 1:
                filters.append(QueryFilter(
                    field='state',
                    operator=FilterOperator.EQUALS,
                    value=entities['states'][0],
                    table='trades'
                ))
            else:
                filters.append(QueryFilter(
                    field='state',
                    operator=FilterOperator.IN,
                    value=entities['states'],
                    table='trades'
                ))

        # Suspicious filter
        if entities.get('suspicious_filter'):
            filters.append(QueryFilter(
                field='conviction_score',
                operator=FilterOperator.GREATER_EQUAL,
                value=60,
                table='analysis'
            ))

        return filters

    def _determine_ordering(self, query: str, entities: Dict) -> Tuple[Optional[str], str]:
        """Determine the ORDER BY clause."""
        query_lower = query.lower()

        # Default ordering based on intent
        if entities.get('suspicious_filter'):
            return 'conviction_score', 'DESC'

        if 'most' in query_lower or 'top' in query_lower:
            if 'recent' in query_lower or 'latest' in query_lower:
                return 'transaction_date', 'DESC'
            if 'amount' in query_lower or 'large' in query_lower:
                return 'amount_from', 'DESC'
            if 'trade' in query_lower:
                return 'trade_count', 'DESC'
            if 'suspicious' in query_lower:
                return 'conviction_score', 'DESC'
            return 'amount_from', 'DESC'

        if 'least' in query_lower or 'bottom' in query_lower:
            if 'recent' in query_lower:
                return 'transaction_date', 'ASC'
            if 'amount' in query_lower:
                return 'amount_from', 'ASC'
            return 'amount_from', 'ASC'

        if 'oldest' in query_lower:
            return 'transaction_date', 'ASC'

        if 'newest' in query_lower or 'latest' in query_lower or 'recent' in query_lower:
            return 'transaction_date', 'DESC'

        # Default to transaction date
        return 'transaction_date', 'DESC'

    def _extract_limit(self, query: str) -> int:
        """Extract result limit from query."""
        # Look for explicit limits
        limit_pattern = re.compile(r'\b(top|first|last)\s+(\d+)\b', re.IGNORECASE)
        match = limit_pattern.search(query)
        if match:
            return min(int(match.group(2)), 100)  # Cap at 100

        # Check for number patterns like "10 trades"
        count_pattern = re.compile(r'\b(\d+)\s+(trade[s]?|member[s]?|result[s]?)\b', re.IGNORECASE)
        match = count_pattern.search(query)
        if match:
            return min(int(match.group(1)), 100)

        # Default limits based on query type
        if 'all' in query.lower():
            return 100
        return 50

    def _calculate_confidence(self, result: ParsedQuery) -> float:
        """Calculate confidence score for the parsed query."""
        confidence = 0.5  # Base confidence

        # Increase confidence for recognized entities
        if result.entities.get('members'):
            confidence += 0.2
        if result.entities.get('stocks'):
            confidence += 0.15
        if result.entities.get('sectors'):
            confidence += 0.15
        if result.entities.get('parties'):
            confidence += 0.1
        if result.entities.get('time_filters'):
            confidence += 0.1
        if result.entities.get('transaction_types'):
            confidence += 0.1

        # Decrease confidence for unknown intent
        if result.intent == QueryIntent.UNKNOWN:
            confidence -= 0.3

        # Increase confidence if we have filters
        if len(result.filters) > 0:
            confidence += 0.1 * min(len(result.filters), 3)

        return min(max(confidence, 0.0), 1.0)

    def _generate_explanation(self, result: ParsedQuery) -> str:
        """Generate human-readable explanation of the parsed query."""
        parts = []

        if result.intent == QueryIntent.UNKNOWN:
            return "Unable to determine query intent. Please try rephrasing."

        parts.append(f"Intent: {result.intent.value.replace('_', ' ').title()}")

        if result.entities.get('members'):
            parts.append(f"Members: {', '.join(result.entities['members'])}")
        if result.entities.get('stocks'):
            parts.append(f"Stocks: {', '.join(result.entities['stocks'])}")
        if result.entities.get('sectors'):
            parts.append(f"Sectors: {', '.join(result.entities['sectors'])}")
        if result.entities.get('parties'):
            party_names = {'D': 'Democrat', 'R': 'Republican', 'I': 'Independent'}
            parts.append(f"Parties: {', '.join(party_names.get(p, p) for p in result.entities['parties'])}")
        if result.entities.get('transaction_types'):
            parts.append(f"Transaction types: {', '.join(result.entities['transaction_types'])}")
        if result.entities.get('time_filters'):
            tf = result.entities['time_filters'][0]
            parts.append(f"Time range: {tf.get('start', 'N/A')} to {tf.get('end', 'N/A')}")

        parts.append(f"Sorting by: {result.order_by or 'default'} ({result.order_direction})")
        parts.append(f"Limit: {result.limit} results")
        parts.append(f"Confidence: {result.confidence:.0%}")

        return " | ".join(parts)

    def _generate_suggestions(self, query: str, entities: Dict) -> List[str]:
        """Generate query suggestions when confidence is low."""
        suggestions = []

        if not entities.get('members'):
            suggestions.append("Try specifying a member name, e.g., 'Show me Pelosi's trades'")

        if not entities.get('stocks') and not entities.get('sectors'):
            suggestions.append("Try adding a stock or sector, e.g., 'tech trades' or 'NVDA purchases'")

        if not entities.get('time_filters'):
            suggestions.append("Try adding a time period, e.g., 'this month' or 'in 2023'")

        if not entities.get('transaction_types'):
            suggestions.append("Try specifying buy/sell, e.g., 'purchases' or 'sales'")

        # Add example queries
        examples = [
            "Show me Pelosi's tech trades",
            "Who bought NVDA this year?",
            "Most suspicious trades this month",
            "Republican vs Democrat trading volume",
            "Senate trades in the banking sector",
            "Trades over $1 million in 2023"
        ]

        suggestions.append(f"Example queries: {', '.join(examples[:3])}")

        return suggestions

    def to_sql(self, result: ParsedQuery, table: str = "trades") -> Tuple[str, List[Any]]:
        """
        Generate SQL query from parsed result.

        Args:
            result: ParsedQuery object
            table: Base table name

        Returns:
            Tuple of (SQL query string, list of parameters)
        """
        where_clause, params = result.to_sql_where()

        query = f"SELECT * FROM {table}"

        if where_clause:
            query += f" WHERE {where_clause}"

        if result.order_by:
            query += f" ORDER BY {result.order_by} {result.order_direction}"

        query += f" LIMIT {result.limit}"

        return query, params

    def to_filter_dict(self, result: ParsedQuery) -> Dict:
        """
        Convert parsed query to filter dictionary for in-memory filtering.

        Useful for filtering lists of dictionaries without a database.
        """
        return {
            'members': result.entities.get('members', []),
            'stocks': result.entities.get('stocks', []) + result.entities.get('sector_stocks', []),
            'parties': result.entities.get('parties', []),
            'chambers': result.entities.get('chambers', []),
            'transaction_types': result.entities.get('transaction_types', []),
            'time_range': result.entities.get('time_filters', [{}])[0] if result.entities.get('time_filters') else None,
            'min_amount': next((a['value'] for a in result.entities.get('amounts', []) if a['type'] == 'min'), None),
            'max_amount': next((a['value'] for a in result.entities.get('amounts', []) if a['type'] == 'max'), None),
            'states': result.entities.get('states', []),
            'suspicious_only': result.entities.get('suspicious_filter', False),
            'order_by': result.order_by,
            'order_direction': result.order_direction,
            'limit': result.limit
        }


class NLQueryService:
    """
    High-level service for processing natural language queries.

    Combines parsing with data retrieval for complete query handling.
    """

    def __init__(self, parser: Optional[NLQueryParser] = None):
        """Initialize the service."""
        self.parser = parser or NLQueryParser()
        self._query_history: List[ParsedQuery] = []

    def query(self, natural_query: str) -> ParsedQuery:
        """
        Process a natural language query.

        Args:
            natural_query: Natural language query string

        Returns:
            ParsedQuery with filters and metadata
        """
        result = self.parser.parse(natural_query)
        self._query_history.append(result)
        return result

    def filter_trades(self, trades: List[Dict], query: str) -> List[Dict]:
        """
        Filter a list of trades based on natural language query.

        Args:
            trades: List of trade dictionaries
            query: Natural language query

        Returns:
            Filtered list of trades
        """
        parsed = self.parser.parse(query)
        filters = self.parser.to_filter_dict(parsed)

        filtered = trades

        # Apply member filter
        if filters['members']:
            filtered = [t for t in filtered
                       if any(m.lower() in t.get('member_name', '').lower() for m in filters['members'])]

        # Apply stock filter
        if filters['stocks']:
            stock_set = set(s.upper() for s in filters['stocks'])
            filtered = [t for t in filtered if t.get('symbol', '').upper() in stock_set]

        # Apply party filter
        if filters['parties']:
            filtered = [t for t in filtered if t.get('party') in filters['parties']]

        # Apply chamber filter
        if filters['chambers']:
            filtered = [t for t in filtered if t.get('chamber') in filters['chambers']]

        # Apply transaction type filter
        if filters['transaction_types']:
            filtered = [t for t in filtered if t.get('transaction_type') in filters['transaction_types']]

        # Apply time range filter
        if filters['time_range']:
            start = filters['time_range'].get('start')
            end = filters['time_range'].get('end')
            if start and end:
                def in_range(trade):
                    trade_date = trade.get('transaction_date', '')
                    if isinstance(trade_date, str):
                        try:
                            trade_date = datetime.strptime(trade_date, '%Y-%m-%d').date()
                        except ValueError:
                            return True
                    return start <= trade_date <= end
                filtered = [t for t in filtered if in_range(t)]

        # Apply amount filters
        if filters['min_amount']:
            filtered = [t for t in filtered
                       if (t.get('amount_from', 0) or 0) >= filters['min_amount']]
        if filters['max_amount']:
            filtered = [t for t in filtered
                       if (t.get('amount_to', float('inf')) or float('inf')) <= filters['max_amount']]

        # Apply state filter
        if filters['states']:
            filtered = [t for t in filtered if t.get('state') in filters['states']]

        # Apply ordering
        if filters['order_by']:
            reverse = filters['order_direction'] == 'DESC'
            key_field = filters['order_by']
            # Map common fields
            field_map = {
                'transaction_date': 'transaction_date',
                'amount_from': 'amount_from',
                'conviction_score': 'conviction_score',
            }
            actual_field = field_map.get(key_field, key_field)
            filtered = sorted(filtered, key=lambda x: x.get(actual_field, 0) or 0, reverse=reverse)

        # Apply limit
        filtered = filtered[:filters['limit']]

        return filtered

    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """Get recent query history."""
        return [q.to_dict() for q in self._query_history[-limit:]]


def main():
    """Demo and testing for the NL Query parser."""
    print("=" * 60)
    print("Congressional Trading Natural Language Query Parser - Demo")
    print("=" * 60)

    parser = NLQueryParser()
    service = NLQueryService(parser)

    # Test queries
    test_queries = [
        "Show me Pelosi's tech trades",
        "Who bought NVDA before the earnings?",
        "Which Republicans trade the most?",
        "Most suspicious trade this month",
        "Senate banking committee trades in 2023",
        "Trades over $1 million by Democrats",
        "What did Manchin sell this year?",
        "Top 10 trades in the energy sector",
        "COVID trading activity",
        "Compare Republican vs Democrat trading",
        "Large purchases in semiconductor stocks",
        "Pelosi Apple trades last year",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("-" * 60)

        result = service.query(query)

        print(f"Intent: {result.intent.value}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"\nExtracted Entities:")
        for key, value in result.entities.items():
            if value:
                print(f"  - {key}: {value}")

        print(f"\nFilters ({len(result.filters)}):")
        for f in result.filters:
            print(f"  - {f.field} {f.operator.value} {f.value}")

        print(f"\nOrdering: {result.order_by} {result.order_direction}")
        print(f"Limit: {result.limit}")

        if result.suggestions:
            print(f"\nSuggestions:")
            for s in result.suggestions:
                print(f"  - {s}")

        # Generate SQL
        sql, params = parser.to_sql(result)
        print(f"\nGenerated SQL:\n{sql}")
        print(f"Parameters: {params}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
