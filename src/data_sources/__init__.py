# Congressional Trading Intelligence System - Data Sources Package
# API client integrations for disclosure scraping, market data, and external APIs

"""
Data Sources Package
====================

This package provides comprehensive data collection and integration capabilities
for the Congressional Trading Intelligence System.

Modules:
--------

Disclosure Scrapers:
- house_disclosure_scraper: Real-time House STOCK Act filing detection
- senate_disclosure_scraper: Real-time Senate STOCK Act filing detection
- disclosure_scheduler: Automated 15-minute polling for new filings
- filing_pipeline: Parse, validate, and process disclosure data

Market Data:
- market_data: Yahoo Finance integration for prices and company info
- finnhub_client: Finnhub API client for congressional trading data

Government APIs:
- congress_gov_client: Congress.gov API for member and committee data
- congress_gov_scraper: Scraper for additional congressional data
- propublica_client: ProPublica API for legislative data

Usage:
------

# Real-time disclosure detection
from src.data_sources import HouseDisclosureScraper, SenateDisclosureScraper, DisclosureScheduler

# Start automated polling
scheduler = DisclosureScheduler(poll_interval_minutes=15)
scheduler.start()

# Market data enrichment
from src.data_sources import MarketDataService

service = MarketDataService()
enrichment = service.enrich_trade("AAPL", trade_date, filing_date)

# Filing pipeline
from src.data_sources import FilingPipeline

pipeline = FilingPipeline()
parsed_trade = pipeline.process(raw_disclosure_data)
"""

# Disclosure Scrapers
from .house_disclosure_scraper import (
    HouseDisclosureScraper,
    DisclosureRecord,
    TradeRecord,
    ReportType,
    TransactionType,
)

from .senate_disclosure_scraper import (
    SenateDisclosureScraper,
    SenateDisclosureRecord,
    SenateTradeRecord,
    SenateReportType,
)

from .disclosure_scheduler import (
    DisclosureScheduler,
    FilingAlert,
)

from .filing_pipeline import (
    FilingPipeline,
    ParsedTrade,
    ValidationResult,
    ValidationLevel,
    TransactionType as PipelineTransactionType,
    AssetType,
)

# Market Data
from .market_data import (
    MarketDataService,
    MarketDataCache,
    StockPrice,
    StockQuote,
    CompanyInfo,
    TradeEnrichment,
    Sector,
)

# External API Clients
from .finnhub_client import (
    FinnhubAPIClient,
    CongressionalTrade,
    StockPrice as FinnhubStockPrice,
    CompanyProfile,
    RateLimiter,
)

from .congress_gov_client import CongressGovAPIClient
from .propublica_client import ProPublicaAPIClient
from .congress_gov_scraper import CongressGovScraper

__all__ = [
    # House Disclosure
    'HouseDisclosureScraper',
    'DisclosureRecord',
    'TradeRecord',
    'ReportType',
    'TransactionType',

    # Senate Disclosure
    'SenateDisclosureScraper',
    'SenateDisclosureRecord',
    'SenateTradeRecord',
    'SenateReportType',

    # Scheduler
    'DisclosureScheduler',
    'FilingAlert',

    # Pipeline
    'FilingPipeline',
    'ParsedTrade',
    'ValidationResult',
    'ValidationLevel',
    'AssetType',

    # Market Data
    'MarketDataService',
    'MarketDataCache',
    'StockPrice',
    'StockQuote',
    'CompanyInfo',
    'TradeEnrichment',
    'Sector',

    # External APIs
    'FinnhubAPIClient',
    'CongressionalTrade',
    'CompanyProfile',
    'CongressGovAPIClient',
    'ProPublicaAPIClient',
    'CongressGovScraper',
    'RateLimiter',
]

# Version
__version__ = '2.0.0'
