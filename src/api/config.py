"""
Configuration settings for the Congressional Trading Intelligence API.
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Congressional Trading Intelligence API"
    app_version: str = "2.0.0"
    debug: bool = False
    environment: str = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Database
    database_url: str = "sqlite:///./congressional_trading.db"

    # Redis (for rate limiting and caching)
    redis_url: str = "redis://localhost:6379/0"

    # Security
    secret_key: str = "change-me-in-production-use-strong-secret"
    api_key_header: str = "X-API-Key"

    # Rate Limiting (requests per day by tier)
    rate_limit_public: int = 100
    rate_limit_journalist: int = 1000
    rate_limit_academic: int = 5000
    rate_limit_premium: int = 100000  # Effectively unlimited

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Caching
    cache_ttl_seconds: int = 300  # 5 minutes

    # Pagination
    default_page_size: int = 20
    max_page_size: int = 100

    # External APIs (for enrichment)
    finnhub_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Rate limit mapping by tier
RATE_LIMITS = {
    "public": get_settings().rate_limit_public,
    "journalist": get_settings().rate_limit_journalist,
    "academic": get_settings().rate_limit_academic,
    "premium": get_settings().rate_limit_premium,
}

# API description for OpenAPI docs
API_DESCRIPTION = """
# Congressional Trading Intelligence API

Advanced REST API for analyzing congressional trading patterns and promoting financial transparency.

## Features

- **Member Data**: Access information about all 535+ congressional members
- **Trading Data**: Search and analyze stock trades disclosed under the STOCK Act
- **Conviction Scoring**: AI-powered analysis indicating how suspicious a trade appears
- **Timing Analysis**: Correlation between trades and market-moving events
- **Story Generation**: Generate journalist-ready narratives about trading patterns

## Authentication

Most endpoints require an API key passed in the `X-API-Key` header.

### Access Tiers

| Tier | Rate Limit | Features |
|------|------------|----------|
| Public | 100/day | Basic read-only access |
| Journalist | 1,000/day | Full access + story generation |
| Academic | 5,000/day | Bulk exports + research features |
| Premium | Unlimited | All features + priority support |

## Data Sources

All data comes from official STOCK Act disclosures and public records.

## Disclaimer

This API provides educational analysis and is not financial advice. High conviction scores indicate patterns worth investigating, not proof of wrongdoing.
"""

API_TAGS_METADATA = [
    {
        "name": "members",
        "description": "Congressional member information and profiles",
    },
    {
        "name": "trades",
        "description": "Stock trade disclosures and analysis",
    },
    {
        "name": "analysis",
        "description": "Statistical analysis, leaderboards, and insights",
    },
    {
        "name": "stories",
        "description": "AI-generated narrative content (authenticated)",
    },
    {
        "name": "auth",
        "description": "Authentication and API key management",
    },
    {
        "name": "health",
        "description": "Service health and status",
    },
]
