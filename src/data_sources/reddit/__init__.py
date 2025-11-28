"""
Reddit Intelligence Module
Tracks retail investor sentiment and detects predatory congressional trading
"""

from src.data_sources.reddit.reddit_scraper import (
    RedditScraper,
    RedditCongressCorrelator
)

__all__ = [
    'RedditScraper',
    'RedditCongressCorrelator'
]
