#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Real-Time News Monitoring
Advanced news monitoring and sentiment analysis for congressional trading intelligence.
"""

import os
import logging
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import json
import re
from urllib.parse import urlencode
import hashlib

import numpy as np
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from transformers import pipeline
import feedparser
import requests
from bs4 import BeautifulSoup
import yfinance as yf

logger = logging.getLogger(__name__)

class SentimentScore(Enum):
    """Sentiment classification levels."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"

class NewsSource(Enum):
    """Supported news sources."""
    FINANCIAL_NEWS = "financial_news"
    POLITICAL_NEWS = "political_news"
    CONGRESSIONAL_PRESS = "congressional_press"
    SOCIAL_MEDIA = "social_media"
    RSS_FEEDS = "rss_feeds"

@dataclass
class NewsArticle:
    """Data model for news articles."""
    article_id: str
    title: str
    content: str
    url: str
    source: str
    author: Optional[str]
    published_date: str
    collected_date: str
    entities: List[str]
    keywords: List[str]
    sentiment_score: float
    sentiment_label: SentimentScore
    relevance_score: float
    category: str

@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis results."""
    overall_sentiment: float
    compound_score: float
    positive_score: float
    negative_score: float
    neutral_score: float
    confidence: float
    sentiment_label: SentimentScore
    key_phrases: List[str]
    emotions: Dict[str, float]

@dataclass
class MarketCorrelation:
    """Market correlation with news sentiment."""
    symbol: str
    correlation_coefficient: float
    p_value: float
    time_lag_hours: int
    sentiment_impact: float
    statistical_significance: bool

class EntityExtractor:
    """Extracts entities and keywords from news text."""
    
    def __init__(self):
        """Initialize entity extractor."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic extraction")
            self.nlp = None
        
        # Congressional members and stock symbols for targeted extraction
        self.congressional_keywords = [
            'congress', 'senator', 'representative', 'house', 'senate',
            'committee', 'hearing', 'bill', 'vote', 'legislation'
        ]
        
        self.financial_keywords = [
            'stock', 'trade', 'invest', 'portfolio', 'disclosure',
            'SEC', 'insider', 'earnings', 'market', 'financial'
        ]
    
    def extract_entities(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract entities and keywords from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (entities, keywords)
        """
        entities = []
        keywords = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE']:  # Person, Organization, Location
                    entities.append(ent.text.strip())
            
            # Extract keywords based on POS tags
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN'] and 
                    len(token.text) > 2 and 
                    not token.is_stop and 
                    not token.is_punct):
                    keywords.append(token.lemma_.lower())
        else:
            # Fallback: simple keyword extraction
            text_lower = text.lower()
            
            # Find congressional keywords
            for keyword in self.congressional_keywords:
                if keyword in text_lower:
                    keywords.append(keyword)
            
            # Find financial keywords
            for keyword in self.financial_keywords:
                if keyword in text_lower:
                    keywords.append(keyword)
            
            # Simple entity extraction using patterns
            entities.extend(self._extract_person_names(text))
            entities.extend(self._extract_stock_symbols(text))
        
        # Remove duplicates and clean
        entities = list(set([e.strip() for e in entities if len(e.strip()) > 1]))
        keywords = list(set([k.strip() for k in keywords if len(k.strip()) > 2]))
        
        return entities, keywords
    
    def _extract_person_names(self, text: str) -> List[str]:
        """Extract person names using simple patterns."""
        # Pattern for titles followed by names
        pattern = r'(?:Sen\.|Rep\.|Senator|Representative)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)'
        matches = re.findall(pattern, text)
        return matches
    
    def _extract_stock_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        # Pattern for stock symbols (3-5 uppercase letters)
        pattern = r'\b[A-Z]{3,5}\b'
        potential_symbols = re.findall(pattern, text)
        
        # Filter out common false positives
        false_positives = {'THE', 'AND', 'FOR', 'SEC', 'CEO', 'CFO', 'USA', 'USD'}
        return [symbol for symbol in potential_symbols if symbol not in false_positives]

class SentimentAnalyzer:
    """Advanced sentiment analysis for financial and political news."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Try to load transformer model for more accurate sentiment analysis
        try:
            self.transformer_analyzer = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",  # FinBERT for financial sentiment
                return_all_scores=True
            )
            self.has_transformer = True
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}")
            self.transformer_analyzer = None
            self.has_transformer = False
    
    def analyze_sentiment(self, text: str) -> SentimentAnalysis:
        """
        Perform comprehensive sentiment analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Comprehensive sentiment analysis results
        """
        # VADER sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        # TextBlob sentiment analysis
        blob = TextBlob(text)
        textblob_polarity = blob.sentiment.polarity
        textblob_subjectivity = blob.sentiment.subjectivity
        
        # Transformer-based analysis (if available)
        transformer_scores = None
        if self.has_transformer and self.transformer_analyzer:
            try:
                transformer_results = self.transformer_analyzer(text[:512])  # Limit length
                transformer_scores = {
                    result['label'].lower(): result['score'] 
                    for result in transformer_results[0]
                }
            except Exception as e:
                logger.warning(f"Transformer analysis failed: {e}")
        
        # Combine scores
        if transformer_scores:
            # Use transformer scores as primary, VADER as secondary
            positive_score = transformer_scores.get('positive', vader_scores['pos'])
            negative_score = transformer_scores.get('negative', vader_scores['neg'])
            neutral_score = transformer_scores.get('neutral', vader_scores['neu'])
            compound_score = vader_scores['compound']  # VADER's compound is reliable
        else:
            # Use VADER scores
            positive_score = vader_scores['pos']
            negative_score = vader_scores['neg']
            neutral_score = vader_scores['neu']
            compound_score = vader_scores['compound']
        
        # Overall sentiment (weighted combination)
        overall_sentiment = (compound_score + textblob_polarity) / 2
        
        # Determine sentiment label
        if overall_sentiment >= 0.5:
            sentiment_label = SentimentScore.VERY_POSITIVE
        elif overall_sentiment >= 0.1:
            sentiment_label = SentimentScore.POSITIVE
        elif overall_sentiment <= -0.5:
            sentiment_label = SentimentScore.VERY_NEGATIVE
        elif overall_sentiment <= -0.1:
            sentiment_label = SentimentScore.NEGATIVE
        else:
            sentiment_label = SentimentScore.NEUTRAL
        
        # Calculate confidence (based on score magnitude and subjectivity)
        confidence = min(1.0, abs(overall_sentiment) + (1 - textblob_subjectivity))
        
        # Extract key phrases (simple approach)
        key_phrases = self._extract_key_phrases(text)
        
        # Basic emotion detection (simplified)
        emotions = self._detect_emotions(text, vader_scores)
        
        return SentimentAnalysis(
            overall_sentiment=overall_sentiment,
            compound_score=compound_score,
            positive_score=positive_score,
            negative_score=negative_score,
            neutral_score=neutral_score,
            confidence=confidence,
            sentiment_label=sentiment_label,
            key_phrases=key_phrases,
            emotions=emotions
        )
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases that drive sentiment."""
        # Simple approach: find adjectives and adverbs near important nouns
        blob = TextBlob(text)
        key_phrases = []
        
        for sentence in blob.sentences:
            # Find sentiment-bearing phrases
            if abs(sentence.sentiment.polarity) > 0.1:
                # Extract phrases with strong sentiment
                phrase = str(sentence)[:100]  # Limit length
                key_phrases.append(phrase.strip())
        
        return key_phrases[:5]  # Return top 5 phrases
    
    def _detect_emotions(self, text: str, vader_scores: Dict[str, float]) -> Dict[str, float]:
        """Detect basic emotions from text."""
        # Simplified emotion detection based on keywords and sentiment
        emotions = {
            'anger': 0.0,
            'fear': 0.0,
            'joy': 0.0,
            'sadness': 0.0,
            'surprise': 0.0,
            'disgust': 0.0
        }
        
        text_lower = text.lower()
        
        # Keyword-based emotion detection
        emotion_keywords = {
            'anger': ['angry', 'furious', 'outraged', 'mad', 'annoyed'],
            'fear': ['afraid', 'scared', 'worried', 'anxious', 'concerned'],
            'joy': ['happy', 'excited', 'pleased', 'delighted', 'optimistic'],
            'sadness': ['sad', 'disappointed', 'depressed', 'upset', 'pessimistic'],
            'surprise': ['surprised', 'shocked', 'amazed', 'unexpected', 'sudden'],
            'disgust': ['disgusted', 'revolted', 'appalled', 'repulsed', 'sickened']
        }
        
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            emotions[emotion] = min(1.0, score / len(keywords))
        
        # Adjust based on VADER sentiment
        if vader_scores['compound'] > 0.1:
            emotions['joy'] = max(emotions['joy'], vader_scores['pos'])
        elif vader_scores['compound'] < -0.1:
            emotions['sadness'] = max(emotions['sadness'], vader_scores['neg'])
            emotions['anger'] = max(emotions['anger'], vader_scores['neg'] * 0.5)
        
        return emotions

class NewsSourceManager:
    """Manages multiple news sources and data collection."""
    
    def __init__(self):
        """Initialize news source manager."""
        self.session = aiohttp.ClientSession()
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # News source configurations
        self.sources = {
            NewsSource.FINANCIAL_NEWS: {
                'urls': [
                    'https://feeds.finance.yahoo.com/rss/2.0/headline',
                    'https://feeds.bloomberg.com/markets/news.rss',
                    'https://www.reuters.com/business/finance/rss'
                ],
                'keywords': ['stock', 'market', 'trading', 'finance', 'earnings']
            },
            NewsSource.POLITICAL_NEWS: {
                'urls': [
                    'https://rss.cnn.com/rss/edition.rss',
                    'https://feeds.npr.org/1001/rss.xml',
                    'https://www.politico.com/rss/politics.xml'
                ],
                'keywords': ['congress', 'senate', 'house', 'representative', 'senator']
            },
            NewsSource.RSS_FEEDS: {
                'urls': [
                    'https://feeds.washingtonpost.com/rss/business',
                    'https://www.wsj.com/xml/rss/3_7085.xml'  # WSJ Markets
                ],
                'keywords': ['congressional', 'insider trading', 'stock act']
            }
        }
        
        # Rate limiting
        self.last_request_time = {}
        self.min_request_interval = 1.0  # Seconds between requests
    
    async def collect_news_articles(self, 
                                  sources: Optional[List[NewsSource]] = None,
                                  hours_back: int = 24) -> List[NewsArticle]:
        """
        Collect news articles from multiple sources.
        
        Args:
            sources: List of news sources to query
            hours_back: How many hours back to collect news
            
        Returns:
            List of collected news articles
        """
        if sources is None:
            sources = list(self.sources.keys())
        
        logger.info(f"Collecting news from {len(sources)} sources for last {hours_back} hours")
        
        all_articles = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for source in sources:
            try:
                articles = await self._collect_from_source(source, cutoff_time)
                all_articles.extend(articles)
                logger.info(f"Collected {len(articles)} articles from {source.value}")
            except Exception as e:
                logger.error(f"Error collecting from {source.value}: {e}")
        
        # Remove duplicates based on content similarity
        unique_articles = self._deduplicate_articles(all_articles)
        
        logger.info(f"Collected {len(unique_articles)} unique articles total")
        return unique_articles
    
    async def _collect_from_source(self, source: NewsSource, cutoff_time: datetime) -> List[NewsArticle]:
        """Collect articles from a specific news source."""
        articles = []
        source_config = self.sources[source]
        
        for url in source_config['urls']:
            try:
                # Rate limiting
                await self._respect_rate_limit(url)
                
                # Fetch RSS feed
                feed_data = feedparser.parse(url)
                
                for entry in feed_data.entries:
                    # Parse publication date
                    try:
                        pub_date = datetime(*entry.published_parsed[:6])
                    except:
                        pub_date = datetime.now()
                    
                    # Skip old articles
                    if pub_date < cutoff_time:
                        continue
                    
                    # Extract content
                    content = entry.get('summary', entry.get('description', ''))
                    if not content:
                        continue
                    
                    # Check relevance
                    if not self._is_relevant_article(content, source_config['keywords']):
                        continue
                    
                    # Extract entities and analyze sentiment
                    entities, keywords = self.entity_extractor.extract_entities(content)
                    sentiment = self.sentiment_analyzer.analyze_sentiment(content)
                    
                    # Calculate relevance score
                    relevance_score = self._calculate_relevance_score(content, entities, keywords)
                    
                    # Create article object
                    article = NewsArticle(
                        article_id=self._generate_article_id(entry.link, pub_date),
                        title=entry.title,
                        content=content,
                        url=entry.link,
                        source=source.value,
                        author=entry.get('author'),
                        published_date=pub_date.isoformat(),
                        collected_date=datetime.now().isoformat(),
                        entities=entities,
                        keywords=keywords,
                        sentiment_score=sentiment.overall_sentiment,
                        sentiment_label=sentiment.sentiment_label,
                        relevance_score=relevance_score,
                        category=self._categorize_article(content, entities)
                    )
                    
                    articles.append(article)
                    
            except Exception as e:
                logger.warning(f"Error processing URL {url}: {e}")
        
        return articles
    
    async def _respect_rate_limit(self, url: str):
        """Ensure rate limiting between requests."""
        now = datetime.now().timestamp()
        last_request = self.last_request_time.get(url, 0)
        
        if now - last_request < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - (now - last_request))
        
        self.last_request_time[url] = now
    
    def _is_relevant_article(self, content: str, keywords: List[str]) -> bool:
        """Check if article is relevant based on keywords."""
        content_lower = content.lower()
        
        # Must contain at least one keyword
        keyword_matches = sum(1 for keyword in keywords if keyword in content_lower)
        
        # Additional relevance checks
        congressional_terms = ['congress', 'senator', 'representative', 'committee']
        trading_terms = ['stock', 'trade', 'investment', 'portfolio', 'sec']
        
        has_congressional = any(term in content_lower for term in congressional_terms)
        has_trading = any(term in content_lower for term in trading_terms)
        
        return keyword_matches > 0 or (has_congressional and has_trading)
    
    def _calculate_relevance_score(self, content: str, entities: List[str], keywords: List[str]) -> float:
        """Calculate relevance score for an article."""
        score = 0.0
        content_lower = content.lower()
        
        # Keyword relevance
        congressional_keywords = ['congress', 'senator', 'representative', 'committee', 'house', 'senate']
        trading_keywords = ['stock', 'trade', 'investment', 'disclosure', 'insider']
        
        congressional_score = sum(1 for kw in congressional_keywords if kw in content_lower) / len(congressional_keywords)
        trading_score = sum(1 for kw in trading_keywords if kw in content_lower) / len(trading_keywords)
        
        score += (congressional_score + trading_score) * 0.4
        
        # Entity relevance
        entity_score = min(1.0, len(entities) / 5.0)  # Normalize by 5 entities
        score += entity_score * 0.3
        
        # Content length bonus (more detailed articles)
        length_score = min(1.0, len(content) / 1000.0)  # Normalize by 1000 chars
        score += length_score * 0.1
        
        # Recency bonus (implicit - newer articles already filtered)
        score += 0.2
        
        return min(1.0, score)
    
    def _categorize_article(self, content: str, entities: List[str]) -> str:
        """Categorize article based on content."""
        content_lower = content.lower()
        
        if any(term in content_lower for term in ['earnings', 'quarterly', 'revenue', 'profit']):
            return 'earnings'
        elif any(term in content_lower for term in ['committee', 'hearing', 'bill', 'vote']):
            return 'legislation'
        elif any(term in content_lower for term in ['trade', 'stock', 'investment', 'portfolio']):
            return 'trading'
        elif any(term in content_lower for term in ['election', 'campaign', 'candidate']):
            return 'politics'
        else:
            return 'general'
    
    def _generate_article_id(self, url: str, pub_date: datetime) -> str:
        """Generate unique article ID."""
        content = f"{url}{pub_date.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on content similarity."""
        if len(articles) <= 1:
            return articles
        
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Simple deduplication based on title similarity
            title_key = article.title.lower().strip()
            
            # Remove common prefixes/suffixes that cause false duplicates
            title_key = re.sub(r'^(breaking|news|update):\s*', '', title_key)
            title_key = re.sub(r'\s*-\s*(reuters|bloomberg|cnn|npr).*$', '', title_key)
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)
        
        return unique_articles
    
    async def close(self):
        """Close the session."""
        await self.session.close()

class MarketCorrelationAnalyzer:
    """Analyzes correlation between news sentiment and market movements."""
    
    def __init__(self):
        """Initialize market correlation analyzer."""
        self.stock_data_cache = {}
        
    def analyze_sentiment_market_correlation(self, 
                                          articles: List[NewsArticle],
                                          symbols: List[str],
                                          time_window_hours: int = 24) -> List[MarketCorrelation]:
        """
        Analyze correlation between news sentiment and stock price movements.
        
        Args:
            articles: List of news articles with sentiment
            symbols: Stock symbols to analyze
            time_window_hours: Time window for correlation analysis
            
        Returns:
            List of market correlations
        """
        logger.info(f"Analyzing sentiment-market correlation for {len(symbols)} symbols")
        
        correlations = []
        
        for symbol in symbols:
            try:
                correlation = self._calculate_symbol_correlation(articles, symbol, time_window_hours)
                if correlation:
                    correlations.append(correlation)
            except Exception as e:
                logger.error(f"Error analyzing correlation for {symbol}: {e}")
        
        return correlations
    
    def _calculate_symbol_correlation(self, 
                                    articles: List[NewsArticle], 
                                    symbol: str, 
                                    time_window_hours: int) -> Optional[MarketCorrelation]:
        """Calculate correlation for a specific symbol."""
        # Filter articles mentioning the symbol
        relevant_articles = [
            article for article in articles 
            if symbol.upper() in article.content.upper() or symbol.upper() in article.entities
        ]
        
        if len(relevant_articles) < 5:  # Need minimum articles for correlation
            return None
        
        # Get stock price data
        try:
            stock_data = self._get_stock_data(symbol, days=7)
            if stock_data is None or len(stock_data) < 2:
                return None
        except Exception as e:
            logger.warning(f"Could not get stock data for {symbol}: {e}")
            return None
        
        # Aggregate sentiment by time periods
        sentiment_timeline = self._aggregate_sentiment_timeline(relevant_articles, time_window_hours)
        
        # Align sentiment and price data
        aligned_data = self._align_sentiment_price_data(sentiment_timeline, stock_data)
        
        if len(aligned_data) < 3:  # Need minimum data points
            return None
        
        # Calculate correlation
        sentiments = [point['sentiment'] for point in aligned_data]
        returns = [point['return'] for point in aligned_data]
        
        correlation_coeff = np.corrcoef(sentiments, returns)[0, 1]
        
        # Simple p-value approximation (would use scipy.stats in production)
        n = len(aligned_data)
        t_stat = correlation_coeff * np.sqrt((n - 2) / (1 - correlation_coeff**2))
        p_value = 2 * (1 - abs(t_stat) / np.sqrt(n))  # Simplified approximation
        
        # Determine statistical significance
        significant = p_value < 0.05 and abs(correlation_coeff) > 0.3
        
        return MarketCorrelation(
            symbol=symbol,
            correlation_coefficient=correlation_coeff,
            p_value=p_value,
            time_lag_hours=0,  # Simplified - no lag analysis
            sentiment_impact=abs(correlation_coeff),
            statistical_significance=significant
        )
    
    def _get_stock_data(self, symbol: str, days: int = 7) -> Optional[pd.DataFrame]:
        """Get stock price data."""
        if symbol in self.stock_data_cache:
            return self.stock_data_cache[symbol]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=f"{days}d", interval="1h")
            
            if len(data) == 0:
                return None
            
            # Calculate returns
            data['return'] = data['Close'].pct_change()
            data = data.dropna()
            
            self.stock_data_cache[symbol] = data
            return data
            
        except Exception as e:
            logger.warning(f"Error fetching stock data for {symbol}: {e}")
            return None
    
    def _aggregate_sentiment_timeline(self, articles: List[NewsArticle], window_hours: int) -> List[Dict]:
        """Aggregate sentiment scores by time periods."""
        timeline = []
        
        # Group articles by time windows
        time_groups = {}
        
        for article in articles:
            pub_date = datetime.fromisoformat(article.published_date.replace('Z', '+00:00'))
            # Round to nearest hour for grouping
            hour_key = pub_date.replace(minute=0, second=0, microsecond=0)
            
            if hour_key not in time_groups:
                time_groups[hour_key] = []
            time_groups[hour_key].append(article)
        
        # Calculate average sentiment for each time period
        for time_key, group_articles in time_groups.items():
            avg_sentiment = np.mean([article.sentiment_score for article in group_articles])
            timeline.append({
                'timestamp': time_key,
                'sentiment': avg_sentiment,
                'article_count': len(group_articles)
            })
        
        return sorted(timeline, key=lambda x: x['timestamp'])
    
    def _align_sentiment_price_data(self, sentiment_timeline: List[Dict], stock_data: pd.DataFrame) -> List[Dict]:
        """Align sentiment and price data by timestamps."""
        aligned_data = []
        
        for sentiment_point in sentiment_timeline:
            sentiment_time = sentiment_point['timestamp']
            
            # Find closest stock data point (within 2 hours)
            stock_times = stock_data.index
            time_diffs = [abs((stock_time - sentiment_time).total_seconds()) for stock_time in stock_times]
            
            if time_diffs:
                min_diff_idx = np.argmin(time_diffs)
                min_diff_seconds = time_diffs[min_diff_idx]
                
                if min_diff_seconds <= 7200:  # Within 2 hours
                    stock_return = stock_data.iloc[min_diff_idx]['return']
                    
                    if not np.isnan(stock_return):
                        aligned_data.append({
                            'timestamp': sentiment_time,
                            'sentiment': sentiment_point['sentiment'],
                            'return': stock_return,
                            'article_count': sentiment_point['article_count']
                        })
        
        return aligned_data

class NewsIntelligenceMonitor:
    """Main class for real-time news monitoring and intelligence analysis."""
    
    def __init__(self):
        """Initialize news intelligence monitor."""
        self.news_manager = NewsSourceManager()
        self.correlation_analyzer = MarketCorrelationAnalyzer()
        self.is_running = False
        
    async def start_monitoring(self, 
                             update_interval_minutes: int = 30,
                             symbols_to_track: Optional[List[str]] = None):
        """
        Start continuous news monitoring.
        
        Args:
            update_interval_minutes: How often to collect news
            symbols_to_track: Stock symbols to track for correlation
        """
        if symbols_to_track is None:
            symbols_to_track = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        
        logger.info(f"Starting news monitoring with {update_interval_minutes}min intervals")
        self.is_running = True
        
        while self.is_running:
            try:
                # Collect news articles
                articles = await self.news_manager.collect_news_articles(hours_back=2)
                
                if articles:
                    logger.info(f"Collected {len(articles)} articles")
                    
                    # Analyze market correlations
                    correlations = self.correlation_analyzer.analyze_sentiment_market_correlation(
                        articles, symbols_to_track
                    )
                    
                    # Process results
                    await self._process_intelligence_results(articles, correlations)
                
                # Wait for next update
                await asyncio.sleep(update_interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _process_intelligence_results(self, 
                                          articles: List[NewsArticle], 
                                          correlations: List[MarketCorrelation]):
        """Process and store intelligence results."""
        # High-impact articles (strong sentiment or high relevance)
        high_impact_articles = [
            article for article in articles 
            if abs(article.sentiment_score) > 0.5 or article.relevance_score > 0.7
        ]
        
        if high_impact_articles:
            logger.info(f"Found {len(high_impact_articles)} high-impact articles")
            
            for article in high_impact_articles[:5]:  # Show top 5
                logger.info(f"  {article.sentiment_label.value}: {article.title[:80]}...")
        
        # Significant correlations
        significant_correlations = [
            corr for corr in correlations 
            if corr.statistical_significance and abs(corr.correlation_coefficient) > 0.4
        ]
        
        if significant_correlations:
            logger.info(f"Found {len(significant_correlations)} significant correlations")
            
            for corr in significant_correlations:
                logger.info(f"  {corr.symbol}: {corr.correlation_coefficient:.3f} (p={corr.p_value:.3f})")
        
        # In production, this would store results in database
        # and potentially trigger alerts or notifications
    
    def stop_monitoring(self):
        """Stop the monitoring loop."""
        logger.info("Stopping news monitoring")
        self.is_running = False
    
    async def get_latest_intelligence_summary(self) -> Dict[str, Any]:
        """Get latest intelligence summary."""
        # Collect recent articles
        articles = await self.news_manager.collect_news_articles(hours_back=6)
        
        # Analyze sentiment distribution
        sentiment_counts = {}
        for article in articles:
            label = article.sentiment_label.value
            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1
        
        # Top entities mentioned
        all_entities = []
        for article in articles:
            all_entities.extend(article.entities)
        
        from collections import Counter
        entity_counts = Counter(all_entities)
        top_entities = entity_counts.most_common(10)
        
        # Category distribution
        category_counts = {}
        for article in articles:
            category = article.category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return {
            'total_articles': len(articles),
            'sentiment_distribution': sentiment_counts,
            'top_entities': top_entities,
            'category_distribution': category_counts,
            'avg_sentiment': np.mean([article.sentiment_score for article in articles]) if articles else 0,
            'last_updated': datetime.now().isoformat()
        }
    
    async def close(self):
        """Clean up resources."""
        await self.news_manager.close()

async def main():
    """Test function for news intelligence monitor."""
    logging.basicConfig(level=logging.INFO)
    
    monitor = NewsIntelligenceMonitor()
    
    try:
        print("Testing news collection and sentiment analysis...")
        
        # Get intelligence summary
        summary = await monitor.get_latest_intelligence_summary()
        
        print(f"\nIntelligence Summary:")
        print(f"Total articles: {summary['total_articles']}")
        print(f"Average sentiment: {summary['avg_sentiment']:.3f}")
        print(f"Sentiment distribution: {summary['sentiment_distribution']}")
        print(f"Top entities: {summary['top_entities'][:5]}")
        print(f"Categories: {summary['category_distribution']}")
        
        # Test correlation analysis with sample symbols
        if summary['total_articles'] > 0:
            print("\nTesting market correlation analysis...")
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            
            # This would normally use the collected articles
            print(f"Would analyze correlations for: {symbols}")
        
    except Exception as e:
        print(f"Error in testing: {e}")
    finally:
        await monitor.close()

if __name__ == "__main__":
    asyncio.run(main())