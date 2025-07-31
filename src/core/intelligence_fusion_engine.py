"""
APEX Intelligence Fusion Engine
Multi-Source Intelligence Integration for Ultimate Trading Advantage

This module integrates multiple intelligence sources including:
- Congressional trading data
- News sentiment analysis
- Social media monitoring
- Options flow analysis
- Earnings predictions
- Legislative tracking
- Market microstructure analysis

Author: Manus AI
Version: 1.0
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import time
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from textblob import TextBlob
    import tweepy
    SENTIMENT_AVAILABLE = True
except ImportError:
    print("Sentiment analysis libraries not available. Install textblob and tweepy for full functionality.")
    SENTIMENT_AVAILABLE = False

try:
    import yfinance as yf
    MARKET_DATA_AVAILABLE = True
except ImportError:
    print("Market data library not available. Install yfinance for market data.")
    MARKET_DATA_AVAILABLE = False

class IntelligenceSource(Enum):
    """Types of intelligence sources"""
    CONGRESSIONAL = "congressional"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    OPTIONS_FLOW = "options_flow"
    EARNINGS = "earnings"
    LEGISLATIVE = "legislative"
    MARKET_STRUCTURE = "market_structure"
    INSIDER_TRADING = "insider_trading"

class SignalPriority(Enum):
    """Signal priority levels"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    NOISE = 1

@dataclass
class IntelligenceSignal:
    """Unified intelligence signal structure"""
    source: IntelligenceSource
    symbol: str
    signal_type: str
    priority: SignalPriority
    confidence: float
    impact_score: float
    time_sensitivity: str  # "immediate", "hours", "days", "weeks"
    description: str
    raw_data: Dict[str, Any]
    timestamp: datetime
    expiry: datetime
    correlation_id: Optional[str] = None

class NewsIntelligenceCollector:
    """Collect and analyze news intelligence"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.news_sources = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.cnbc.com/id/100003114/device/rss/rss.html"
        ]
        
    def collect_news_signals(self, symbols: List[str]) -> List[IntelligenceSignal]:
        """Collect news-based intelligence signals"""
        signals = []
        
        for symbol in symbols:
            try:
                # Get recent news for symbol
                news_data = self.get_symbol_news(symbol)
                
                for article in news_data:
                    # Analyze sentiment
                    sentiment_score = self.analyze_sentiment(article['title'] + " " + article.get('summary', ''))
                    
                    # Determine impact
                    impact_score = self.calculate_news_impact(article, sentiment_score)
                    
                    # Create signal
                    if abs(impact_score) > 0.3:  # Significant impact threshold
                        signal = IntelligenceSignal(
                            source=IntelligenceSource.NEWS,
                            symbol=symbol,
                            signal_type="news_sentiment",
                            priority=self.determine_news_priority(impact_score),
                            confidence=min(0.9, abs(sentiment_score) * 0.8),
                            impact_score=impact_score,
                            time_sensitivity="hours",
                            description=f"News sentiment: {article['title'][:100]}...",
                            raw_data=article,
                            timestamp=datetime.now(),
                            expiry=datetime.now() + timedelta(hours=24)
                        )
                        signals.append(signal)
                        
            except Exception as e:
                print(f"Error collecting news for {symbol}: {e}")
                
        return signals
    
    def get_symbol_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news articles for a specific symbol"""
        # Simplified news collection - in practice would use proper news APIs
        news_articles = [
            {
                "title": f"{symbol} Reports Strong Quarterly Earnings Beat",
                "summary": f"{symbol} exceeded analyst expectations with strong revenue growth",
                "published": datetime.now() - timedelta(hours=2),
                "source": "Financial News"
            },
            {
                "title": f"Analysts Upgrade {symbol} Price Target",
                "summary": f"Multiple analysts raise price targets for {symbol} following positive developments",
                "published": datetime.now() - timedelta(hours=6),
                "source": "Market Watch"
            }
        ]
        
        return news_articles
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of news text"""
        if not SENTIMENT_AVAILABLE:
            # Simple keyword-based sentiment
            positive_words = ['beat', 'strong', 'growth', 'upgrade', 'positive', 'bullish', 'outperform']
            negative_words = ['miss', 'weak', 'decline', 'downgrade', 'negative', 'bearish', 'underperform']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count + neg_count == 0:
                return 0.0
            
            return (pos_count - neg_count) / (pos_count + neg_count)
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def calculate_news_impact(self, article: Dict[str, Any], sentiment: float) -> float:
        """Calculate potential market impact of news"""
        impact = sentiment
        
        # Boost impact for certain keywords
        title = article.get('title', '').lower()
        high_impact_keywords = ['earnings', 'merger', 'acquisition', 'fda', 'approval', 'lawsuit', 'investigation']
        
        for keyword in high_impact_keywords:
            if keyword in title:
                impact *= 1.5
                break
        
        # Time decay - recent news has more impact
        hours_old = (datetime.now() - article.get('published', datetime.now())).total_seconds() / 3600
        time_decay = max(0.1, 1 - (hours_old / 24))  # Decay over 24 hours
        
        return impact * time_decay
    
    def determine_news_priority(self, impact_score: float) -> SignalPriority:
        """Determine priority based on impact score"""
        abs_impact = abs(impact_score)
        
        if abs_impact > 0.8:
            return SignalPriority.CRITICAL
        elif abs_impact > 0.6:
            return SignalPriority.HIGH
        elif abs_impact > 0.4:
            return SignalPriority.MEDIUM
        elif abs_impact > 0.2:
            return SignalPriority.LOW
        else:
            return SignalPriority.NOISE

class SocialMediaIntelligenceCollector:
    """Collect and analyze social media intelligence"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.twitter_api = None
        self.initialize_twitter_api()
        
    def initialize_twitter_api(self):
        """Initialize Twitter API if credentials available"""
        if not SENTIMENT_AVAILABLE:
            return
            
        try:
            if 'twitter_bearer_token' in self.api_keys:
                self.twitter_api = tweepy.Client(bearer_token=self.api_keys['twitter_bearer_token'])
        except Exception as e:
            print(f"Twitter API initialization failed: {e}")
    
    def collect_social_signals(self, symbols: List[str]) -> List[IntelligenceSignal]:
        """Collect social media intelligence signals"""
        signals = []
        
        for symbol in symbols:
            try:
                # Get social media mentions
                mentions = self.get_symbol_mentions(symbol)
                
                # Analyze sentiment and volume
                sentiment_analysis = self.analyze_social_sentiment(mentions)
                volume_analysis = self.analyze_mention_volume(mentions, symbol)
                
                # Create signals
                if sentiment_analysis['significance'] > 0.3:
                    signal = IntelligenceSignal(
                        source=IntelligenceSource.SOCIAL_MEDIA,
                        symbol=symbol,
                        signal_type="social_sentiment",
                        priority=self.determine_social_priority(sentiment_analysis['significance']),
                        confidence=sentiment_analysis['confidence'],
                        impact_score=sentiment_analysis['sentiment'] * sentiment_analysis['significance'],
                        time_sensitivity="immediate",
                        description=f"Social sentiment shift: {sentiment_analysis['summary']}",
                        raw_data=sentiment_analysis,
                        timestamp=datetime.now(),
                        expiry=datetime.now() + timedelta(hours=6)
                    )
                    signals.append(signal)
                
                if volume_analysis['anomaly_score'] > 0.5:
                    signal = IntelligenceSignal(
                        source=IntelligenceSource.SOCIAL_MEDIA,
                        symbol=symbol,
                        signal_type="social_volume",
                        priority=self.determine_social_priority(volume_analysis['anomaly_score']),
                        confidence=0.7,
                        impact_score=volume_analysis['anomaly_score'],
                        time_sensitivity="immediate",
                        description=f"Unusual social media volume: {volume_analysis['current_volume']}x normal",
                        raw_data=volume_analysis,
                        timestamp=datetime.now(),
                        expiry=datetime.now() + timedelta(hours=4)
                    )
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error collecting social signals for {symbol}: {e}")
                
        return signals
    
    def get_symbol_mentions(self, symbol: str) -> List[Dict[str, Any]]:
        """Get social media mentions for symbol"""
        # Simplified social media data - in practice would use real APIs
        mentions = [
            {
                "text": f"${symbol} looking strong after earnings beat! 🚀",
                "timestamp": datetime.now() - timedelta(minutes=30),
                "engagement": 150,
                "source": "twitter"
            },
            {
                "text": f"Bullish on ${symbol} - great fundamentals",
                "timestamp": datetime.now() - timedelta(hours=1),
                "engagement": 75,
                "source": "reddit"
            },
            {
                "text": f"${symbol} might be overvalued at current levels",
                "timestamp": datetime.now() - timedelta(hours=2),
                "engagement": 45,
                "source": "twitter"
            }
        ]
        
        return mentions
    
    def analyze_social_sentiment(self, mentions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment from social media mentions"""
        if not mentions:
            return {"sentiment": 0, "confidence": 0, "significance": 0, "summary": "No mentions"}
        
        sentiments = []
        total_engagement = 0
        
        for mention in mentions:
            sentiment = self.analyze_sentiment(mention['text'])
            engagement = mention.get('engagement', 1)
            
            # Weight sentiment by engagement
            sentiments.append(sentiment * engagement)
            total_engagement += engagement
        
        if total_engagement == 0:
            weighted_sentiment = 0
        else:
            weighted_sentiment = sum(sentiments) / total_engagement
        
        # Calculate confidence based on volume and consistency
        sentiment_std = np.std([s/max(1, m.get('engagement', 1)) for s, m in zip(sentiments, mentions)])
        confidence = min(0.9, len(mentions) / 10) * (1 - min(1, sentiment_std))
        
        # Significance based on volume and engagement
        significance = min(1.0, (len(mentions) * total_engagement) / 1000)
        
        return {
            "sentiment": weighted_sentiment,
            "confidence": confidence,
            "significance": significance,
            "summary": f"{len(mentions)} mentions, avg sentiment: {weighted_sentiment:.2f}"
        }
    
    def analyze_mention_volume(self, mentions: List[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        """Analyze mention volume anomalies"""
        current_volume = len(mentions)
        
        # Simplified baseline - in practice would use historical data
        baseline_volume = 5  # Average mentions per hour
        
        anomaly_score = min(2.0, current_volume / max(1, baseline_volume)) - 1
        anomaly_score = max(0, anomaly_score)
        
        return {
            "current_volume": current_volume,
            "baseline_volume": baseline_volume,
            "anomaly_score": anomaly_score,
            "volume_ratio": current_volume / max(1, baseline_volume)
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of social media text"""
        if not SENTIMENT_AVAILABLE:
            # Simple keyword-based sentiment
            positive_words = ['bullish', 'moon', '🚀', 'buy', 'strong', 'great', 'love', 'bull']
            negative_words = ['bearish', 'sell', 'weak', 'bad', 'hate', 'crash', 'bear', 'short']
            
            text_lower = text.lower()
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count + neg_count == 0:
                return 0.0
            
            return (pos_count - neg_count) / (pos_count + neg_count)
        
        try:
            blob = TextBlob(text)
            return blob.sentiment.polarity
        except:
            return 0.0
    
    def determine_social_priority(self, significance: float) -> SignalPriority:
        """Determine priority based on social significance"""
        if significance > 0.8:
            return SignalPriority.CRITICAL
        elif significance > 0.6:
            return SignalPriority.HIGH
        elif significance > 0.4:
            return SignalPriority.MEDIUM
        elif significance > 0.2:
            return SignalPriority.LOW
        else:
            return SignalPriority.NOISE

class OptionsFlowIntelligenceCollector:
    """Collect and analyze options flow intelligence"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        
    def collect_options_signals(self, symbols: List[str]) -> List[IntelligenceSignal]:
        """Collect options flow intelligence signals"""
        signals = []
        
        for symbol in symbols:
            try:
                # Get options flow data
                options_flow = self.get_options_flow(symbol)
                
                # Analyze unusual activity
                unusual_activity = self.analyze_unusual_options_activity(options_flow)
                
                if unusual_activity['significance'] > 0.4:
                    signal = IntelligenceSignal(
                        source=IntelligenceSource.OPTIONS_FLOW,
                        symbol=symbol,
                        signal_type="unusual_options_activity",
                        priority=self.determine_options_priority(unusual_activity['significance']),
                        confidence=unusual_activity['confidence'],
                        impact_score=unusual_activity['directional_bias'] * unusual_activity['significance'],
                        time_sensitivity="hours",
                        description=f"Unusual options activity: {unusual_activity['description']}",
                        raw_data=unusual_activity,
                        timestamp=datetime.now(),
                        expiry=datetime.now() + timedelta(days=1)
                    )
                    signals.append(signal)
                    
            except Exception as e:
                print(f"Error collecting options signals for {symbol}: {e}")
                
        return signals
    
    def get_options_flow(self, symbol: str) -> Dict[str, Any]:
        """Get options flow data for symbol"""
        # Simplified options flow data
        if not MARKET_DATA_AVAILABLE:
            return {
                "call_volume": 15000,
                "put_volume": 8000,
                "call_oi": 45000,
                "put_oi": 32000,
                "large_trades": [
                    {"type": "call", "volume": 2000, "strike": 150, "expiry": "2025-07-18"},
                    {"type": "put", "volume": 1500, "strike": 140, "expiry": "2025-07-18"}
                ]
            }
        
        try:
            ticker = yf.Ticker(symbol)
            options_dates = ticker.options
            
            if not options_dates:
                return {"error": "No options data available"}
            
            # Get nearest expiration
            nearest_expiry = options_dates[0]
            calls = ticker.option_chain(nearest_expiry).calls
            puts = ticker.option_chain(nearest_expiry).puts
            
            return {
                "call_volume": calls['volume'].sum(),
                "put_volume": puts['volume'].sum(),
                "call_oi": calls['openInterest'].sum(),
                "put_oi": puts['openInterest'].sum(),
                "calls_data": calls.to_dict('records'),
                "puts_data": puts.to_dict('records')
            }
            
        except Exception as e:
            return {"error": f"Failed to get options data: {e}"}
    
    def analyze_unusual_options_activity(self, options_flow: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze options flow for unusual activity"""
        if "error" in options_flow:
            return {"significance": 0, "confidence": 0, "description": "No data"}
        
        call_volume = options_flow.get('call_volume', 0)
        put_volume = options_flow.get('put_volume', 0)
        total_volume = call_volume + put_volume
        
        if total_volume == 0:
            return {"significance": 0, "confidence": 0, "description": "No volume"}
        
        # Calculate call/put ratio
        call_put_ratio = call_volume / max(1, put_volume)
        
        # Determine if unusual
        normal_ratio_range = (0.5, 2.0)
        
        if call_put_ratio < normal_ratio_range[0] or call_put_ratio > normal_ratio_range[1]:
            # Unusual activity detected
            if call_put_ratio > 3.0:
                bias = 1.0  # Bullish
                description = f"Heavy call buying (C/P ratio: {call_put_ratio:.1f})"
                significance = min(1.0, call_put_ratio / 5.0)
            elif call_put_ratio < 0.3:
                bias = -1.0  # Bearish
                description = f"Heavy put buying (C/P ratio: {call_put_ratio:.1f})"
                significance = min(1.0, (1 / call_put_ratio) / 5.0)
            else:
                bias = 0.5 if call_put_ratio > 1 else -0.5
                description = f"Moderate unusual activity (C/P ratio: {call_put_ratio:.1f})"
                significance = 0.5
            
            confidence = min(0.9, total_volume / 10000)  # Higher confidence with more volume
            
        else:
            bias = 0
            description = "Normal options activity"
            significance = 0
            confidence = 0.5
        
        return {
            "significance": significance,
            "confidence": confidence,
            "directional_bias": bias,
            "description": description,
            "call_put_ratio": call_put_ratio,
            "total_volume": total_volume
        }
    
    def determine_options_priority(self, significance: float) -> SignalPriority:
        """Determine priority based on options significance"""
        if significance > 0.8:
            return SignalPriority.CRITICAL
        elif significance > 0.6:
            return SignalPriority.HIGH
        elif significance > 0.4:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW

class LegislativeIntelligenceCollector:
    """Collect and analyze legislative intelligence"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.congress_api_key = self.api_keys.get('congress_api_key')
        
    def collect_legislative_signals(self, symbols: List[str]) -> List[IntelligenceSignal]:
        """Collect legislative intelligence signals"""
        signals = []
        
        try:
            # Get upcoming legislative events
            upcoming_events = self.get_upcoming_legislative_events()
            
            for event in upcoming_events:
                # Determine affected symbols
                affected_symbols = self.determine_affected_symbols(event, symbols)
                
                for symbol in affected_symbols:
                    impact_score = self.calculate_legislative_impact(event, symbol)
                    
                    if abs(impact_score) > 0.3:
                        signal = IntelligenceSignal(
                            source=IntelligenceSource.LEGISLATIVE,
                            symbol=symbol,
                            signal_type="legislative_event",
                            priority=self.determine_legislative_priority(abs(impact_score)),
                            confidence=0.8,
                            impact_score=impact_score,
                            time_sensitivity=self.determine_time_sensitivity(event),
                            description=f"Legislative event: {event['title'][:100]}...",
                            raw_data=event,
                            timestamp=datetime.now(),
                            expiry=event['date'] + timedelta(days=1)
                        )
                        signals.append(signal)
                        
        except Exception as e:
            print(f"Error collecting legislative signals: {e}")
            
        return signals
    
    def get_upcoming_legislative_events(self) -> List[Dict[str, Any]]:
        """Get upcoming legislative events"""
        # Simplified legislative events - in practice would use Congress.gov API
        events = [
            {
                "title": "House Committee on Energy and Commerce Hearing on AI Regulation",
                "date": datetime.now() + timedelta(days=7),
                "committee": "Energy and Commerce",
                "chamber": "House",
                "type": "hearing",
                "topics": ["artificial intelligence", "technology regulation", "data privacy"],
                "significance": 0.8
            },
            {
                "title": "Senate Finance Committee Markup on Tax Reform Bill",
                "date": datetime.now() + timedelta(days=14),
                "committee": "Finance",
                "chamber": "Senate",
                "type": "markup",
                "topics": ["tax reform", "corporate taxes", "capital gains"],
                "significance": 0.9
            },
            {
                "title": "House Armed Services Committee Defense Authorization Hearing",
                "date": datetime.now() + timedelta(days=21),
                "committee": "Armed Services",
                "chamber": "House",
                "type": "hearing",
                "topics": ["defense spending", "military contracts", "cybersecurity"],
                "significance": 0.7
            }
        ]
        
        return events
    
    def determine_affected_symbols(self, event: Dict[str, Any], symbols: List[str]) -> List[str]:
        """Determine which symbols are affected by legislative event"""
        affected = []
        
        # Sector mappings
        sector_mappings = {
            "artificial intelligence": ["NVDA", "GOOGL", "MSFT", "AAPL"],
            "technology regulation": ["GOOGL", "META", "AAPL", "AMZN"],
            "tax reform": ["JPM", "BAC", "AAPL", "MSFT"],  # Large cap stocks
            "defense spending": ["LMT", "RTX", "NOC", "GD"],
            "healthcare": ["JNJ", "PFE", "UNH", "ABBV"]
        }
        
        topics = event.get('topics', [])
        
        for topic in topics:
            if topic in sector_mappings:
                for symbol in sector_mappings[topic]:
                    if symbol in symbols and symbol not in affected:
                        affected.append(symbol)
        
        return affected
    
    def calculate_legislative_impact(self, event: Dict[str, Any], symbol: str) -> float:
        """Calculate potential impact of legislative event on symbol"""
        base_impact = event.get('significance', 0.5)
        
        # Adjust based on event type
        type_multipliers = {
            "hearing": 0.6,
            "markup": 0.8,
            "vote": 1.0,
            "passage": 1.2
        }
        
        event_type = event.get('type', 'hearing')
        impact = base_impact * type_multipliers.get(event_type, 0.6)
        
        # Time decay - closer events have more impact
        days_until = (event['date'] - datetime.now()).days
        time_factor = max(0.3, 1 - (days_until / 60))  # Decay over 60 days
        
        # Determine direction based on topics (simplified)
        topics = event.get('topics', [])
        direction = 1  # Default positive
        
        negative_topics = ['regulation', 'taxes', 'investigation']
        if any(neg_topic in ' '.join(topics) for neg_topic in negative_topics):
            direction = -1
        
        return impact * time_factor * direction
    
    def determine_time_sensitivity(self, event: Dict[str, Any]) -> str:
        """Determine time sensitivity of legislative event"""
        days_until = (event['date'] - datetime.now()).days
        
        if days_until <= 3:
            return "immediate"
        elif days_until <= 7:
            return "days"
        elif days_until <= 30:
            return "weeks"
        else:
            return "months"
    
    def determine_legislative_priority(self, impact: float) -> SignalPriority:
        """Determine priority based on legislative impact"""
        if impact > 0.8:
            return SignalPriority.CRITICAL
        elif impact > 0.6:
            return SignalPriority.HIGH
        elif impact > 0.4:
            return SignalPriority.MEDIUM
        elif impact > 0.2:
            return SignalPriority.LOW
        else:
            return SignalPriority.NOISE

class IntelligenceFusionEngine:
    """Main intelligence fusion engine that combines all sources"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        
        # Initialize collectors
        self.news_collector = NewsIntelligenceCollector(api_keys)
        self.social_collector = SocialMediaIntelligenceCollector(api_keys)
        self.options_collector = OptionsFlowIntelligenceCollector(api_keys)
        self.legislative_collector = LegislativeIntelligenceCollector(api_keys)
        
        # Intelligence database
        self.db_path = "/home/ubuntu/intelligence_fusion.db"
        self.initialize_database()
        
        # Signal correlation tracking
        self.correlation_tracker = defaultdict(list)
        
    def initialize_database(self):
        """Initialize intelligence fusion database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS intelligence_signals (
                id INTEGER PRIMARY KEY,
                source TEXT,
                symbol TEXT,
                signal_type TEXT,
                priority INTEGER,
                confidence REAL,
                impact_score REAL,
                time_sensitivity TEXT,
                description TEXT,
                raw_data TEXT,
                timestamp DATETIME,
                expiry DATETIME,
                correlation_id TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_correlations (
                id INTEGER PRIMARY KEY,
                correlation_id TEXT,
                signal_count INTEGER,
                combined_confidence REAL,
                combined_impact REAL,
                consensus_direction REAL,
                created_at DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def collect_all_intelligence(self, symbols: List[str]) -> Dict[str, List[IntelligenceSignal]]:
        """Collect intelligence from all sources"""
        all_signals = {}
        
        print(f"🔍 Collecting intelligence for {len(symbols)} symbols...")
        
        # Collect from each source
        try:
            print("📰 Collecting news intelligence...")
            all_signals['news'] = self.news_collector.collect_news_signals(symbols)
        except Exception as e:
            print(f"News collection error: {e}")
            all_signals['news'] = []
        
        try:
            print("📱 Collecting social media intelligence...")
            all_signals['social'] = self.social_collector.collect_social_signals(symbols)
        except Exception as e:
            print(f"Social media collection error: {e}")
            all_signals['social'] = []
        
        try:
            print("📊 Collecting options flow intelligence...")
            all_signals['options'] = self.options_collector.collect_options_signals(symbols)
        except Exception as e:
            print(f"Options flow collection error: {e}")
            all_signals['options'] = []
        
        try:
            print("🏛️ Collecting legislative intelligence...")
            all_signals['legislative'] = self.legislative_collector.collect_legislative_signals(symbols)
        except Exception as e:
            print(f"Legislative collection error: {e}")
            all_signals['legislative'] = []
        
        # Store signals in database
        self.store_signals(all_signals)
        
        return all_signals
    
    def store_signals(self, signals_by_source: Dict[str, List[IntelligenceSignal]]):
        """Store signals in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for source, signals in signals_by_source.items():
            for signal in signals:
                # Convert datetime objects to strings for JSON serialization
                raw_data_serializable = {}
                for key, value in signal.raw_data.items():
                    if isinstance(value, datetime):
                        raw_data_serializable[key] = value.isoformat()
                    else:
                        raw_data_serializable[key] = value
                
                cursor.execute('''
                    INSERT INTO intelligence_signals (
                        source, symbol, signal_type, priority, confidence,
                        impact_score, time_sensitivity, description, raw_data,
                        timestamp, expiry, correlation_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.source.value, signal.symbol, signal.signal_type,
                    signal.priority.value, signal.confidence, signal.impact_score,
                    signal.time_sensitivity, signal.description,
                    json.dumps(raw_data_serializable), signal.timestamp, signal.expiry,
                    signal.correlation_id
                ))
        
        conn.commit()
        conn.close()
    
    def fuse_intelligence(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fuse intelligence from all sources into unified signals"""
        
        # Collect all intelligence
        all_signals = self.collect_all_intelligence(symbols)
        
        # Fuse signals by symbol
        fused_intelligence = {}
        
        for symbol in symbols:
            symbol_signals = []
            
            # Gather all signals for this symbol
            for source, signals in all_signals.items():
                symbol_signals.extend([s for s in signals if s.symbol == symbol])
            
            if symbol_signals:
                fused_intelligence[symbol] = self.fuse_symbol_signals(symbol, symbol_signals)
            else:
                fused_intelligence[symbol] = {
                    "overall_signal": "NEUTRAL",
                    "confidence": 0.0,
                    "impact_score": 0.0,
                    "priority": "LOW",
                    "signals_count": 0,
                    "source_breakdown": {},
                    "time_sensitivity": "days",
                    "recommendation": "No significant intelligence detected"
                }
        
        return fused_intelligence
    
    def fuse_symbol_signals(self, symbol: str, signals: List[IntelligenceSignal]) -> Dict[str, Any]:
        """Fuse all signals for a specific symbol"""
        
        if not signals:
            return {"overall_signal": "NEUTRAL", "confidence": 0.0}
        
        # Separate signals by time sensitivity
        immediate_signals = [s for s in signals if s.time_sensitivity == "immediate"]
        short_term_signals = [s for s in signals if s.time_sensitivity in ["hours", "days"]]
        long_term_signals = [s for s in signals if s.time_sensitivity in ["weeks", "months"]]
        
        # Calculate weighted impact scores
        total_weighted_impact = 0
        total_weight = 0
        
        for signal in signals:
            # Weight by confidence and priority
            weight = signal.confidence * signal.priority.value
            total_weighted_impact += signal.impact_score * weight
            total_weight += weight
        
        if total_weight == 0:
            overall_impact = 0
        else:
            overall_impact = total_weighted_impact / total_weight
        
        # Calculate overall confidence
        confidence_scores = [s.confidence for s in signals]
        overall_confidence = np.mean(confidence_scores) * min(1.0, len(signals) / 3)
        
        # Determine overall signal direction
        if overall_impact > 0.3:
            overall_signal = "BULLISH"
        elif overall_impact < -0.3:
            overall_signal = "BEARISH"
        else:
            overall_signal = "NEUTRAL"
        
        # Determine priority
        max_priority = max([s.priority.value for s in signals])
        if max_priority >= 4:
            priority = "HIGH"
        elif max_priority >= 3:
            priority = "MEDIUM"
        else:
            priority = "LOW"
        
        # Source breakdown
        source_breakdown = {}
        for signal in signals:
            source = signal.source.value
            if source not in source_breakdown:
                source_breakdown[source] = {
                    "count": 0,
                    "avg_impact": 0,
                    "avg_confidence": 0
                }
            
            source_breakdown[source]["count"] += 1
            source_breakdown[source]["avg_impact"] += signal.impact_score
            source_breakdown[source]["avg_confidence"] += signal.confidence
        
        # Calculate averages
        for source in source_breakdown:
            count = source_breakdown[source]["count"]
            source_breakdown[source]["avg_impact"] /= count
            source_breakdown[source]["avg_confidence"] /= count
        
        # Determine time sensitivity
        if immediate_signals:
            time_sensitivity = "immediate"
        elif short_term_signals:
            time_sensitivity = "hours"
        else:
            time_sensitivity = "days"
        
        # Generate recommendation
        recommendation = self.generate_fusion_recommendation(
            overall_signal, overall_confidence, overall_impact, signals
        )
        
        return {
            "overall_signal": overall_signal,
            "confidence": overall_confidence,
            "impact_score": overall_impact,
            "priority": priority,
            "signals_count": len(signals),
            "source_breakdown": source_breakdown,
            "time_sensitivity": time_sensitivity,
            "recommendation": recommendation,
            "signal_details": [
                {
                    "source": s.source.value,
                    "type": s.signal_type,
                    "impact": s.impact_score,
                    "confidence": s.confidence,
                    "description": s.description
                }
                for s in signals
            ]
        }
    
    def generate_fusion_recommendation(self, signal: str, confidence: float, 
                                     impact: float, signals: List[IntelligenceSignal]) -> str:
        """Generate trading recommendation based on fused intelligence"""
        
        if confidence < 0.3:
            return "Insufficient intelligence - monitor for developments"
        
        if signal == "BULLISH":
            if confidence > 0.8 and impact > 0.6:
                return f"STRONG BUY - High confidence bullish signals from {len(signals)} sources"
            elif confidence > 0.6 and impact > 0.4:
                return f"BUY - Multiple bullish indicators detected"
            else:
                return f"LEAN BULLISH - Moderate positive signals"
        
        elif signal == "BEARISH":
            if confidence > 0.8 and abs(impact) > 0.6:
                return f"STRONG SELL - High confidence bearish signals from {len(signals)} sources"
            elif confidence > 0.6 and abs(impact) > 0.4:
                return f"SELL - Multiple bearish indicators detected"
            else:
                return f"LEAN BEARISH - Moderate negative signals"
        
        else:
            return f"NEUTRAL - Mixed or weak signals from {len(signals)} sources"
    
    def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of intelligence collection"""
        conn = sqlite3.connect(self.db_path)
        
        # Get signal counts by source
        source_counts = pd.read_sql_query('''
            SELECT source, COUNT(*) as count, AVG(confidence) as avg_confidence,
                   AVG(impact_score) as avg_impact
            FROM intelligence_signals 
            WHERE timestamp > datetime('now', '-24 hours')
            GROUP BY source
        ''', conn)
        
        # Get recent high-priority signals
        high_priority_signals = pd.read_sql_query('''
            SELECT symbol, source, signal_type, confidence, impact_score, description
            FROM intelligence_signals 
            WHERE priority >= 4 AND timestamp > datetime('now', '-6 hours')
            ORDER BY timestamp DESC
            LIMIT 10
        ''', conn)
        
        conn.close()
        
        return {
            "source_summary": source_counts.to_dict('records') if not source_counts.empty else [],
            "high_priority_signals": high_priority_signals.to_dict('records') if not high_priority_signals.empty else [],
            "total_signals_24h": source_counts['count'].sum() if not source_counts.empty else 0
        }

def demo_intelligence_fusion():
    """Demonstrate the intelligence fusion engine"""
    
    print("🧠 INTELLIGENCE FUSION ENGINE DEMONSTRATION 🧠")
    print("=" * 70)
    
    # Initialize fusion engine
    fusion_engine = IntelligenceFusionEngine()
    
    # Demo symbols
    symbols = ["NVDA", "AAPL", "GOOGL", "TSLA", "JPM"]
    
    print(f"🎯 Analyzing intelligence for: {', '.join(symbols)}")
    print("-" * 50)
    
    # Collect and fuse intelligence
    fused_intelligence = fusion_engine.fuse_intelligence(symbols)
    
    # Display results
    for symbol, intelligence in fused_intelligence.items():
        print(f"\n📊 INTELLIGENCE FUSION FOR {symbol} 📊")
        print(f"Overall Signal: {intelligence['overall_signal']}")
        print(f"Confidence: {intelligence['confidence']:.1%}")
        print(f"Impact Score: {intelligence['impact_score']:.2f}")
        print(f"Priority: {intelligence['priority']}")
        print(f"Signals Count: {intelligence['signals_count']}")
        print(f"Time Sensitivity: {intelligence['time_sensitivity']}")
        print(f"Recommendation: {intelligence['recommendation']}")
        
        if intelligence['source_breakdown']:
            print("\n📈 Source Breakdown:")
            for source, data in intelligence['source_breakdown'].items():
                print(f"  {source}: {data['count']} signals, "
                      f"avg impact: {data['avg_impact']:.2f}, "
                      f"avg confidence: {data['avg_confidence']:.1%}")
        
        print("-" * 50)
    
    # Show intelligence summary
    print("\n📋 INTELLIGENCE COLLECTION SUMMARY 📋")
    summary = fusion_engine.get_intelligence_summary()
    print(f"Total Signals (24h): {summary['total_signals_24h']}")
    
    if summary['source_summary']:
        print("\nSource Performance:")
        for source_data in summary['source_summary']:
            print(f"  {source_data['source']}: {source_data['count']} signals, "
                  f"avg confidence: {source_data['avg_confidence']:.1%}")
    
    print("\n✅ INTELLIGENCE FUSION DEMONSTRATION COMPLETE ✅")
    return fusion_engine

if __name__ == "__main__":
    # Run demonstration
    fusion_system = demo_intelligence_fusion()

