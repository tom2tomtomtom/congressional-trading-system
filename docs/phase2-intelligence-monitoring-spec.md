# Phase 2 Intelligence & Monitoring - Technical Specification

**Component**: Real-time Intelligence & News Monitoring  
**Phase**: 2 - Intelligence & Analytics  
**Status**: Implemented  
**Version**: 2.0  

## Overview

The Intelligence and Monitoring system provides real-time news aggregation, sentiment analysis, and market correlation tracking for congressional trading activities. This specification details the comprehensive intelligence pipeline that transforms raw news data into actionable insights for transparency research.

## Technical Architecture

### Core Components
- **News Intelligence Monitor** (`src/intelligence/news_monitor.py`)
- **Multi-source News Aggregation** (247+ sources)
- **BERT-based Sentiment Analysis** (Fine-tuned transformer)
- **Market Correlation Engine** (Statistical correlation analysis)
- **React Intelligence Dashboard** (`src/pages/Intelligence/Intelligence.tsx`)

### Data Flow Pipeline
```
News Sources → Content Extraction → Entity Recognition → Sentiment Analysis → Market Correlation → Dashboard Display
     ↓               ↓                    ↓                 ↓                    ↓                  ↓
RSS Feeds      Article Parsing      Member/Stock IDs    BERT Scoring     Statistical Tests    Real-time Updates
Web APIs       Content Cleaning     NER Processing      Confidence        Correlation Metrics  Alert Generation
Social Media   Language Detection   Relevance Filter    Score (-1 to +1)  P-value Testing     Export Functions
```

## Implementation Details

### 1. News Source Management

#### 1.1 Multi-Source Aggregation System
**File**: `src/intelligence/news_monitor.py`

**Class Architecture**:
```python
class NewsSourceManager:
    def __init__(self, max_sources=247):
        self.max_sources = max_sources
        self.active_sources = {}
        self.source_reliability = {}
        self.update_frequencies = {}
        
    def initialize_sources(self):
        """Initialize all news sources with configuration"""
        self.sources = {
            'rss_feeds': self._setup_rss_sources(),
            'news_apis': self._setup_api_sources(),
            'social_media': self._setup_social_sources(),
            'government_feeds': self._setup_government_sources()
        }
```

#### 1.2 Source Categories and Configuration

**RSS Feed Sources** (150+ sources):
```python
def _setup_rss_sources(self):
    """Configure RSS feed sources"""
    rss_sources = {
        # Major News Outlets
        'reuters_politics': {
            'url': 'https://feeds.reuters.com/reuters/politicsNews',
            'reliability': 0.95,
            'update_frequency': 300,  # seconds
            'category': 'politics'
        },
        'bloomberg_markets': {
            'url': 'https://feeds.bloomberg.com/markets/news.rss',
            'reliability': 0.93,
            'update_frequency': 180,
            'category': 'market'
        },
        'wsj_politics': {
            'url': 'https://feeds.a.dj.com/rss/RSSPolitics.xml',
            'reliability': 0.92,
            'update_frequency': 240,
            'category': 'politics'
        },
        
        # Financial News
        'financial_times': {
            'url': 'https://www.ft.com/rss/home/us',
            'reliability': 0.91,
            'update_frequency': 300,
            'category': 'market'
        },
        'cnbc_politics': {
            'url': 'https://www.cnbc.com/id/10000113/device/rss/rss.html',
            'reliability': 0.88,
            'update_frequency': 180,
            'category': 'politics'
        },
        
        # Specialized Congressional Coverage
        'politico_congress': {
            'url': 'https://www.politico.com/rss/congress.xml',
            'reliability': 0.89,
            'update_frequency': 120,
            'category': 'politics'
        },
        'the_hill_senate': {
            'url': 'https://thehill.com/rss/syndicator/19109',
            'reliability': 0.86,
            'update_frequency': 180,
            'category': 'politics'
        }
    }
    
    return rss_sources
```

**API-based Sources** (47+ sources):
```python
def _setup_api_sources(self):
    """Configure API-based news sources"""
    api_sources = {
        'newsapi_org': {
            'endpoint': 'https://newsapi.org/v2/everything',
            'api_key': os.getenv('NEWSAPI_KEY'),
            'queries': [
                'congressional trading',
                'stock act disclosure',
                'congress ethics',
                'insider trading congress'
            ],
            'reliability': 0.87,
            'rate_limit': 1000,  # requests per day
            'category': 'mixed'
        },
        'alpha_vantage_news': {
            'endpoint': 'https://www.alphavantage.co/query',
            'api_key': os.getenv('ALPHA_VANTAGE_KEY'),
            'function': 'NEWS_SENTIMENT',
            'reliability': 0.82,
            'rate_limit': 100,
            'category': 'market'
        }
    }
    
    return api_sources
```

**Social Media Monitoring** (25+ sources):
```python
def _setup_social_sources(self):
    """Configure social media monitoring"""
    social_sources = {
        'twitter_api': {
            'endpoint': 'https://api.twitter.com/2/tweets/search/recent',
            'api_key': os.getenv('TWITTER_API_KEY'),
            'queries': [
                '#CongressionalTrading',
                '#STOCKAct',
                'congress insider trading',
                'congressional ethics'
            ],
            'reliability': 0.65,  # Lower reliability for social media
            'rate_limit': 300,
            'category': 'social'
        },
        'reddit_api': {
            'endpoint': 'https://www.reddit.com/r/politics+investing+SecurityAnalysis/.json',
            'reliability': 0.58,
            'rate_limit': 60,
            'category': 'social'
        }
    }
    
    return social_sources
```

**Government Sources** (25+ sources):
```python
def _setup_government_sources(self):
    """Configure official government news sources"""
    government_sources = {
        'house_gov': {
            'url': 'https://www.house.gov/rss.xml',
            'reliability': 0.98,  # Highest reliability for official sources
            'update_frequency': 600,
            'category': 'government'
        },
        'senate_gov': {
            'url': 'https://www.senate.gov/rss/feeds/news.xml',
            'reliability': 0.98,
            'update_frequency': 600,
            'category': 'government'
        },
        'sec_gov': {
            'url': 'https://www.sec.gov/rss/news/press-release-2023.xml',
            'reliability': 0.99,
            'update_frequency': 1800,
            'category': 'regulatory'
        }
    }
    
    return government_sources
```

### 2. Content Processing Pipeline

#### 2.1 Article Extraction and Cleaning
**Purpose**: Extract clean, structured content from various news formats

```python
class ContentProcessor:
    def __init__(self):
        self.html_parser = BeautifulSoup
        self.text_cleaner = TextCleaner()
        self.language_detector = langdetect
        
    def process_article(self, raw_content, source_info):
        """Process raw article content into structured format"""
        try:
            # HTML parsing and content extraction
            if source_info['format'] == 'html':
                content = self._extract_from_html(raw_content)
            elif source_info['format'] == 'rss':
                content = self._extract_from_rss(raw_content)
            elif source_info['format'] == 'json':
                content = self._extract_from_json(raw_content)
            
            # Content cleaning and validation
            cleaned_content = self._clean_content(content)
            
            # Language detection (English only for now)
            if self._detect_language(cleaned_content) != 'en':
                return None
                
            # Content validation
            if not self._validate_content_quality(cleaned_content):
                return None
                
            return {
                'title': cleaned_content['title'],
                'content': cleaned_content['body'],
                'published_at': cleaned_content['publish_date'],
                'author': cleaned_content.get('author'),
                'source': source_info['name'],
                'url': cleaned_content['url'],
                'word_count': len(cleaned_content['body'].split()),
                'language': 'en'
            }
            
        except Exception as e:
            self.logger.error(f"Content processing error: {e}")
            return None
```

#### 2.2 Entity Recognition and Filtering
**Purpose**: Identify congressional members and stock symbols in news content

```python
class EntityRecognizer:
    def __init__(self):
        self.member_names = self._load_congressional_members()
        self.stock_symbols = self._load_stock_symbols()
        self.nlp_model = spacy.load('en_core_web_sm')
        
    def extract_entities(self, article_content):
        """Extract relevant entities from article content"""
        doc = self.nlp_model(article_content['content'])
        
        # Congressional member detection
        detected_members = []
        for member in self.member_names:
            # Full name matching
            if member['full_name'].lower() in article_content['content'].lower():
                detected_members.append({
                    'bioguide_id': member['bioguide_id'],
                    'name': member['full_name'],
                    'confidence': 0.95,
                    'mention_count': article_content['content'].lower().count(member['full_name'].lower())
                })
            
            # Last name matching with context validation
            elif member['last_name'].lower() in article_content['content'].lower():
                # Validate context (should be near political terms)
                if self._validate_political_context(article_content['content'], member['last_name']):
                    detected_members.append({
                        'bioguide_id': member['bioguide_id'],
                        'name': member['full_name'],
                        'confidence': 0.75,
                        'mention_count': article_content['content'].lower().count(member['last_name'].lower())
                    })
        
        # Stock symbol detection
        detected_stocks = []
        for ent in doc.ents:
            if ent.label_ == 'ORG' and ent.text.upper() in self.stock_symbols:
                detected_stocks.append({
                    'symbol': ent.text.upper(),
                    'company_name': self.stock_symbols[ent.text.upper()],
                    'confidence': 0.90,
                    'mention_count': article_content['content'].count(ent.text)
                })
        
        # Manual pattern matching for stock symbols
        import re
        stock_pattern = r'\b[A-Z]{1,5}\b'
        potential_symbols = re.findall(stock_pattern, article_content['content'])
        
        for symbol in potential_symbols:
            if symbol in self.stock_symbols and symbol not in [s['symbol'] for s in detected_stocks]:
                detected_stocks.append({
                    'symbol': symbol,
                    'company_name': self.stock_symbols[symbol],
                    'confidence': 0.80,
                    'mention_count': article_content['content'].count(symbol)
                })
        
        return {
            'members': detected_members,
            'stocks': detected_stocks,
            'relevance_score': self._calculate_relevance_score(detected_members, detected_stocks)
        }
```

### 3. Sentiment Analysis Engine

#### 3.1 BERT-based Model Implementation
**Purpose**: Fine-tuned transformer model for financial and political sentiment analysis

```python
class BERTSentimentAnalyzer:
    def __init__(self, model_name='nlptown/bert-base-multilingual-uncased-sentiment'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def analyze_sentiment(self, text, entities=None):
        """Analyze sentiment with context awareness"""
        # Preprocess text
        processed_text = self._preprocess_text(text)
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Model inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Convert to sentiment score (-1 to +1)
        sentiment_score = self._convert_to_sentiment_score(predictions)
        
        # Context-aware sentiment adjustment
        if entities:
            sentiment_score = self._adjust_for_context(sentiment_score, text, entities)
        
        # Confidence calculation
        confidence = self._calculate_confidence(predictions)
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'sentiment_label': self._get_sentiment_label(sentiment_score),
            'raw_probabilities': predictions.cpu().numpy().tolist()
        }
```

#### 3.2 Context-Aware Sentiment Adjustment
**Purpose**: Adjust sentiment based on congressional and financial context

```python
def _adjust_for_context(self, base_sentiment, text, entities):
    """Adjust sentiment based on contextual factors"""
    adjustment_factors = {
        'investigation_keywords': -0.2,  # More negative for investigations
        'ethics_keywords': -0.15,       # Negative for ethics issues
        'market_positive_keywords': 0.1, # Positive for market growth
        'bipartisan_keywords': 0.05,    # Slightly positive for bipartisan action
        'scandal_keywords': -0.3        # Very negative for scandals
    }
    
    text_lower = text.lower()
    total_adjustment = 0.0
    
    # Investigation context
    investigation_terms = ['investigation', 'probe', 'inquiry', 'subpoena', 'violation']
    if any(term in text_lower for term in investigation_terms):
        total_adjustment += adjustment_factors['investigation_keywords']
    
    # Ethics context
    ethics_terms = ['ethics', 'conflict of interest', 'insider trading', 'disclosure']
    if any(term in text_lower for term in ethics_terms):
        total_adjustment += adjustment_factors['ethics_keywords']
    
    # Market context
    market_positive_terms = ['growth', 'profit', 'gain', 'bullish', 'positive outlook']
    if any(term in text_lower for term in market_positive_terms):
        total_adjustment += adjustment_factors['market_positive_keywords']
    
    # Bipartisan context
    bipartisan_terms = ['bipartisan', 'across the aisle', 'cooperation', 'unity']
    if any(term in text_lower for term in bipartisan_terms):
        total_adjustment += adjustment_factors['bipartisan_keywords']
    
    # Scandal context
    scandal_terms = ['scandal', 'corruption', 'fraud', 'illegal', 'criminal']
    if any(term in text_lower for term in scandal_terms):
        total_adjustment += adjustment_factors['scandal_keywords']
    
    # Entity-specific adjustments
    if entities and entities['members']:
        # More conservative adjustment for high-profile members
        high_profile_members = ['pelosi', 'mccarthy', 'schumer', 'mcconnell']
        for member in entities['members']:
            if any(hp_member in member['name'].lower() for hp_member in high_profile_members):
                total_adjustment *= 0.8  # Reduce adjustment magnitude
    
    # Apply adjustment with bounds
    adjusted_sentiment = base_sentiment + total_adjustment
    return max(-1.0, min(1.0, adjusted_sentiment))
```

### 4. Market Correlation Analysis

#### 4.1 Statistical Correlation Engine
**Purpose**: Analyze correlation between news sentiment and market movements

```python
class MarketCorrelationAnalyzer:
    def __init__(self):
        self.market_data_client = MarketDataClient()
        self.correlation_cache = {}
        self.significance_threshold = 0.05
        
    def analyze_sentiment_market_correlation(self, news_data, lookback_days=30):
        """Analyze correlation between news sentiment and market performance"""
        correlation_results = {}
        
        # Group news by related stocks
        stock_news = self._group_news_by_stocks(news_data)
        
        for stock_symbol, articles in stock_news.items():
            # Get market data for the stock
            market_data = self.market_data_client.get_stock_data(
                stock_symbol, 
                days=lookback_days
            )
            
            if market_data is None:
                continue
            
            # Aggregate daily sentiment scores
            daily_sentiment = self._aggregate_daily_sentiment(articles)
            
            # Align sentiment and market data by date
            aligned_data = self._align_sentiment_market_data(daily_sentiment, market_data)
            
            if len(aligned_data) < 10:  # Minimum data points for correlation
                continue
            
            # Calculate correlations
            correlations = self._calculate_correlations(aligned_data)
            
            # Statistical significance testing
            significance_results = self._test_correlation_significance(aligned_data)
            
            correlation_results[stock_symbol] = {
                'correlations': correlations,
                'significance': significance_results,
                'sample_size': len(aligned_data),
                'date_range': {
                    'start': min(aligned_data.keys()),
                    'end': max(aligned_data.keys())
                }
            }
        
        return correlation_results
```

#### 4.2 Advanced Correlation Metrics
**Purpose**: Multiple correlation measures for comprehensive analysis

```python
def _calculate_correlations(self, aligned_data):
    """Calculate multiple correlation metrics"""
    dates = sorted(aligned_data.keys())
    sentiment_scores = [aligned_data[date]['sentiment'] for date in dates]
    price_returns = [aligned_data[date]['price_return'] for date in dates]
    volume_changes = [aligned_data[date]['volume_change'] for date in dates]
    
    correlations = {
        # Pearson correlation (linear relationship)
        'sentiment_price_pearson': stats.pearsonr(sentiment_scores, price_returns)[0],
        'sentiment_volume_pearson': stats.pearsonr(sentiment_scores, volume_changes)[0],
        
        # Spearman correlation (monotonic relationship)
        'sentiment_price_spearman': stats.spearmanr(sentiment_scores, price_returns)[0],
        'sentiment_volume_spearman': stats.spearmanr(sentiment_scores, volume_changes)[0],
        
        # Kendall's tau (rank correlation)
        'sentiment_price_kendall': stats.kendalltau(sentiment_scores, price_returns)[0],
        
        # Lead-lag correlations
        'sentiment_leads_price': self._calculate_lead_lag_correlation(
            sentiment_scores, price_returns, lag=1
        ),
        'price_leads_sentiment': self._calculate_lead_lag_correlation(
            price_returns, sentiment_scores, lag=1
        )
    }
    
    return correlations

def _calculate_lead_lag_correlation(self, series1, series2, lag=1):
    """Calculate correlation with time lag"""
    if lag > 0:
        # series1 leads series2
        series1_lagged = series1[:-lag]
        series2_lagged = series2[lag:]
    else:
        # series2 leads series1
        series1_lagged = series1[-lag:]
        series2_lagged = series2[:lag]
    
    if len(series1_lagged) < 5:  # Minimum for meaningful correlation
        return None
    
    correlation, p_value = stats.pearsonr(series1_lagged, series2_lagged)
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'lag_days': lag,
        'sample_size': len(series1_lagged)
    }
```

### 5. Real-time Processing Pipeline

#### 5.1 Asynchronous News Collection
**Purpose**: Continuously collect and process news from all sources

```python
import asyncio
import aiohttp

class RealTimeNewsProcessor:
    def __init__(self, max_concurrent_requests=50):
        self.max_concurrent_requests = max_concurrent_requests
        self.processing_queue = asyncio.Queue()
        self.active_tasks = set()
        
    async def start_real_time_processing(self):
        """Start continuous news processing"""
        # Start collector tasks for different source types
        collector_tasks = [
            asyncio.create_task(self._rss_collector()),
            asyncio.create_task(self._api_collector()),
            asyncio.create_task(self._social_collector())
        ]
        
        # Start processor tasks
        processor_tasks = [
            asyncio.create_task(self._process_news_queue())
            for _ in range(self.max_concurrent_requests)
        ]
        
        # Run all tasks concurrently
        await asyncio.gather(*collector_tasks, *processor_tasks)
    
    async def _rss_collector(self):
        """Continuously collect RSS feed updates"""
        while True:
            for source_name, source_config in self.rss_sources.items():
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(source_config['url']) as response:
                            if response.status == 200:
                                content = await response.text()
                                await self.processing_queue.put({
                                    'type': 'rss',
                                    'source': source_name,
                                    'content': content,
                                    'timestamp': datetime.utcnow()
                                })
                except Exception as e:
                    self.logger.error(f"RSS collection error for {source_name}: {e}")
                
                # Wait according to source update frequency
                await asyncio.sleep(source_config['update_frequency'])
    
    async def _process_news_queue(self):
        """Process news items from the queue"""
        while True:
            try:
                news_item = await self.processing_queue.get()
                
                # Process the news item
                processed_article = await self._process_single_article(news_item)
                
                if processed_article:
                    # Store in database
                    await self._store_processed_article(processed_article)
                    
                    # Check for alerts
                    await self._check_for_alerts(processed_article)
                    
                    # Update dashboard
                    await self._update_dashboard(processed_article)
                
                self.processing_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"News processing error: {e}")
```

#### 5.2 Alert Generation System
**Purpose**: Generate alerts for significant news events and sentiment changes

```python
class AlertSystem:
    def __init__(self):
        self.alert_thresholds = {
            'high_impact_sentiment': 0.8,    # |sentiment| > 0.8
            'high_correlation': 0.7,         # |correlation| > 0.7
            'volume_spike': 2.0,             # 2x normal volume
            'breaking_news_keywords': ['investigation', 'indictment', 'resignation']
        }
        
    async def check_for_alerts(self, processed_article):
        """Check if article triggers any alerts"""
        alerts = []
        
        # High-impact sentiment alert
        if abs(processed_article['sentiment_score']) > self.alert_thresholds['high_impact_sentiment']:
            alerts.append({
                'type': 'high_impact_sentiment',
                'severity': 'high' if abs(processed_article['sentiment_score']) > 0.9 else 'medium',
                'message': f"High sentiment impact detected: {processed_article['sentiment_score']:.2f}",
                'article_id': processed_article['id'],
                'related_members': processed_article['entities']['members'],
                'related_stocks': processed_article['entities']['stocks']
            })
        
        # Breaking news keywords
        for keyword in self.alert_thresholds['breaking_news_keywords']:
            if keyword.lower() in processed_article['content'].lower():
                alerts.append({
                    'type': 'breaking_news',
                    'severity': 'high',
                    'message': f"Breaking news keyword detected: {keyword}",
                    'keyword': keyword,
                    'article_id': processed_article['id'],
                    'related_members': processed_article['entities']['members']
                })
        
        # Market correlation alert
        for stock in processed_article['entities']['stocks']:
            correlation = await self._get_recent_correlation(stock['symbol'])
            if correlation and abs(correlation) > self.alert_thresholds['high_correlation']:
                alerts.append({
                    'type': 'high_correlation',
                    'severity': 'medium',
                    'message': f"High sentiment-market correlation: {correlation:.2f}",
                    'stock_symbol': stock['symbol'],
                    'correlation': correlation,
                    'article_id': processed_article['id']
                })
        
        # Send alerts
        for alert in alerts:
            await self._send_alert(alert)
        
        return alerts
```

### 6. Dashboard Integration

#### 6.1 React Intelligence Page
**File**: `src/pages/Intelligence/Intelligence.tsx`

**Key Features**:
- **Real-time News Feed**: Live updates from 247+ sources
- **Sentiment Analysis Display**: Visual sentiment indicators and trends
- **Market Correlation Dashboard**: Statistical correlation metrics
- **Advanced Filtering**: Multi-dimensional filtering system
- **Export Capabilities**: Research data export functionality

**Component Structure**:
```tsx
const Intelligence: React.FC = () => {
  const [newsItems, setNewsItems] = useState<NewsItem[]>([]);
  const [sentimentTrends, setSentimentTrends] = useState<SentimentTrend[]>([]);
  const [marketCorrelations, setMarketCorrelations] = useState<MarketCorrelation[]>([]);
  
  // Tab system for different intelligence views
  const [tabValue, setTabValue] = useState(0);
  
  // Real-time data loading
  useEffect(() => {
    const loadIntelligenceData = async () => {
      const data = await fetchIntelligenceData();
      setNewsItems(data.news);
      setSentimentTrends(data.sentiment);
      setMarketCorrelations(data.correlations);
    };
    
    loadIntelligenceData();
    
    // Set up real-time updates
    const interval = setInterval(loadIntelligenceData, 30000); // 30 seconds
    return () => clearInterval(interval);
  }, []);
  
  return (
    <Box>
      {/* Summary metrics */}
      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard 
            title="News Articles" 
            value={newsItems.length} 
            subtitle="This week" 
          />
        </Grid>
        {/* Additional metric cards */}
      </Grid>
      
      {/* Tabbed interface */}
      <Tabs value={tabValue} onChange={handleTabChange}>
        <Tab label="News Feed" />
        <Tab label="Sentiment Analysis" />
        <Tab label="Market Correlations" />
      </Tabs>
      
      {/* Tab panels with content */}
      <TabPanel value={tabValue} index={0}>
        <NewsFeedComponent newsItems={filteredNews} />
      </TabPanel>
      
      <TabPanel value={tabValue} index={1}>
        <SentimentAnalysisComponent trends={sentimentTrends} />
      </TabPanel>
      
      <TabPanel value={tabValue} index={2}>
        <MarketCorrelationComponent correlations={marketCorrelations} />
      </TabPanel>
    </Box>
  );
};
```

#### 6.2 Advanced Filtering System
**Purpose**: Multi-dimensional filtering for news intelligence

```tsx
const useIntelligenceFilters = () => {
  const [filters, setFilters] = useState({
    search: '',
    category: 'all',
    sentiment: 'all',
    dateRange: {
      start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
      end: new Date()
    },
    members: [],
    stocks: [],
    sources: [],
    minConfidence: 0.5
  });
  
  const applyFilters = useCallback((newsItems: NewsItem[]) => {
    return newsItems.filter(item => {
      // Search filter
      if (filters.search && 
          !item.title.toLowerCase().includes(filters.search.toLowerCase()) &&
          !item.content.toLowerCase().includes(filters.search.toLowerCase())) {
        return false;
      }
      
      // Category filter
      if (filters.category !== 'all' && item.category !== filters.category) {
        return false;
      }
      
      // Sentiment filter
      if (filters.sentiment !== 'all' && item.sentiment !== filters.sentiment) {
        return false;
      }
      
      // Date range filter
      const itemDate = new Date(item.published_at);
      if (itemDate < filters.dateRange.start || itemDate > filters.dateRange.end) {
        return false;
      }
      
      // Member filter
      if (filters.members.length > 0) {
        const itemMembers = item.entities.members.map(m => m.bioguide_id);
        if (!filters.members.some(member => itemMembers.includes(member))) {
          return false;
        }
      }
      
      // Stock filter
      if (filters.stocks.length > 0) {
        const itemStocks = item.entities.stocks.map(s => s.symbol);
        if (!filters.stocks.some(stock => itemStocks.includes(stock))) {
          return false;
        }
      }
      
      // Confidence filter
      if (item.sentiment_confidence < filters.minConfidence) {
        return false;
      }
      
      return true;
    });
  }, [filters]);
  
  return { filters, setFilters, applyFilters };
};
```

### 7. Performance Optimization

#### 7.1 Caching Strategy
**Purpose**: Optimize performance for high-volume news processing

```python
class IntelligenceCacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = {
            'sentiment_analysis': 3600,    # 1 hour
            'market_correlation': 1800,    # 30 minutes
            'news_content': 86400,         # 24 hours
            'entity_recognition': 7200     # 2 hours
        }
    
    def get_cached_sentiment(self, content_hash):
        """Get cached sentiment analysis result"""
        cache_key = f"sentiment:{content_hash}"
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        return None
    
    def cache_sentiment_result(self, content_hash, sentiment_result):
        """Cache sentiment analysis result"""
        cache_key = f"sentiment:{content_hash}"
        self.redis_client.setex(
            cache_key,
            self.cache_ttl['sentiment_analysis'],
            json.dumps(sentiment_result)
        )
    
    def get_cached_correlation(self, stock_symbol, timeframe):
        """Get cached market correlation data"""
        cache_key = f"correlation:{stock_symbol}:{timeframe}"
        cached_result = self.redis_client.get(cache_key)
        
        if cached_result:
            return json.loads(cached_result)
        return None
```

#### 7.2 Batch Processing Optimization
**Purpose**: Efficient processing of large news volumes

```python
class BatchProcessor:
    def __init__(self, batch_size=100):
        self.batch_size = batch_size
        self.processing_pool = ProcessPoolExecutor(max_workers=8)
        
    async def process_news_batch(self, news_items):
        """Process news items in optimized batches"""
        batches = [
            news_items[i:i + self.batch_size] 
            for i in range(0, len(news_items), self.batch_size)
        ]
        
        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_single_batch(batch))
            tasks.append(task)
        
        # Wait for all batches to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        processed_items = []
        for batch_result in results:
            processed_items.extend(batch_result)
        
        return processed_items
    
    async def _process_single_batch(self, batch):
        """Process a single batch of news items"""
        loop = asyncio.get_event_loop()
        
        # Run CPU-intensive processing in thread pool
        processed_batch = await loop.run_in_executor(
            self.processing_pool,
            self._cpu_intensive_processing,
            batch
        )
        
        return processed_batch
```

### 8. Quality Assurance and Validation

#### 8.1 Data Quality Monitoring
**Purpose**: Ensure high-quality intelligence data

```python
class DataQualityMonitor:
    def __init__(self):
        self.quality_metrics = {
            'content_length_min': 100,     # Minimum article length
            'sentiment_confidence_min': 0.6, # Minimum sentiment confidence
            'entity_detection_min': 1,      # Minimum entities detected
            'source_reliability_min': 0.7   # Minimum source reliability
        }
    
    def validate_article_quality(self, processed_article):
        """Validate processed article meets quality standards"""
        quality_score = 0.0
        issues = []
        
        # Content length check
        if len(processed_article['content']) >= self.quality_metrics['content_length_min']:
            quality_score += 0.2
        else:
            issues.append('Content too short')
        
        # Sentiment confidence check
        if processed_article['sentiment_confidence'] >= self.quality_metrics['sentiment_confidence_min']:
            quality_score += 0.2
        else:
            issues.append('Low sentiment confidence')
        
        # Entity detection check
        total_entities = (len(processed_article['entities']['members']) + 
                         len(processed_article['entities']['stocks']))
        if total_entities >= self.quality_metrics['entity_detection_min']:
            quality_score += 0.2
        else:
            issues.append('Insufficient entity detection')
        
        # Source reliability check
        if processed_article['source_reliability'] >= self.quality_metrics['source_reliability_min']:
            quality_score += 0.2
        else:
            issues.append('Low source reliability')
        
        # Relevance score check
        if processed_article['relevance_score'] >= 0.5:
            quality_score += 0.2
        else:
            issues.append('Low relevance score')
        
        return {
            'quality_score': quality_score,
            'passes_quality_check': quality_score >= 0.6,
            'issues': issues
        }
```

#### 8.2 Correlation Validation
**Purpose**: Validate statistical significance of market correlations

```python
def validate_correlation_significance(self, correlation_data, min_sample_size=20):
    """Validate statistical significance of correlations"""
    validation_results = {}
    
    for stock_symbol, data in correlation_data.items():
        sample_size = data['sample_size']
        correlations = data['correlations']
        
        # Sample size validation
        if sample_size < min_sample_size:
            validation_results[stock_symbol] = {
                'valid': False,
                'reason': f'Insufficient sample size: {sample_size} < {min_sample_size}'
            }
            continue
        
        # Statistical significance testing
        significant_correlations = {}
        for corr_type, corr_value in correlations.items():
            if corr_value is None:
                continue
                
            # Calculate p-value for correlation
            t_statistic = corr_value * np.sqrt((sample_size - 2) / (1 - corr_value**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), sample_size - 2))
            
            significant_correlations[corr_type] = {
                'correlation': corr_value,
                'p_value': p_value,
                'significant': p_value < 0.05,
                't_statistic': t_statistic
            }
        
        validation_results[stock_symbol] = {
            'valid': True,
            'sample_size': sample_size,
            'correlations': significant_correlations,
            'overall_significance': any(
                corr['significant'] for corr in significant_correlations.values()
            )
        }
    
    return validation_results
```

This comprehensive intelligence and monitoring specification provides the foundation for real-time congressional trading intelligence, combining advanced NLP, statistical analysis, and market correlation tracking to deliver actionable insights for transparency research.