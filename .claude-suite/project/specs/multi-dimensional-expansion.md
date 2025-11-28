# Multi-Dimensional Intelligence Platform - Expansion Specification

## Executive Summary
Transform the Congressional Trading Intelligence System into a **multi-dimensional influence intelligence platform** that tracks, correlates, and analyzes influence patterns across political, social, corporate, and retail investor dimensions.

**Core Thesis:** Congressional trading is one signal in a complex web of influence, information flow, and market manipulation. By adding social media, Reddit, corporate insiders, and other data sources, we can detect:
- **Information asymmetry** (who knows what, when)
- **Influence patterns** (who influences whom)
- **Coordinated behavior** (groups acting together)
- **Retail vs. institutional dynamics** (predatory behavior detection)

---

## 🎯 Vision: From Single to Multi-Dimensional Intelligence

### Current State (1 Dimension)
```
Congressional Trading ↔ Market Data
```

### Target State (7+ Dimensions)
```
┌─────────────────────────────────────────────────────────────┐
│                  INFLUENCE INTELLIGENCE HUB                  │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Congressional│  │   Corporate  │  │    Retail    │     │
│  │   Members    │  │   Insiders   │  │   Investors  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│  ┌──────▼──────────────────▼──────────────────▼───────┐    │
│  │           CORRELATION ENGINE                        │    │
│  │  - Timing Analysis    - Influence Scoring           │    │
│  │  - Pattern Detection  - Network Analysis            │    │
│  └──────┬──────────────────┬──────────────────┬───────┘    │
│         │                  │                  │              │
│  ┌──────▼───────┐  ┌──────▼───────┐  ┌──────▼───────┐     │
│  │Social Media  │  │    Reddit    │  │   Lobbying   │     │
│  │(Twitter/X)   │  │  Sentiment   │  │   Spending   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dimension 1: Social Media Intelligence (Twitter/X)

### Data to Track
1. **Congressional Member Activity**
   - Tweet content and timing
   - Mentions of companies/sectors
   - Sentiment toward industries
   - Engagement metrics (likes, retweets, replies)
   - Follower demographics

2. **Corporate Executive Activity**
   - CEO/CFO tweets about company performance
   - Pre-announcement messaging
   - Crisis communication timing
   - Coordinated executive messaging

3. **Media Personality Influence**
   - Financial media mentions (Cramer, financial pundits)
   - Sector enthusiasm/FUD creation
   - Pump-and-dump patterns

### Analysis Capabilities

#### A. **Tweet-to-Trade Timing Analysis**
```python
# Detect: Member tweets positively about sector → Trades in sector
class TweetTradingCorrelator:
    def analyze_tweet_trade_timing(self, member):
        """
        Detect suspicious patterns:
        1. Negative tweet → Sell trade (within 48 hours)
        2. Positive tweet → Buy trade (within 48 hours)
        3. Silence before major trade → Hiding intentions
        4. Tweet storm after trade → Justifying after the fact
        """

# Example Detection:
# Senator X tweets: "Big Tech is overvalued and needs regulation"
# 3 days later: Sells $500K in GOOGL, META, AMZN
# Suspicion Score: 85/100
```

#### B. **Coordinated Messaging Detection**
```python
class CoordinatedMessagingDetector:
    def detect_coordinated_behavior(self, members: List[Member]):
        """
        Detect when multiple members tweet similar messages:
        - Hashtag coordination
        - Talking point synchronization
        - Simultaneous sector criticism/praise
        - Cross-party coordination (unusual alliances)
        """

# Example:
# 5 senators from different committees tweet about China threat
# All 5 buy defense stocks within 7 days
# Network analysis reveals coordinated campaign
```

#### C. **Sentiment-Trade Divergence**
```python
class SentimentDivergenceAnalyzer:
    def detect_public_vs_private_sentiment(self, member):
        """
        Compare public messaging to private trading:
        - Says "bullish on American manufacturing"
        - Sells all domestic manufacturing stocks
        - Hypocrisy score: 95/100
        """
```

### Data Sources & APIs
- **Twitter/X API v2** (paid tier for historical data)
  - `/tweets/search/recent` - Recent tweets
  - `/users/{id}/tweets` - User timeline
  - `/tweets/{id}/metrics` - Engagement data

- **Alternative: Nitter Scraping** (free but rate-limited)
- **Archive Services:** Wayback Machine, Politwoops (deleted tweets)

### Database Schema Extension
```python
# New models in src/models/social_media.py
class SocialMediaPost(Base, TimestampMixin, IDMixin):
    __tablename__ = "social_media_posts"

    member_id = Column(Integer, ForeignKey("members.id"))
    platform = Column(String)  # twitter, truth_social, etc.
    post_id = Column(String, unique=True)
    content = Column(Text)
    posted_at = Column(DateTime)
    engagement_score = Column(Float)  # Likes + RTs + Replies

    # Extracted metadata
    mentioned_companies = Column(ARRAY(String))  # GOOGL, TSLA
    mentioned_sectors = Column(ARRAY(String))  # tech, energy
    sentiment_score = Column(Float)  # -1 to +1

    # Relationships
    member = relationship("Member")
    related_trades = relationship("Trade", secondary="social_trade_correlations")

class SocialTradeCorrelation(Base):
    __tablename__ = "social_trade_correlations"

    post_id = Column(Integer, ForeignKey("social_media_posts.id"))
    trade_id = Column(Integer, ForeignKey("trades.id"))

    correlation_type = Column(String)  # "pre-trade", "post-trade", "divergence"
    time_delta_hours = Column(Float)  # Hours between post and trade
    suspicion_score = Column(Float)  # 0-100
    explanation = Column(Text)
```

---

## 📊 Dimension 2: Reddit Intelligence

### Why Reddit?
- **Retail investor sentiment** (r/wallstreetbets, r/stocks, r/investing)
- **Early signals** before mainstream media coverage
- **Coordinated movements** (GME, AMC short squeezes)
- **Political discussion** (r/politics for congressional sentiment)

### Data to Track

1. **Subreddit Activity**
   - Post volume by ticker symbol
   - Sentiment trends (bullish/bearish)
   - "DD" (due diligence) posts quality
   - Meme stock identification

2. **Congressional-Reddit Timing**
   - Does congress trade *against* WSB sentiment? (predatory)
   - Does congress trade *with* WSB? (following retail)
   - Do trades precede Reddit discussions? (insider info)

3. **Coordinated Pump-and-Dump Detection**
   - Sudden Reddit hype + Congressional trade = Suspicious
   - Pattern: Member buys → Reddit campaign → Member sells

### Analysis Capabilities

#### A. **Retail Sentiment Correlation**
```python
class RetailSentimentAnalyzer:
    def analyze_congress_vs_retail(self, symbol: str, timeframe: int):
        """
        Detect predatory behavior:
        1. Reddit: Bullish sentiment rising
        2. Congress: Selling the same stock
        3. Interpretation: Dumping on retail investors

        Or insider advantage:
        1. Congress: Buying a stock heavily
        2. Reddit: No discussion (yet)
        3. One week later: Reddit explodes with hype
        4. Interpretation: Congress knew before retail
        """
```

#### B. **Meme Stock Congressional Behavior**
```python
class MemeStockTracker:
    def track_meme_stock_trades(self):
        """
        Track congressional trades in r/wsb favorites:
        - GME, AMC, BBBY (classic memes)
        - NVDA, TSLA (current favorites)

        Generate alerts:
        - "Senator X bought GME 3 days before WSB surge"
        - "Rep Y sold AMC at peak Reddit enthusiasm"
        """
```

#### C. **Information Advantage Scoring**
```python
class InformationAdvantageDetector:
    def calculate_information_advantage(self, trade: Trade):
        """
        Score formula:
        - Trade before Reddit discussion: +30 points
        - Trade against Reddit sentiment: +20 points (if profitable)
        - Trade matches Reddit but earlier: +25 points
        - Exit at Reddit sentiment peak: +40 points (dump on retail)

        Information Advantage Score: 0-100
        """
```

### Data Sources

1. **Reddit API (PRAW - Python Reddit API Wrapper)**
   ```python
   # Free tier with rate limits
   import praw

   reddit = praw.Reddit(
       client_id="YOUR_CLIENT_ID",
       client_secret="YOUR_SECRET",
       user_agent="CongressionalTrading/1.0"
   )

   # Track specific subreddits
   subreddits = ['wallstreetbets', 'stocks', 'investing', 'politics', 'StockMarket']
   ```

2. **Pushshift API** (Historical Reddit data)
   - Archive of all Reddit posts/comments
   - Time-based queries
   - Sentiment analysis of historical discussions

3. **Alternative:** Web scraping with `BeautifulSoup` + `requests`

### Database Schema Extension
```python
class RedditPost(Base, TimestampMixin, IDMixin):
    __tablename__ = "reddit_posts"

    post_id = Column(String, unique=True)
    subreddit = Column(String, index=True)
    title = Column(Text)
    content = Column(Text)
    author = Column(String)
    posted_at = Column(DateTime, index=True)
    score = Column(Integer)  # Upvotes - Downvotes
    num_comments = Column(Integer)

    # Extracted metadata
    mentioned_tickers = Column(ARRAY(String), index=True)
    sentiment_score = Column(Float)  # -1 to +1
    is_dd_post = Column(Boolean)  # Due Diligence post
    hype_score = Column(Float)  # 0-100 based on engagement + sentiment

class RedditTickerSentiment(Base):
    """Aggregated daily sentiment per ticker"""
    __tablename__ = "reddit_ticker_sentiment"

    ticker = Column(String, index=True)
    date = Column(Date, index=True)

    # Aggregated metrics
    post_count = Column(Integer)
    total_score = Column(Integer)  # Sum of upvotes
    avg_sentiment = Column(Float)
    hype_level = Column(String)  # low, medium, high, extreme

    # Most influential posts
    top_posts = Column(JSON)  # List of top 5 post IDs

class CongressRedditCorrelation(Base):
    """Links congressional trades to Reddit activity"""
    __tablename__ = "congress_reddit_correlations"

    trade_id = Column(Integer, ForeignKey("trades.id"))
    ticker = Column(String, index=True)

    # Reddit metrics at time of trade
    reddit_sentiment_at_trade = Column(Float)
    reddit_hype_level = Column(String)
    reddit_post_count_7d = Column(Integer)  # Posts in prior 7 days

    # Timing analysis
    time_delta_vs_reddit_peak = Column(Integer)  # Days before/after Reddit peak
    traded_before_reddit_hype = Column(Boolean)
    traded_against_sentiment = Column(Boolean)

    # Scoring
    information_advantage_score = Column(Float)  # 0-100
    predatory_behavior_score = Column(Float)  # 0-100
```

---

## 📊 Dimension 3: Corporate Insider Trading

### Why Track Corporate Insiders?
- **Compare patterns:** Do congressional members trade like insiders?
- **Coordination detection:** Do insiders and congress trade simultaneously?
- **Market manipulation:** Cross-dimensional pump-and-dump schemes

### Data to Track
1. **SEC Form 4 Filings** (corporate insider trades)
   - CEO, CFO, directors, 10%+ owners
   - Transaction types (purchase, sale, option exercise)
   - Filing delays (same 45-day rule as congress)

2. **SEC Form 144** (planned insider sales)
   - Pre-announced sale plans (10b5-1)
   - Detect deviations from plans

### Analysis Capabilities

#### A. **Congress-Insider Coordination**
```python
class CongressInsiderCorrelator:
    def detect_coordinated_trading(self, symbol: str):
        """
        Pattern detection:
        1. Corporate insider sells heavily
        2. Congressional member sells same stock (within 7 days)
        3. Suspicion: Shared information or coordination

        Or:
        1. Congressional member buys
        2. Company announces major contract (in member's oversight area)
        3. Corporate insiders sell at peak
        4. Congressional member exits with insiders
        5. Suspicion: Coordinated pump-and-dump
        """
```

#### B. **Insider Trading Pattern Comparison**
```python
class InsiderPatternComparator:
    def compare_trading_patterns(self, member: Member, symbol: str):
        """
        Compare member's trading pattern to corporate insiders:
        - Do they trade at similar times?
        - Do they exit before bad news?
        - Do they enter before good news?

        If patterns match closely → Possible information sharing
        """
```

### Data Sources
- **SEC EDGAR API** (free, official)
  - Form 4: Insider trades
  - Form 144: Planned sales
  - 13F: Institutional holdings (quarterly)

- **Commercial APIs:**
  - OpenInsider.com (parsed data)
  - Finviz (insider trading screener)
  - QuiverQuant API (paid, aggregated)

### Database Schema
```python
class CorporateInsider(Base, TimestampMixin, IDMixin):
    __tablename__ = "corporate_insiders"

    name = Column(String)
    company_ticker = Column(String, index=True)
    position = Column(String)  # CEO, CFO, Director, etc.
    is_director = Column(Boolean)
    is_officer = Column(Boolean)
    is_ten_percent_owner = Column(Boolean)

class InsiderTrade(Base, TimestampMixin, IDMixin):
    __tablename__ = "insider_trades"

    insider_id = Column(Integer, ForeignKey("corporate_insiders.id"))
    ticker = Column(String, index=True)
    transaction_type = Column(String)
    shares = Column(Integer)
    price_per_share = Column(Numeric)
    transaction_date = Column(Date, index=True)
    filing_date = Column(Date)
    sec_form_url = Column(String)

    # Relationships
    insider = relationship("CorporateInsider")

class CongressInsiderCorrelation(Base):
    __tablename__ = "congress_insider_correlations"

    congress_trade_id = Column(Integer, ForeignKey("trades.id"))
    insider_trade_id = Column(Integer, ForeignKey("insider_trades.id"))

    time_delta_days = Column(Integer)
    same_direction = Column(Boolean)  # Both buy or both sell
    coordination_score = Column(Float)  # 0-100
```

---

## 📊 Dimension 4: Lobbying Intelligence

### Data to Track
1. **Lobbying Disclosure Reports** (LDA)
   - Who is lobbying whom
   - Issue areas (healthcare, defense, tech)
   - Spending amounts
   - Lobbyist-client relationships

2. **Congress-Lobbying Correlation**
   - Does member trade in sectors they're lobbied on?
   - Does lobbying spending correlate with favorable trades?
   - Do members trade before/after lobbying meetings?

### Analysis Capabilities
```python
class LobbyingTradingCorrelator:
    def analyze_lobbying_influence(self, member: Member):
        """
        Pattern detection:
        1. Tech companies spend $500K lobbying member
        2. Member sits on tech oversight committee
        3. Member buys tech stocks heavily
        4. Member votes favorably on tech bills

        Corruption Score: High
        """
```

### Data Sources
- **Senate Lobbying Database** (free, public)
- **OpenSecrets.org API** (LDA data)
- **ProPublica Congress API** (lobbying + voting records)

---

## 📊 Dimension 5: Campaign Finance

### Data to Track
1. **FEC Campaign Contributions**
   - PAC donations
   - Individual contributions (itemized $200+)
   - Super PAC spending

2. **Donor-Trading Correlation**
   - Do members trade in donor company stocks?
   - Do contributions influence votes + trades?

### Analysis Capabilities
```python
class CampaignFinanceAnalyzer:
    def detect_quid_pro_quo_patterns(self, member: Member):
        """
        Pattern:
        1. Defense contractor donates $50K to member's campaign
        2. Member buys defense contractor stock
        3. Member votes for defense spending increase
        4. Stock rises 20%
        5. Member sells at peak

        Quid Pro Quo Score: 90/100
        """
```

---

## 🏗️ Technical Architecture for Multi-Dimensional Platform

### 1. Unified Data Model: "Actor-Event-Correlation" Pattern

```python
# Base Actor Model (People/Entities that take actions)
class Actor(Base):
    __tablename__ = "actors"

    id = Column(Integer, primary_key=True)
    actor_type = Column(String)  # congressional_member, corporate_insider,
                                 # media_personality, retail_investor_group
    name = Column(String)

    # Polymorphic relationship to specific actor types
    __mapper_args__ = {
        'polymorphic_identity': 'actor',
        'polymorphic_on': actor_type
    }

class CongressionalMemberActor(Actor):
    __mapper_args__ = {'polymorphic_identity': 'congressional_member'}
    member_id = Column(Integer, ForeignKey("members.id"))
    member = relationship("Member")

class CorporateInsiderActor(Actor):
    __mapper_args__ = {'polymorphic_identity': 'corporate_insider'}
    insider_id = Column(Integer, ForeignKey("corporate_insiders.id"))
    insider = relationship("CorporateInsider")

# Unified Event Timeline
class Event(Base, TimestampMixin):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    event_type = Column(String)  # trade, tweet, reddit_post, lobbying_meeting,
                                 # vote, announcement, filing
    event_timestamp = Column(DateTime, index=True)
    actor_id = Column(Integer, ForeignKey("actors.id"))

    # Polymorphic to specific event types
    actor = relationship("Actor")

    __mapper_args__ = {
        'polymorphic_identity': 'event',
        'polymorphic_on': event_type
    }

class TradeEvent(Event):
    __mapper_args__ = {'polymorphic_identity': 'trade'}
    trade_id = Column(Integer, ForeignKey("trades.id"))
    trade = relationship("Trade")

class SocialMediaEvent(Event):
    __mapper_args__ = {'polymorphic_identity': 'social_media_post'}
    post_id = Column(Integer, ForeignKey("social_media_posts.id"))
    post = relationship("SocialMediaPost")

class RedditEvent(Event):
    __mapper_args__ = {'polymorphic_identity': 'reddit_activity'}
    # Aggregated Reddit activity for a ticker on a specific day
    ticker = Column(String)
    sentiment_shift = Column(Float)
    hype_level_change = Column(String)

# Cross-Dimensional Correlations
class EventCorrelation(Base):
    __tablename__ = "event_correlations"

    id = Column(Integer, primary_key=True)

    # The two events being correlated
    event1_id = Column(Integer, ForeignKey("events.id"))
    event2_id = Column(Integer, ForeignKey("events.id"))

    # Correlation metadata
    correlation_type = Column(String)  # timing, coordination, divergence, predatory
    time_delta_hours = Column(Float)
    correlation_strength = Column(Float)  # 0-100

    # Analysis results
    suspicion_score = Column(Float)  # 0-100
    explanation = Column(Text)
    detected_by = Column(String)  # Which analyzer detected this

    # Relationships
    event1 = relationship("Event", foreign_keys=[event1_id])
    event2 = relationship("Event", foreign_keys=[event2_id])
```

### 2. Modular Data Source Architecture

```
src/data_sources/
├── __init__.py
├── base_scraper.py           # Abstract base class for all scrapers
├── congressional/
│   ├── house_scraper.py
│   └── senate_scraper.py
├── social_media/
│   ├── twitter_scraper.py
│   ├── truth_social_scraper.py
│   └── sentiment_analyzer.py
├── reddit/
│   ├── reddit_scraper.py
│   ├── subreddit_tracker.py
│   └── ticker_extractor.py
├── corporate/
│   ├── sec_edgar_scraper.py
│   ├── form4_parser.py
│   └── insider_tracker.py
├── lobbying/
│   ├── senate_lda_scraper.py
│   └── opensecrets_client.py
└── campaign_finance/
    └── fec_scraper.py
```

### 3. Unified Intelligence Engine

```python
# src/intelligence/multi_dimensional_analyzer.py

class MultiDimensionalAnalyzer:
    """
    Analyzes correlations across ALL dimensions simultaneously
    """

    def analyze_member_influence_network(self, member: Member) -> InfluenceNetwork:
        """
        Build complete influence graph for a member:

        1. Congressional Trading Dimension
           - What they trade, when, amounts

        2. Social Media Dimension
           - What they tweet about
           - Sentiment toward sectors

        3. Reddit Dimension
           - Are they trading with or against retail?

        4. Lobbying Dimension
           - Who lobbies them
           - Do they trade in lobbied sectors?

        5. Campaign Finance Dimension
           - Who funds them
           - Do they trade in donor stocks?

        6. Legislative Dimension
           - What bills they sponsor/vote on
           - Do trades correlate with votes?

        7. Corporate Insider Dimension
           - Do they trade like insiders?
           - Coordinated behavior?

        Returns: Network graph showing all influences and correlations
        """

    def detect_cross_dimensional_patterns(self, timeframe: int) -> List[Pattern]:
        """
        Detect complex patterns across dimensions:

        Example Pattern: "The Coordinated Pump"
        1. Corporate insider starts buying (Dimension 3)
        2. Member tweets positively about sector (Dimension 1)
        3. Reddit hype increases (Dimension 2)
        4. Member buys stock (Congressional Trading)
        5. Good news announced
        6. Reddit reaches peak hype (Dimension 2)
        7. Corporate insider sells (Dimension 3)
        8. Member sells (Congressional Trading)
        9. Member stops tweeting about sector (Dimension 1)

        Pattern Detected: Coordinated pump-and-dump
        Suspicion Score: 95/100
        """
```

### 4. API Endpoint Structure

```
/api/v2/
├── /actors                           # All actors (members, insiders, etc.)
│   ├── /congressional-members
│   ├── /corporate-insiders
│   └── /media-personalities
│
├── /events                           # Unified event timeline
│   ├── /trades
│   ├── /social-media
│   ├── /reddit-activity
│   ├── /lobbying-meetings
│   └── /timeline?start=...&end=...  # All events in timeframe
│
├── /correlations                     # Cross-dimensional analysis
│   ├── /social-trading              # Social media ↔ Trading
│   ├── /reddit-trading              # Reddit ↔ Trading
│   ├── /insider-congress            # Insiders ↔ Congress
│   └── /multi-dimensional           # Full network analysis
│
├── /analysis                         # Intelligence reports
│   ├── /influence-networks          # Network graphs
│   ├── /predatory-behavior          # Retail investor protection
│   ├── /information-advantage       # Who knows what, when
│   └── /coordinated-activity        # Group behavior detection
│
└── /dashboards                       # Pre-built dashboard configs
    ├── /congressional-overview      # Current dashboard
    ├── /social-sentiment            # Social media focus
    ├── /retail-protection           # WSB vs Congress
    └── /corruption-index            # Multi-dimensional scoring
```

### 5. Dashboard Architecture

```
src/dashboard/
├── index.html                        # Main entry point
├── css/
│   ├── main.css
│   └── components/
│       ├── timeline.css
│       ├── network-graph.css
│       └── correlation-matrix.css
├── js/
│   ├── app.js                        # Main application
│   ├── components/
│   │   ├── Timeline.js               # Unified event timeline
│   │   ├── NetworkGraph.js           # D3.js influence network
│   │   ├── CorrelationMatrix.js      # Heatmap of correlations
│   │   ├── ActorProfile.js           # Multi-dimensional actor view
│   │   └── PatternDetector.js        # Real-time pattern alerts
│   ├── dimensions/
│   │   ├── CongressionalDimension.js
│   │   ├── SocialMediaDimension.js
│   │   ├── RedditDimension.js
│   │   ├── InsiderDimension.js
│   │   └── LobbyingDimension.js
│   └── services/
│       └── MultiDimensionalAPI.js
```

---

## 🚀 Implementation Roadmap

### Phase 1: Social Media Integration (Weeks 1-3)
**Goal:** Add Twitter/X dimension for congressional members

1. **Week 1: Data Collection**
   - Set up Twitter API v2 access
   - Build `TwitterScraper` in `src/data_sources/social_media/`
   - Create `SocialMediaPost` model
   - Implement sentiment analysis (TextBlob or VADER)
   - Store 30 days of historical tweets for all members

2. **Week 2: Analysis Engine**
   - Build `TweetTradingCorrelator`
   - Implement timing analysis (tweet ↔ trade correlation)
   - Add sentiment divergence detection
   - Create alerts for suspicious patterns

3. **Week 3: Dashboard Integration**
   - Add "Social Media" tab to dashboard
   - Timeline view: Tweets + Trades on same timeline
   - Sentiment charts: Member sentiment over time
   - Correlation alerts: Suspicious tweet-trade patterns

**Success Metrics:**
- 535 members' Twitter accounts tracked
- Real-time tweet ingestion (< 5 min delay)
- 10+ suspicious tweet-trade patterns detected

### Phase 2: Reddit Sentiment (Weeks 4-6)
**Goal:** Track retail investor sentiment and detect predatory behavior

1. **Week 4: Reddit Data Pipeline**
   - Set up Reddit API (PRAW)
   - Build `RedditScraper` for r/wallstreetbets, r/stocks
   - Create `RedditPost` and `RedditTickerSentiment` models
   - Implement ticker extraction (regex + NLP)
   - Calculate daily sentiment aggregates

2. **Week 5: Retail Correlation Analysis**
   - Build `RetailSentimentAnalyzer`
   - Detect congress trading against retail sentiment
   - Identify information advantage patterns
   - Create "predatory behavior" scoring

3. **Week 6: Retail Protection Dashboard**
   - Add "Retail vs Congress" dashboard tab
   - Show tickers where congress trades against WSB
   - Alert retail investors to potential manipulation
   - Leaderboard: "Most Predatory Members"

**Success Metrics:**
- 5 subreddits monitored 24/7
- Daily sentiment scores for 500+ tickers
- Detect 5+ cases of congress dumping on retail

### Phase 3: Corporate Insider Integration (Weeks 7-9)
**Goal:** Add corporate insider dimension and detect coordination

1. **Week 7: SEC EDGAR Integration**
   - Build `SECEdgarScraper` for Form 4 filings
   - Create `CorporateInsider` and `InsiderTrade` models
   - Parse Form 4 XML/HTML
   - Store 2 years of insider trading history

2. **Week 8: Congress-Insider Correlation**
   - Build `CongressInsiderCorrelator`
   - Detect simultaneous trading patterns
   - Identify possible information sharing
   - Score coordination likelihood

3. **Week 9: Insider Dashboard**
   - Add "Insider Coordination" tab
   - Side-by-side: Congress vs Corporate Insiders
   - Coordination alerts
   - Network graph: Who trades with whom

**Success Metrics:**
- 10,000+ corporate insiders tracked
- 100+ companies with insider data
- Detect 10+ coordinated trading patterns

### Phase 4: Multi-Dimensional Intelligence (Weeks 10-12)
**Goal:** Unified analysis across all dimensions

1. **Week 10: Unified Event Timeline**
   - Implement `Actor-Event-Correlation` data model
   - Migrate existing data to new schema
   - Build `MultiDimensionalAnalyzer`
   - Create unified event API

2. **Week 11: Cross-Dimensional Pattern Detection**
   - Implement complex pattern detectors:
     - Coordinated pump-and-dump
     - Information cascade detection
     - Influence network analysis
   - Build ML models for pattern recognition

3. **Week 12: Ultimate Intelligence Dashboard**
   - Completely redesigned dashboard
   - Network graph: All dimensions visualized
   - Timeline: All events on single timeline
   - Correlation matrix: All dimensions cross-referenced
   - Real-time alerts for multi-dimensional patterns

**Success Metrics:**
- 7 dimensions fully integrated
- < 3 second query times for complex correlations
- 50+ complex patterns detected
- Network graph with 1,000+ nodes

---

## 💡 Example Use Cases

### Use Case 1: The Social Media Pump
**Scenario:**
1. Senator tweets: "Excited about American AI leadership! 🇺🇸"
2. Buys $250K in NVDA (AI chip maker)
3. Reddit r/wallstreetbets starts NVDA hype train
4. Retail investors pile in, price surges 15%
5. Senator sells at peak
6. Reddit sentiment crashes
7. Senator stops tweeting about AI

**Detection:**
- Social media dimension: Positive AI tweet
- Congressional trading: Buy → Sell in 14 days
- Reddit dimension: Hype surge started 2 days after buy
- Timing: Sold exactly at Reddit sentiment peak
- **Suspicion Score: 92/100** (Coordinated pump, dumped on retail)

### Use Case 2: The Insider Information Cascade
**Scenario:**
1. Pharmaceutical company CEO sells 50% of holdings (Form 4 filed)
2. Congressional member on Health Committee sells same stock 3 days later
3. No public tweets or announcements
4. 1 week later: FDA rejects company's drug application
5. Stock crashes 40%

**Detection:**
- Corporate insider dimension: Unusual sale by CEO
- Congressional trading: Suspicious timing and same stock
- Social media dimension: Complete silence (hiding)
- Time delta: 3 days (possible information sharing)
- **Coordination Score: 88/100**

### Use Case 3: The Lobbying Trade
**Scenario:**
1. Tech companies spend $2M lobbying member's committee
2. Member tweets criticizing Big Tech regulation
3. Member buys $500K in GOOGL, META, AMZN
4. Member votes against tech regulation bill
5. Stocks rise 10%
6. Member sells

**Detection:**
- Lobbying dimension: Heavy lobbying spending
- Social media dimension: Public support for tech companies
- Congressional trading: Bought stocks of lobbying companies
- Legislative dimension: Voted favorably for those companies
- **Corruption Score: 95/100**

---

## 🔒 Ethical & Legal Considerations

### 1. Data Privacy
- **Only public data:** All data sources are public records (STOCK Act, SEC filings, public tweets, Reddit)
- **No private information:** No DMs, private meetings, non-public communications
- **Aggregation focus:** Focus on patterns, not individual doxing

### 2. Retail Investor Protection
- **Educational mission:** Help retail investors identify when they're being manipulated
- **Transparency:** Show when congress trades against retail sentiment
- **No financial advice:** Platform shows data, users make own decisions

### 3. Platform Disclaimers
```
DISCLAIMER: This platform is for educational and research purposes only.
- All data from public sources (STOCK Act, SEC, Twitter, Reddit)
- Correlation does not imply causation
- Suspicion scores are algorithmic, not legal determinations
- Not financial advice
- For transparency and accountability research
```

---

## 📈 Success Metrics

### Technical Metrics
- **Data Coverage:** 535 members + 10K insiders + 5 subreddits
- **Update Frequency:** < 15 min for new data
- **API Performance:** < 3 sec for complex correlations
- **Pattern Detection:** 100+ patterns detected per month

### Impact Metrics
- **Academic Citations:** Research papers using the platform
- **Journalism:** Investigative reports based on platform data
- **Legislative Reform:** Bills introduced based on detected patterns
- **Public Awareness:** Media coverage and public engagement

### Engagement Metrics
- **API Users:** 1,000+ API keys issued
- **Dashboard Users:** 10,000+ monthly active users
- **Data Downloads:** 500+ research datasets downloaded
- **Media Mentions:** 50+ articles citing the platform

---

## 🎯 Competitive Advantages

### vs. Existing Platforms
1. **Quiver Quantitative:** Only tracks congressional trades, no social media/Reddit
2. **Unusual Whales:** Focuses on options flow, not multi-dimensional correlations
3. **Capitol Trades:** Basic trade tracking, no intelligence engine
4. **OpenSecrets:** Lobbying + finance, but no trading correlation

### Our Unique Value
- **Multi-dimensional:** Only platform correlating 7+ dimensions
- **Retail protection:** First platform detecting congress vs retail patterns
- **Real-time alerts:** Live pattern detection and notifications
- **Open source:** Transparent methodology, reproducible research
- **Academic rigor:** Statistical significance testing, not just anecdotes

---

## 🔮 Future Dimensions (Phase 5+)

### Dimension 8: Media Appearance Tracking
- Track congressional members on CNBC, Bloomberg, Fox Business
- Correlate media appearances with trading
- Detect "talking your book" behavior

### Dimension 9: Think Tank & Policy Influencers
- Track think tank reports and recommendations
- Correlate with congressional trading and voting
- Identify revolving door patterns

### Dimension 10: Real Estate & Property
- Track congressional real estate transactions
- Correlate with zoning votes and development projects
- Identify NIMBY vs YIMBY trading patterns

### Dimension 11: Cryptocurrency & DeFi
- Track congressional crypto holdings
- Correlate with crypto regulation votes
- Detect pump-and-dump in crypto space

### Dimension 12: International Dimension
- Track foreign government actions
- Correlate with congressional trades (defense, energy)
- Detect geopolitical trading patterns

---

## 🛠️ Technical Stack Additions

### Current Stack
- Backend: Python 3.13, FastAPI, SQLAlchemy
- Frontend: HTML/CSS/JS (migrating to React)
- Database: PostgreSQL
- ML: scikit-learn, TensorFlow

### New Additions for Multi-Dimensional Platform
- **Social Media APIs:** Tweepy (Twitter), PRAW (Reddit)
- **NLP:** spaCy, Hugging Face Transformers (sentiment, entity extraction)
- **Network Analysis:** NetworkX, graph-tool (influence networks)
- **Visualization:** D3.js (timeline, network graphs), Plotly (correlation matrices)
- **Time Series:** Prophet (Facebook), statsmodels (correlation analysis)
- **Caching:** Redis (real-time data), Memcached (API responses)
- **Task Queue:** Celery (background scraping), RabbitMQ (message broker)
- **Monitoring:** Prometheus (metrics), Grafana (dashboards)

---

## 📝 Summary

**Transformation:**
```
Congressional Trading Tracker
         ↓
Multi-Dimensional Influence Intelligence Platform
```

**Key Dimensions:**
1. ✅ Congressional Trading (existing)
2. 🆕 Social Media (Twitter/X) - Phase 1
3. 🆕 Reddit Sentiment - Phase 2
4. 🆕 Corporate Insiders - Phase 3
5. 🆕 Lobbying Intelligence - Phase 4
6. 🆕 Campaign Finance - Phase 4
7. 🆕 Multi-Dimensional Analysis - Phase 4

**Timeline:** 12 weeks to full multi-dimensional platform

**Expected Impact:**
- **Retail Protection:** Help millions of retail investors avoid manipulation
- **Transparency:** Unprecedented visibility into influence networks
- **Accountability:** Data-driven evidence for legislative reform
- **Research:** Enable academic and journalistic investigations

**Next Steps:**
1. Review and approve this specification
2. Start Phase 1: Social Media Integration
3. Set up Twitter API access
4. Build initial correlators
5. Launch beta dashboard with social media dimension

---

**Questions for Discussion:**
1. Should we prioritize social media or Reddit first? (Spec suggests social media)
2. Do we want to support additional platforms (Truth Social, Mastodon)?
3. Should we add cryptocurrency dimension earlier?
4. What's our stance on monetization (API tiers, premium features)?
5. How aggressively should we market "retail protection" angle?
