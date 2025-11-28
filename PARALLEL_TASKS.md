# Parallel Implementation Tasks

This document contains ready-to-execute task descriptions for parallel Claude agents. Each section is self-contained and can be assigned to a separate agent.

---

## How to Use This Document

Run each major section as a separate parallel Claude task. Tasks within each track have dependencies noted - run them in order within a track, but tracks can run in parallel with each other.

---

# TRACK A: Intelligence Engine

## A1: Conviction Score Algorithm
**Priority**: P0 | **Complexity**: High | **Dependencies**: None

### Objective
Build the core conviction scoring engine that analyzes trades and outputs a 0-100 score indicating how likely a trade appears to be based on non-public information.

### Requirements

1. **Create `/src/intelligence/conviction_scorer.py`**:
   - Class `ConvictionScorer` with method `score_trade(trade, member) -> ConvictionResult`
   - Return object with: `score` (0-100), `factors` (dict of contributing elements), `explanation` (human-readable string)

2. **Scoring Factors** (weights totaling 100):
   - **Committee Access** (25 points): Does member sit on committee overseeing this sector?
   - **Timing Proximity** (25 points): How close to relevant announcements/hearings?
   - **Filing Delay** (15 points): How late was the disclosure filed?
   - **Trade Size Anomaly** (15 points): Is this unusually large for this member?
   - **Historical Pattern** (10 points): Does this deviate from normal behavior?
   - **Sector Concentration** (10 points): Over-weighted in oversight sectors?

3. **Explanation Generator**:
   - Generate natural language explanation: "This trade scores 82/100 because..."
   - List top 3 contributing factors with specific values

4. **Integration**:
   - Add method to batch score all trades: `score_all_trades() -> List[ConvictionResult]`
   - Store results in database with timestamp
   - Add API endpoint: `GET /api/trades/{id}/conviction`

### Acceptance Criteria
- [ ] All 1,755 existing trades have conviction scores
- [ ] Scores are deterministic (same input = same output)
- [ ] Top 10 highest scores manually reviewed for reasonableness
- [ ] Unit tests for each scoring factor

---

## A2: Timing Analysis Engine
**Priority**: P0 | **Complexity**: High | **Dependencies**: A1

### Objective
Build an engine that identifies trades occurring suspiciously close to market-moving events the member had plausible access to.

### Requirements

1. **Create `/src/intelligence/timing_analyzer.py`**:
   - Class `TimingAnalyzer`
   - Method `analyze_trade_timing(trade) -> TimingAnalysis`

2. **Event Types to Correlate**:
   - Committee hearings (from congress.gov)
   - Bill introductions/votes affecting sector
   - Classified briefings (by committee membership)
   - Earnings announcements of traded stock
   - FDA decisions (for pharma)
   - Defense contract announcements

3. **Analysis Output**:
   ```python
   class TimingAnalysis:
       trade_id: str
       suspicious_events: List[SuspiciousEvent]
       timing_score: float  # 0-1
       summary: str

   class SuspiciousEvent:
       event_type: str
       event_date: date
       days_before_trade: int
       days_after_trade: int
       member_access_level: str  # "direct", "committee", "public"
       description: str
   ```

4. **Integration with Conviction Score**:
   - Feed timing analysis into conviction scorer as factor
   - Update A1 to consume timing data

### Acceptance Criteria
- [ ] Timing analysis for all trades with known correlated events
- [ ] COVID-19 trades (Jan-Mar 2020) all analyzed with briefing correlation
- [ ] API endpoint: `GET /api/trades/{id}/timing`

---

## A3: Committee-Trade Correlation
**Priority**: P0 | **Complexity**: Medium | **Dependencies**: None

### Objective
Map every member's committee assignments to sectors and flag trades in their oversight areas.

### Requirements

1. **Create `/src/intelligence/committee_correlator.py`**:
   - Class `CommitteeCorrelator`
   - Method `get_oversight_sectors(member) -> List[Sector]`
   - Method `is_oversight_trade(trade, member) -> CorrelationResult`

2. **Committee to Sector Mapping**:
   ```python
   COMMITTEE_SECTORS = {
       "Financial Services": ["XLF", "Banks", "Insurance", "Fintech"],
       "Armed Services": ["XLI", "Defense", "Aerospace"],
       "Energy and Commerce": ["XLE", "XLU", "Pharma", "Tech"],
       "Agriculture": ["Commodities", "Food", "Farming"],
       # ... complete mapping for all committees
   }
   ```

3. **Output**:
   ```python
   class CorrelationResult:
       is_oversight: bool
       committee: str
       sector: str
       overlap_score: float  # 0-1
       explanation: str
   ```

4. **Dashboard Data**:
   - Generate summary: "X% of Member Y's trades are in their oversight sectors"
   - Rank members by oversight trading percentage

### Acceptance Criteria
- [ ] All committees mapped to sectors
- [ ] Every trade flagged for oversight correlation
- [ ] Member ranking by oversight trading %
- [ ] Visualization data ready for Track C

---

## A4: Story Generator AI
**Priority**: P1 | **Complexity**: High | **Dependencies**: A1, A2

### Objective
Build an AI-powered system that generates journalist-ready narratives about suspicious trading patterns.

### Requirements

1. **Create `/src/intelligence/story_generator.py`**:
   - Class `StoryGenerator`
   - Integration with Claude API for LLM generation

2. **Story Templates**:
   - **Tweet Thread**: 280 char segments, 3-5 tweets
   - **News Brief**: 200 words, inverted pyramid style
   - **Deep Dive**: 1000 words, full investigation format
   - **Data Card**: Key stats for social sharing

3. **Method Signatures**:
   ```python
   def generate_trade_story(trade_id: str, format: StoryFormat) -> Story
   def generate_member_profile(member_id: str) -> Story
   def generate_pattern_alert(pattern_type: str, trades: List) -> Story
   def generate_weekly_roundup() -> Story
   ```

4. **Story Elements**:
   - Lead with most compelling fact
   - Include conviction score and explanation
   - Add relevant timing analysis
   - Cite specific dates, amounts, percentages
   - Include standard disclaimer

5. **Output Formats**:
   - Markdown
   - HTML with styling
   - Plain text
   - JSON (for API)

### Acceptance Criteria
- [ ] Generate stories for top 20 highest conviction trades
- [ ] Stories pass manual review for accuracy
- [ ] API endpoint: `POST /api/stories/generate`
- [ ] Social sharing formatted correctly

---

## A5: Swamp Score Calculator
**Priority**: P1 | **Complexity**: Medium | **Dependencies**: A1, A2, A3

### Objective
Create a composite ethics/corruption indicator per member, aggregating all analysis into a single 0-100 score.

### Requirements

1. **Create `/src/intelligence/swamp_scorer.py`**:
   - Class `SwampScorer`
   - Method `calculate_swamp_score(member) -> SwampScore`

2. **Score Components** (weights = 100):
   - Average conviction score of trades (40%)
   - Filing compliance rate (20%)
   - Oversight sector trading % (20%)
   - Timing suspicion patterns (10%)
   - Volume anomalies (10%)

3. **Output**:
   ```python
   class SwampScore:
       member_id: str
       total_score: int  # 0-100
       rank: int  # out of 535
       percentile: int
       components: Dict[str, float]
       trend: str  # "improving", "worsening", "stable"
       explanation: str
   ```

4. **Rankings**:
   - Overall ranking
   - By party
   - By chamber
   - By state
   - Historical trend (if data available)

### Acceptance Criteria
- [ ] Swamp scores for all 535 members
- [ ] Rankings published and sortable
- [ ] API endpoint: `GET /api/members/{id}/swamp-score`
- [ ] Weekly recalculation scheduled

---

## A6: Copy Congress Backtester
**Priority**: P1 | **Complexity**: Medium | **Dependencies**: None

### Objective
Build a backtesting engine that calculates returns if you had copied congressional trades.

### Requirements

1. **Create `/src/analysis/backtester.py`**:
   - Class `CopyCongressBacktester`
   - Method `backtest_member(member_id, start_date, end_date) -> BacktestResult`

2. **Backtest Logic**:
   - Simulate buying on filing date (realistic delay)
   - Calculate returns vs. S&P 500 benchmark
   - Track win/loss rate on individual trades
   - Calculate Sharpe ratio, max drawdown, etc.

3. **Output**:
   ```python
   class BacktestResult:
       member_id: str
       total_return_pct: float
       benchmark_return_pct: float
       alpha: float  # excess return
       sharpe_ratio: float
       win_rate: float
       avg_trade_return: float
       max_drawdown: float
       trades_analyzed: int
   ```

4. **Comparisons**:
   - vs. S&P 500
   - vs. Sector ETFs
   - vs. Random stock picks
   - vs. Average investor

5. **Leaderboard**:
   - Best performers ranked
   - Party comparison
   - Chamber comparison

### Acceptance Criteria
- [ ] Backtest results for all trading members
- [ ] Results validated against 3 known cases manually
- [ ] API endpoint: `GET /api/members/{id}/performance`
- [ ] Leaderboard data for dashboard

---

# TRACK B: Data Pipeline

## B1: Real-Time Disclosure Scraping
**Priority**: P0 | **Complexity**: High | **Dependencies**: None

### Objective
Build automated scrapers that detect new STOCK Act filings within 1 hour of publication.

### Requirements

1. **Enhance `/src/data_sources/`**:
   - Improve `house_disclosure_scraper.py` for reliability
   - Add Senate disclosure scraper
   - Implement polling mechanism (every 15 min)

2. **Scraper Features**:
   - Detect new filings since last check
   - Parse PDF disclosures where needed
   - Extract: member, stock, date, amount, transaction type
   - Handle amendments and corrections

3. **Pipeline**:
   ```
   Scraper → Parser → Validator → Database → Trigger Alerts
   ```

4. **Monitoring**:
   - Log all scraping runs
   - Alert on failures
   - Track data freshness

5. **Historical Backfill**:
   - Script to backfill all 2023-2024 data
   - Deduplication logic

### Acceptance Criteria
- [ ] New filings detected within 1 hour
- [ ] 99%+ parsing accuracy
- [ ] No duplicate entries
- [ ] Backfill complete for current Congress

---

## B2: Market Data Integration
**Priority**: P0 | **Complexity**: Medium | **Dependencies**: None

### Objective
Integrate real-time and historical market data for enriching trade analysis.

### Requirements

1. **Create `/src/data_sources/market_data.py`**:
   - Integrate free API (Yahoo Finance, Alpha Vantage, or Finnhub)
   - Get historical prices for all traded symbols
   - Get current prices for portfolio valuation

2. **Data Points Needed**:
   - Price at trade date
   - Price at filing date
   - Current price
   - 1-day, 1-week, 1-month returns post-trade
   - Sector classification
   - Company info (name, industry)

3. **Enrichment Pipeline**:
   - Batch enrich existing trades
   - Auto-enrich new trades on ingest

4. **Caching**:
   - Redis cache for frequent lookups
   - Daily refresh of current prices

### Acceptance Criteria
- [ ] All traded symbols enriched with prices
- [ ] Return calculations for all trades
- [ ] Sector classification for all symbols
- [ ] API for price lookups

---

## B3: Committee Hearing Calendar
**Priority**: P1 | **Complexity**: Medium | **Dependencies**: None

### Objective
Ingest committee hearing schedules for timing correlation.

### Requirements

1. **Create `/src/data_sources/committee_calendar.py`**:
   - Scrape or API from congress.gov
   - Get all committee hearings with dates, topics, witnesses

2. **Data Model**:
   ```python
   class Hearing:
       committee: str
       subcommittee: Optional[str]
       date: date
       topic: str
       witnesses: List[str]
       related_sectors: List[str]  # derived
   ```

3. **Sector Tagging**:
   - NLP to extract sector relevance from hearing topic
   - Manual mapping for major hearings

### Acceptance Criteria
- [ ] All 2023-2024 hearings ingested
- [ ] Sector tagging for 90%+ of hearings
- [ ] API endpoint for hearing lookup

---

## B4: News Correlation Engine
**Priority**: P1 | **Complexity**: High | **Dependencies**: B2

### Objective
Track news events that correlate with congressional trades.

### Requirements

1. **Create `/src/data_sources/news_tracker.py`**:
   - Integrate news API (NewsAPI, GDELT, or similar)
   - Track major market-moving events
   - Correlate with trades

2. **Event Types**:
   - Earnings announcements
   - FDA decisions
   - Defense contracts
   - Regulatory announcements
   - Major deals/M&A

3. **Correlation Logic**:
   - Find events within ±7 days of trades
   - Score relevance to traded stock
   - Feed into timing analysis

### Acceptance Criteria
- [ ] News events tracked for top 100 traded symbols
- [ ] Correlation with existing trades
- [ ] API for event lookup

---

# TRACK C: Dashboard & Frontend

## C1: Conviction Score UI
**Priority**: P0 | **Complexity**: Medium | **Dependencies**: A1 (backend)

### Objective
Add conviction score display to the dashboard with visual indicators and explanations.

### Requirements

1. **Trade Table Enhancement**:
   - Add conviction score column
   - Color coding: 0-30 green, 30-60 yellow, 60-100 red
   - Tooltip with score breakdown

2. **Score Card Component**:
   - Large score display (e.g., "82/100")
   - Factor breakdown chart (radar or bar)
   - Natural language explanation
   - "Learn More" expandable section

3. **Sort and Filter**:
   - Sort all trades by conviction score
   - Filter by score range
   - "Most Suspicious" quick filter

4. **Styling**:
   - Consistent with existing dashboard
   - Mobile responsive
   - Loading states

### Acceptance Criteria
- [ ] All trades display conviction scores
- [ ] Score breakdown accessible
- [ ] Sortable and filterable
- [ ] Mobile responsive

---

## C2: Committee Connection Visualization
**Priority**: P0 | **Complexity**: High | **Dependencies**: A3 (backend)

### Objective
Create interactive visualization showing committee-to-trading connections.

### Requirements

1. **Network Graph**:
   - Nodes: Members, Committees, Sectors, Stocks
   - Edges: Membership, Trading, Oversight
   - Interactive: Click to filter, hover for details
   - Use D3.js or vis.js

2. **Sankey Diagram**:
   - Flow: Committee → Sector → Trades
   - Width proportional to trade volume

3. **Heat Map**:
   - X-axis: Committees
   - Y-axis: Sectors
   - Color: Trading volume in oversight areas

4. **Controls**:
   - Filter by party, chamber, committee
   - Time range selector
   - Toggle visualization types

### Acceptance Criteria
- [ ] Network graph functional with all data
- [ ] At least 2 visualization types
- [ ] Interactive filtering
- [ ] Sharable/embeddable

---

## C3: Copy Congress Simulator UI
**Priority**: P0 | **Complexity**: Medium | **Dependencies**: A6 (backend)

### Objective
Build interactive "What If" portfolio simulator.

### Requirements

1. **Member Selector**:
   - Search/select member to follow
   - Multi-select for portfolio mixing
   - Quick picks: "Top 10 Performers", "Finance Committee", etc.

2. **Performance Dashboard**:
   - Total return chart over time
   - Comparison to S&P 500
   - Individual trade history
   - Key stats: return, alpha, Sharpe, win rate

3. **Calculator**:
   - "If you invested $10,000..."
   - Date range selector
   - Hypothetical current value

4. **Leaderboard**:
   - Best performing members
   - Worst performing members
   - Party comparison
   - This week/month/year/all-time tabs

### Acceptance Criteria
- [ ] Simulator functional for all trading members
- [ ] Comparison charts working
- [ ] Leaderboard populated
- [ ] Sharable results

---

## C4: Hall of Shame/Fame
**Priority**: P1 | **Complexity**: Low | **Dependencies**: A5 (swamp scores)

### Objective
Create leaderboards for most and least ethical traders.

### Requirements

1. **Hall of Shame**:
   - Top 20 highest swamp scores
   - Top 20 most conviction score flags
   - Top 20 worst filing compliance
   - Top 20 most oversight trading

2. **Hall of Fame**:
   - Members with no individual stock trading
   - Fastest filers
   - Lowest swamp scores

3. **Design**:
   - Card-based layout
   - Photo, name, party, score
   - Click to member profile
   - Share button for each card

4. **Filters**:
   - By party
   - By chamber
   - By time period

### Acceptance Criteria
- [ ] Both leaderboards populated
- [ ] Member cards with key info
- [ ] Sharable individual cards
- [ ] Mobile responsive

---

## C5: Timing Analysis Timeline
**Priority**: P1 | **Complexity**: Medium | **Dependencies**: A2 (backend)

### Objective
Visual timeline showing trade timing relative to events.

### Requirements

1. **Timeline Component**:
   - Horizontal timeline
   - Markers for: trades, hearings, announcements, filings
   - Zoom in/out
   - Click for details

2. **Trade-Event Pairing**:
   - Visual connection between suspicious trade and event
   - Time delta displayed
   - Color coded by suspicion level

3. **Filters**:
   - By member
   - By sector
   - By event type
   - Date range

### Acceptance Criteria
- [ ] Timeline displays all major suspicious timings
- [ ] Interactive zoom and pan
- [ ] Event details accessible
- [ ] Exportable as image

---

# TRACK D: Distribution & Engagement

## D1: Twitter/X Bot
**Priority**: P0 | **Complexity**: Medium | **Dependencies**: A1, A4

### Objective
Automated bot that posts about new filings and suspicious patterns.

### Requirements

1. **Create `/src/bots/twitter_bot.py`**:
   - Twitter API v2 integration
   - Scheduled posting
   - Rate limiting compliance

2. **Post Types**:
   - New filing alerts: "BREAKING: Rep. X just disclosed $Y purchase of $STOCK"
   - High conviction alerts: "⚠️ This trade scored 87/100 on our suspicion meter..."
   - Weekly roundup: "This week's most suspicious trades..."
   - Leaderboard updates: "Updated: Congress's best stock pickers..."

3. **Formatting**:
   - Thread support for longer analyses
   - Image/chart attachments
   - Hashtags: #CongressTrading #StockAct

4. **Moderation**:
   - Human approval queue for high-stakes posts
   - Accuracy verification before posting
   - Reply monitoring

### Acceptance Criteria
- [ ] Bot posts new filings within 1 hour
- [ ] High conviction trades get threads
- [ ] Weekly roundup posts
- [ ] 1,000+ followers within first month

---

## D2: Discord Bot
**Priority**: P1 | **Complexity**: Low | **Dependencies**: D1

### Objective
Discord bot for real-time alerts in community servers.

### Requirements

1. **Create `/src/bots/discord_bot.py`**:
   - discord.py integration
   - Slash commands

2. **Commands**:
   - `/trade {member}` - Latest trades
   - `/score {member}` - Swamp score
   - `/alerts subscribe` - Channel subscription
   - `/leaderboard` - Top traders

3. **Auto-Alerts**:
   - New filings
   - High conviction scores
   - Pattern detections

### Acceptance Criteria
- [ ] Bot functional in test server
- [ ] Commands working
- [ ] Alert channel configured
- [ ] Public invite available

---

## D3: Email Alert System
**Priority**: P1 | **Complexity**: Medium | **Dependencies**: A1

### Objective
Email subscription system for trade alerts.

### Requirements

1. **Create `/src/notifications/email_service.py`**:
   - Sendgrid or SES integration
   - Template management
   - Subscription management

2. **Email Types**:
   - Instant alerts (high conviction trades)
   - Daily digest
   - Weekly roundup
   - Member-specific alerts

3. **Subscription Management**:
   - Subscribe/unsubscribe
   - Preference center
   - Member watchlist

4. **Templates**:
   - HTML email with charts
   - Plain text fallback
   - Mobile optimized

### Acceptance Criteria
- [ ] Subscription flow working
- [ ] Daily digest sending
- [ ] Unsubscribe working
- [ ] Templates approved

---

# TRACK E: API & Platform

## E1: REST API Design
**Priority**: P0 | **Complexity**: Medium | **Dependencies**: None

### Objective
Design and implement public REST API.

### Requirements

1. **Migrate to FastAPI** in `/src/api/`:
   - OpenAPI documentation
   - Async handlers
   - Response models

2. **Endpoints**:
   ```
   GET  /api/members                    - List all members
   GET  /api/members/{id}               - Member details
   GET  /api/members/{id}/trades        - Member's trades
   GET  /api/members/{id}/score         - Member's swamp score

   GET  /api/trades                     - List trades (paginated)
   GET  /api/trades/{id}                - Trade details
   GET  /api/trades/{id}/conviction     - Conviction analysis
   GET  /api/trades/{id}/timing         - Timing analysis

   GET  /api/analysis/leaderboard       - Rankings
   GET  /api/analysis/stats             - Aggregate stats

   POST /api/stories/generate           - Generate story (authed)
   ```

3. **Response Format**:
   - JSON with consistent structure
   - Pagination for lists
   - Error handling

### Acceptance Criteria
- [ ] All endpoints implemented
- [ ] OpenAPI docs generated
- [ ] Rate limiting in place
- [ ] Test coverage

---

## E2: API Authentication
**Priority**: P0 | **Complexity**: Medium | **Dependencies**: E1

### Objective
Implement API authentication and access tiers.

### Requirements

1. **Auth Methods**:
   - API key authentication
   - Optional OAuth for premium features

2. **Access Tiers**:
   - Public: Basic endpoints, 100 req/day
   - Journalist: Full access, 1000 req/day
   - Academic: Bulk export, 5000 req/day
   - Premium: Unlimited, SLA

3. **Key Management**:
   - Self-service registration
   - Key rotation
   - Usage tracking

### Acceptance Criteria
- [ ] API keys working
- [ ] Rate limiting by tier
- [ ] Usage dashboard
- [ ] Key rotation

---

# TRACK F: ML/AI Enhancement

## F1: LLM Story Generation
**Priority**: P0 | **Complexity**: High | **Dependencies**: A4

### Objective
Integrate Claude API for high-quality story generation.

### Requirements

1. **Create `/src/ml_models/llm_service.py`**:
   - Claude API client
   - Prompt engineering
   - Response parsing

2. **Prompts**:
   - Trade story template
   - Member profile template
   - Pattern analysis template
   - Weekly roundup template

3. **Quality Controls**:
   - Fact verification step
   - Disclaimer injection
   - Tone consistency

### Acceptance Criteria
- [ ] Claude integration working
- [ ] Stories factually accurate
- [ ] Consistent voice and style
- [ ] API costs tracked

---

## F2: Natural Language Query
**Priority**: P1 | **Complexity**: High | **Dependencies**: None

### Objective
Allow users to query the database using natural language.

### Requirements

1. **Create `/src/ml_models/nl_query.py`**:
   - Parse natural language to SQL/filters
   - Handle common query patterns
   - Fallback to suggestions

2. **Example Queries**:
   - "Show me Pelosi's tech trades"
   - "Who bought NVDA before the earnings?"
   - "Which Republicans trade the most?"
   - "Most suspicious trade this month"

3. **UI Integration**:
   - Search bar in dashboard
   - Query suggestions
   - Results display

### Acceptance Criteria
- [ ] 90%+ accuracy on test queries
- [ ] Graceful fallback on failures
- [ ] Search bar integrated
- [ ] Query history

---

# Coordination Notes

## Dependency Graph

```
      A1 (Conviction) ──────┐
           │                │
           ▼                ▼
      A2 (Timing) ───► A4 (Stories) ───► D1 (Twitter)
           │                │                 │
           ▼                ▼                 ▼
      A5 (Swamp) ────► C4 (Hall) ────────► D2 (Discord)
           │
           ▼
      C1, C5 (UI)

      A3 (Committee) ──────► C2 (Viz)

      A6 (Backtest) ───────► C3 (Simulator)

      B1, B2 (Data) ───────► All Analysis

      E1, E2 (API) ─────────► All Distribution
```

## Suggested Execution Order

**Week 1**:
- Start: A1, A3, A6, B1, B2, E1 (all independent)

**Week 2**:
- Start: A2, C1, C2, C3, E2 (depend on Week 1)
- Continue: B1, B2

**Week 3**:
- Start: A4, A5, F1, D1 (depend on Week 2)
- Start: C4, C5, D3

**Week 4**:
- Start: F2, D2, remaining items
- Polish and integration

---

*Document generated for parallel Claude agent execution.*
