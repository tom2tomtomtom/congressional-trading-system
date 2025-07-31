# Congressional Trading Agent Research

## CapitolTrades.com Analysis

### Overview
CapitolTrades is a platform that tracks congressional trading activity, providing transparency into stock trades made by U.S. politicians.

### Key Features Observed:
1. **Latest Trades Section**: Shows recent buy/sell transactions with:
   - Trade type (buy/sell)
   - Date (e.g., "2 days ago")
   - Company name and ticker (e.g., Lululemon LULU:US, Snowflake SNOW:US)
   - Politician name and party affiliation
   - Position (House/Senate)

2. **Featured Politicians**: Profiles showing:
   - Total trades count
   - Number of filings
   - Number of issuers
   - Trading volume
   - Performance charts

3. **Featured Issuers**: Popular stocks being traded by congress members:
   - Tesla (TSLA) - 164 trades, 340.47% performance
   - NVIDIA (NVDA) - 248 trades, 147.97% performance
   - FedEx (FDX) - 67 trades, 229.51% performance
   - Meta (META) - 220 trades, 712.23% performance
   - Amazon (AMZN) - 244 trades, 212.77% performance

4. **Insights Section**: Analysis articles on trading patterns and policy implications

5. **Buzz Section**: Real-time updates on trading activity and market news

### Recent Notable Trades:
- Marjorie Taylor Greene: Multiple recent trades in Lululemon (sell), Snowflake (buy), Tractor Supply (buy)
- Rob Bresnahan: Recent sells in 3M and Arista Networks

### Data Richness:
The platform appears to have comprehensive data including historical performance, trade timing, and detailed politician profiles.


### CapitolTrades Technical Details:

**Data Source & Quality:**
- Backed by 2iQ Research with 20+ years of expertise in insider transaction data
- Combines automated and manual record collection for accuracy
- Claims to have the most historical data available in the market (past 3 years restriction noted)
- Industry-leading filtering capabilities
- Highest data volume for Senate and Congress representatives

**Key Features:**
- **Fast Delivery**: Real-time or near real-time trade publication
- **Enriched Data**: Leverages 2iQ data factory to enrich transactions with additional information and classifications
- **Quality Assurance**: Manual data analyst review for accuracy
- **Twitter Integration**: Real-time trade notifications via Twitter
- **Detail Pages**: Comprehensive politician profiles with bio, committee memberships, tweets, trading activity, and preferred sectors

**Data Access:**
- Platform is free for public use
- No obvious API documentation found on About page
- Twitter feed provides real-time notifications
- Historical data limited to past 3 years
- May need to explore 2iQ Research for potential API access

**Regulatory Compliance:**
- Supports STOCK Act (Stop Trading on Congressional Knowledge Act)
- Focuses on transparency and accountability
- Advocates for better reporting systems and penalties



## Polymarket.com Analysis

### Overview
Polymarket is the world's largest prediction market platform where users can bet on the outcomes of real-world events using cryptocurrency.

### How It Works:
1. **Pick a Market**: Choose from various prediction markets on politics, economics, sports, culture, etc.
2. **Place a Bet**: Fund account with crypto, credit/debit card, or bank transfer. Buy "Yes" or "No" shares based on your prediction.
3. **Profit**: Sell shares at any time or wait until market resolution. Winning shares redeem for $1 each.

### Key Features:
- **Real-time Odds**: Prices shift dynamically based on market sentiment and trading activity
- **Multiple Categories**: Politics, Middle East, Sports, Crypto, Tech, Culture, World, Economy
- **Political Markets**: Elections, Trump-related events, policy decisions, geopolitical events
- **Economic Markets**: Fed decisions, recession predictions, trade wars
- **No Betting Limits**: No restrictions on bet sizes
- **No Fees**: Platform claims no fees for trading

### Market Examples Observed:
- NYC Mayoral Election predictions
- Federal Reserve interest rate decisions
- Geopolitical events (Iran, Israel conflicts)
- Economic indicators (US recession predictions)
- Political outcomes (Congressional elections, policy decisions)

### Trading Volume:
- High-volume markets with millions in trading volume
- Individual markets ranging from $50k to $54m in volume
- Active real-time trading with percentage-based odds

### Potential Integration Points:
- Could create prediction markets based on congressional trading patterns
- Economic policy markets could correlate with congressional trades
- Stock-specific prediction markets could be influenced by congressional activity
- API access would be needed for automated trading based on congressional data


## API Access Analysis

### Congressional Trading Data APIs:

#### 1. Finnhub Congressional Trading API (Premium)
- **Endpoint**: `/stock/congressional-trading?symbol=AAPL`
- **Authentication**: API key required
- **Data Format**: JSON with fields:
  - amountFrom, amountTo, assetName, filingDate, name, ownerType, position, symbol, transactionDate, transactionType
- **Pricing**: Premium subscription required
- **Coverage**: Congressional trading data disclosed by members of congress

#### 2. Financial Modeling Prep Senate Trading API
- **Endpoint**: `https://financialmodelingprep.com/api/v4/senate-trading?symbol=AAPL`
- **Authentication**: API key required
- **Data Format**: JSON with fields:
  - firstName, lastName, office, link, dateReceived, transactionDate, owner, assetDescription, assetType, type, amount, comment, symbol
- **Coverage**: US Senate trading activity
- **Compliance**: Based on STOCK Act disclosures

#### 3. Third-party Scrapers:
- **Apify CapitolTrades Scraper**: Programmatic access to CapitolTrades data
- **GitHub Projects**: Community-built APIs and scrapers for congressional data

### Polymarket API:

#### CLOB (Central Limit Order Book) API
- **Architecture**: Hybrid-decentralized with off-chain matching, on-chain settlement
- **Authentication**: EIP712-signed structured data
- **Features**:
  - REST and WebSocket endpoints
  - Market data access (all markets, prices, order history)
  - Order management (create, list, fetch, cancel)
  - Real-time market updates
  - Programmatic trading capabilities

#### Key Capabilities:
- **Market Discovery**: Access to all available prediction markets
- **Order Management**: Place, modify, cancel orders programmatically
- **Real-time Data**: WebSocket feeds for live market updates
- **Trading**: Automated trading via API
- **Fees**: Currently 0 bps for both maker and taker fees

#### Technical Details:
- **Tokens**: Binary Outcome Tokens (CTF ERC1155) and collateral (ERC20)
- **Settlement**: Non-custodial, on-chain via smart contracts
- **Security**: Audited by Chainsecurity
- **Rate Limits**: Documented API rate limits available

### Integration Potential:
1. **Data Pipeline**: Congressional trading data → Analysis → Polymarket predictions
2. **Automated Trading**: React to congressional trades by placing prediction market bets
3. **Market Creation**: Create new prediction markets based on congressional activity patterns
4. **Cross-platform Analysis**: Correlate congressional trades with existing political/economic markets

