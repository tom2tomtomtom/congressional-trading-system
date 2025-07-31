# CongressWatch: Hobby Development Plan

**Project:** Personal Congressional Trading Agent  
**Timeline:** 16 weeks (4 months)  
**Effort:** 5-10 hours/week  
**Budget:** Under $500 total

## Quick Start Guide

### Weekend 1: Setup and First Data Pull

**Goal**: Get your first congressional trading data and see what's available.

**Setup Steps:**
1. Install Python 3.9+ and create a virtual environment
2. Sign up for free Finnhub account (60 API calls/minute)
3. Install basic packages: `pip install requests pandas sqlite3 matplotlib`
4. Write your first data collection script

**Sample Code to Get Started:**
```python
import requests
import pandas as pd
import sqlite3
from datetime import datetime

# Your Finnhub API key (free tier)
API_KEY = "your_finnhub_key_here"

def fetch_congressional_trades(symbol="AAPL"):
    url = f"https://finnhub.io/api/v1/stock/congressional-trading"
    params = {"symbol": symbol, "token": API_KEY}
    response = requests.get(url, params=params)
    return response.json()

# Try it out
trades = fetch_congressional_trades("AAPL")
print(f"Found {len(trades.get('data', []))} Apple trades")
```

**Weekend Goal**: Successfully pull and examine congressional trading data for 5-10 popular stocks.

### Weekend 2-3: Build Data Collection System

**Goal**: Create a systematic way to collect and store congressional trading data.

**Database Setup:**
```python
import sqlite3

def setup_database():
    conn = sqlite3.connect('congress_trades.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY,
        symbol TEXT,
        member_name TEXT,
        transaction_date TEXT,
        amount_from INTEGER,
        amount_to INTEGER,
        transaction_type TEXT,
        filing_date TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    return conn
```

**Data Collection Script:**
```python
def collect_all_trades():
    # List of popular stocks to monitor
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
    
    conn = setup_database()
    
    for symbol in symbols:
        trades = fetch_congressional_trades(symbol)
        for trade in trades.get('data', []):
            # Insert into database (add error handling)
            cursor.execute('''
            INSERT OR IGNORE INTO trades 
            (symbol, member_name, transaction_date, amount_from, amount_to, transaction_type, filing_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['symbol'],
                trade['name'],
                trade['transactionDate'],
                trade['amountFrom'],
                trade['amountTo'],
                trade['transactionType'],
                trade['filingDate']
            ))
    
    conn.commit()
    print(f"Collected trades for {len(symbols)} symbols")
```

**Weekend Goal**: Automated data collection running daily via cron job.

## Phase 1: Data Foundation (Weeks 1-4)

### Week 1: Environment and Basic Collection
- Set up development environment
- Create Finnhub and Alpha Vantage accounts
- Build basic data collection for 10-20 stocks
- Set up SQLite database

### Week 2: Data Analysis and Exploration
- Analyze collected data for patterns
- Create basic visualizations (who trades what, when)
- Identify interesting trends and potential signals
- Document findings in a Jupyter notebook

### Week 3: Expand Data Collection
- Add Financial Modeling Prep as backup data source
- Implement error handling and retry logic
- Add more stocks and historical data collection
- Create data quality checks

### Week 4: Basic Signal Research
- Implement simple signal ideas (recent large trades, consensus trades)
- Backtest signals against historical stock performance
- Create performance metrics and visualization
- Document which signals look promising

**Phase 1 Deliverables:**
- Working data collection system
- Database with 2-3 months of historical data
- Analysis notebook with insights
- 2-3 promising signal concepts

## Phase 2: Signal Development (Weeks 5-8)

### Week 5: Signal Implementation
```python
def calculate_signals(symbol, days_back=30):
    conn = sqlite3.connect('congress_trades.db')
    
    # Get recent trades for this symbol
    query = '''
    SELECT * FROM trades 
    WHERE symbol = ? AND transaction_date > date('now', '-{} days')
    ORDER BY transaction_date DESC
    '''.format(days_back)
    
    trades_df = pd.read_sql(query, conn, params=[symbol])
    
    signals = {}
    
    # Signal 1: Recent large purchases
    large_buys = trades_df[
        (trades_df['transaction_type'] == 'Purchase') & 
        (trades_df['amount_from'] > 50000)
    ]
    signals['large_buy_score'] = len(large_buys)
    
    # Signal 2: Multiple members buying
    unique_buyers = trades_df[
        trades_df['transaction_type'] == 'Purchase'
    ]['member_name'].nunique()
    signals['consensus_score'] = unique_buyers
    
    # Signal 3: Recent activity vs historical
    # (implement based on your analysis)
    
    return signals
```

### Week 6: Backtesting Framework
```python
def backtest_signal(signal_func, start_date, end_date):
    results = []
    
    # For each day in the period
    for date in pd.date_range(start_date, end_date):
        # Calculate signals for all stocks
        for symbol in STOCK_LIST:
            signal = signal_func(symbol, date)
            
            # If signal is strong enough, simulate a trade
            if signal['total_score'] > THRESHOLD:
                # Get stock performance over next 30 days
                future_return = get_stock_return(symbol, date, days=30)
                
                results.append({
                    'date': date,
                    'symbol': symbol,
                    'signal_score': signal['total_score'],
                    'return': future_return
                })
    
    return pd.DataFrame(results)
```

### Week 7: Signal Optimization
- Test different signal combinations and thresholds
- Analyze which types of congressional trades are most predictive
- Implement position sizing based on signal strength
- Create risk-adjusted performance metrics

### Week 8: Strategy Validation
- Run comprehensive backtests on 1-2 years of data
- Calculate Sharpe ratio, max drawdown, win rate
- Compare against buy-and-hold S&P 500
- Document final strategy parameters

**Phase 2 Deliverables:**
- Signal generation system with 3-5 signals
- Backtesting framework with historical validation
- Strategy parameters with documented performance
- Risk management rules

## Phase 3: Trading Integration (Weeks 9-12)

### Week 9: Alpaca Integration Setup
```python
import alpaca_trade_api as tradeapi

# Paper trading credentials
API_KEY = "your_paper_key"
SECRET_KEY = "your_paper_secret"
BASE_URL = "https://paper-api.alpaca.markets"

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def get_account_info():
    account = api.get_account()
    print(f"Account Status: {account.status}")
    print(f"Buying Power: ${account.buying_power}")
    print(f"Portfolio Value: ${account.portfolio_value}")

def place_order(symbol, qty, side='buy'):
    try:
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        print(f"Order placed: {side} {qty} shares of {symbol}")
        return order
    except Exception as e:
        print(f"Order failed: {e}")
        return None
```

### Week 10: Basic Trading Logic
```python
def execute_trading_strategy():
    # Get current signals for all stocks
    signals = {}
    for symbol in STOCK_LIST:
        signals[symbol] = calculate_signals(symbol)
    
    # Sort by signal strength
    ranked_signals = sorted(signals.items(), 
                          key=lambda x: x[1]['total_score'], 
                          reverse=True)
    
    # Current positions
    positions = {pos.symbol: int(pos.qty) for pos in api.list_positions()}
    
    # Trading logic
    for symbol, signal in ranked_signals[:10]:  # Top 10 signals
        current_qty = positions.get(symbol, 0)
        target_qty = calculate_position_size(signal['total_score'])
        
        if target_qty > current_qty:
            # Buy more
            qty_to_buy = target_qty - current_qty
            place_order(symbol, qty_to_buy, 'buy')
        elif target_qty < current_qty:
            # Sell some
            qty_to_sell = current_qty - target_qty
            place_order(symbol, qty_to_sell, 'sell')

def calculate_position_size(signal_score, max_position_value=200):
    # Simple position sizing: $100-$200 per position based on signal strength
    if signal_score > 5:
        return max_position_value
    elif signal_score > 3:
        return max_position_value * 0.7
    elif signal_score > 1:
        return max_position_value * 0.4
    else:
        return 0
```

### Week 11: Paper Trading System
- Deploy complete system for paper trading
- Run daily for 2 weeks with monitoring
- Track performance vs benchmarks
- Debug and refine based on results

### Week 12: Risk Management and Monitoring
```python
def check_risk_limits():
    account = api.get_account()
    positions = api.list_positions()
    
    total_value = float(account.portfolio_value)
    
    # Risk checks
    for position in positions:
        position_value = float(position.market_value)
        position_pct = position_value / total_value
        
        # No single position > 10% of portfolio
        if position_pct > 0.10:
            print(f"WARNING: {position.symbol} is {position_pct:.1%} of portfolio")
        
        # Check stop loss (down 15% from entry)
        unrealized_pl_pct = float(position.unrealized_plpc)
        if unrealized_pl_pct < -0.15:
            print(f"STOP LOSS: {position.symbol} down {unrealized_pl_pct:.1%}")
            # Implement stop loss logic
```

**Phase 3 Deliverables:**
- Working paper trading system
- 2 weeks of paper trading results
- Risk management and monitoring
- Performance tracking dashboard

## Phase 4: Live Trading and Optimization (Weeks 13-16)

### Week 13: Live Trading Transition
- Switch from paper to live trading with small amounts
- Start with $500-$1000 total portfolio
- Position sizes of $50-$100 per trade
- Daily monitoring and manual override capability

### Week 14: Performance Monitoring
```python
def generate_performance_report():
    # Get all trades from last 30 days
    trades = api.list_orders(status='filled', limit=100)
    
    # Calculate metrics
    total_return = calculate_total_return()
    sharpe_ratio = calculate_sharpe_ratio()
    max_drawdown = calculate_max_drawdown()
    
    # Generate report
    report = f"""
    CongressWatch Performance Report
    ================================
    Period: Last 30 days
    Total Return: {total_return:.2%}
    Sharpe Ratio: {sharpe_ratio:.2f}
    Max Drawdown: {max_drawdown:.2%}
    
    Top Performing Trades:
    {get_top_trades()}
    
    Worst Performing Trades:
    {get_worst_trades()}
    """
    
    return report
```

### Week 15: Strategy Optimization
- Analyze live trading results vs backtests
- Identify areas for improvement
- Adjust signal weights and thresholds
- Implement lessons learned

### Week 16: Documentation and Future Planning
- Create comprehensive documentation
- Document lessons learned and insights
- Plan next phase improvements
- Consider scaling up if performance is good

**Phase 4 Deliverables:**
- Live trading system with 4 weeks of results
- Performance analysis and optimization
- Complete documentation
- Future development roadmap

## Simple Deployment Guide

### Option 1: Home Computer (Easiest)
```bash
# Set up cron job to run daily at 9 AM
crontab -e

# Add this line:
0 9 * * 1-5 /usr/bin/python3 /path/to/your/congress_trader.py
```

### Option 2: Cloud VPS ($5/month)
```bash
# DigitalOcean droplet setup
ssh root@your_droplet_ip

# Install dependencies
apt update
apt install python3 python3-pip
pip3 install pandas requests alpaca-trade-api

# Upload your code
scp -r congress_trader/ root@your_droplet_ip:/home/

# Set up cron job
crontab -e
0 9 * * 1-5 cd /home/congress_trader && python3 main.py
```

## Budget Breakdown

### Development Phase (One-time costs)
- **Learning Resources**: $50-100 (books, courses)
- **Development Tools**: $0 (using free tools)
- **API Access**: $0 (free tiers sufficient)
- **Total**: $50-100

### Operational Costs (Monthly)
- **Cloud Server**: $5-20/month (optional, can run from home)
- **API Costs**: $0-10/month (free tiers usually sufficient)
- **Trading Account**: $0 (Alpaca has no minimums)
- **Total**: $5-30/month

### Trading Capital
- **Initial Portfolio**: $500-2000 (start small)
- **Position Sizes**: $50-200 per trade
- **Max Positions**: 5-10 at a time

## Success Metrics for Hobby Project

### Technical Success
- [ ] System runs reliably with minimal maintenance
- [ ] Data collection works consistently
- [ ] Trading integration executes orders correctly
- [ ] Basic monitoring and alerting functional

### Educational Success
- [ ] Understanding of congressional trading patterns
- [ ] Experience with quantitative trading concepts
- [ ] Practical API integration skills
- [ ] Basic machine learning and backtesting experience

### Financial Success (Secondary)
- [ ] Positive returns over 6-month period
- [ ] Outperform S&P 500 by 2-5% annually
- [ ] Maximum drawdown under 20%
- [ ] Sharpe ratio > 1.0

## Common Pitfalls to Avoid

### Over-Engineering
- Don't build complex infrastructure you don't need
- Start simple and add complexity only when necessary
- Focus on learning over perfect code

### Unrealistic Expectations
- This is a learning project, not a get-rich-quick scheme
- Expect modest returns and occasional losses
- Value the experience over the profits

### Insufficient Risk Management
- Always use position sizing limits
- Implement stop losses
- Never risk more than you can afford to lose

### Ignoring Costs
- Watch out for API rate limits and costs
- Consider transaction costs in backtesting
- Monitor cloud infrastructure costs

## Next Steps After Completion

### Potential Enhancements
- Add more sophisticated ML models
- Integrate news sentiment analysis
- Implement options trading strategies
- Add cryptocurrency or forex markets

### Career Applications
- Portfolio project for fintech job applications
- Demonstration of quantitative and programming skills
- Foundation for more advanced trading systems
- Experience with financial APIs and data

### Scaling Considerations
- Increase portfolio size gradually based on performance
- Add more data sources and signals
- Consider forming investment club with friends
- Explore regulatory requirements for larger scale

This hobby-scale approach provides a realistic path to building a congressional trading agent while learning valuable skills and keeping costs minimal. The key is to start simple, learn continuously, and gradually increase complexity as your understanding grows.

