import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time

# Note: This would require a Finnhub API key for real usage
# For demonstration, we'll use sample data structure

def get_congressional_trades_sample():
    """
    Sample congressional trading data structure based on Finnhub API format
    In real usage, this would call: https://finnhub.io/api/v1/stock/congressional-trading
    """
    
    # Sample data representing actual patterns found in research
    sample_trades = [
        {
            "symbol": "NVDA",
            "name": "Nancy Pelosi",
            "transactionDate": "2023-12-15",
            "filingDate": "2024-01-20",
            "transactionType": "Purchase",
            "amountFrom": 1000000,
            "amountTo": 5000000,
            "assetName": "NVIDIA Corporation",
            "ownerType": "Spouse"
        },
        {
            "symbol": "AMZN", 
            "name": "Dan Crenshaw",
            "transactionDate": "2020-03-20",
            "filingDate": "2020-09-15",
            "transactionType": "Purchase", 
            "amountFrom": 15000,
            "amountTo": 50000,
            "assetName": "Amazon.com Inc",
            "ownerType": "Self"
        },
        {
            "symbol": "HCA",
            "name": "Richard Burr",
            "transactionDate": "2020-02-13",
            "filingDate": "2020-03-19",
            "transactionType": "Sale",
            "amountFrom": 628000,
            "amountTo": 1720000,
            "assetName": "HCA Healthcare Inc",
            "ownerType": "Self"
        },
        {
            "symbol": "GOOGL",
            "name": "Paul Pelosi",
            "transactionDate": "2020-12-22",
            "filingDate": "2021-01-15",
            "transactionType": "Purchase",
            "amountFrom": 500000,
            "amountTo": 1000000,
            "assetName": "Alphabet Inc Class A",
            "ownerType": "Self"
        },
        {
            "symbol": "TSLA",
            "name": "Austin Scott",
            "transactionDate": "2022-01-10",
            "filingDate": "2022-02-14",
            "transactionType": "Purchase",
            "amountFrom": 1000,
            "amountTo": 15000,
            "assetName": "Tesla Inc",
            "ownerType": "Self"
        }
    ]
    
    return sample_trades

def analyze_trading_patterns(trades_data):
    """
    Analyze congressional trading data for suspicious patterns
    """
    df = pd.DataFrame(trades_data)
    
    # Convert dates
    df['transactionDate'] = pd.to_datetime(df['transactionDate'])
    df['filingDate'] = pd.to_datetime(df['filingDate'])
    
    # Calculate filing delay
    df['filing_delay_days'] = (df['filingDate'] - df['transactionDate']).dt.days
    
    # Calculate average trade size
    df['avg_amount'] = (df['amountFrom'] + df['amountTo']) / 2
    
    # Analyze patterns
    analysis = {
        'total_trades': len(df),
        'unique_members': df['name'].nunique(),
        'avg_filing_delay': df['filing_delay_days'].mean(),
        'late_filings': len(df[df['filing_delay_days'] > 45]),  # STOCK Act requires 45 days
        'large_trades': len(df[df['avg_amount'] > 100000]),
        'by_member': df.groupby('name').agg({
            'avg_amount': ['count', 'mean', 'sum'],
            'filing_delay_days': 'mean'
        }).round(2),
        'by_transaction_type': df['transactionType'].value_counts(),
        'top_stocks': df['symbol'].value_counts()
    }
    
    return df, analysis

def calculate_suspicious_score(member_data):
    """
    Calculate a suspicion score based on various factors
    """
    score = 0
    
    # Large trade amounts (higher amounts = more suspicious)
    avg_amount = member_data['avg_amount'].mean()
    if avg_amount > 1000000:
        score += 3
    elif avg_amount > 100000:
        score += 2
    elif avg_amount > 50000:
        score += 1
    
    # Filing delays (longer delays = more suspicious)
    avg_delay = member_data['filing_delay_days'].mean()
    if avg_delay > 100:
        score += 3
    elif avg_delay > 60:
        score += 2
    elif avg_delay > 45:
        score += 1
    
    # Frequency of trading (more trades = potentially more suspicious)
    trade_count = len(member_data)
    if trade_count > 20:
        score += 2
    elif trade_count > 10:
        score += 1
    
    # Timing around major events (would need additional data)
    # This would require cross-referencing with market events, committee schedules, etc.
    
    return score

# Run the analysis
if __name__ == "__main__":
    print("Congressional Trading Analysis")
    print("=" * 50)
    
    # Get sample data
    trades = get_congressional_trades_sample()
    df, analysis = analyze_trading_patterns(trades)
    
    print(f"Total trades analyzed: {analysis['total_trades']}")
    print(f"Unique members: {analysis['unique_members']}")
    print(f"Average filing delay: {analysis['avg_filing_delay']:.1f} days")
    print(f"Late filings (>45 days): {analysis['late_filings']}")
    print(f"Large trades (>$100k): {analysis['large_trades']}")
    
    print("\nTrading by Member:")
    print(analysis['by_member'])
    
    print("\nTransaction Types:")
    print(analysis['by_transaction_type'])
    
    print("\nMost Traded Stocks:")
    print(analysis['top_stocks'])
    
    # Calculate suspicion scores
    print("\nSuspicion Scores by Member:")
    print("-" * 30)
    
    for member in df['name'].unique():
        member_data = df[df['name'] == member]
        score = calculate_suspicious_score(member_data)
        trade_count = len(member_data)
        avg_amount = member_data['avg_amount'].mean()
        avg_delay = member_data['filing_delay_days'].mean()
        
        print(f"{member}:")
        print(f"  Suspicion Score: {score}/10")
        print(f"  Trades: {trade_count}")
        print(f"  Avg Amount: ${avg_amount:,.0f}")
        print(f"  Avg Filing Delay: {avg_delay:.0f} days")
        print()

