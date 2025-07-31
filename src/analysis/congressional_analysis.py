import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time

# Note: This would require a Finnhub API key for real usage
# For demonstration, we'll use sample data structure

def get_congressional_trades_sample():
    """
    Expanded congressional trading data structure based on Finnhub API format
    In real usage, this would call: https://finnhub.io/api/v1/stock/congressional-trading
    """
    
    # Expanded sample data representing actual patterns found in research
    sample_trades = [
        {
            "symbol": "NVDA",
            "name": "Nancy Pelosi",
            "party": "D",
            "state": "CA",
            "chamber": "House",
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
            "party": "R",
            "state": "TX",
            "chamber": "House",
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
            "party": "R",
            "state": "NC",
            "chamber": "Senate",
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
            "party": "D",
            "state": "CA", 
            "chamber": "House",
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
            "party": "R",
            "state": "GA",
            "chamber": "House",
            "transactionDate": "2022-01-10",
            "filingDate": "2022-02-14",
            "transactionType": "Purchase",
            "amountFrom": 1000,
            "amountTo": 15000,
            "assetName": "Tesla Inc",
            "ownerType": "Self"
        },
        # Additional congressional members
        {
            "symbol": "MSFT",
            "name": "Josh Gottheimer",
            "party": "D",
            "state": "NJ",
            "chamber": "House",
            "transactionDate": "2024-01-15",
            "filingDate": "2024-02-28",
            "transactionType": "Purchase",
            "amountFrom": 50000,
            "amountTo": 100000,
            "assetName": "Microsoft Corporation",
            "ownerType": "Self"
        },
        {
            "symbol": "AAPL",
            "name": "Ro Khanna",
            "party": "D",
            "state": "CA",
            "chamber": "House",
            "transactionDate": "2024-06-23",
            "filingDate": "2024-07-15",
            "transactionType": "Sale",
            "amountFrom": 15000,
            "amountTo": 50000,
            "assetName": "Apple Inc",
            "ownerType": "Self"
        },
        {
            "symbol": "JPM",
            "name": "Pat Toomey",
            "party": "R",
            "state": "PA",
            "chamber": "Senate",
            "transactionDate": "2023-11-10",
            "filingDate": "2023-12-20",
            "transactionType": "Purchase",
            "amountFrom": 100000,
            "amountTo": 250000,
            "assetName": "JPMorgan Chase & Co",
            "ownerType": "Self"
        },
        {
            "symbol": "BAC",
            "name": "Sherrod Brown",
            "party": "D",
            "state": "OH",
            "chamber": "Senate",
            "transactionDate": "2024-03-05",
            "filingDate": "2024-04-12",
            "transactionType": "Sale",
            "amountFrom": 75000,
            "amountTo": 150000,
            "assetName": "Bank of America Corp",
            "ownerType": "Self"
        },
        {
            "symbol": "XOM",
            "name": "Joe Manchin",
            "party": "D",
            "state": "WV",
            "chamber": "Senate",
            "transactionDate": "2024-02-14",
            "filingDate": "2024-03-30",
            "transactionType": "Purchase",
            "amountFrom": 250000,
            "amountTo": 500000,
            "assetName": "Exxon Mobil Corporation",
            "ownerType": "Self"
        },
        {
            "symbol": "PFE",
            "name": "Susan Collins",
            "party": "R",
            "state": "ME",
            "chamber": "Senate",
            "transactionDate": "2023-09-20",
            "filingDate": "2023-10-25",
            "transactionType": "Purchase",
            "amountFrom": 25000,
            "amountTo": 75000,
            "assetName": "Pfizer Inc",
            "ownerType": "Self"
        },
        {
            "symbol": "DIS",
            "name": "Kevin McCarthy",
            "party": "R",
            "state": "CA",
            "chamber": "House",
            "transactionDate": "2024-04-10",
            "filingDate": "2024-05-20",
            "transactionType": "Purchase",
            "amountFrom": 15000,
            "amountTo": 50000,
            "assetName": "The Walt Disney Company",
            "ownerType": "Self"
        },
        {
            "symbol": "META",
            "name": "Alexandria Ocasio-Cortez",
            "party": "D",
            "state": "NY",
            "chamber": "House",
            "transactionDate": "2024-05-15",
            "filingDate": "2024-06-25",
            "transactionType": "Sale",
            "amountFrom": 5000,
            "amountTo": 15000,
            "assetName": "Meta Platforms Inc",
            "ownerType": "Self"
        },
        {
            "symbol": "COIN",
            "name": "Ted Cruz",
            "party": "R",
            "state": "TX",
            "chamber": "Senate",
            "transactionDate": "2024-01-08",
            "filingDate": "2024-02-15",
            "transactionType": "Purchase",
            "amountFrom": 50000,
            "amountTo": 100000,
            "assetName": "Coinbase Global Inc",
            "ownerType": "Self"
        },
        {
            "symbol": "RBLX",
            "name": "Mark Warner",
            "party": "D",
            "state": "VA",
            "chamber": "Senate",
            "transactionDate": "2023-12-01",
            "filingDate": "2024-01-10",
            "transactionType": "Purchase",
            "amountFrom": 15000,
            "amountTo": 50000,
            "assetName": "Roblox Corporation",
            "ownerType": "Self"
        }
    ]
    
    return sample_trades

def get_committee_assignments():
    """
    Congressional committee assignments and oversight areas
    """
    committee_data = {
        "Nancy Pelosi": {
            "committees": ["House Financial Services (Former)", "House Intelligence"],
            "leadership": "Former Speaker of the House",
            "oversight_areas": ["Financial Services", "Intelligence", "Technology Policy"],
            "recent_legislation": ["AI Safety Act", "CHIPS Act Implementation", "Financial Innovation"]
        },
        "Dan Crenshaw": {
            "committees": ["House Energy and Commerce", "House Budget"],
            "leadership": "House Energy Subcommittee Member",
            "oversight_areas": ["Energy Policy", "Healthcare", "Technology"],
            "recent_legislation": ["Energy Independence Act", "Healthcare Innovation", "Tech Regulation"]
        },
        "Richard Burr": {
            "committees": ["Senate Health, Education, Labor and Pensions", "Senate Intelligence (Former Chair)"],
            "leadership": "Former Intelligence Committee Chair",
            "oversight_areas": ["Healthcare", "Intelligence", "Cybersecurity"],
            "recent_legislation": ["Healthcare Price Transparency", "Cybersecurity Standards", "Biotech Regulation"]
        },
        "Josh Gottheimer": {
            "committees": ["House Financial Services", "House Homeland Security"],
            "leadership": "Problem Solvers Caucus Co-Chair",
            "oversight_areas": ["Banking", "Fintech", "Cybersecurity"],
            "recent_legislation": ["Banking Innovation Act", "Cryptocurrency Regulation", "Fintech Oversight"]
        },
        "Ro Khanna": {
            "committees": ["House Oversight", "House Armed Services"],
            "leadership": "Progressive Caucus Deputy Whip",
            "oversight_areas": ["Technology Policy", "Defense", "Trade"],
            "recent_legislation": ["Big Tech Antitrust", "Trade Policy Reform", "AI Ethics Act"]
        },
        "Pat Toomey": {
            "committees": ["Senate Banking (Former Chair)", "Senate Finance"],
            "leadership": "Former Banking Committee Chair",
            "oversight_areas": ["Banking", "Financial Services", "Monetary Policy"],
            "recent_legislation": ["Banking Deregulation", "Cryptocurrency Framework", "Fed Oversight"]
        },
        "Sherrod Brown": {
            "committees": ["Senate Banking (Chair)", "Senate Finance"],
            "leadership": "Banking Committee Chair",
            "oversight_areas": ["Banking", "Consumer Protection", "Housing"],
            "recent_legislation": ["Bank Accountability Act", "Consumer Protection Reform", "Housing Policy"]
        },
        "Joe Manchin": {
            "committees": ["Senate Energy and Natural Resources (Chair)", "Senate Appropriations"],
            "leadership": "Energy Committee Chair",
            "oversight_areas": ["Energy Policy", "Climate", "Infrastructure"],
            "recent_legislation": ["Energy Security Act", "Infrastructure Investment", "Climate Policy"]
        },
        "Susan Collins": {
            "committees": ["Senate Appropriations", "Senate Health, Education, Labor and Pensions"],
            "leadership": "Appropriations Subcommittee Chair",
            "oversight_areas": ["Healthcare", "Education", "Government Funding"],
            "recent_legislation": ["Healthcare Access Act", "Education Funding", "Biomedical Research"]
        },
        "Kevin McCarthy": {
            "committees": ["House Republican Leadership"],
            "leadership": "Former House Speaker",
            "oversight_areas": ["All House Legislative Activity", "Budget", "Strategy"],
            "recent_legislation": ["Budget Reform", "Government Efficiency", "Economic Policy"]
        },
        "Alexandria Ocasio-Cortez": {
            "committees": ["House Oversight", "House Financial Services"],
            "leadership": "Progressive Caucus Member",
            "oversight_areas": ["Financial Services", "Climate", "Social Media"],
            "recent_legislation": ["Green New Deal", "Social Media Regulation", "Financial Justice"]
        },
        "Ted Cruz": {
            "committees": ["Senate Commerce", "Senate Judiciary"],
            "leadership": "Commerce Subcommittee Ranking Member",
            "oversight_areas": ["Technology", "Judiciary", "Commerce"],
            "recent_legislation": ["Big Tech Regulation", "Cryptocurrency Protection", "Free Speech Online"]
        },
        "Mark Warner": {
            "committees": ["Senate Intelligence (Chair)", "Senate Banking"],
            "leadership": "Intelligence Committee Chair",
            "oversight_areas": ["Intelligence", "Cybersecurity", "Technology"],
            "recent_legislation": ["Cybersecurity Standards", "AI Oversight", "Tech Competition"]
        }
    }
    
    return committee_data

def get_current_legislation():
    """
    Current legislation being debated with market implications
    """
    legislation_data = [
        {
            "bill": "AI Safety and Innovation Act (H.R. 2847)",
            "status": "House Committee Markup",
            "committees": ["House Energy and Commerce", "House Science"],
            "market_impact": ["NVDA", "GOOGL", "MSFT", "META"],
            "key_dates": ["Committee Vote: Aug 15, 2025", "Floor Vote: Sept 10, 2025"],
            "description": "Comprehensive AI regulation framework affecting major tech companies"
        },
        {
            "bill": "Banking Innovation Act (S. 1523)",
            "status": "Senate Banking Committee",
            "committees": ["Senate Banking"],
            "market_impact": ["JPM", "BAC", "WFC", "COIN"],
            "key_dates": ["Hearing: Aug 8, 2025", "Markup: Aug 22, 2025"],
            "description": "Modernizes banking regulations for digital assets and fintech"
        },
        {
            "bill": "Energy Security and Independence Act (H.R. 3456)",
            "status": "House-Senate Conference",
            "committees": ["House Energy and Commerce", "Senate Energy"],
            "market_impact": ["XOM", "CVX", "COP", "TSLA"],
            "key_dates": ["Conference Report: Aug 20, 2025", "Final Passage: Sept 5, 2025"],
            "description": "Comprehensive energy policy including fossil fuels and renewables"
        },
        {
            "bill": "Healthcare Price Transparency Act (S. 892)",
            "status": "Senate HELP Committee",
            "committees": ["Senate Health, Education, Labor and Pensions"],
            "market_impact": ["UNH", "ANTM", "CVS", "PFE"],
            "key_dates": ["Committee Vote: Aug 12, 2025"],
            "description": "Requires healthcare price disclosure and reduces prescription costs"
        },
        {
            "bill": "Cryptocurrency Regulation Framework (H.R. 4123)",
            "status": "House Financial Services",
            "committees": ["House Financial Services"],
            "market_impact": ["COIN", "MSTR", "RIOT", "MARA"],
            "key_dates": ["Subcommittee Hearing: Aug 18, 2025", "Full Committee: Sept 3, 2025"],
            "description": "Establishes federal framework for cryptocurrency regulation"
        },
        {
            "bill": "Social Media Accountability Act (S. 2156)",
            "status": "Senate Commerce Committee",
            "committees": ["Senate Commerce"],
            "market_impact": ["META", "GOOGL", "SNAP", "PINS"],
            "key_dates": ["Hearing: Aug 25, 2025", "Markup: Sept 15, 2025"],
            "description": "Content moderation and algorithmic transparency requirements"
        },
        {
            "bill": "Infrastructure Maintenance Act (H.R. 5678)",
            "status": "House Transportation",
            "committees": ["House Transportation and Infrastructure"],
            "market_impact": ["CAT", "DE", "VMC", "MLM"],
            "key_dates": ["Committee Vote: Aug 30, 2025"],
            "description": "Multi-year infrastructure investment and maintenance program"
        }
    ]
    
    return legislation_data

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

