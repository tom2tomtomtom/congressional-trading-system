import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np

def get_enhanced_congressional_data():
    """
    Enhanced dataset with more examples from research findings
    """
    trades = [
        # Nancy Pelosi / Paul Pelosi - High-profile tech trades
        {"symbol": "NVDA", "name": "Nancy Pelosi", "transactionDate": "2023-12-15", "filingDate": "2024-01-20", 
         "transactionType": "Purchase", "amountFrom": 1000000, "amountTo": 5000000, "ownerType": "Spouse",
         "committee": "House Speaker", "sector": "Technology"},
        
        {"symbol": "GOOGL", "name": "Paul Pelosi", "transactionDate": "2020-12-22", "filingDate": "2021-01-15",
         "transactionType": "Purchase", "amountFrom": 500000, "amountTo": 1000000, "ownerType": "Self",
         "committee": "House Speaker", "sector": "Technology"},
         
        {"symbol": "AAPL", "name": "Nancy Pelosi", "transactionDate": "2022-07-01", "filingDate": "2022-08-15",
         "transactionType": "Purchase", "amountFrom": 1000000, "amountTo": 5000000, "ownerType": "Spouse",
         "committee": "House Speaker", "sector": "Technology"},
        
        # Dan Crenshaw - Pandemic timing
        {"symbol": "AMZN", "name": "Dan Crenshaw", "transactionDate": "2020-03-20", "filingDate": "2020-09-15",
         "transactionType": "Purchase", "amountFrom": 15000, "amountTo": 50000, "ownerType": "Self",
         "committee": "Energy and Commerce", "sector": "Technology"},
         
        {"symbol": "ZM", "name": "Dan Crenshaw", "transactionDate": "2020-03-25", "filingDate": "2020-09-15",
         "transactionType": "Purchase", "amountFrom": 1000, "amountTo": 15000, "ownerType": "Self",
         "committee": "Energy and Commerce", "sector": "Technology"},
        
        # Richard Burr - COVID-19 sell-off
        {"symbol": "HCA", "name": "Richard Burr", "transactionDate": "2020-02-13", "filingDate": "2020-03-19",
         "transactionType": "Sale", "amountFrom": 628000, "amountTo": 1720000, "ownerType": "Self",
         "committee": "Intelligence Committee", "sector": "Healthcare"},
         
        {"symbol": "VFC", "name": "Richard Burr", "transactionDate": "2020-02-13", "filingDate": "2020-03-19",
         "transactionType": "Sale", "amountFrom": 1000, "amountTo": 15000, "ownerType": "Self",
         "committee": "Intelligence Committee", "sector": "Consumer"},
        
        # Kelly Loeffler - COVID-19 trades
        {"symbol": "ICE", "name": "Kelly Loeffler", "transactionDate": "2020-01-24", "filingDate": "2020-02-26",
         "transactionType": "Purchase", "amountFrom": 50000, "amountTo": 100000, "ownerType": "Self",
         "committee": "Agriculture Committee", "sector": "Financial"},
         
        {"symbol": "CARR", "name": "Kelly Loeffler", "transactionDate": "2020-03-11", "filingDate": "2020-04-15",
         "transactionType": "Sale", "amountFrom": 15000, "amountTo": 50000, "ownerType": "Self",
         "committee": "Agriculture Committee", "sector": "Industrial"},
        
        # John Kerry - Healthcare (historical example)
        {"symbol": "UNH", "name": "John Kerry", "transactionDate": "2009-06-15", "filingDate": "2009-07-30",
         "transactionType": "Purchase", "amountFrom": 100000, "amountTo": 250000, "ownerType": "Self",
         "committee": "Health Subcommittee", "sector": "Healthcare"},
         
        # More recent examples
        {"symbol": "RBLX", "name": "Austin Scott", "transactionDate": "2022-01-10", "filingDate": "2022-02-14",
         "transactionType": "Purchase", "amountFrom": 1000, "amountTo": 15000, "ownerType": "Self",
         "committee": "Agriculture Committee", "sector": "Technology"},
         
        {"symbol": "DIS", "name": "Josh Gottheimer", "transactionDate": "2021-11-05", "filingDate": "2021-12-20",
         "transactionType": "Purchase", "amountFrom": 15000, "amountTo": 50000, "ownerType": "Self",
         "committee": "Financial Services", "sector": "Media"}
    ]
    
    return trades

def enhanced_suspicious_scoring(df):
    """
    More sophisticated scoring algorithm
    """
    scores = {}
    
    for member in df['name'].unique():
        member_data = df[df['name'] == member]
        score = 0
        factors = []
        
        # Factor 1: Trade size relative to typical congressional wealth
        avg_amount = member_data['avg_amount'].mean()
        if avg_amount > 2000000:
            score += 4
            factors.append("Very large trades (>$2M)")
        elif avg_amount > 500000:
            score += 3
            factors.append("Large trades (>$500K)")
        elif avg_amount > 100000:
            score += 2
            factors.append("Moderate trades (>$100K)")
        
        # Factor 2: Filing compliance
        avg_delay = member_data['filing_delay_days'].mean()
        max_delay = member_data['filing_delay_days'].max()
        if max_delay > 180:
            score += 4
            factors.append("Severely late filings (>180 days)")
        elif max_delay > 90:
            score += 3
            factors.append("Very late filings (>90 days)")
        elif max_delay > 45:
            score += 2
            factors.append("Late filings (>45 days)")
        
        # Factor 3: Trading frequency
        trade_count = len(member_data)
        if trade_count > 10:
            score += 3
            factors.append("High frequency trading")
        elif trade_count > 5:
            score += 2
            factors.append("Moderate frequency trading")
        elif trade_count > 2:
            score += 1
            factors.append("Multiple trades")
        
        # Factor 4: Sector concentration (potential committee conflicts)
        sectors = member_data['sector'].value_counts()
        if len(sectors) == 1 and trade_count > 1:
            score += 2
            factors.append("Concentrated in single sector")
        
        # Factor 5: Timing patterns (simplified - would need market data)
        # Check for trades during major market events
        covid_period = (member_data['transactionDate'] >= '2020-02-01') & (member_data['transactionDate'] <= '2020-04-30')
        if covid_period.any():
            score += 2
            factors.append("Traded during COVID-19 crisis")
        
        # Factor 6: Use of spouse/family accounts
        if 'Spouse' in member_data['ownerType'].values:
            score += 1
            factors.append("Uses spouse account")
        
        scores[member] = {
            'score': min(score, 10),  # Cap at 10
            'factors': factors,
            'trade_count': trade_count,
            'avg_amount': avg_amount,
            'avg_delay': avg_delay,
            'max_delay': max_delay
        }
    
    return scores

def create_visualizations(df, scores):
    """
    Create visualizations of the analysis
    """
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Congressional Trading Analysis', fontsize=16, fontweight='bold')
    
    # 1. Suspicion scores by member
    members = list(scores.keys())
    score_values = [scores[m]['score'] for m in members]
    
    axes[0,0].barh(members, score_values, color=['red' if s >= 7 else 'orange' if s >= 4 else 'yellow' if s >= 2 else 'green' for s in score_values])
    axes[0,0].set_xlabel('Suspicion Score (0-10)')
    axes[0,0].set_title('Suspicion Scores by Member')
    axes[0,0].grid(axis='x', alpha=0.3)
    
    # 2. Trade amounts vs filing delays
    axes[0,1].scatter(df['filing_delay_days'], df['avg_amount'], 
                     c=[scores[name]['score'] for name in df['name']], 
                     cmap='Reds', s=100, alpha=0.7)
    axes[0,1].set_xlabel('Filing Delay (days)')
    axes[0,1].set_ylabel('Average Trade Amount ($)')
    axes[0,1].set_title('Trade Amount vs Filing Delay')
    axes[0,1].axvline(x=45, color='red', linestyle='--', alpha=0.5, label='Legal limit (45 days)')
    axes[0,1].legend()
    axes[0,1].set_yscale('log')
    
    # 3. Trading by sector
    sector_counts = df['sector'].value_counts()
    axes[1,0].pie(sector_counts.values, labels=sector_counts.index, autopct='%1.1f%%')
    axes[1,0].set_title('Trading by Sector')
    
    # 4. Timeline of trades
    df_sorted = df.sort_values('transactionDate')
    trade_dates = pd.to_datetime(df_sorted['transactionDate'])
    
    # Color by suspicion score
    colors = [scores[name]['score'] for name in df_sorted['name']]
    scatter = axes[1,1].scatter(trade_dates, df_sorted['avg_amount'], 
                               c=colors, cmap='Reds', s=100, alpha=0.7)
    axes[1,1].set_xlabel('Transaction Date')
    axes[1,1].set_ylabel('Trade Amount ($)')
    axes[1,1].set_title('Timeline of Trades')
    axes[1,1].set_yscale('log')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # Add COVID-19 period shading
    covid_start = pd.to_datetime('2020-02-01')
    covid_end = pd.to_datetime('2020-04-30')
    axes[1,1].axvspan(covid_start, covid_end, alpha=0.2, color='red', label='COVID-19 Crisis')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/congressional_trading_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_report(df, scores):
    """
    Generate a detailed report of findings
    """
    print("CONGRESSIONAL TRADING SUSPICION ANALYSIS")
    print("=" * 60)
    print()
    
    # Overall statistics
    total_trades = len(df)
    total_volume = df['avg_amount'].sum()
    late_filings = len(df[df['filing_delay_days'] > 45])
    
    print(f"OVERVIEW:")
    print(f"Total trades analyzed: {total_trades}")
    print(f"Total trading volume: ${total_volume:,.0f}")
    print(f"Late filings (>45 days): {late_filings} ({late_filings/total_trades*100:.1f}%)")
    print(f"Average filing delay: {df['filing_delay_days'].mean():.1f} days")
    print()
    
    # Rank members by suspicion score
    ranked_members = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    
    print("SUSPICION RANKING:")
    print("-" * 40)
    
    for i, (member, data) in enumerate(ranked_members, 1):
        risk_level = "HIGH" if data['score'] >= 7 else "MEDIUM" if data['score'] >= 4 else "LOW"
        
        print(f"{i}. {member}")
        print(f"   Suspicion Score: {data['score']}/10 ({risk_level} RISK)")
        print(f"   Trades: {data['trade_count']}")
        print(f"   Avg Amount: ${data['avg_amount']:,.0f}")
        print(f"   Max Filing Delay: {data['max_delay']} days")
        print(f"   Suspicious Factors:")
        for factor in data['factors']:
            print(f"     • {factor}")
        print()
    
    # Sector analysis
    print("SECTOR ANALYSIS:")
    print("-" * 20)
    sector_analysis = df.groupby('sector').agg({
        'avg_amount': ['count', 'sum', 'mean'],
        'filing_delay_days': 'mean'
    }).round(2)
    print(sector_analysis)
    print()
    
    # Committee conflicts
    print("POTENTIAL COMMITTEE CONFLICTS:")
    print("-" * 35)
    covid_start = pd.to_datetime('2020-02-01')
    covid_end = pd.to_datetime('2020-04-30')
    
    for _, row in df.iterrows():
        if (row['committee'] == 'Intelligence Committee' and row['transactionDate'] >= covid_start and row['transactionDate'] <= covid_end):
            print(f"⚠️  {row['name']}: {row['transactionType']} {row['symbol']} during COVID briefings")
        elif (row['committee'] == 'Health Subcommittee' and row['sector'] == 'Healthcare'):
            print(f"⚠️  {row['name']}: Healthcare trades while on health committee")
        elif (row['committee'] == 'House Speaker' and row['avg_amount'] > 1000000):
            print(f"⚠️  {row['name']}: Large trades with access to all legislative information")

# Run enhanced analysis
if __name__ == "__main__":
    # Load enhanced dataset
    trades = get_enhanced_congressional_data()
    df = pd.DataFrame(trades)
    
    # Process data
    df['transactionDate'] = pd.to_datetime(df['transactionDate'])
    df['filingDate'] = pd.to_datetime(df['filingDate'])
    df['filing_delay_days'] = (df['filingDate'] - df['transactionDate']).dt.days
    df['avg_amount'] = (df['amountFrom'] + df['amountTo']) / 2
    
    # Calculate suspicion scores
    scores = enhanced_suspicious_scoring(df)
    
    # Generate report
    generate_report(df, scores)
    
    # Create visualizations
    create_visualizations(df, scores)
    
    print("\nVisualization saved as 'congressional_trading_analysis.png'")

