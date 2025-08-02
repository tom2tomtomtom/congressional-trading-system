#!/usr/bin/env python3
"""
Comprehensive Congressional Trading Analysis - All 535+ Members
Enhanced analysis system using the full congressional database.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_full_congressional_data():
    """Load the comprehensive congressional database."""
    
    # Try to load from generated files first
    members_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'congressional_members_full.json')
    trades_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'congressional_trades_full.json')
    
    if os.path.exists(members_file) and os.path.exists(trades_file):
        with open(members_file, 'r') as f:
            members_data = json.load(f)
        
        with open(trades_file, 'r') as f:
            trades_data = json.load(f)
        
        print(f"✅ Loaded full congressional database:")
        print(f"   • {len(members_data)} congressional members")
        print(f"   • {len(trades_data)} trades")
        
        return members_data, trades_data
    else:
        print("❌ Full congressional database not found. Generating now...")
        # Import and generate the database
        from data.full_congressional_database import save_congressional_data
        return save_congressional_data()

def analyze_full_congressional_trading():
    """Comprehensive analysis of all congressional trading data."""
    
    print("🏛️ COMPREHENSIVE CONGRESSIONAL TRADING ANALYSIS")
    print("=" * 65)
    print("Analysis of ALL 535+ Congressional Members")
    print()
    
    # Load full data
    members_data, trades_data = load_full_congressional_data()
    
    # Convert to DataFrames
    members_df = pd.DataFrame(members_data)
    trades_df = pd.DataFrame(trades_data)
    
    # Process trading data
    trades_df['transaction_date'] = pd.to_datetime(trades_df['transaction_date'])
    trades_df['filing_date'] = pd.to_datetime(trades_df['filing_date'])
    trades_df['filing_delay_days'] = (trades_df['filing_date'] - trades_df['transaction_date']).dt.days
    trades_df['avg_amount'] = (trades_df['amount_from'] + trades_df['amount_to']) / 2
    
    # Generate comprehensive analysis
    print("📊 OVERALL STATISTICS")
    print("-" * 25)
    print(f"Total Congressional Members: {len(members_df)}")
    print(f"  • House Representatives: {len(members_df[members_df['chamber'] == 'House'])}")
    print(f"  • Senators: {len(members_df[members_df['chamber'] == 'Senate'])}")
    print()
    
    print(f"Party Breakdown:")
    party_counts = members_df['party'].value_counts()
    for party, count in party_counts.items():
        percentage = (count / len(members_df)) * 100
        print(f"  • {party}: {count} members ({percentage:.1f}%)")
    print()
    
    print(f"Trading Activity Overview:")
    print(f"  • Total trades analyzed: {len(trades_df)}")
    print(f"  • Members with trading activity: {trades_df['member_id'].nunique()}")
    print(f"  • Members with no trades: {len(members_df) - trades_df['member_id'].nunique()}")
    
    # Trading volume analysis
    total_volume = trades_df['avg_amount'].sum()
    print(f"  • Total trading volume: ${total_volume:,.0f}")
    print(f"  • Average trade amount: ${trades_df['avg_amount'].mean():,.0f}")
    print(f"  • Median trade amount: ${trades_df['avg_amount'].median():,.0f}")
    print()
    
    # Filing compliance analysis
    late_filings = len(trades_df[trades_df['filing_delay_days'] > 45])
    compliance_rate = ((len(trades_df) - late_filings) / len(trades_df)) * 100
    print(f"Filing Compliance (STOCK Act):")
    print(f"  • Average filing delay: {trades_df['filing_delay_days'].mean():.1f} days")
    print(f"  • Late filings (>45 days): {late_filings} ({(late_filings/len(trades_df)*100):.1f}%)")
    print(f"  • Compliance rate: {compliance_rate:.1f}%")
    print()
    
    # High-volume traders analysis
    member_trading_summary = trades_df.groupby(['member_id', 'member_name']).agg({
        'avg_amount': ['count', 'sum', 'mean'],
        'filing_delay_days': 'mean'
    }).round(2)
    
    member_trading_summary.columns = ['trade_count', 'total_volume', 'avg_trade_size', 'avg_filing_delay']
    member_trading_summary = member_trading_summary.sort_values('total_volume', ascending=False)
    
    # Calculate suspicion scores for all members
    suspicion_scores = calculate_comprehensive_suspicion_scores(trades_df, members_df)
    
    print("🚨 TOP 20 HIGH-VOLUME TRADERS")
    print("-" * 35)
    for i, (_, member) in enumerate(member_trading_summary.head(20).iterrows()):
        member_id = member.name[0]
        member_name = member.name[1]
        suspicion_score = suspicion_scores.get(member_id, 0)
        
        print(f"{i+1:2d}. {member_name}")
        print(f"     Trades: {member['trade_count']:3.0f} | Volume: ${member['total_volume']:>10,.0f} | "
              f"Avg: ${member['avg_trade_size']:>8,.0f}")
        print(f"     Filing Delay: {member['avg_filing_delay']:4.1f} days | "
              f"Suspicion Score: {suspicion_score:.1f}/10")
        print()
    
    # Sector analysis
    print("📈 SECTOR ANALYSIS")
    print("-" * 18)
    sector_mapping = get_sector_mapping()
    trades_df['sector'] = trades_df['symbol'].map(sector_mapping)
    sector_analysis = trades_df.groupby('sector').agg({
        'avg_amount': ['count', 'sum'],
        'member_id': 'nunique'
    }).round(2)
    sector_analysis.columns = ['trade_count', 'total_volume', 'unique_members']
    sector_analysis = sector_analysis.sort_values('total_volume', ascending=False)
    
    for sector, data in sector_analysis.iterrows():
        percentage = (data['total_volume'] / total_volume) * 100
        print(f"{sector:15s}: {data['trade_count']:4.0f} trades | "
              f"${data['total_volume']:>12,.0f} ({percentage:4.1f}%) | "
              f"{data['unique_members']:3.0f} members")
    print()
    
    # Committee analysis
    print("🏛️ COMMITTEE ANALYSIS")
    print("-" * 20)
    committee_analysis = analyze_committee_trading_patterns(trades_df, members_df)
    
    for idx, data in committee_analysis.head(15).iterrows():
        committee = data['committee']
        print(f"{committee[:25]:25s}: {data['member_count']:2.0f} members | "
              f"{data['total_trades']:4.0f} trades | "
              f"${data['avg_trade_size']:>8,.0f} avg | "
              f"Risk: {data['avg_suspicion']:3.1f}/10")
    print()
    
    # High-risk members
    high_risk_members = [(member_id, score) for member_id, score in suspicion_scores.items() if score >= 7.0]
    high_risk_members.sort(key=lambda x: x[1], reverse=True)
    
    print("⚠️ HIGH-RISK MEMBERS (Score ≥ 7.0)")
    print("-" * 30)
    if high_risk_members:
        for member_id, score in high_risk_members[:15]:
            member_info = members_df[members_df['id'] == member_id].iloc[0]
            member_trades = trades_df[trades_df['member_id'] == member_id]
            
            print(f"{member_info['name']} ({member_info['party']}-{member_info['state']})")
            print(f"  Suspicion Score: {score:.1f}/10")
            print(f"  Trades: {len(member_trades)} | Volume: ${member_trades['avg_amount'].sum():,.0f}")
            print(f"  Committee: {member_info.get('committee', 'Unknown')}")
            print(f"  Avg Filing Delay: {member_trades['filing_delay_days'].mean():.1f} days")
            print()
    else:
        print("No members with suspicion scores ≥ 7.0")
        print()
    
    # Generate summary statistics
    summary_stats = {
        'total_members': len(members_df),
        'total_trades': len(trades_df),
        'trading_members': trades_df['member_id'].nunique(),
        'total_volume': total_volume,
        'compliance_rate': compliance_rate,
        'high_risk_members': len(high_risk_members),
        'avg_suspicion_score': np.mean(list(suspicion_scores.values()))
    }
    
    return members_df, trades_df, summary_stats, suspicion_scores

def calculate_comprehensive_suspicion_scores(trades_df, members_df):
    """Calculate suspicion scores for all trading members."""
    
    suspicion_scores = {}
    
    for member_id in trades_df['member_id'].unique():
        member_trades = trades_df[trades_df['member_id'] == member_id]
        member_info = members_df[members_df['id'] == member_id].iloc[0]
        
        score = 0
        
        # Factor 1: Trade size relative to net worth
        avg_amount = member_trades['avg_amount'].mean()
        net_worth = member_info.get('net_worth', 1000000)
        
        if avg_amount > net_worth * 0.15:  # >15% of net worth
            score += 3
        elif avg_amount > net_worth * 0.08:  # >8% of net worth
            score += 2
        elif avg_amount > net_worth * 0.03:  # >3% of net worth
            score += 1
        
        # Factor 2: Filing delays
        avg_delay = member_trades['filing_delay_days'].mean()
        max_delay = member_trades['filing_delay_days'].max()
        
        if max_delay > 120:
            score += 3
        elif max_delay > 60:
            score += 2
        elif max_delay > 45:
            score += 1
        
        # Factor 3: Trading frequency
        trade_count = len(member_trades)
        if trade_count > 20:
            score += 2
        elif trade_count > 10:
            score += 1
        
        # Factor 4: Committee conflicts
        committee = member_info.get('committee', '').lower()
        sectors = member_trades['symbol'].map(get_sector_mapping()).unique()
        
        # Check for potential conflicts
        if ('financial' in committee or 'banking' in committee) and 'Financial' in sectors:
            score += 2
        if 'intelligence' in committee:
            score += 1  # Intelligence committee members get extra scrutiny
        if 'energy' in committee and 'Energy' in sectors:
            score += 1
        
        # Factor 5: Spouse/family trading
        if 'Spouse' in member_trades['owner_type'].values:
            score += 1
        
        suspicion_scores[member_id] = min(score, 10)  # Cap at 10
    
    return suspicion_scores

def analyze_committee_trading_patterns(trades_df, members_df):
    """Analyze trading patterns by committee assignment."""
    
    # Create committee trading summary
    committee_data = []
    
    for committee in members_df['committee'].unique():
        if pd.isna(committee):
            continue
            
        committee_members = members_df[members_df['committee'] == committee]
        committee_member_ids = committee_members['id'].tolist()
        committee_trades = trades_df[trades_df['member_id'].isin(committee_member_ids)]
        
        if len(committee_trades) == 0:
            continue
        
        # Calculate suspicion scores for committee members
        committee_suspicion_scores = []
        for member_id in committee_member_ids:
            member_trades = trades_df[trades_df['member_id'] == member_id]
            if len(member_trades) > 0:
                member_info = members_df[members_df['id'] == member_id].iloc[0]
                # Simplified suspicion calculation
                avg_amount = member_trades['avg_amount'].mean()
                avg_delay = member_trades['filing_delay_days'].mean()
                score = min(10, (avg_amount / 500000) * 3 + (max(0, avg_delay - 30) / 30) * 2)
                committee_suspicion_scores.append(score)
        
        committee_data.append({
            'committee': committee,
            'member_count': len(committee_members),
            'trading_members': len(set(committee_trades['member_id'])),
            'total_trades': len(committee_trades),
            'total_volume': committee_trades['avg_amount'].sum(),
            'avg_trade_size': committee_trades['avg_amount'].mean(),
            'avg_filing_delay': committee_trades['filing_delay_days'].mean(),
            'avg_suspicion': np.mean(committee_suspicion_scores) if committee_suspicion_scores else 0
        })
    
    committee_df = pd.DataFrame(committee_data)
    return committee_df.sort_values('avg_suspicion', ascending=False)

def get_sector_mapping():
    """Get sector mapping for stock symbols."""
    return {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
        'META': 'Technology', 'NVDA': 'Technology', 'TSLA': 'Technology', 'ORCL': 'Technology',
        'CRM': 'Technology', 'ADBE': 'Technology',
        'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial', 'GS': 'Financial',
        'MS': 'Financial', 'C': 'Financial', 'USB': 'Financial', 'PNC': 'Financial',
        'TFC': 'Financial', 'COF': 'Financial',
        'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'ABT': 'Healthcare',
        'TMO': 'Healthcare', 'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare',
        'GILD': 'Healthcare', 'BIIB': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
        'SLB': 'Energy', 'MPC': 'Energy', 'VLO': 'Energy', 'PSX': 'Energy',
        'KMI': 'Energy', 'OKE': 'Energy',
        'WMT': 'Consumer', 'HD': 'Consumer', 'PG': 'Consumer', 'KO': 'Consumer',
        'PEP': 'Consumer', 'COST': 'Consumer', 'LOW': 'Consumer', 'TGT': 'Consumer',
        'SBUX': 'Consumer', 'MCD': 'Consumer',
        'BA': 'Industrial', 'CAT': 'Industrial', 'GE': 'Industrial', 'MMM': 'Industrial',
        'HON': 'Industrial', 'UPS': 'Industrial', 'LMT': 'Industrial', 'RTX': 'Industrial',
        'DE': 'Industrial', 'EMR': 'Industrial'
    }

def export_comprehensive_analysis(members_df, trades_df, output_dir='analysis_output'):
    """Export comprehensive analysis results."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Export member summary
    member_summary = members_df.copy()
    member_summary.to_csv(f'{output_dir}/congressional_members_summary.csv', index=False)
    
    # Export trading data
    trades_df.to_csv(f'{output_dir}/congressional_trades_analysis.csv', index=False)
    
    # Export high-level statistics
    stats = {
        'analysis_date': datetime.now().isoformat(),
        'total_members': len(members_df),
        'total_trades': len(trades_df),
        'trading_members': trades_df['member_id'].nunique(),
        'total_volume': float(trades_df['avg_amount'].sum()),
        'avg_trade_amount': float(trades_df['avg_amount'].mean()),
        'compliance_rate': float(((len(trades_df) - len(trades_df[trades_df['filing_delay_days'] > 45])) / len(trades_df)) * 100)
    }
    
    with open(f'{output_dir}/analysis_summary.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n📁 Analysis exported to {output_dir}/")
    print(f"   • congressional_members_summary.csv")
    print(f"   • congressional_trades_analysis.csv") 
    print(f"   • analysis_summary.json")

if __name__ == "__main__":
    try:
        members_df, trades_df, summary_stats, suspicion_scores = analyze_full_congressional_trading()
        
        print("🎊 COMPREHENSIVE ANALYSIS COMPLETE")
        print("=" * 40)
        print(f"✅ Analyzed {summary_stats['total_members']} congressional members")
        print(f"✅ Processed {summary_stats['total_trades']} trades")
        print(f"✅ Identified {summary_stats['high_risk_members']} high-risk members")
        print(f"✅ Overall compliance rate: {summary_stats['compliance_rate']:.1f}%")
        
        # Export results
        export_comprehensive_analysis(members_df, trades_df)
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()