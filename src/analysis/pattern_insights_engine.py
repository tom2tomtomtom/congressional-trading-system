#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - AI-Powered Pattern Insights
Educational research insights based on AI/ML analysis of congressional trading patterns.

IMPORTANT: This system provides educational insights for transparency and research.
NOT financial advice or trading recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json

class PatternInsightsEngine:
    """
    Generate educational insights about congressional trading patterns
    using AI/ML analysis for transparency and accountability research.
    """

    def __init__(self):
        self.insights = []
        self.confidence_threshold = 0.7

    def generate_comprehensive_insights(self, trades_df, committee_data, legislation_data):
        """
        Generate comprehensive AI-powered insights about trading patterns.
        Returns educational research findings, NOT trading recommendations.
        """
        insights = {
            'timing_patterns': self._analyze_timing_patterns(trades_df, legislation_data),
            'committee_correlations': self._analyze_committee_correlations(trades_df, committee_data),
            'unusual_activity': self._detect_unusual_activity(trades_df),
            'sector_trends': self._analyze_sector_trends(trades_df),
            'filing_compliance': self._analyze_filing_patterns(trades_df),
            'educational_insights': self._generate_educational_insights(trades_df, committee_data, legislation_data),
            'research_questions': self._generate_research_questions(trades_df, committee_data),
            'transparency_metrics': self._calculate_transparency_metrics(trades_df)
        }

        return insights

    def _analyze_timing_patterns(self, trades_df, legislation_data):
        """
        Analyze timing patterns between trades and legislative events.
        Educational insight: Understanding potential information advantages.
        """
        patterns = []

        # Group trades by member
        for member in trades_df['name'].unique():
            member_trades = trades_df[trades_df['name'] == member]

            # Analyze trade timing relative to market events
            for _, trade in member_trades.iterrows():
                trade_date = pd.to_datetime(trade['transactionDate'])

                # Check for legislation correlation
                for legislation in legislation_data:
                    leg_dates = self._extract_legislation_dates(legislation)

                    for leg_date_str, event_type in leg_dates:
                        leg_date = pd.to_datetime(leg_date_str)
                        days_before = (leg_date - trade_date).days

                        # Trade occurred 1-90 days before legislative event
                        if 1 <= days_before <= 90:
                            # Check if stock is related to legislation
                            if trade['symbol'] in legislation.get('market_impact', []):
                                pattern = {
                                    'type': 'legislative_timing',
                                    'member': member,
                                    'stock': trade['symbol'],
                                    'trade_date': trade_date.strftime('%Y-%m-%d'),
                                    'trade_type': trade['transactionType'],
                                    'amount': trade['avg_amount'],
                                    'legislation': legislation['bill'],
                                    'event_type': event_type,
                                    'days_advance': days_before,
                                    'confidence': self._calculate_timing_confidence(days_before, trade['avg_amount']),
                                    'insight': f"Trade occurred {days_before} days before {event_type}",
                                    'educational_note': "Pattern suggests potential information advantage - for research purposes only"
                                }
                                patterns.append(pattern)

        # Sort by confidence
        patterns.sort(key=lambda x: x['confidence'], reverse=True)
        return patterns[:10]  # Top 10 patterns

    def _extract_legislation_dates(self, legislation):
        """Extract key dates from legislation."""
        dates = []
        for key_date in legislation.get('key_dates', []):
            # Parse "Committee Vote: Aug 15, 2025" format
            if ':' in key_date:
                event_type, date_str = key_date.split(':', 1)
                try:
                    # Simple date parsing for common formats
                    date_str = date_str.strip()
                    # Convert "Aug 15, 2025" to "2025-08-15"
                    parsed_date = pd.to_datetime(date_str)
                    dates.append((parsed_date.strftime('%Y-%m-%d'), event_type.strip()))
                except:
                    pass
        return dates

    def _calculate_timing_confidence(self, days_before, amount):
        """Calculate confidence score for timing pattern."""
        # Higher confidence for trades closer to events and larger amounts
        recency_score = max(0, 1 - (days_before / 90))  # 0-1 scale
        amount_score = min(1.0, amount / 1000000)  # 0-1 scale, normalized by $1M

        confidence = (recency_score * 0.6 + amount_score * 0.4)
        return round(confidence, 2)

    def _analyze_committee_correlations(self, trades_df, committee_data):
        """
        Analyze correlation between committee assignments and trading sectors.
        Educational insight: Understanding oversight conflicts.
        """
        correlations = []

        sector_mapping = {
            'NVDA': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'AAPL': 'Technology', 'META': 'Technology', 'AMZN': 'Technology',
            'TSLA': 'Automotive/Energy', 'COIN': 'Financial/Crypto', 'RBLX': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'XOM': 'Energy', 'CVX': 'Energy', 'PFE': 'Healthcare',
            'UNH': 'Healthcare', 'DIS': 'Media', 'HCA': 'Healthcare'
        }

        for member, committees in committee_data.items():
            member_trades = trades_df[trades_df['name'] == member]

            if len(member_trades) == 0:
                continue

            # Analyze overlap between committee oversight and trading sectors
            oversight_areas = committees.get('oversight_areas', [])

            for _, trade in member_trades.iterrows():
                stock_sector = sector_mapping.get(trade['symbol'], 'Other')

                # Check for oversight-sector overlap
                oversight_match = any(
                    area.lower() in stock_sector.lower() or stock_sector.lower() in area.lower()
                    for area in oversight_areas
                )

                if oversight_match:
                    correlation = {
                        'member': member,
                        'committee_role': committees.get('leadership', 'Member'),
                        'oversight_areas': oversight_areas,
                        'stock': trade['symbol'],
                        'sector': stock_sector,
                        'amount': trade['avg_amount'],
                        'trade_type': trade['transactionType'],
                        'confidence': 0.85 if 'Chair' in committees.get('leadership', '') else 0.70,
                        'insight': f"Committee oversight of {stock_sector} aligns with {trade['symbol']} trading",
                        'educational_note': "Potential conflict of interest - educational transparency finding"
                    }
                    correlations.append(correlation)

        # Sort by amount and confidence
        correlations.sort(key=lambda x: (x['confidence'], x['amount']), reverse=True)
        return correlations[:15]  # Top 15 correlations

    def _detect_unusual_activity(self, trades_df):
        """
        Detect statistically unusual trading activity.
        Educational insight: Identifying outliers for research.
        """
        unusual_patterns = []

        # Calculate statistical baselines
        mean_amount = trades_df['avg_amount'].mean()
        std_amount = trades_df['avg_amount'].std()
        mean_delay = trades_df['filing_delay_days'].mean()
        std_delay = trades_df['filing_delay_days'].std()

        for _, trade in trades_df.iterrows():
            anomalies = []
            confidence = 0.5

            # Unusually large trade (>2 std deviations)
            if trade['avg_amount'] > mean_amount + (2 * std_amount):
                anomalies.append('exceptionally_large_amount')
                confidence += 0.2

            # Unusually late filing (>2 std deviations)
            if trade['filing_delay_days'] > mean_delay + (2 * std_delay):
                anomalies.append('exceptionally_late_filing')
                confidence += 0.15

            # Very late filing (>45 days = STOCK Act violation)
            if trade['filing_delay_days'] > 45:
                anomalies.append('stock_act_violation')
                confidence += 0.25

            if anomalies:
                pattern = {
                    'member': trade['name'],
                    'stock': trade['symbol'],
                    'amount': trade['avg_amount'],
                    'filing_delay': trade['filing_delay_days'],
                    'anomalies': anomalies,
                    'confidence': min(1.0, confidence),
                    'z_score_amount': (trade['avg_amount'] - mean_amount) / std_amount if std_amount > 0 else 0,
                    'z_score_delay': (trade['filing_delay_days'] - mean_delay) / std_delay if std_delay > 0 else 0,
                    'insight': f"Statistical outlier: {', '.join(anomalies)}",
                    'educational_note': "Unusual pattern detected - warrants further research"
                }
                unusual_patterns.append(pattern)

        # Sort by confidence
        unusual_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        return unusual_patterns[:10]

    def _analyze_sector_trends(self, trades_df):
        """
        Analyze sector-level trading trends.
        Educational insight: Identifying sectoral focus areas.
        """
        sector_mapping = {
            'NVDA': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'AAPL': 'Technology', 'META': 'Technology', 'AMZN': 'Technology',
            'TSLA': 'Automotive/Energy', 'COIN': 'Financial', 'RBLX': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'XOM': 'Energy', 'CVX': 'Energy', 'PFE': 'Healthcare',
            'UNH': 'Healthcare', 'DIS': 'Media', 'HCA': 'Healthcare'
        }

        # Add sector column
        trades_with_sector = trades_df.copy()
        trades_with_sector['sector'] = trades_with_sector['symbol'].map(sector_mapping).fillna('Other')

        # Analyze by sector
        sector_analysis = trades_with_sector.groupby('sector').agg({
            'avg_amount': ['sum', 'mean', 'count'],
            'name': 'nunique'
        }).round(2)

        trends = []
        for sector in sector_analysis.index:
            total_volume = sector_analysis.loc[sector, ('avg_amount', 'sum')]
            avg_trade = sector_analysis.loc[sector, ('avg_amount', 'mean')]
            trade_count = sector_analysis.loc[sector, ('avg_amount', 'count')]
            unique_members = sector_analysis.loc[sector, ('name', 'nunique')]

            # Calculate trend strength
            volume_score = min(1.0, total_volume / 10000000)  # Normalize by $10M
            participation_score = min(1.0, unique_members / 10)  # Normalize by 10 members

            trend = {
                'sector': sector,
                'total_volume': total_volume,
                'avg_trade_size': avg_trade,
                'trade_count': int(trade_count),
                'unique_members': int(unique_members),
                'trend_strength': round((volume_score + participation_score) / 2, 2),
                'insight': f"{sector} sector: ${total_volume:,.0f} across {int(unique_members)} members",
                'educational_note': "Sector concentration analysis for research"
            }
            trends.append(trend)

        # Sort by total volume
        trends.sort(key=lambda x: x['total_volume'], reverse=True)
        return trends

    def _analyze_filing_patterns(self, trades_df):
        """
        Analyze filing compliance patterns.
        Educational insight: STOCK Act compliance metrics.
        """
        total_trades = len(trades_df)
        late_filings = len(trades_df[trades_df['filing_delay_days'] > 45])
        very_late_filings = len(trades_df[trades_df['filing_delay_days'] > 100])

        # Member-level compliance
        member_compliance = []
        for member in trades_df['name'].unique():
            member_trades = trades_df[trades_df['name'] == member]
            member_late = len(member_trades[member_trades['filing_delay_days'] > 45])

            compliance_rate = 1 - (member_late / len(member_trades))

            member_compliance.append({
                'member': member,
                'total_trades': len(member_trades),
                'late_filings': member_late,
                'compliance_rate': round(compliance_rate, 2),
                'avg_delay': round(member_trades['filing_delay_days'].mean(), 1),
                'max_delay': int(member_trades['filing_delay_days'].max()),
                'insight': f"Compliance rate: {compliance_rate*100:.0f}% ({member_late} late out of {len(member_trades)})",
                'educational_note': "STOCK Act requires filing within 45 days"
            })

        # Sort by compliance rate (worst first for accountability)
        member_compliance.sort(key=lambda x: x['compliance_rate'])

        return {
            'overall': {
                'total_trades': total_trades,
                'late_filings': late_filings,
                'very_late_filings': very_late_filings,
                'overall_compliance_rate': round(1 - (late_filings / total_trades), 2),
                'avg_delay': round(trades_df['filing_delay_days'].mean(), 1)
            },
            'by_member': member_compliance[:10]  # Top 10 worst compliance
        }

    def _generate_educational_insights(self, trades_df, committee_data, legislation_data):
        """
        Generate high-level educational insights for public understanding.
        These are research findings, NOT investment advice.
        """
        insights = []

        # Insight 1: Committee-Trading Alignment
        tech_committee_members = [
            member for member, data in committee_data.items()
            if any('Tech' in area or 'Intelligence' in area for area in data.get('oversight_areas', []))
        ]
        tech_stocks = ['NVDA', 'GOOGL', 'MSFT', 'AAPL', 'META', 'AMZN']
        tech_committee_tech_trades = trades_df[
            (trades_df['name'].isin(tech_committee_members)) &
            (trades_df['symbol'].isin(tech_stocks))
        ]

        if len(tech_committee_tech_trades) > 0:
            insights.append({
                'category': 'Oversight Alignment',
                'finding': f"{len(tech_committee_tech_trades)} tech stock trades by tech oversight members",
                'significance': 'high',
                'educational_value': "Shows potential information advantages from committee assignments",
                'context': "Members with tech oversight trading tech stocks raises transparency questions"
            })

        # Insight 2: High-Value Trades During Legislative Activity
        high_value_trades = trades_df[trades_df['avg_amount'] > 500000]
        if len(high_value_trades) > 0:
            insights.append({
                'category': 'Large Transactions',
                'finding': f"{len(high_value_trades)} trades exceeding $500,000",
                'significance': 'high',
                'educational_value': "Large trades may indicate high-confidence positions or insider information",
                'context': "Exceptionally large trades warrant closer research scrutiny"
            })

        # Insight 3: Filing Delays
        violation_trades = trades_df[trades_df['filing_delay_days'] > 45]
        if len(violation_trades) > 0:
            insights.append({
                'category': 'STOCK Act Compliance',
                'finding': f"{len(violation_trades)} STOCK Act filing violations detected",
                'significance': 'medium',
                'educational_value': "Late filings reduce transparency and accountability",
                'context': "STOCK Act requires filing within 45 days of trade"
            })

        # Insight 4: Sector Concentration
        sector_diversity = self._calculate_sector_diversity(trades_df)
        if sector_diversity < 0.5:  # Low diversity
            insights.append({
                'category': 'Sector Concentration',
                'finding': f"Trading concentrated in limited sectors (diversity: {sector_diversity:.2f})",
                'significance': 'medium',
                'educational_value': "Concentration suggests coordinated or shared information sources",
                'context': "Diverse trading would suggest independent decision-making"
            })

        return insights

    def _calculate_sector_diversity(self, trades_df):
        """Calculate Shannon diversity index for sector distribution."""
        sector_mapping = {
            'NVDA': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'AAPL': 'Technology', 'META': 'Technology', 'AMZN': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'COIN': 'Financial',
            'XOM': 'Energy', 'CVX': 'Energy', 'PFE': 'Healthcare'
        }

        trades_df['sector'] = trades_df['symbol'].map(sector_mapping).fillna('Other')
        sector_counts = trades_df['sector'].value_counts()
        total = len(trades_df)

        # Shannon diversity
        diversity = -sum((count/total) * np.log(count/total) for count in sector_counts)
        # Normalize to 0-1 scale (max diversity for 7 sectors = log(7) ≈ 1.95)
        normalized_diversity = diversity / np.log(len(sector_counts)) if len(sector_counts) > 1 else 0

        return round(normalized_diversity, 2)

    def _generate_research_questions(self, trades_df, committee_data):
        """
        Generate research questions for further investigation.
        Educational output for researchers and journalists.
        """
        questions = []

        # Q1: Information Advantage
        questions.append({
            'question': "Do committee members have systematic information advantages?",
            'context': "Analyzing trade timing relative to committee hearings and votes",
            'methodology': "Statistical analysis of trade dates vs. legislative calendar",
            'significance': "Core democratic accountability question"
        })

        # Q2: Party Differences
        dem_trades = trades_df[trades_df['party'] == 'D']
        rep_trades = trades_df[trades_df['party'] == 'R']

        if len(dem_trades) > 0 and len(rep_trades) > 0:
            dem_avg = dem_trades['avg_amount'].mean()
            rep_avg = rep_trades['avg_amount'].mean()

            questions.append({
                'question': "Are there partisan differences in trading patterns?",
                'context': f"Dem avg: ${dem_avg:,.0f}, Rep avg: ${rep_avg:,.0f}",
                'methodology': "Compare trade frequency, size, and timing by party",
                'significance': "Understanding institutional vs. individual behavior"
            })

        # Q3: Leadership Premium
        leadership_members = [
            member for member, data in committee_data.items()
            if 'Chair' in data.get('leadership', '') or 'Speaker' in data.get('leadership', '')
        ]

        if len(leadership_members) > 0:
            leadership_trades = trades_df[trades_df['name'].isin(leadership_members)]

            questions.append({
                'question': "Do committee leaders show different trading patterns?",
                'context': f"{len(leadership_trades)} trades by {len(leadership_members)} leaders",
                'methodology': "Compare leadership vs. rank-and-file trading performance",
                'significance': "Leadership positions may provide greater information access"
            })

        return questions

    def _calculate_transparency_metrics(self, trades_df):
        """
        Calculate transparency and accountability metrics.
        """
        metrics = {
            'total_disclosure_volume': trades_df['avg_amount'].sum(),
            'public_filing_rate': len(trades_df) / (len(trades_df) + 0),  # Assuming all disclosed
            'avg_disclosure_delay': trades_df['filing_delay_days'].mean(),
            'compliance_rate': len(trades_df[trades_df['filing_delay_days'] <= 45]) / len(trades_df),
            'data_quality_score': self._calculate_data_quality(trades_df),
            'transparency_grade': self._assign_transparency_grade(trades_df)
        }

        return metrics

    def _calculate_data_quality(self, trades_df):
        """Calculate data quality score based on completeness and timeliness."""
        # Score based on filing delays (lower = better)
        delay_score = max(0, 1 - (trades_df['filing_delay_days'].mean() / 90))

        # Score based on data completeness (assuming all fields present)
        completeness_score = 1.0

        quality_score = (delay_score * 0.6 + completeness_score * 0.4)
        return round(quality_score, 2)

    def _assign_transparency_grade(self, trades_df):
        """Assign letter grade for transparency."""
        compliance_rate = len(trades_df[trades_df['filing_delay_days'] <= 45]) / len(trades_df)

        if compliance_rate >= 0.95:
            return 'A'
        elif compliance_rate >= 0.90:
            return 'B'
        elif compliance_rate >= 0.80:
            return 'C'
        elif compliance_rate >= 0.70:
            return 'D'
        else:
            return 'F'

    def export_insights_to_json(self, insights, output_file='pattern_insights.json'):
        """Export insights to JSON for dashboard consumption."""
        with open(output_file, 'w') as f:
            json.dump(insights, f, indent=2, default=str)

        print(f"Insights exported to {output_file}")
        return output_file


def generate_pattern_insights_report(trades_df, committee_data, legislation_data):
    """
    Generate comprehensive pattern insights report.
    Main entry point for creating educational research insights.
    """
    print("=" * 70)
    print("AI-POWERED PATTERN INSIGHTS - EDUCATIONAL RESEARCH SYSTEM")
    print("=" * 70)
    print()
    print("DISCLAIMER: These insights are for educational and transparency purposes.")
    print("NOT financial advice. NOT trading recommendations.")
    print("For research, journalism, and democratic accountability only.")
    print()
    print("=" * 70)
    print()

    engine = PatternInsightsEngine()
    insights = engine.generate_comprehensive_insights(trades_df, committee_data, legislation_data)

    # Print Summary
    print("📊 PATTERN INSIGHTS SUMMARY")
    print("-" * 70)
    print()

    print("1️⃣  TIMING PATTERNS")
    print(f"   Found {len(insights['timing_patterns'])} significant timing correlations")
    for pattern in insights['timing_patterns'][:3]:
        print(f"   • {pattern['member']}: {pattern['stock']} {pattern['days_advance']} days before {pattern['legislation'][:30]}...")
        print(f"     Confidence: {pattern['confidence']:.0%}")
    print()

    print("2️⃣  COMMITTEE CORRELATIONS")
    print(f"   Found {len(insights['committee_correlations'])} oversight-trading alignments")
    for corr in insights['committee_correlations'][:3]:
        print(f"   • {corr['member']}: ${corr['amount']:,.0f} in {corr['stock']}")
        print(f"     Oversight: {', '.join(corr['oversight_areas'][:2])}")
    print()

    print("3️⃣  UNUSUAL ACTIVITY")
    print(f"   Detected {len(insights['unusual_activity'])} statistical anomalies")
    for anomaly in insights['unusual_activity'][:3]:
        print(f"   • {anomaly['member']}: {', '.join(anomaly['anomalies'])}")
        print(f"     Confidence: {anomaly['confidence']:.0%}")
    print()

    print("4️⃣  SECTOR TRENDS")
    for trend in insights['sector_trends'][:5]:
        print(f"   • {trend['sector']}: ${trend['total_volume']:,.0f} ({trend['trade_count']} trades)")
    print()

    print("5️⃣  FILING COMPLIANCE")
    compliance = insights['filing_compliance']['overall']
    print(f"   Overall Compliance Rate: {compliance['overall_compliance_rate']:.0%}")
    print(f"   Late Filings: {compliance['late_filings']} of {compliance['total_trades']}")
    print(f"   Average Delay: {compliance['avg_delay']:.1f} days")
    print()

    print("6️⃣  EDUCATIONAL INSIGHTS")
    for insight in insights['educational_insights']:
        print(f"   📚 {insight['category']}: {insight['finding']}")
        print(f"      {insight['educational_value']}")
    print()

    print("7️⃣  TRANSPARENCY METRICS")
    metrics = insights['transparency_metrics']
    print(f"   Transparency Grade: {metrics['transparency_grade']}")
    print(f"   Data Quality Score: {metrics['data_quality_score']:.0%}")
    print(f"   Total Disclosure Volume: ${metrics['total_disclosure_volume']:,.0f}")
    print()

    print("8️⃣  RESEARCH QUESTIONS")
    for i, question in enumerate(insights['research_questions'], 1):
        print(f"   {i}. {question['question']}")
        print(f"      Context: {question['context']}")
    print()

    print("=" * 70)
    print("✅ Pattern insights generation complete!")
    print("=" * 70)

    # Export to JSON
    engine.export_insights_to_json(insights)

    return insights


if __name__ == "__main__":
    # Import base data
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    from congressional_analysis import (
        get_congressional_trades_sample,
        get_committee_assignments,
        get_current_legislation,
        analyze_trading_patterns
    )

    # Load data
    trades = get_congressional_trades_sample()
    committee_data = get_committee_assignments()
    legislation_data = get_current_legislation()

    # Process trades
    df, analysis = analyze_trading_patterns(trades)

    # Generate insights
    insights = generate_pattern_insights_report(df, committee_data, legislation_data)

    print("\n✅ Insights available in 'pattern_insights.json'")
