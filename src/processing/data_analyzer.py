#!/usr/bin/env python3
"""
Congressional Trading Data Analysis Engine
Comprehensive data processing and analysis for congressional trading patterns
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CongressionalTradingAnalyzer:
    """
    Comprehensive analyzer for congressional trading data
    Provides statistical analysis, pattern detection, and risk assessment
    """
    
    def __init__(self, members_file: str = None, trades_file: str = None):
        self.members_df = None
        self.trades_df = None
        self.analysis_results = {}
        
        # Default file paths
        self.members_file = members_file or 'src/data/congressional_members_full.json'
        self.trades_file = trades_file or 'src/data/congressional_trades_full.json'
        
        # Analysis parameters
        self.risk_thresholds = {
            'high_amount': 100000,
            'extreme_amount': 1000000,
            'late_filing': 45,  # days
            'extreme_delay': 90,  # days
            'high_frequency': 10,  # trades per year
            'extreme_frequency': 25
        }
        
    def load_data(self) -> bool:
        """Load congressional members and trading data from JSON files"""
        try:
            # Load members data
            if os.path.exists(self.members_file):
                with open(self.members_file, 'r') as f:
                    members_data = json.load(f)
                self.members_df = pd.DataFrame(members_data)
                logger.info(f"✅ Loaded {len(self.members_df)} congressional members")
            else:
                logger.warning(f"Members file not found: {self.members_file}")
                return False
            
            # Load trades data
            if os.path.exists(self.trades_file):
                with open(self.trades_file, 'r') as f:
                    trades_data = json.load(f)
                self.trades_df = pd.DataFrame(trades_data)
                
                # Process date columns
                self.trades_df['transaction_date'] = pd.to_datetime(
                    self.trades_df['transaction_date'], errors='coerce'
                )
                self.trades_df['filing_date'] = pd.to_datetime(
                    self.trades_df['filing_date'], errors='coerce'
                )
                
                # Calculate derived metrics
                self.trades_df['filing_delay_days'] = (
                    self.trades_df['filing_date'] - self.trades_df['transaction_date']
                ).dt.days
                
                self.trades_df['amount_avg'] = (
                    self.trades_df['amount_from'] + self.trades_df['amount_to']
                ) / 2
                
                self.trades_df['amount_range'] = (
                    self.trades_df['amount_to'] - self.trades_df['amount_from']
                )
                
                # Add year and month columns
                self.trades_df['transaction_year'] = self.trades_df['transaction_date'].dt.year
                self.trades_df['transaction_month'] = self.trades_df['transaction_date'].dt.month
                
                logger.info(f"✅ Loaded {len(self.trades_df)} congressional trades")
            else:
                logger.warning(f"Trades file not found: {self.trades_file}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def calculate_member_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive statistics for each member"""
        if self.trades_df is None:
            return {}
        
        member_stats = {}
        
        for member_name in self.trades_df['member_name'].unique():
            member_trades = self.trades_df[self.trades_df['member_name'] == member_name]
            
            # Basic statistics
            total_trades = len(member_trades)
            total_volume = member_trades['amount_avg'].sum()
            avg_trade_size = member_trades['amount_avg'].mean()
            largest_trade = member_trades['amount_avg'].max()
            
            # Timing analysis
            avg_filing_delay = member_trades['filing_delay_days'].mean()
            late_filings = len(member_trades[member_trades['filing_delay_days'] > 45])
            compliance_rate = ((total_trades - late_filings) / total_trades * 100) if total_trades > 0 else 100
            
            # Portfolio diversity
            unique_symbols = member_trades['symbol'].nunique()
            unique_sectors = member_trades['asset_name'].nunique()  # Assuming asset_name indicates sector
            
            # Transaction patterns
            purchases = len(member_trades[member_trades['transaction_type'] == 'Purchase'])
            sales = len(member_trades[member_trades['transaction_type'] == 'Sale'])
            purchase_ratio = (purchases / total_trades * 100) if total_trades > 0 else 0
            
            # Temporal patterns
            trades_by_month = member_trades['transaction_month'].value_counts().to_dict()
            most_active_month = member_trades['transaction_month'].mode().iloc[0] if len(member_trades) > 0 else 0
            
            # Risk indicators
            high_amount_trades = len(member_trades[member_trades['amount_avg'] > self.risk_thresholds['high_amount']])
            extreme_amount_trades = len(member_trades[member_trades['amount_avg'] > self.risk_thresholds['extreme_amount']])
            
            member_stats[member_name] = {
                'basic_metrics': {
                    'total_trades': total_trades,
                    'total_volume': total_volume,
                    'avg_trade_size': avg_trade_size,
                    'largest_trade': largest_trade,
                    'trading_period_days': (member_trades['transaction_date'].max() - 
                                          member_trades['transaction_date'].min()).days if len(member_trades) > 1 else 0
                },
                'compliance_metrics': {
                    'avg_filing_delay': avg_filing_delay,
                    'late_filings': late_filings,
                    'compliance_rate': compliance_rate,
                    'worst_delay': member_trades['filing_delay_days'].max()
                },
                'portfolio_metrics': {
                    'unique_symbols': unique_symbols,
                    'unique_sectors': unique_sectors,
                    'concentration_ratio': (member_trades['symbol'].value_counts().iloc[0] / total_trades * 100) if total_trades > 0 else 0
                },
                'behavioral_patterns': {
                    'purchase_ratio': purchase_ratio,
                    'sale_ratio': 100 - purchase_ratio,
                    'most_active_month': most_active_month,
                    'monthly_distribution': trades_by_month,
                    'trading_frequency': total_trades / max(1, (member_trades['transaction_date'].max() - 
                                                            member_trades['transaction_date'].min()).days / 365.25) if len(member_trades) > 1 else 0
                },
                'risk_indicators': {
                    'high_amount_trades': high_amount_trades,
                    'extreme_amount_trades': extreme_amount_trades,
                    'high_amount_ratio': (high_amount_trades / total_trades * 100) if total_trades > 0 else 0,
                    'extreme_delay_count': len(member_trades[member_trades['filing_delay_days'] > self.risk_thresholds['extreme_delay']])
                }
            }
        
        return member_stats
    
    def calculate_risk_scores(self) -> Dict[str, float]:
        """Calculate comprehensive risk scores for all members"""
        member_stats = self.calculate_member_statistics()
        risk_scores = {}
        
        for member_name, stats in member_stats.items():
            risk_score = 0
            max_score = 0
            
            # Trading frequency risk (0-2 points)
            frequency = stats['behavioral_patterns']['trading_frequency']
            if frequency > self.risk_thresholds['extreme_frequency']:
                risk_score += 2
            elif frequency > self.risk_thresholds['high_frequency']:
                risk_score += 1
            max_score += 2
            
            # Trade size risk (0-3 points)
            high_ratio = stats['risk_indicators']['high_amount_ratio']
            if high_ratio > 50:
                risk_score += 3
            elif high_ratio > 25:
                risk_score += 2
            elif high_ratio > 10:
                risk_score += 1
            max_score += 3
            
            # Filing compliance risk (0-2 points)
            compliance_rate = stats['compliance_metrics']['compliance_rate']
            if compliance_rate < 70:
                risk_score += 2
            elif compliance_rate < 85:
                risk_score += 1
            max_score += 2
            
            # Portfolio concentration risk (0-2 points)
            concentration = stats['portfolio_metrics']['concentration_ratio']
            if concentration > 75:
                risk_score += 2
            elif concentration > 50:
                risk_score += 1
            max_score += 2
            
            # Extreme behavior risk (0-1 points)
            if stats['risk_indicators']['extreme_amount_trades'] > 0:
                risk_score += 1
            max_score += 1
            
            # Normalize to 0-10 scale
            normalized_score = (risk_score / max_score * 10) if max_score > 0 else 0
            risk_scores[member_name] = round(normalized_score, 1)
        
        return risk_scores
    
    def detect_trading_patterns(self) -> Dict[str, Any]:
        """Detect significant trading patterns and anomalies"""
        if self.trades_df is None:
            return {}
        
        patterns = {
            'temporal_patterns': {},
            'symbol_patterns': {},
            'party_patterns': {},
            'chamber_patterns': {},
            'anomalies': []
        }
        
        # Temporal patterns
        monthly_volume = self.trades_df.groupby('transaction_month')['amount_avg'].sum()
        patterns['temporal_patterns']['peak_month'] = monthly_volume.idxmax()
        patterns['temporal_patterns']['lowest_month'] = monthly_volume.idxmin()
        patterns['temporal_patterns']['monthly_volumes'] = monthly_volume.to_dict()
        
        yearly_counts = self.trades_df.groupby('transaction_year').size()
        patterns['temporal_patterns']['yearly_trade_counts'] = yearly_counts.to_dict()
        
        # Symbol patterns
        symbol_volumes = self.trades_df.groupby('symbol')['amount_avg'].sum().sort_values(ascending=False)
        patterns['symbol_patterns']['top_symbols'] = symbol_volumes.head(10).to_dict()
        
        symbol_counts = self.trades_df['symbol'].value_counts()
        patterns['symbol_patterns']['most_traded'] = symbol_counts.head(10).to_dict()
        
        # Party patterns
        party_stats = self.trades_df.groupby('party').agg({
            'amount_avg': ['mean', 'sum', 'count'],
            'filing_delay_days': 'mean'
        }).round(2)
        
        patterns['party_patterns'] = {
            'average_trade_size': party_stats[('amount_avg', 'mean')].to_dict(),
            'total_volume': party_stats[('amount_avg', 'sum')].to_dict(),
            'trade_count': party_stats[('amount_avg', 'count')].to_dict(),
            'avg_filing_delay': party_stats[('filing_delay_days', 'mean')].to_dict()
        }
        
        # Chamber patterns
        chamber_stats = self.trades_df.groupby('chamber').agg({
            'amount_avg': ['mean', 'sum', 'count'],
            'filing_delay_days': 'mean'
        }).round(2)
        
        patterns['chamber_patterns'] = {
            'average_trade_size': chamber_stats[('amount_avg', 'mean')].to_dict(),
            'total_volume': chamber_stats[('amount_avg', 'sum')].to_dict(),
            'trade_count': chamber_stats[('amount_avg', 'count')].to_dict(),
            'avg_filing_delay': chamber_stats[('filing_delay_days', 'mean')].to_dict()
        }
        
        # Anomaly detection
        # Large trades (top 5%)
        large_threshold = self.trades_df['amount_avg'].quantile(0.95)
        large_trades = self.trades_df[self.trades_df['amount_avg'] >= large_threshold]
        
        for _, trade in large_trades.iterrows():
            patterns['anomalies'].append({
                'type': 'LARGE_TRADE',
                'member': trade['member_name'],
                'symbol': trade['symbol'],
                'amount': trade['amount_avg'],
                'date': trade['transaction_date'].strftime('%Y-%m-%d'),
                'description': f"Trade amount ${trade['amount_avg']:,.0f} is in top 5% of all trades"
            })
        
        # Extreme filing delays
        extreme_delays = self.trades_df[self.trades_df['filing_delay_days'] > self.risk_thresholds['extreme_delay']]
        
        for _, trade in extreme_delays.iterrows():
            patterns['anomalies'].append({
                'type': 'EXTREME_DELAY',
                'member': trade['member_name'],
                'symbol': trade['symbol'],
                'delay_days': trade['filing_delay_days'],
                'date': trade['transaction_date'].strftime('%Y-%m-%d'),
                'description': f"Filing delayed {trade['filing_delay_days']} days (legal limit: 45 days)"
            })
        
        return patterns
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance analysis"""
        if self.trades_df is None:
            return {}
        
        total_trades = len(self.trades_df)
        late_trades = len(self.trades_df[self.trades_df['filing_delay_days'] > 45])
        
        report = {
            'overview': {
                'total_trades': total_trades,
                'compliant_trades': total_trades - late_trades,
                'late_trades': late_trades,
                'overall_compliance_rate': ((total_trades - late_trades) / total_trades * 100) if total_trades > 0 else 0
            },
            'by_member': {},
            'by_party': {},
            'by_chamber': {},
            'severity_breakdown': {
                'minor_violations': 0,  # 46-60 days
                'moderate_violations': 0,  # 61-90 days
                'severe_violations': 0  # 90+ days
            }
        }
        
        # Severity breakdown
        minor = len(self.trades_df[(self.trades_df['filing_delay_days'] > 45) & 
                                 (self.trades_df['filing_delay_days'] <= 60)])
        moderate = len(self.trades_df[(self.trades_df['filing_delay_days'] > 60) & 
                                    (self.trades_df['filing_delay_days'] <= 90)])
        severe = len(self.trades_df[self.trades_df['filing_delay_days'] > 90])
        
        report['severity_breakdown'] = {
            'minor_violations': minor,
            'moderate_violations': moderate,
            'severe_violations': severe
        }
        
        # By member compliance
        member_compliance = self.trades_df.groupby('member_name').agg({
            'filing_delay_days': ['count', lambda x: (x > 45).sum(), 'mean']
        }).round(2)
        
        for member in member_compliance.index:
            total = member_compliance.loc[member, ('filing_delay_days', 'count')]
            violations = member_compliance.loc[member, ('filing_delay_days', '<lambda_0>')]
            avg_delay = member_compliance.loc[member, ('filing_delay_days', 'mean')]
            
            report['by_member'][member] = {
                'total_trades': total,
                'violations': violations,
                'compliance_rate': ((total - violations) / total * 100) if total > 0 else 100,
                'avg_filing_delay': avg_delay
            }
        
        # By party compliance
        party_compliance = self.trades_df.groupby('party').agg({
            'filing_delay_days': ['count', lambda x: (x > 45).sum(), 'mean']
        }).round(2)
        
        for party in party_compliance.index:
            total = party_compliance.loc[party, ('filing_delay_days', 'count')]
            violations = party_compliance.loc[party, ('filing_delay_days', '<lambda_0>')]
            
            report['by_party'][party] = {
                'total_trades': total,
                'violations': violations,
                'compliance_rate': ((total - violations) / total * 100) if total > 0 else 100
            }
        
        # By chamber compliance
        chamber_compliance = self.trades_df.groupby('chamber').agg({
            'filing_delay_days': ['count', lambda x: (x > 45).sum(), 'mean']
        }).round(2)
        
        for chamber in chamber_compliance.index:
            total = chamber_compliance.loc[chamber, ('filing_delay_days', 'count')]
            violations = chamber_compliance.loc[chamber, ('filing_delay_days', '<lambda_0>')]
            
            report['by_chamber'][chamber] = {
                'total_trades': total,
                'violations': violations,
                'compliance_rate': ((total - violations) / total * 100) if total > 0 else 100
            }
        
        return report
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        logger.info("🔍 Starting comprehensive congressional trading analysis...")
        
        if not self.load_data():
            logger.error("❌ Failed to load data")
            return {}
        
        # Run all analysis components
        analysis_results = {
            'metadata': {
                'analysis_date': datetime.now().isoformat(),
                'data_period': {
                    'start': self.trades_df['transaction_date'].min().strftime('%Y-%m-%d'),
                    'end': self.trades_df['transaction_date'].max().strftime('%Y-%m-%d')
                },
                'total_members': len(self.members_df) if self.members_df is not None else 0,
                'total_trades': len(self.trades_df) if self.trades_df is not None else 0
            },
            'member_statistics': self.calculate_member_statistics(),
            'risk_scores': self.calculate_risk_scores(),
            'trading_patterns': self.detect_trading_patterns(),
            'compliance_report': self.generate_compliance_report()
        }
        
        # Cache results
        self.analysis_results = analysis_results
        
        logger.info("✅ Comprehensive analysis completed")
        return analysis_results
    
    def export_analysis(self, output_dir: str = 'analysis_output') -> Dict[str, str]:
        """Export analysis results to various formats"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.analysis_results:
            self.run_comprehensive_analysis()
        
        exported_files = {}
        
        try:
            # Export full analysis as JSON
            json_file = os.path.join(output_dir, f'comprehensive_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(json_file, 'w') as f:
                json.dump(self.analysis_results, f, indent=2, default=str)
            exported_files['json'] = json_file
            
            # Export member statistics as CSV
            if 'member_statistics' in self.analysis_results:
                member_data = []
                for member, stats in self.analysis_results['member_statistics'].items():
                    row = {'member_name': member}
                    
                    # Flatten nested dictionary
                    for category, metrics in stats.items():
                        for metric, value in metrics.items():
                            row[f'{category}_{metric}'] = value
                    
                    member_data.append(row)
                
                member_df = pd.DataFrame(member_data)
                csv_file = os.path.join(output_dir, f'member_analysis_{datetime.now().strftime("%Y%m%d")}.csv')
                member_df.to_csv(csv_file, index=False)
                exported_files['member_csv'] = csv_file
            
            # Export risk scores
            if 'risk_scores' in self.analysis_results:
                risk_df = pd.DataFrame(list(self.analysis_results['risk_scores'].items()), 
                                     columns=['member_name', 'risk_score'])
                risk_file = os.path.join(output_dir, f'risk_scores_{datetime.now().strftime("%Y%m%d")}.csv')
                risk_df.to_csv(risk_file, index=False)
                exported_files['risk_csv'] = risk_file
            
            logger.info(f"✅ Analysis exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Error exporting analysis: {e}")
        
        return exported_files

def main():
    """Command-line interface for running analysis"""
    analyzer = CongressionalTradingAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()
    
    if results:
        # Print summary
        print("\n" + "="*60)
        print("📊 CONGRESSIONAL TRADING ANALYSIS SUMMARY")
        print("="*60)
        
        metadata = results.get('metadata', {})
        print(f"📅 Analysis Date: {metadata.get('analysis_date', 'N/A')}")
        print(f"👥 Total Members: {metadata.get('total_members', 0)}")
        print(f"💰 Total Trades: {metadata.get('total_trades', 0)}")
        
        if 'data_period' in metadata:
            period = metadata['data_period']
            print(f"📊 Data Period: {period.get('start', 'N/A')} to {period.get('end', 'N/A')}")
        
        # Risk score summary
        risk_scores = results.get('risk_scores', {})
        if risk_scores:
            high_risk_members = [name for name, score in risk_scores.items() if score >= 7]
            print(f"⚠️  High-Risk Members: {len(high_risk_members)}")
            
            if high_risk_members:
                print("\nTop High-Risk Members:")
                for member in sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                    print(f"  • {member[0]}: {member[1]}/10")
        
        # Compliance summary
        compliance = results.get('compliance_report', {}).get('overview', {})
        if compliance:
            compliance_rate = compliance.get('overall_compliance_rate', 0)
            print(f"📋 Overall Compliance: {compliance_rate:.1f}%")
            print(f"⏰ Late Filings: {compliance.get('late_trades', 0)}")
        
        print("="*60)
        
        # Export results
        exported = analyzer.export_analysis()
        if exported:
            print(f"\n📁 Analysis exported to:")
            for file_type, filepath in exported.items():
                print(f"  • {file_type}: {filepath}")
    
    else:
        print("❌ Analysis failed - check data files and try again")

if __name__ == '__main__':
    main()