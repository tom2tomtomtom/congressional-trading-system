#!/usr/bin/env python3
"""
Congressional Trading Statistical Analysis Engine
Advanced statistical analysis and pattern detection for congressional trading data
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

# Statistical analysis imports
from scipy import stats
from scipy.stats import chi2_contingency, kstest, normaltest, pearsonr, spearmanr
from scipy.stats import ttest_ind, mannwhitneyu, kruskal
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CongressionalStatisticalAnalyzer:
    """
    Advanced statistical analysis engine for congressional trading patterns
    Provides hypothesis testing, correlation analysis, and statistical anomaly detection
    """
    
    def __init__(self, data_path: str = 'src/data'):
        self.data_path = data_path
        self.members_df = None
        self.trades_df = None
        self.analysis_results = {}
        
        # Statistical parameters
        self.significance_level = 0.05
        self.confidence_interval = 0.95
        
        # Analysis thresholds
        self.thresholds = {
            'large_trade': 100000,
            'extreme_trade': 1000000,
            'late_filing': 45,
            'extreme_delay': 90,
            'high_frequency': 10,
            'correlation_threshold': 0.3
        }
    
    def load_data(self) -> bool:
        """Load congressional trading data for statistical analysis"""
        try:
            # Load members data
            members_file = os.path.join(self.data_path, 'congressional_members_full.json')
            if os.path.exists(members_file):
                with open(members_file, 'r') as f:
                    members_data = json.load(f)
                self.members_df = pd.DataFrame(members_data)
            
            # Load trades data
            trades_file = os.path.join(self.data_path, 'congressional_trades_full.json')
            if os.path.exists(trades_file):
                with open(trades_file, 'r') as f:
                    trades_data = json.load(f)
                self.trades_df = pd.DataFrame(trades_data)
                
                # Process dates and create derived fields
                self.trades_df['transaction_date'] = pd.to_datetime(
                    self.trades_df['transaction_date'], errors='coerce'
                )
                self.trades_df['filing_date'] = pd.to_datetime(
                    self.trades_df['filing_date'], errors='coerce'
                )
                
                self.trades_df['filing_delay_days'] = (
                    self.trades_df['filing_date'] - self.trades_df['transaction_date']
                ).dt.days
                
                self.trades_df['amount_avg'] = (
                    self.trades_df['amount_from'] + self.trades_df['amount_to']
                ) / 2
                
                self.trades_df['transaction_year'] = self.trades_df['transaction_date'].dt.year
                self.trades_df['transaction_month'] = self.trades_df['transaction_date'].dt.month
                self.trades_df['transaction_quarter'] = self.trades_df['transaction_date'].dt.quarter
                
                logger.info(f"✅ Loaded {len(self.trades_df)} trades for statistical analysis")
                return True
            
            logger.error("❌ Failed to load trading data")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            return False
    
    def test_party_differences(self) -> Dict[str, Any]:
        """Statistical tests for differences between political parties"""
        if self.trades_df is None:
            return {}
        
        results = {
            'test_type': 'Party Comparison Analysis',
            'hypothesis': 'H0: No significant difference between parties',
            'significance_level': self.significance_level,
            'tests_performed': {}
        }
        
        # Separate data by party
        dem_trades = self.trades_df[self.trades_df['party'] == 'D']
        rep_trades = self.trades_df[self.trades_df['party'] == 'R']
        
        if len(dem_trades) == 0 or len(rep_trades) == 0:
            return {'error': 'Insufficient data for party comparison'}
        
        # Test 1: Trade Amount Differences
        dem_amounts = dem_trades['amount_avg'].dropna()
        rep_amounts = rep_trades['amount_avg'].dropna()
        
        # Mann-Whitney U test (non-parametric)
        mw_stat, mw_p = mannwhitneyu(dem_amounts, rep_amounts, alternative='two-sided')
        
        # T-test (parametric)
        t_stat, t_p = ttest_ind(dem_amounts, rep_amounts, equal_var=False)
        
        results['tests_performed']['trade_amounts'] = {
            'democrat_mean': float(dem_amounts.mean()),
            'republican_mean': float(rep_amounts.mean()),
            'democrat_median': float(dem_amounts.median()),
            'republican_median': float(rep_amounts.median()),
            'mann_whitney_u': {
                'statistic': float(mw_stat),
                'p_value': float(mw_p),
                'significant': mw_p < self.significance_level
            },
            't_test': {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': t_p < self.significance_level
            },
            'effect_size': float(abs(dem_amounts.mean() - rep_amounts.mean()) / 
                               np.sqrt((dem_amounts.var() + rep_amounts.var()) / 2))
        }
        
        # Test 2: Filing Delay Differences
        dem_delays = dem_trades['filing_delay_days'].dropna()
        rep_delays = rep_trades['filing_delay_days'].dropna()
        
        mw_delay_stat, mw_delay_p = mannwhitneyu(dem_delays, rep_delays, alternative='two-sided')
        t_delay_stat, t_delay_p = ttest_ind(dem_delays, rep_delays, equal_var=False)
        
        results['tests_performed']['filing_delays'] = {
            'democrat_mean': float(dem_delays.mean()),
            'republican_mean': float(rep_delays.mean()),
            'mann_whitney_u': {
                'statistic': float(mw_delay_stat),
                'p_value': float(mw_delay_p),
                'significant': mw_delay_p < self.significance_level
            },
            't_test': {
                'statistic': float(t_delay_stat),
                'p_value': float(t_delay_p),
                'significant': t_delay_p < self.significance_level
            }
        }
        
        # Test 3: Transaction Type Distribution (Chi-square)
        contingency_table = pd.crosstab(
            self.trades_df['party'], 
            self.trades_df['transaction_type']
        )
        
        if contingency_table.size > 0:
            chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
            
            results['tests_performed']['transaction_type_distribution'] = {
                'contingency_table': contingency_table.to_dict(),
                'chi_square': {
                    'statistic': float(chi2_stat),
                    'p_value': float(chi2_p),
                    'degrees_of_freedom': int(dof),
                    'significant': chi2_p < self.significance_level
                },
                'cramers_v': float(np.sqrt(chi2_stat / (contingency_table.sum().sum() * 
                                                      (min(contingency_table.shape) - 1))))
            }
        
        # Test 4: Large Trade Frequency
        dem_large_trades = len(dem_trades[dem_trades['amount_avg'] > self.thresholds['large_trade']])
        rep_large_trades = len(rep_trades[rep_trades['amount_avg'] > self.thresholds['large_trade']])
        
        dem_large_rate = dem_large_trades / len(dem_trades) if len(dem_trades) > 0 else 0
        rep_large_rate = rep_large_trades / len(rep_trades) if len(rep_trades) > 0 else 0
        
        # Fisher's exact test for proportions
        from scipy.stats import fisher_exact
        contingency_large = np.array([
            [dem_large_trades, len(dem_trades) - dem_large_trades],
            [rep_large_trades, len(rep_trades) - rep_large_trades]
        ])
        
        if contingency_large.min() > 0:
            fisher_odds, fisher_p = fisher_exact(contingency_large)
            
            results['tests_performed']['large_trade_frequency'] = {
                'democrat_rate': float(dem_large_rate),
                'republican_rate': float(rep_large_rate),
                'fisher_exact': {
                    'odds_ratio': float(fisher_odds),
                    'p_value': float(fisher_p),
                    'significant': fisher_p < self.significance_level
                }
            }
        
        logger.info("✅ Party difference analysis completed")
        return results
    
    def test_chamber_differences(self) -> Dict[str, Any]:
        """Statistical tests for differences between House and Senate"""
        if self.trades_df is None:
            return {}
        
        results = {
            'test_type': 'Chamber Comparison Analysis',
            'hypothesis': 'H0: No significant difference between chambers',
            'significance_level': self.significance_level,
            'tests_performed': {}
        }
        
        # Separate data by chamber
        house_trades = self.trades_df[self.trades_df['chamber'] == 'House']
        senate_trades = self.trades_df[self.trades_df['chamber'] == 'Senate']
        
        if len(house_trades) == 0 or len(senate_trades) == 0:
            return {'error': 'Insufficient data for chamber comparison'}
        
        # Trade amount comparison
        house_amounts = house_trades['amount_avg'].dropna()
        senate_amounts = senate_trades['amount_avg'].dropna()
        
        mw_stat, mw_p = mannwhitneyu(house_amounts, senate_amounts, alternative='two-sided')
        t_stat, t_p = ttest_ind(house_amounts, senate_amounts, equal_var=False)
        
        results['tests_performed']['trade_amounts'] = {
            'house_mean': float(house_amounts.mean()),
            'senate_mean': float(senate_amounts.mean()),
            'house_median': float(house_amounts.median()),
            'senate_median': float(senate_amounts.median()),
            'mann_whitney_u': {
                'statistic': float(mw_stat),
                'p_value': float(mw_p),
                'significant': mw_p < self.significance_level
            },
            't_test': {
                'statistic': float(t_stat),
                'p_value': float(t_p),
                'significant': t_p < self.significance_level
            }
        }
        
        # Filing compliance comparison
        house_delays = house_trades['filing_delay_days'].dropna()
        senate_delays = senate_trades['filing_delay_days'].dropna()
        
        mw_delay_stat, mw_delay_p = mannwhitneyu(house_delays, senate_delays, alternative='two-sided')
        
        results['tests_performed']['filing_compliance'] = {
            'house_mean_delay': float(house_delays.mean()),
            'senate_mean_delay': float(senate_delays.mean()),
            'house_compliance_rate': float((house_delays <= 45).mean()),
            'senate_compliance_rate': float((senate_delays <= 45).mean()),
            'mann_whitney_u': {
                'statistic': float(mw_delay_stat),
                'p_value': float(mw_delay_p),
                'significant': mw_delay_p < self.significance_level
            }
        }
        
        logger.info("✅ Chamber difference analysis completed")
        return results
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns and seasonality in trading"""
        if self.trades_df is None:
            return {}
        
        results = {
            'analysis_type': 'Temporal Pattern Analysis',
            'patterns_detected': {}
        }
        
        # Monthly trading patterns
        monthly_counts = self.trades_df['transaction_month'].value_counts().sort_index()
        monthly_amounts = self.trades_df.groupby('transaction_month')['amount_avg'].sum()
        
        # Test for uniform distribution across months (Chi-square goodness of fit)
        expected_per_month = len(self.trades_df) / 12
        chi2_monthly, p_monthly = stats.chisquare(monthly_counts.values)
        
        results['patterns_detected']['monthly_distribution'] = {
            'counts_by_month': monthly_counts.to_dict(),
            'amounts_by_month': monthly_amounts.to_dict(),
            'uniformity_test': {
                'chi_square_statistic': float(chi2_monthly),
                'p_value': float(p_monthly),
                'uniform_distribution': p_monthly > self.significance_level
            },
            'peak_month': int(monthly_counts.idxmax()),
            'lowest_month': int(monthly_counts.idxmin())
        }
        
        # Quarterly patterns
        quarterly_counts = self.trades_df['transaction_quarter'].value_counts().sort_index()
        quarterly_amounts = self.trades_df.groupby('transaction_quarter')['amount_avg'].sum()
        
        results['patterns_detected']['quarterly_distribution'] = {
            'counts_by_quarter': quarterly_counts.to_dict(),
            'amounts_by_quarter': quarterly_amounts.to_dict(),
            'peak_quarter': int(quarterly_counts.idxmax()),
            'lowest_quarter': int(quarterly_counts.idxmin())
        }
        
        # Year-over-year analysis
        if 'transaction_year' in self.trades_df.columns:
            yearly_stats = self.trades_df.groupby('transaction_year').agg({
                'amount_avg': ['count', 'mean', 'sum'],
                'filing_delay_days': 'mean'
            }).round(2)
            
            # Test for trend over time
            years = yearly_stats.index.values
            trade_counts = yearly_stats[('amount_avg', 'count')].values
            
            if len(years) > 2:
                # Spearman correlation for trend
                trend_corr, trend_p = spearmanr(years, trade_counts)
                
                results['patterns_detected']['yearly_trends'] = {
                    'trade_counts_by_year': dict(zip(years, trade_counts)),
                    'trend_analysis': {
                        'correlation': float(trend_corr),
                        'p_value': float(trend_p),
                        'significant_trend': trend_p < self.significance_level,
                        'trend_direction': 'increasing' if trend_corr > 0 else 'decreasing'
                    }
                }
        
        # Day of week analysis
        self.trades_df['weekday'] = self.trades_df['transaction_date'].dt.day_name()
        weekday_counts = self.trades_df['weekday'].value_counts()
        
        results['patterns_detected']['weekday_distribution'] = {
            'counts_by_weekday': weekday_counts.to_dict(),
            'most_active_day': weekday_counts.idxmax(),
            'least_active_day': weekday_counts.idxmin()
        }
        
        logger.info("✅ Temporal pattern analysis completed")
        return results
    
    def correlation_analysis(self) -> Dict[str, Any]:
        """Comprehensive correlation analysis between variables"""
        if self.trades_df is None:
            return {}
        
        results = {
            'analysis_type': 'Correlation Analysis',
            'correlations': {}
        }
        
        # Select numerical variables for correlation
        numerical_vars = [
            'amount_avg', 'filing_delay_days', 'transaction_month', 
            'transaction_quarter', 'transaction_year'
        ]
        
        # Add encoded categorical variables
        if 'party' in self.trades_df.columns:
            self.trades_df['party_numeric'] = self.trades_df['party'].map({'D': 0, 'R': 1, 'I': 2}).fillna(-1)
            numerical_vars.append('party_numeric')
        
        if 'chamber' in self.trades_df.columns:
            self.trades_df['chamber_numeric'] = self.trades_df['chamber'].map({'House': 0, 'Senate': 1}).fillna(-1)
            numerical_vars.append('chamber_numeric')
        
        # Create correlation matrix
        correlation_data = self.trades_df[numerical_vars].dropna()
        
        if len(correlation_data) > 0:
            # Pearson correlations
            pearson_corr = correlation_data.corr(method='pearson')
            
            # Spearman correlations (rank-based)
            spearman_corr = correlation_data.corr(method='spearman')
            
            # Significant correlations
            significant_correlations = []
            
            for i, var1 in enumerate(numerical_vars):
                for j, var2 in enumerate(numerical_vars):
                    if i < j:  # Avoid duplicates
                        pearson_val = pearson_corr.loc[var1, var2]
                        spearman_val = spearman_corr.loc[var1, var2]
                        
                        if abs(pearson_val) > self.thresholds['correlation_threshold']:
                            # Calculate p-value for correlation
                            _, p_value = pearsonr(
                                correlation_data[var1], 
                                correlation_data[var2]
                            )
                            
                            significant_correlations.append({
                                'variable_1': var1,
                                'variable_2': var2,
                                'pearson_correlation': float(pearson_val),
                                'spearman_correlation': float(spearman_val),
                                'p_value': float(p_value),
                                'significant': p_value < self.significance_level,
                                'strength': self._interpret_correlation(abs(pearson_val))
                            })
            
            results['correlations'] = {
                'pearson_matrix': pearson_corr.to_dict(),
                'spearman_matrix': spearman_corr.to_dict(),
                'significant_correlations': significant_correlations,
                'correlation_threshold': self.thresholds['correlation_threshold']
            }
        
        # Special correlations of interest
        
        # Amount vs Filing Delay
        if len(correlation_data) > 10:
            amount_delay_corr, amount_delay_p = pearsonr(
                correlation_data['amount_avg'], 
                correlation_data['filing_delay_days']
            )
            
            results['special_analyses'] = {
                'amount_vs_filing_delay': {
                    'correlation': float(amount_delay_corr),
                    'p_value': float(amount_delay_p),
                    'interpretation': 'Higher amounts correlate with longer delays' if amount_delay_corr > 0.1 else 'No strong relationship'
                }
            }
        
        logger.info("✅ Correlation analysis completed")
        return results
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength"""
        if correlation >= 0.7:
            return 'strong'
        elif correlation >= 0.5:
            return 'moderate'
        elif correlation >= 0.3:
            return 'weak'
        else:
            return 'negligible'
    
    def outlier_detection_analysis(self) -> Dict[str, Any]:
        """Statistical outlier detection using multiple methods"""
        if self.trades_df is None:
            return {}
        
        results = {
            'analysis_type': 'Statistical Outlier Detection',
            'methods_used': ['IQR', 'Z-Score', 'Modified Z-Score'],
            'outliers_detected': {}
        }
        
        # Variables to analyze for outliers
        variables = ['amount_avg', 'filing_delay_days']
        
        for var in variables:
            if var not in self.trades_df.columns:
                continue
            
            data = self.trades_df[var].dropna()
            
            if len(data) == 0:
                continue
            
            outliers = {
                'variable': var,
                'total_observations': len(data),
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'methods': {}
            }
            
            # Method 1: IQR (Interquartile Range)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            outliers['methods']['IQR'] = {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'outlier_count': len(iqr_outliers),
                'outlier_percentage': float(len(iqr_outliers) / len(data) * 100),
                'outlier_values': iqr_outliers.head(10).tolist()  # Top 10 outliers
            }
            
            # Method 2: Z-Score
            z_scores = np.abs(stats.zscore(data))
            z_outliers = data[z_scores > 3]  # |z| > 3
            
            outliers['methods']['Z_Score'] = {
                'threshold': 3.0,
                'outlier_count': len(z_outliers),
                'outlier_percentage': float(len(z_outliers) / len(data) * 100),
                'outlier_values': z_outliers.head(10).tolist()
            }
            
            # Method 3: Modified Z-Score (using median)
            median = data.median()
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            modified_z_outliers = data[np.abs(modified_z_scores) > 3.5]
            
            outliers['methods']['Modified_Z_Score'] = {
                'threshold': 3.5,
                'outlier_count': len(modified_z_outliers),
                'outlier_percentage': float(len(modified_z_outliers) / len(data) * 100),
                'outlier_values': modified_z_outliers.head(10).tolist()
            }
            
            # Combine outliers and get trade details
            all_outlier_values = set(
                list(iqr_outliers.values) + 
                list(z_outliers.values) + 
                list(modified_z_outliers.values)
            )
            
            # Get trade details for outliers
            outlier_trades = self.trades_df[self.trades_df[var].isin(all_outlier_values)]
            outlier_details = []
            
            for _, trade in outlier_trades.head(20).iterrows():  # Top 20 outlier trades
                outlier_details.append({
                    'member_name': trade.get('member_name', 'Unknown'),
                    'symbol': trade.get('symbol', 'Unknown'),
                    'transaction_date': trade.get('transaction_date', '').strftime('%Y-%m-%d') if pd.notna(trade.get('transaction_date')) else '',
                    'value': float(trade[var]),
                    'party': trade.get('party', 'Unknown'),
                    'chamber': trade.get('chamber', 'Unknown')
                })
            
            outliers['outlier_trade_details'] = outlier_details
            results['outliers_detected'][var] = outliers
        
        logger.info("✅ Outlier detection analysis completed")
        return results
    
    def distribution_analysis(self) -> Dict[str, Any]:
        """Analyze distributions and test for normality"""
        if self.trades_df is None:
            return {}
        
        results = {
            'analysis_type': 'Distribution Analysis',
            'variables_analyzed': []
        }
        
        variables = ['amount_avg', 'filing_delay_days']
        
        for var in variables:
            if var not in self.trades_df.columns:
                continue
                
            data = self.trades_df[var].dropna()
            
            if len(data) < 10:
                continue
            
            analysis = {
                'variable': var,
                'sample_size': len(data),
                'descriptive_stats': {
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'mode': float(data.mode().iloc[0]) if len(data.mode()) > 0 else None,
                    'std': float(data.std()),
                    'variance': float(data.var()),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'range': float(data.max() - data.min())
                },
                'percentiles': {
                    '1st': float(data.quantile(0.01)),
                    '5th': float(data.quantile(0.05)),
                    '10th': float(data.quantile(0.10)),
                    '25th': float(data.quantile(0.25)),
                    '50th': float(data.quantile(0.50)),
                    '75th': float(data.quantile(0.75)),
                    '90th': float(data.quantile(0.90)),
                    '95th': float(data.quantile(0.95)),
                    '99th': float(data.quantile(0.99))
                },
                'normality_tests': {}
            }
            
            # Normality tests
            if len(data) >= 20:
                # Shapiro-Wilk test (for smaller samples)
                if len(data) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(data)
                    analysis['normality_tests']['shapiro_wilk'] = {
                        'statistic': float(shapiro_stat),
                        'p_value': float(shapiro_p),
                        'normal_distribution': shapiro_p > self.significance_level
                    }
                
                # D'Agostino-Pearson test
                dagostino_stat, dagostino_p = normaltest(data)
                analysis['normality_tests']['dagostino_pearson'] = {
                    'statistic': float(dagostino_stat),
                    'p_value': float(dagostino_p),
                    'normal_distribution': dagostino_p > self.significance_level
                }
                
                # Kolmogorov-Smirnov test against normal distribution
                ks_stat, ks_p = kstest(data, lambda x: stats.norm.cdf(x, data.mean(), data.std()))
                analysis['normality_tests']['kolmogorov_smirnov'] = {
                    'statistic': float(ks_stat),
                    'p_value': float(ks_p),
                    'normal_distribution': ks_p > self.significance_level
                }
            
            # Distribution interpretation
            if abs(analysis['descriptive_stats']['skewness']) > 1:
                skew_interpretation = 'highly skewed'
            elif abs(analysis['descriptive_stats']['skewness']) > 0.5:
                skew_interpretation = 'moderately skewed'
            else:
                skew_interpretation = 'approximately symmetric'
            
            analysis['interpretation'] = {
                'skewness': f"{skew_interpretation} ({'right' if analysis['descriptive_stats']['skewness'] > 0 else 'left'})",
                'kurtosis': 'heavy-tailed' if analysis['descriptive_stats']['kurtosis'] > 0 else 'light-tailed',
                'likely_distribution': self._suggest_distribution(analysis['descriptive_stats'])
            }
            
            results['variables_analyzed'].append(analysis)
        
        logger.info("✅ Distribution analysis completed")
        return results
    
    def _suggest_distribution(self, stats: Dict[str, float]) -> str:
        """Suggest likely distribution based on descriptive statistics"""
        skewness = abs(stats['skewness'])
        kurtosis = stats['kurtosis']
        
        if skewness < 0.5 and abs(kurtosis) < 0.5:
            return 'Normal'
        elif skewness > 1 and stats['skewness'] > 0:
            return 'Log-normal or Exponential'
        elif skewness > 1 and stats['skewness'] < 0:
            return 'Beta (left-skewed)'
        elif kurtosis > 2:
            return 'Heavy-tailed (possibly t-distribution)'
        else:
            return 'Non-standard distribution'
    
    def run_comprehensive_statistical_analysis(self) -> Dict[str, Any]:
        """Run complete statistical analysis pipeline"""
        logger.info("📊 Starting comprehensive statistical analysis...")
        
        if not self.load_data():
            return {'error': 'Failed to load data'}
        
        analysis_results = {
            'metadata': {
                'analysis_timestamp': datetime.now().isoformat(),
                'total_trades': len(self.trades_df),
                'unique_members': self.trades_df['member_name'].nunique() if 'member_name' in self.trades_df.columns else 0,
                'data_period': {
                    'start': self.trades_df['transaction_date'].min().strftime('%Y-%m-%d') if 'transaction_date' in self.trades_df.columns else None,
                    'end': self.trades_df['transaction_date'].max().strftime('%Y-%m-%d') if 'transaction_date' in self.trades_df.columns else None
                },
                'significance_level': self.significance_level
            }
        }
        
        # Run all statistical analyses
        try:
            logger.info("🗳️ Analyzing party differences...")
            analysis_results['party_differences'] = self.test_party_differences()
            
            logger.info("🏛️ Analyzing chamber differences...")
            analysis_results['chamber_differences'] = self.test_chamber_differences()
            
            logger.info("📅 Analyzing temporal patterns...")
            analysis_results['temporal_patterns'] = self.analyze_temporal_patterns()
            
            logger.info("🔗 Performing correlation analysis...")
            analysis_results['correlations'] = self.correlation_analysis()
            
            logger.info("📈 Detecting statistical outliers...")
            analysis_results['outliers'] = self.outlier_detection_analysis()
            
            logger.info("📊 Analyzing distributions...")
            analysis_results['distributions'] = self.distribution_analysis()
            
            # Summary insights
            analysis_results['key_insights'] = self._generate_insights(analysis_results)
            
            logger.info("✅ Comprehensive statistical analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Error in statistical analysis: {e}")
            analysis_results['error'] = str(e)
        
        return analysis_results
    
    def _generate_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate key insights from statistical analysis"""
        insights = []
        
        # Party differences insights
        if 'party_differences' in results and 'tests_performed' in results['party_differences']:
            party_tests = results['party_differences']['tests_performed']
            
            if 'trade_amounts' in party_tests:
                amount_test = party_tests['trade_amounts']
                if amount_test['mann_whitney_u']['significant']:
                    higher_party = 'Democrats' if amount_test['democrat_mean'] > amount_test['republican_mean'] else 'Republicans'
                    insights.append(f"Significant difference in trade amounts between parties - {higher_party} have higher average trades")
            
            if 'filing_delays' in party_tests:
                delay_test = party_tests['filing_delays']
                if delay_test['mann_whitney_u']['significant']:
                    faster_party = 'Democrats' if delay_test['democrat_mean'] < delay_test['republican_mean'] else 'Republicans'
                    insights.append(f"Significant difference in filing compliance - {faster_party} file faster on average")
        
        # Temporal patterns insights
        if 'temporal_patterns' in results and 'patterns_detected' in results['temporal_patterns']:
            temporal = results['temporal_patterns']['patterns_detected']
            
            if 'monthly_distribution' in temporal:
                monthly = temporal['monthly_distribution']
                if not monthly['uniformity_test']['uniform_distribution']:
                    peak_month = monthly['peak_month']
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    insights.append(f"Trading activity is not uniform throughout the year - peak activity in {month_names[peak_month-1]}")
        
        # Outliers insights
        if 'outliers' in results and 'outliers_detected' in results['outliers']:
            for var, outlier_data in results['outliers']['outliers_detected'].items():
                if var == 'amount_avg':
                    iqr_pct = outlier_data['methods']['IQR']['outlier_percentage']
                    if iqr_pct > 10:
                        insights.append(f"High percentage of outlier trades detected ({iqr_pct:.1f}%) - indicates significant variation in trade amounts")
        
        # Correlation insights
        if 'correlations' in results and 'significant_correlations' in results['correlations']:
            sig_corr = results['correlations']['significant_correlations']
            for corr in sig_corr:
                if corr['significant'] and corr['strength'] in ['moderate', 'strong']:
                    insights.append(f"Significant {corr['strength']} correlation between {corr['variable_1']} and {corr['variable_2']}")
        
        return insights[:10]  # Return top 10 insights
    
    def export_statistical_report(self, output_dir: str = 'analysis_output') -> str:
        """Export comprehensive statistical analysis report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Run analysis if not already done
        if not hasattr(self, 'analysis_results') or not self.analysis_results:
            self.analysis_results = self.run_comprehensive_statistical_analysis()
        
        # Export to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'statistical_analysis_report_{timestamp}.json'
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        logger.info(f"✅ Statistical analysis report exported to {filepath}")
        return filepath

def main():
    """Command-line interface for statistical analysis"""
    analyzer = CongressionalStatisticalAnalyzer()
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_statistical_analysis()
    
    if 'error' not in results:
        print("\n" + "="*70)
        print("📊 CONGRESSIONAL TRADING STATISTICAL ANALYSIS REPORT")
        print("="*70)
        
        metadata = results.get('metadata', {})
        print(f"📅 Analysis Date: {metadata.get('analysis_timestamp', 'N/A')}")
        print(f"💰 Total Trades: {metadata.get('total_trades', 0)}")
        print(f"👥 Unique Members: {metadata.get('unique_members', 0)}")
        
        if 'data_period' in metadata:
            period = metadata['data_period']
            print(f"📊 Data Period: {period.get('start', 'N/A')} to {period.get('end', 'N/A')}")
        
        # Key insights
        if 'key_insights' in results:
            print(f"\n🔍 KEY STATISTICAL INSIGHTS:")
            for i, insight in enumerate(results['key_insights'], 1):
                print(f"  {i}. {insight}")
        
        # Party differences summary
        if 'party_differences' in results:
            party_result = results['party_differences']
            if 'tests_performed' in party_result:
                print(f"\n🗳️ PARTY DIFFERENCES:")
                tests = party_result['tests_performed']
                
                if 'trade_amounts' in tests:
                    amounts = tests['trade_amounts']
                    print(f"  • Trade Amounts: D=${amounts['democrat_mean']:.0f}, R=${amounts['republican_mean']:.0f}")
                    print(f"    Significant difference: {amounts['mann_whitney_u']['significant']}")
        
        # Export report
        report_file = analyzer.export_statistical_report()
        print(f"\n📄 Full report exported to: {report_file}")
        
        print("="*70)
    else:
        print(f"❌ Statistical analysis failed: {results.get('error', 'Unknown error')}")

if __name__ == '__main__':
    main()