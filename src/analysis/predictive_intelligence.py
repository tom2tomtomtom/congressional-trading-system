#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Predictive Intelligence
ML-powered prediction models for congressional trading patterns and market impact analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Using sklearn models only.")
import warnings
warnings.filterwarnings('ignore')

class TradePredictionEngine:
    """Predict likelihood of congressional members making trades based on committee activity."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
    
    def prepare_prediction_features(self, trades_df, committee_data, legislation_data):
        """
        Prepare features for trade prediction model.
        """
        features = []
        labels = []
        
        # Get committee information
        committee_mapping = self._create_committee_mapping(committee_data)
        
        # Create time-based features for each member
        for member in trades_df['name'].unique():
            member_trades = trades_df[trades_df['name'] == member].copy()
            member_trades = member_trades.sort_values('transactionDate')
            
            if len(member_trades) < 2:
                continue
            
            # Create features for each time period
            for i in range(1, len(member_trades)):
                current_trade = member_trades.iloc[i]
                prev_trades = member_trades.iloc[:i]
                
                # Extract features
                feature_vector = self._extract_prediction_features(
                    member, current_trade, prev_trades, committee_mapping, legislation_data
                )
                
                # Label: 1 if trade occurred, 0 otherwise
                features.append(feature_vector)
                labels.append(1)  # Trade occurred
        
        # Add negative samples (no trades) for better model balance
        negative_features = self._generate_negative_samples(
            trades_df, committee_mapping, legislation_data
        )
        features.extend(negative_features)
        labels.extend([0] * len(negative_features))
        
        return np.array(features), np.array(labels)
    
    def _extract_prediction_features(self, member, current_trade, prev_trades, committee_mapping, legislation_data):
        """Extract features for prediction model."""
        features = []
        
        # Historical trading features
        features.append(len(prev_trades))  # Previous trade count
        features.append(prev_trades['avg_amount'].mean() if len(prev_trades) > 0 else 0)  # Avg amount
        features.append(prev_trades['filing_delay_days'].mean() if len(prev_trades) > 0 else 0)  # Avg delay
        
        # Time since last trade
        if len(prev_trades) > 0:
            last_trade_date = pd.to_datetime(prev_trades.iloc[-1]['transactionDate'])
            current_date = pd.to_datetime(current_trade['transactionDate'])
            days_since_last = (current_date - last_trade_date).days
        else:
            days_since_last = 365  # Arbitrary large number
        features.append(days_since_last)
        
        # Committee involvement features
        member_committees = committee_mapping.get(member, {})
        features.append(len(member_committees.get('committees', [])))  # Number of committees
        features.append(1 if 'Chair' in member_committees.get('leadership', '') else 0)  # Leadership position
        
        # Market sector involvement
        stock_sector = self._get_stock_sector(current_trade['symbol'])
        committee_sectors = member_committees.get('oversight_areas', [])
        features.append(1 if any(sector.lower() in stock_sector.lower() for sector in committee_sectors) else 0)
        
        # Legislation activity (simplified)
        features.append(self._count_relevant_legislation(member, current_trade['transactionDate'], legislation_data))
        
        # Market timing features
        trade_date = pd.to_datetime(current_trade['transactionDate'])
        features.append(trade_date.month)  # Month of year
        features.append(trade_date.day)    # Day of month
        features.append(trade_date.weekday())  # Day of week
        
        # Previous trade success (simplified metric)
        if len(prev_trades) > 0:
            # Assume success based on trade amounts increasing over time
            success_rate = 0.5  # Placeholder - would need market return data
        else:
            success_rate = 0.5
        features.append(success_rate)
        
        return features
    
    def _generate_negative_samples(self, trades_df, committee_mapping, legislation_data):
        """Generate negative samples for model training."""
        negative_features = []
        
        # For each member, create samples representing periods where they didn't trade
        for member in trades_df['name'].unique():
            member_trades = trades_df[trades_df['name'] == member].copy()
            
            # Generate random dates between trades where no trading occurred
            if len(member_trades) > 1:
                member_trades = member_trades.sort_values('transactionDate')
                
                for i in range(len(member_trades) - 1):
                    start_date = pd.to_datetime(member_trades.iloc[i]['transactionDate'])
                    end_date = pd.to_datetime(member_trades.iloc[i+1]['transactionDate'])
                    
                    # Generate random date in between
                    days_diff = (end_date - start_date).days
                    if days_diff > 30:  # Only if gap is significant
                        random_days = np.random.randint(10, min(days_diff-10, 60))
                        random_date = start_date + timedelta(days=random_days)
                        
                        # Create dummy trade for feature extraction
                        dummy_trade = member_trades.iloc[i].copy()
                        dummy_trade['transactionDate'] = random_date.strftime('%Y-%m-%d')
                        
                        prev_trades = member_trades.iloc[:i+1]
                        
                        features = self._extract_prediction_features(
                            member, dummy_trade, prev_trades, committee_mapping, legislation_data
                        )
                        negative_features.append(features)
        
        return negative_features
    
    def _create_committee_mapping(self, committee_data):
        """Create mapping of members to committee information."""
        # This would integrate with the actual committee data
        # For now, using simplified structure
        return committee_data if committee_data else {}
    
    def _get_stock_sector(self, symbol):
        """Get sector for stock symbol."""
        sector_mapping = {
            'NVDA': 'Technology', 'GOOGL': 'Technology', 'MSFT': 'Technology',
            'AAPL': 'Technology', 'META': 'Technology', 'AMZN': 'Technology',
            'TSLA': 'Technology', 'COIN': 'Technology', 'RBLX': 'Technology',
            'JPM': 'Financial', 'BAC': 'Financial', 'WFC': 'Financial',
            'XOM': 'Energy', 'CVX': 'Energy', 'PFE': 'Healthcare',
            'UNH': 'Healthcare', 'DIS': 'Media', 'HCA': 'Healthcare'
        }
        return sector_mapping.get(symbol, 'Other')
    
    def _count_relevant_legislation(self, member, trade_date, legislation_data):
        """Count relevant legislation around trade date."""
        # Simplified implementation
        return 1 if legislation_data else 0
    
    def train_model(self, features, labels):
        """Train the trade prediction model."""
        if len(features) == 0:
            print("No features available for training.")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Trade Prediction Model Trained!")
        print(f"Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self._plot_feature_importance()
        
        self.is_trained = True
        return accuracy
    
    def predict_trade_probability(self, member_features):
        """Predict probability of a trade occurring."""
        if not self.is_trained:
            return 0.5  # Default probability
        
        features_scaled = self.scaler.transform([member_features])
        probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of class 1 (trade)
        
        return probability
    
    def _plot_feature_importance(self):
        """Plot feature importance."""
        if not hasattr(self.model, 'feature_importances_'):
            return
        
        feature_names = [
            'prev_trade_count', 'avg_prev_amount', 'avg_filing_delay', 'days_since_last',
            'committee_count', 'is_chair', 'sector_match', 'legislation_count',
            'month', 'day', 'weekday', 'success_rate'
        ]
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importance for Trade Prediction")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig('trade_prediction_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

class MarketImpactPredictor:
    """Predict market impact when congressional trades are disclosed."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_market_impact_data(self, trades_df):
        """
        Prepare data for market impact prediction.
        Note: This is a simplified version - real implementation would need actual market data.
        """
        features = []
        targets = []
        
        for _, trade in trades_df.iterrows():
            # Extract features that might influence market impact
            feature_vector = [
                trade['avg_amount'],  # Trade size
                self._get_member_influence_score(trade['name']),  # Member influence
                self._get_stock_liquidity_score(trade['symbol']),  # Stock liquidity
                (pd.to_datetime(trade['filingDate']) - pd.to_datetime(trade['transactionDate'])).days,  # Filing delay
                1 if trade['transactionType'] == 'Purchase' else 0,  # Buy vs sell
                self._get_market_sentiment_score(trade['transactionDate']),  # Market conditions
            ]
            
            # Simulated market impact (in reality, would use actual price data)
            impact = self._simulate_market_impact(trade)
            
            features.append(feature_vector)
            targets.append(impact)
        
        return np.array(features), np.array(targets)
    
    def _get_member_influence_score(self, member_name):
        """Get influence score for a member (simplified)."""
        influence_scores = {
            'Nancy Pelosi': 10, 'Dan Crenshaw': 6, 'Richard Burr': 8,
            'Paul Pelosi': 9, 'Josh Gottheimer': 5, 'Ro Khanna': 6,
            'Pat Toomey': 7, 'Sherrod Brown': 7, 'Joe Manchin': 8,
            'Susan Collins': 6, 'Kevin McCarthy': 9, 'Alexandria Ocasio-Cortez': 7,
            'Ted Cruz': 7, 'Mark Warner': 7
        }
        return influence_scores.get(member_name, 5)
    
    def _get_stock_liquidity_score(self, symbol):
        """Get liquidity score for a stock (simplified)."""
        # High liquidity stocks
        high_liquidity = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
        return 8 if symbol in high_liquidity else 5
    
    def _get_market_sentiment_score(self, date):
        """Get market sentiment score for a date (simplified)."""
        # Simplified: negative during COVID, positive otherwise
        covid_period = pd.to_datetime('2020-03-01') <= pd.to_datetime(date) <= pd.to_datetime('2020-06-01')
        return 3 if covid_period else 7
    
    def _simulate_market_impact(self, trade):
        """Simulate market impact based on trade characteristics."""
        # This is a simplified simulation - real implementation would use historical data
        base_impact = min(5.0, trade['avg_amount'] / 1000000)  # Up to 5% impact
        
        # Adjust for member influence
        member_multiplier = self._get_member_influence_score(trade['name']) / 10
        
        # Random component
        random_factor = np.random.normal(1.0, 0.3)
        
        impact = base_impact * member_multiplier * random_factor
        return max(0, min(10, impact))  # Cap between 0-10%
    
    def train_impact_model(self, features, targets):
        """Train market impact prediction model."""
        if len(features) == 0:
            print("No features available for impact model training.")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Market Impact Model Trained!")
        print(f"MSE: {mse:.3f}")
        print(f"R²: {r2:.3f}")
        
        self.is_trained = True
        return r2
    
    def predict_market_impact(self, trade_features):
        """Predict market impact for a trade."""
        if not self.is_trained:
            return 1.0  # Default impact
        
        features_scaled = self.scaler.transform([trade_features])
        impact = self.model.predict(features_scaled)[0]
        
        return max(0, min(10, impact))

class LegislationOutcomePredictor:
    """Predict legislation outcomes based on congressional trading patterns."""
    
    def __init__(self):
        self.model = None
        self.is_trained = False
    
    def analyze_trading_legislation_correlation(self, trades_df, legislation_data):
        """
        Analyze correlation between trading patterns and legislation outcomes.
        """
        correlations = {}
        
        # For each piece of legislation, analyze related trading
        for legislation in legislation_data:
            bill_name = legislation['bill']
            affected_stocks = legislation.get('market_impact', [])
            
            # Find trades in affected stocks around legislation timeline
            relevant_trades = trades_df[
                trades_df['symbol'].isin(affected_stocks)
            ].copy()
            
            if len(relevant_trades) == 0:
                continue
            
            # Analyze trading patterns
            analysis = {
                'bill': bill_name,
                'affected_stocks': affected_stocks,
                'total_trades': len(relevant_trades),
                'total_volume': relevant_trades['avg_amount'].sum(),
                'unique_members': relevant_trades['name'].nunique(),
                'avg_suspicion_score': relevant_trades.apply(
                    lambda x: self._calculate_simple_suspicion(x), axis=1
                ).mean(),
                'prediction_confidence': self._calculate_outcome_confidence(relevant_trades)
            }
            
            correlations[bill_name] = analysis
        
        return correlations
    
    def _calculate_simple_suspicion(self, trade_row):
        """Calculate simple suspicion score for a trade."""
        score = 0
        if trade_row['avg_amount'] > 100000:
            score += 2
        if trade_row['filing_delay_days'] > 45:
            score += 2
        return score
    
    def _calculate_outcome_confidence(self, trades):
        """Calculate confidence in legislation outcome based on trading."""
        if len(trades) == 0:
            return 0.5
        
        # Simplified: more trading = higher confidence in outcome
        volume_factor = min(1.0, trades['avg_amount'].sum() / 10000000)
        member_diversity = trades['name'].nunique() / 15  # Normalize by sample size
        
        confidence = (volume_factor + member_diversity) / 2
        return max(0.1, min(0.9, confidence))

def create_prediction_visualizations(trade_predictions, impact_predictions, legislation_correlations):
    """Create visualizations for predictive intelligence results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Predictive Intelligence Analysis', fontsize=16, fontweight='bold')
    
    # 1. Trade Prediction Probabilities
    if trade_predictions:
        members = list(trade_predictions.keys())
        probabilities = list(trade_predictions.values())
        
        bars = axes[0,0].bar(range(len(members)), probabilities)
        axes[0,0].set_xlabel('Congressional Members')
        axes[0,0].set_ylabel('Trade Probability')
        axes[0,0].set_title('Predicted Trade Probabilities')
        axes[0,0].set_xticks(range(len(members)))
        axes[0,0].set_xticklabels([m.split()[0] for m in members], rotation=45)
        
        # Color bars by probability
        for bar, prob in zip(bars, probabilities):
            if prob > 0.7:
                bar.set_color('red')
            elif prob > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('green')
    
    # 2. Market Impact Predictions
    if impact_predictions:
        impacts = list(impact_predictions.values())
        axes[0,1].hist(impacts, bins=10, alpha=0.7, color='skyblue')
        axes[0,1].set_xlabel('Predicted Market Impact (%)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Distribution of Predicted Market Impacts')
        axes[0,1].axvline(np.mean(impacts), color='red', linestyle='--', 
                         label=f'Mean: {np.mean(impacts):.2f}%')
        axes[0,1].legend()
    
    # 3. Legislation Outcome Confidence
    if legislation_correlations:
        bills = list(legislation_correlations.keys())
        confidences = [legislation_correlations[bill]['prediction_confidence'] 
                      for bill in bills]
        
        bars = axes[1,0].bar(range(len(bills)), confidences)
        axes[1,0].set_xlabel('Legislation')
        axes[1,0].set_ylabel('Outcome Confidence')
        axes[1,0].set_title('Legislation Outcome Prediction Confidence')
        axes[1,0].set_xticks(range(len(bills)))
        axes[1,0].set_xticklabels([bill.split()[0] for bill in bills], rotation=45)
    
    # 4. Trading Volume vs Suspicion Score
    axes[1,1].scatter([0.5], [0.5], s=100, alpha=0.7)  # Placeholder
    axes[1,1].set_xlabel('Trading Volume (Log Scale)')
    axes[1,1].set_ylabel('Average Suspicion Score')
    axes[1,1].set_title('Volume vs Suspicion Correlation')
    axes[1,1].text(0.5, 0.5, 'Placeholder\n(Needs real data)', 
                   ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('predictive_intelligence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_predictive_analysis(trades_df, committee_data=None, legislation_data=None):
    """Run comprehensive predictive intelligence analysis."""
    
    print("PREDICTIVE INTELLIGENCE ANALYSIS")
    print("=" * 50)
    print()
    
    # Initialize predictors
    trade_predictor = TradePredictionEngine()
    impact_predictor = MarketImpactPredictor()
    legislation_predictor = LegislationOutcomePredictor()
    
    # 1. Trade Prediction Analysis
    print("1. TRADE PREDICTION MODEL")
    print("-" * 25)
    
    if committee_data and legislation_data:
        features, labels = trade_predictor.prepare_prediction_features(
            trades_df, committee_data, legislation_data
        )
        
        if len(features) > 10:  # Need minimum samples
            accuracy = trade_predictor.train_model(features, labels)
            print(f"Model trained with {len(features)} samples")
        else:
            print("Insufficient data for trade prediction model")
            accuracy = None
    else:
        print("Committee and legislation data needed for trade prediction")
        accuracy = None
    
    # 2. Market Impact Prediction
    print("\n2. MARKET IMPACT PREDICTION")
    print("-" * 28)
    
    impact_features, impact_targets = impact_predictor.prepare_market_impact_data(trades_df)
    
    if len(impact_features) > 5:
        r2_score = impact_predictor.train_impact_model(impact_features, impact_targets)
        
        # Generate sample predictions
        impact_predictions = {}
        for i, (_, trade) in enumerate(trades_df.head(10).iterrows()):
            features = impact_features[i] if i < len(impact_features) else impact_features[0]
            impact = impact_predictor.predict_market_impact(features)
            impact_predictions[trade['name']] = impact
        
        print(f"Sample Impact Predictions:")
        for member, impact in list(impact_predictions.items())[:5]:
            print(f"  {member}: {impact:.2f}% market impact")
    else:
        print("Insufficient data for market impact prediction")
        impact_predictions = {}
    
    # 3. Legislation Outcome Analysis
    print("\n3. LEGISLATION OUTCOME ANALYSIS")
    print("-" * 30)
    
    if legislation_data:
        legislation_correlations = legislation_predictor.analyze_trading_legislation_correlation(
            trades_df, legislation_data
        )
        
        print("Trading-Legislation Correlations:")
        for bill, analysis in legislation_correlations.items():
            print(f"  {bill}:")
            print(f"    Trading Volume: ${analysis['total_volume']:,.0f}")
            print(f"    Unique Traders: {analysis['unique_members']}")
            print(f"    Outcome Confidence: {analysis['prediction_confidence']:.2f}")
            print()
    else:
        print("Legislation data needed for outcome analysis")
        legislation_correlations = {}
    
    # Generate placeholder trade predictions for visualization
    trade_predictions = {}
    for member in trades_df['name'].unique():
        # Simplified prediction based on trading frequency
        member_trades = trades_df[trades_df['name'] == member]
        frequency_score = len(member_trades) / 10  # Normalize
        amount_score = member_trades['avg_amount'].mean() / 1000000  # Normalize
        probability = min(1.0, (frequency_score + amount_score) / 2)
        trade_predictions[member] = probability
    
    # Create visualizations
    create_prediction_visualizations(trade_predictions, impact_predictions, legislation_correlations)
    
    print("\nPREDICTIVE ANALYSIS SUMMARY:")
    print("-" * 30)
    print(f"Trade Prediction Accuracy: {accuracy:.3f}" if accuracy else "Trade Prediction: Not available")
    print(f"Impact Model R²: {r2_score:.3f}" if 'r2_score' in locals() else "Impact Model: Not available")
    print(f"Legislation Correlations: {len(legislation_correlations)} analyzed")
    
    return {
        'trade_predictions': trade_predictions,
        'impact_predictions': impact_predictions,
        'legislation_correlations': legislation_correlations,
        'model_performance': {
            'trade_accuracy': accuracy,
            'impact_r2': r2_score if 'r2_score' in locals() else None
        }
    }

if __name__ == "__main__":
    # Import base data
    import sys
    sys.path.append('..')
    from congressional_analysis import get_congressional_trades_sample, get_committee_assignments, get_current_legislation
    
    # Load data
    trades = get_congressional_trades_sample()
    committee_data = get_committee_assignments()
    legislation_data = get_current_legislation()
    
    # Process trades data
    df = pd.DataFrame(trades)
    df['transactionDate'] = pd.to_datetime(df['transactionDate'])
    df['filingDate'] = pd.to_datetime(df['filingDate'])
    df['filing_delay_days'] = (df['filingDate'] - df['transactionDate']).dt.days
    df['avg_amount'] = (df['amountFrom'] + df['amountTo']) / 2
    
    # Run predictive analysis
    results = run_predictive_analysis(df, committee_data, legislation_data)
    
    print("\nPredictive intelligence analysis complete!")
    print("Visualizations saved as 'predictive_intelligence_analysis.png' and 'trade_prediction_feature_importance.png'")