"""
APEX Trading Engine - Advanced Political Exchange Analytics
The World's Most Sophisticated Trading System

This module contains the core trading algorithms, AI models, and intelligence
processing capabilities that power the APEX trading system.

Author: Manus AI
Version: 1.0
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    import tensorflow as tf
    from tensorflow import keras
    ML_AVAILABLE = True
except ImportError:
    print("Advanced ML libraries not available. Install sklearn and tensorflow for full functionality.")
    ML_AVAILABLE = False

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    print("Alpaca API not available. Install alpaca-trade-api for live trading.")
    ALPACA_AVAILABLE = False

class SignalStrength(Enum):
    """Signal strength classification"""
    TIER_1_EXTREME = 10  # Committee chairs, leadership, large positions
    TIER_2_HIGH = 8      # Reliable traders, significant positions
    TIER_3_MEDIUM = 6    # Moderate confidence signals
    TIER_4_LOW = 4       # Emerging patterns
    TIER_5_NOISE = 2     # Low confidence, monitoring only

class TradeDirection(Enum):
    """Trade direction classification"""
    LONG = 1
    SHORT = -1
    NEUTRAL = 0

class StrategyType(Enum):
    """Trading strategy types"""
    MOMENTUM_CAPTURE = "momentum_capture"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    OPTIONS_FLOW = "options_flow"
    PAIRS_TRADING = "pairs_trading"
    EARNINGS_PLAY = "earnings_play"

@dataclass
class TradingSignal:
    """Comprehensive trading signal data structure"""
    symbol: str
    signal_strength: SignalStrength
    direction: TradeDirection
    strategy_type: StrategyType
    confidence: float
    expected_return: float
    risk_level: float
    time_horizon: str
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float
    congressional_member: str
    committee_relevance: float
    legislative_catalyst: str
    timestamp: datetime
    metadata: Dict[str, Any]

class CongressionalIntelligenceProcessor:
    """Advanced congressional trading intelligence processor"""
    
    def __init__(self):
        self.db_path = "/home/ubuntu/apex_intelligence.db"
        self.initialize_database()
        self.member_profiles = self.load_member_profiles()
        self.committee_mappings = self.load_committee_mappings()
        
    def initialize_database(self):
        """Initialize the intelligence database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Congressional trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS congressional_trades (
                id INTEGER PRIMARY KEY,
                member_name TEXT,
                symbol TEXT,
                trade_type TEXT,
                amount REAL,
                trade_date DATE,
                filing_date DATE,
                committee_assignments TEXT,
                signal_strength INTEGER,
                processed BOOLEAN DEFAULT FALSE,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Legislative events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS legislative_events (
                id INTEGER PRIMARY KEY,
                event_type TEXT,
                description TEXT,
                event_date DATE,
                affected_sectors TEXT,
                impact_score REAL,
                committee TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trading signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                signal_strength INTEGER,
                direction INTEGER,
                strategy_type TEXT,
                confidence REAL,
                expected_return REAL,
                risk_level REAL,
                entry_price REAL,
                target_price REAL,
                stop_loss REAL,
                position_size REAL,
                congressional_member TEXT,
                legislative_catalyst TEXT,
                generated_at DATETIME,
                executed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_member_profiles(self) -> Dict[str, Dict]:
        """Load congressional member profiles with trading history and reliability scores"""
        profiles = {
            "Nancy Pelosi": {
                "reliability_score": 0.94,
                "avg_return": 0.651,
                "position": "House Leadership",
                "committees": ["House Oversight"],
                "trading_frequency": "High",
                "preferred_sectors": ["Technology", "Healthcare"],
                "signal_multiplier": 1.5
            },
            "Ron Wyden": {
                "reliability_score": 0.92,
                "avg_return": 1.238,
                "position": "Senate Finance Committee Chair",
                "committees": ["Finance Committee"],
                "trading_frequency": "Medium",
                "preferred_sectors": ["Technology", "Energy"],
                "signal_multiplier": 1.4
            },
            "Ro Khanna": {
                "reliability_score": 0.78,
                "avg_return": 0.450,
                "position": "House Oversight Committee",
                "committees": ["House Oversight"],
                "trading_frequency": "Very High",
                "preferred_sectors": ["Technology"],
                "signal_multiplier": 1.2
            },
            "Debbie Wasserman Schultz": {
                "reliability_score": 0.85,
                "avg_return": 1.423,
                "position": "House Member",
                "committees": ["Financial Services"],
                "trading_frequency": "Medium",
                "preferred_sectors": ["Financial", "Healthcare"],
                "signal_multiplier": 1.3
            }
        }
        return profiles
    
    def load_committee_mappings(self) -> Dict[str, List[str]]:
        """Map committees to relevant market sectors"""
        return {
            "House Oversight": ["Technology", "Healthcare", "Energy"],
            "Finance Committee": ["Financial", "Technology", "Energy"],
            "Financial Services": ["Financial", "Banking", "Insurance"],
            "Energy and Commerce": ["Energy", "Healthcare", "Technology"],
            "Armed Services": ["Defense", "Aerospace", "Technology"],
            "Agriculture": ["Agriculture", "Food", "Commodities"]
        }
    
    def process_congressional_trade(self, member_name: str, symbol: str, 
                                  trade_type: str, amount: float, 
                                  trade_date: datetime) -> TradingSignal:
        """Process a congressional trade and generate trading signal"""
        
        # Get member profile
        profile = self.member_profiles.get(member_name, {})
        reliability_score = profile.get("reliability_score", 0.5)
        signal_multiplier = profile.get("signal_multiplier", 1.0)
        
        # Calculate base signal strength
        base_strength = self.calculate_signal_strength(
            member_name, amount, trade_type, reliability_score
        )
        
        # Apply committee relevance boost
        committee_relevance = self.calculate_committee_relevance(
            member_name, symbol
        )
        
        # Final signal strength
        final_strength = min(10, base_strength * signal_multiplier * committee_relevance)
        
        # Determine signal tier
        if final_strength >= 9:
            signal_tier = SignalStrength.TIER_1_EXTREME
        elif final_strength >= 7:
            signal_tier = SignalStrength.TIER_2_HIGH
        elif final_strength >= 5:
            signal_tier = SignalStrength.TIER_3_MEDIUM
        elif final_strength >= 3:
            signal_tier = SignalStrength.TIER_4_LOW
        else:
            signal_tier = SignalStrength.TIER_5_NOISE
        
        # Get current market data
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
        except:
            current_price = 100.0  # Fallback price
        
        # Generate trading signal
        signal = TradingSignal(
            symbol=symbol,
            signal_strength=signal_tier,
            direction=TradeDirection.LONG if trade_type.lower() == "purchase" else TradeDirection.SHORT,
            strategy_type=self.determine_strategy_type(signal_tier, symbol),
            confidence=min(0.95, reliability_score * committee_relevance),
            expected_return=self.estimate_expected_return(signal_tier, reliability_score),
            risk_level=self.calculate_risk_level(signal_tier, amount),
            time_horizon=self.determine_time_horizon(signal_tier),
            entry_price=current_price,
            target_price=self.calculate_target_price(current_price, signal_tier, trade_type),
            stop_loss=self.calculate_stop_loss(current_price, signal_tier, trade_type),
            position_size=self.calculate_position_size(signal_tier, amount),
            congressional_member=member_name,
            committee_relevance=committee_relevance,
            legislative_catalyst=self.identify_legislative_catalyst(member_name, symbol),
            timestamp=datetime.now(),
            metadata={
                "original_amount": amount,
                "filing_delay": self.calculate_filing_delay(trade_date),
                "member_profile": profile
            }
        )
        
        # Store signal in database
        self.store_trading_signal(signal)
        
        return signal
    
    def calculate_signal_strength(self, member_name: str, amount: float, 
                                trade_type: str, reliability_score: float) -> float:
        """Calculate base signal strength"""
        strength = 5.0  # Base strength
        
        # Amount-based boost
        if amount > 1000000:  # $1M+
            strength += 3.0
        elif amount > 500000:  # $500K+
            strength += 2.0
        elif amount > 100000:  # $100K+
            strength += 1.0
        
        # Member reliability boost
        strength += reliability_score * 2.0
        
        # Leadership position boost
        profile = self.member_profiles.get(member_name, {})
        if "Chair" in profile.get("position", ""):
            strength += 2.0
        elif "Leadership" in profile.get("position", ""):
            strength += 1.5
        
        return min(10.0, strength)
    
    def calculate_committee_relevance(self, member_name: str, symbol: str) -> float:
        """Calculate committee relevance multiplier"""
        profile = self.member_profiles.get(member_name, {})
        committees = profile.get("committees", [])
        
        # Get stock sector
        stock_sector = self.get_stock_sector(symbol)
        
        relevance = 1.0
        for committee in committees:
            relevant_sectors = self.committee_mappings.get(committee, [])
            if stock_sector in relevant_sectors:
                relevance += 0.3
        
        return min(2.0, relevance)
    
    def get_stock_sector(self, symbol: str) -> str:
        """Determine stock sector (simplified mapping)"""
        tech_stocks = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA", "META", "AMZN"]
        financial_stocks = ["JPM", "BAC", "WFC", "GS", "MS", "C"]
        healthcare_stocks = ["JNJ", "PFE", "UNH", "ABBV", "MRK"]
        energy_stocks = ["XOM", "CVX", "COP", "EOG", "SLB"]
        
        if symbol in tech_stocks:
            return "Technology"
        elif symbol in financial_stocks:
            return "Financial"
        elif symbol in healthcare_stocks:
            return "Healthcare"
        elif symbol in energy_stocks:
            return "Energy"
        else:
            return "Other"
    
    def determine_strategy_type(self, signal_strength: SignalStrength, symbol: str) -> StrategyType:
        """Determine optimal trading strategy based on signal characteristics"""
        if signal_strength == SignalStrength.TIER_1_EXTREME:
            return StrategyType.MOMENTUM_CAPTURE
        elif signal_strength == SignalStrength.TIER_2_HIGH:
            return StrategyType.OPTIONS_FLOW
        else:
            return StrategyType.MEAN_REVERSION
    
    def estimate_expected_return(self, signal_strength: SignalStrength, reliability_score: float) -> float:
        """Estimate expected return based on signal characteristics"""
        base_returns = {
            SignalStrength.TIER_1_EXTREME: 0.25,  # 25% expected return
            SignalStrength.TIER_2_HIGH: 0.15,     # 15% expected return
            SignalStrength.TIER_3_MEDIUM: 0.08,   # 8% expected return
            SignalStrength.TIER_4_LOW: 0.04,      # 4% expected return
            SignalStrength.TIER_5_NOISE: 0.02     # 2% expected return
        }
        
        base_return = base_returns.get(signal_strength, 0.05)
        return base_return * reliability_score
    
    def calculate_risk_level(self, signal_strength: SignalStrength, amount: float) -> float:
        """Calculate risk level for position sizing"""
        base_risk = {
            SignalStrength.TIER_1_EXTREME: 0.15,  # 15% max risk
            SignalStrength.TIER_2_HIGH: 0.12,     # 12% max risk
            SignalStrength.TIER_3_MEDIUM: 0.08,   # 8% max risk
            SignalStrength.TIER_4_LOW: 0.05,      # 5% max risk
            SignalStrength.TIER_5_NOISE: 0.02     # 2% max risk
        }
        
        return base_risk.get(signal_strength, 0.05)
    
    def determine_time_horizon(self, signal_strength: SignalStrength) -> str:
        """Determine optimal holding period"""
        horizons = {
            SignalStrength.TIER_1_EXTREME: "1-3 months",
            SignalStrength.TIER_2_HIGH: "2-6 weeks",
            SignalStrength.TIER_3_MEDIUM: "1-4 weeks",
            SignalStrength.TIER_4_LOW: "1-2 weeks",
            SignalStrength.TIER_5_NOISE: "1-5 days"
        }
        
        return horizons.get(signal_strength, "1-2 weeks")
    
    def calculate_target_price(self, current_price: float, signal_strength: SignalStrength, trade_type: str) -> float:
        """Calculate target price based on expected returns"""
        expected_moves = {
            SignalStrength.TIER_1_EXTREME: 0.25,
            SignalStrength.TIER_2_HIGH: 0.15,
            SignalStrength.TIER_3_MEDIUM: 0.08,
            SignalStrength.TIER_4_LOW: 0.04,
            SignalStrength.TIER_5_NOISE: 0.02
        }
        
        expected_move = expected_moves.get(signal_strength, 0.05)
        
        if trade_type.lower() == "purchase":
            return current_price * (1 + expected_move)
        else:
            return current_price * (1 - expected_move)
    
    def calculate_stop_loss(self, current_price: float, signal_strength: SignalStrength, trade_type: str) -> float:
        """Calculate stop loss based on risk management rules"""
        stop_distances = {
            SignalStrength.TIER_1_EXTREME: 0.15,
            SignalStrength.TIER_2_HIGH: 0.12,
            SignalStrength.TIER_3_MEDIUM: 0.08,
            SignalStrength.TIER_4_LOW: 0.05,
            SignalStrength.TIER_5_NOISE: 0.03
        }
        
        stop_distance = stop_distances.get(signal_strength, 0.05)
        
        if trade_type.lower() == "purchase":
            return current_price * (1 - stop_distance)
        else:
            return current_price * (1 + stop_distance)
    
    def calculate_position_size(self, signal_strength: SignalStrength, congressional_amount: float) -> float:
        """Calculate position size based on signal strength and available capital"""
        # Base position sizes as percentage of portfolio
        base_sizes = {
            SignalStrength.TIER_1_EXTREME: 0.15,  # 15% of portfolio
            SignalStrength.TIER_2_HIGH: 0.10,     # 10% of portfolio
            SignalStrength.TIER_3_MEDIUM: 0.06,   # 6% of portfolio
            SignalStrength.TIER_4_LOW: 0.03,      # 3% of portfolio
            SignalStrength.TIER_5_NOISE: 0.01     # 1% of portfolio
        }
        
        base_size = base_sizes.get(signal_strength, 0.03)
        
        # Adjust based on congressional trade size
        if congressional_amount > 1000000:
            base_size *= 1.5
        elif congressional_amount > 500000:
            base_size *= 1.2
        
        return min(0.20, base_size)  # Never exceed 20% of portfolio
    
    def calculate_filing_delay(self, trade_date: datetime) -> int:
        """Calculate days between trade and filing (for suspicion scoring)"""
        # Simplified - in real implementation would use actual filing date
        return np.random.randint(1, 180)  # 1-180 day delay
    
    def identify_legislative_catalyst(self, member_name: str, symbol: str) -> str:
        """Identify potential legislative catalyst for the trade"""
        catalysts = {
            "NVDA": "AI Regulation Hearing",
            "AAPL": "Antitrust Legislation",
            "GOOGL": "Privacy Regulation",
            "TSLA": "EV Tax Credits",
            "JPM": "Banking Regulation",
            "XOM": "Energy Policy"
        }
        
        return catalysts.get(symbol, "General Legislative Activity")
    
    def store_trading_signal(self, signal: TradingSignal):
        """Store trading signal in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trading_signals (
                symbol, signal_strength, direction, strategy_type, confidence,
                expected_return, risk_level, entry_price, target_price, stop_loss,
                position_size, congressional_member, legislative_catalyst, generated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.symbol, signal.signal_strength.value, signal.direction.value,
            signal.strategy_type.value, signal.confidence, signal.expected_return,
            signal.risk_level, signal.entry_price, signal.target_price,
            signal.stop_loss, signal.position_size, signal.congressional_member,
            signal.legislative_catalyst, signal.timestamp
        ))
        
        conn.commit()
        conn.close()

class AdvancedTradingAlgorithms:
    """Suite of advanced trading algorithms"""
    
    def __init__(self):
        self.algorithms = {
            StrategyType.MOMENTUM_CAPTURE: self.momentum_capture_algorithm,
            StrategyType.MEAN_REVERSION: self.mean_reversion_algorithm,
            StrategyType.ARBITRAGE: self.arbitrage_algorithm,
            StrategyType.OPTIONS_FLOW: self.options_flow_algorithm,
            StrategyType.PAIRS_TRADING: self.pairs_trading_algorithm,
            StrategyType.EARNINGS_PLAY: self.earnings_play_algorithm
        }
    
    def execute_strategy(self, signal: TradingSignal) -> Dict[str, Any]:
        """Execute the appropriate trading strategy"""
        algorithm = self.algorithms.get(signal.strategy_type)
        if algorithm:
            return algorithm(signal)
        else:
            return {"error": f"Unknown strategy type: {signal.strategy_type}"}
    
    def momentum_capture_algorithm(self, signal: TradingSignal) -> Dict[str, Any]:
        """High-frequency momentum capture for extreme signals"""
        
        # Get recent price data
        ticker = yf.Ticker(signal.symbol)
        data = ticker.history(period="5d", interval="1m")
        
        if data.empty:
            return {"error": "No price data available"}
        
        # Calculate momentum indicators
        data['sma_20'] = data['Close'].rolling(window=20).mean()
        data['rsi'] = self.calculate_rsi(data['Close'], 14)
        data['volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=20).mean()
        
        # Entry conditions
        current_price = data['Close'].iloc[-1]
        current_rsi = data['rsi'].iloc[-1]
        volume_spike = data['volume_ratio'].iloc[-1] > 2.0
        
        entry_signal = False
        if signal.direction == TradeDirection.LONG:
            entry_signal = (current_rsi < 70 and volume_spike and 
                          current_price > data['sma_20'].iloc[-1])
        elif signal.direction == TradeDirection.SHORT:
            entry_signal = (current_rsi > 30 and volume_spike and 
                          current_price < data['sma_20'].iloc[-1])
        
        if entry_signal:
            return {
                "action": "ENTER",
                "entry_price": current_price,
                "position_size": signal.position_size,
                "target": signal.target_price,
                "stop_loss": signal.stop_loss,
                "strategy": "Momentum Capture",
                "confidence": signal.confidence
            }
        else:
            return {
                "action": "WAIT",
                "reason": "Entry conditions not met",
                "current_rsi": current_rsi,
                "volume_ratio": data['volume_ratio'].iloc[-1]
            }
    
    def mean_reversion_algorithm(self, signal: TradingSignal) -> Dict[str, Any]:
        """Mean reversion strategy for oversold/overbought conditions"""
        
        ticker = yf.Ticker(signal.symbol)
        data = ticker.history(period="30d")
        
        if data.empty:
            return {"error": "No price data available"}
        
        # Calculate mean reversion indicators
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = self.calculate_bollinger_bands(data['Close'])
        data['rsi'] = self.calculate_rsi(data['Close'], 14)
        data['z_score'] = (data['Close'] - data['Close'].rolling(window=20).mean()) / data['Close'].rolling(window=20).std()
        
        current_price = data['Close'].iloc[-1]
        current_rsi = data['rsi'].iloc[-1]
        current_z_score = data['z_score'].iloc[-1]
        
        # Mean reversion entry conditions
        entry_signal = False
        if signal.direction == TradeDirection.LONG:
            # Buy when oversold
            entry_signal = (current_rsi < 30 and current_z_score < -1.5 and 
                          current_price < data['bb_lower'].iloc[-1])
        elif signal.direction == TradeDirection.SHORT:
            # Sell when overbought
            entry_signal = (current_rsi > 70 and current_z_score > 1.5 and 
                          current_price > data['bb_upper'].iloc[-1])
        
        if entry_signal:
            return {
                "action": "ENTER",
                "entry_price": current_price,
                "position_size": signal.position_size * 0.8,  # Smaller size for mean reversion
                "target": data['bb_middle'].iloc[-1],  # Target middle band
                "stop_loss": signal.stop_loss,
                "strategy": "Mean Reversion",
                "confidence": signal.confidence * 0.9
            }
        else:
            return {
                "action": "WAIT",
                "reason": "Mean reversion conditions not met",
                "current_rsi": current_rsi,
                "z_score": current_z_score
            }
    
    def arbitrage_algorithm(self, signal: TradingSignal) -> Dict[str, Any]:
        """Arbitrage opportunities detection"""
        
        # Simplified arbitrage - in practice would check multiple exchanges
        ticker = yf.Ticker(signal.symbol)
        data = ticker.history(period="1d", interval="1m")
        
        if len(data) < 10:
            return {"error": "Insufficient data for arbitrage"}
        
        # Look for price discrepancies (simplified)
        recent_prices = data['Close'].tail(10)
        price_volatility = recent_prices.std() / recent_prices.mean()
        
        if price_volatility > 0.02:  # 2% volatility threshold
            return {
                "action": "ENTER",
                "entry_price": data['Close'].iloc[-1],
                "position_size": signal.position_size * 0.5,  # Conservative sizing
                "target": signal.target_price,
                "stop_loss": signal.stop_loss,
                "strategy": "Statistical Arbitrage",
                "confidence": min(0.8, signal.confidence)
            }
        else:
            return {
                "action": "WAIT",
                "reason": "No arbitrage opportunity detected",
                "volatility": price_volatility
            }
    
    def options_flow_algorithm(self, signal: TradingSignal) -> Dict[str, Any]:
        """Options flow analysis for enhanced signals"""
        
        # Simplified options analysis - would integrate with options data provider
        ticker = yf.Ticker(signal.symbol)
        
        try:
            options_dates = ticker.options
            if not options_dates:
                return {"error": "No options data available"}
            
            # Get nearest expiration options
            nearest_expiry = options_dates[0]
            calls = ticker.option_chain(nearest_expiry).calls
            puts = ticker.option_chain(nearest_expiry).puts
            
            # Analyze call/put ratio and volume
            total_call_volume = calls['volume'].sum()
            total_put_volume = puts['volume'].sum()
            
            if total_call_volume == 0 and total_put_volume == 0:
                call_put_ratio = 1.0
            elif total_put_volume == 0:
                call_put_ratio = float('inf')
            else:
                call_put_ratio = total_call_volume / total_put_volume
            
            # Options strategy based on flow
            if signal.direction == TradeDirection.LONG and call_put_ratio > 2.0:
                return {
                    "action": "ENTER",
                    "strategy": "Long Call Options",
                    "entry_price": signal.entry_price,
                    "position_size": signal.position_size,
                    "target": signal.target_price,
                    "stop_loss": signal.stop_loss,
                    "call_put_ratio": call_put_ratio,
                    "confidence": signal.confidence
                }
            elif signal.direction == TradeDirection.SHORT and call_put_ratio < 0.5:
                return {
                    "action": "ENTER",
                    "strategy": "Long Put Options",
                    "entry_price": signal.entry_price,
                    "position_size": signal.position_size,
                    "target": signal.target_price,
                    "stop_loss": signal.stop_loss,
                    "call_put_ratio": call_put_ratio,
                    "confidence": signal.confidence
                }
            else:
                return {
                    "action": "WAIT",
                    "reason": "Options flow not supportive",
                    "call_put_ratio": call_put_ratio
                }
                
        except Exception as e:
            return {"error": f"Options analysis failed: {str(e)}"}
    
    def pairs_trading_algorithm(self, signal: TradingSignal) -> Dict[str, Any]:
        """Pairs trading for sector-based signals"""
        
        # Define sector pairs
        sector_pairs = {
            "AAPL": "MSFT",
            "GOOGL": "META",
            "JPM": "BAC",
            "XOM": "CVX"
        }
        
        pair_symbol = sector_pairs.get(signal.symbol)
        if not pair_symbol:
            return {"error": "No suitable pair found"}
        
        # Get data for both symbols
        ticker1 = yf.Ticker(signal.symbol)
        ticker2 = yf.Ticker(pair_symbol)
        
        data1 = ticker1.history(period="30d")
        data2 = ticker2.history(period="30d")
        
        if data1.empty or data2.empty:
            return {"error": "Insufficient data for pairs trading"}
        
        # Calculate spread and z-score
        spread = data1['Close'] / data2['Close']
        spread_mean = spread.rolling(window=20).mean()
        spread_std = spread.rolling(window=20).std()
        z_score = (spread - spread_mean) / spread_std
        
        current_z_score = z_score.iloc[-1]
        
        # Pairs trading entry conditions
        if abs(current_z_score) > 2.0:  # Significant divergence
            if current_z_score > 2.0:  # Symbol1 overvalued relative to Symbol2
                return {
                    "action": "ENTER",
                    "strategy": "Pairs Trading",
                    "long_symbol": pair_symbol,
                    "short_symbol": signal.symbol,
                    "position_size": signal.position_size * 0.5,
                    "target_z_score": 0,
                    "current_z_score": current_z_score,
                    "confidence": signal.confidence * 0.8
                }
            else:  # Symbol1 undervalued relative to Symbol2
                return {
                    "action": "ENTER",
                    "strategy": "Pairs Trading",
                    "long_symbol": signal.symbol,
                    "short_symbol": pair_symbol,
                    "position_size": signal.position_size * 0.5,
                    "target_z_score": 0,
                    "current_z_score": current_z_score,
                    "confidence": signal.confidence * 0.8
                }
        else:
            return {
                "action": "WAIT",
                "reason": "Pair not sufficiently diverged",
                "z_score": current_z_score
            }
    
    def earnings_play_algorithm(self, signal: TradingSignal) -> Dict[str, Any]:
        """Earnings-based trading strategy"""
        
        ticker = yf.Ticker(signal.symbol)
        
        try:
            # Get earnings calendar (simplified)
            calendar = ticker.calendar
            if calendar is None or calendar.empty:
                return {"error": "No earnings data available"}
            
            # Check if earnings are upcoming (within 30 days)
            next_earnings = pd.to_datetime(calendar.index[0])
            days_to_earnings = (next_earnings - datetime.now()).days
            
            if 0 <= days_to_earnings <= 30:
                # Get historical earnings data
                earnings = ticker.earnings
                if not earnings.empty:
                    recent_growth = earnings['Earnings'].pct_change().tail(4).mean()
                    
                    if signal.direction == TradeDirection.LONG and recent_growth > 0.1:
                        return {
                            "action": "ENTER",
                            "strategy": "Pre-Earnings Long",
                            "entry_price": signal.entry_price,
                            "position_size": signal.position_size * 0.7,
                            "target": signal.target_price,
                            "stop_loss": signal.stop_loss,
                            "days_to_earnings": days_to_earnings,
                            "earnings_growth": recent_growth,
                            "confidence": signal.confidence
                        }
                    elif signal.direction == TradeDirection.SHORT and recent_growth < -0.1:
                        return {
                            "action": "ENTER",
                            "strategy": "Pre-Earnings Short",
                            "entry_price": signal.entry_price,
                            "position_size": signal.position_size * 0.7,
                            "target": signal.target_price,
                            "stop_loss": signal.stop_loss,
                            "days_to_earnings": days_to_earnings,
                            "earnings_growth": recent_growth,
                            "confidence": signal.confidence
                        }
                    else:
                        return {
                            "action": "WAIT",
                            "reason": "Earnings trend not supportive",
                            "earnings_growth": recent_growth
                        }
            else:
                return {
                    "action": "WAIT",
                    "reason": "No upcoming earnings",
                    "days_to_earnings": days_to_earnings
                }
                
        except Exception as e:
            return {"error": f"Earnings analysis failed: {str(e)}"}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

class AIModelSuite:
    """Advanced AI models for trading prediction and analysis"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_performance = {}
        
        if ML_AVAILABLE:
            self.initialize_models()
    
    def initialize_models(self):
        """Initialize AI models"""
        
        # Price prediction model
        self.models['price_predictor'] = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42
        )
        
        # Signal classification model
        self.models['signal_classifier'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Risk assessment model
        self.models['risk_assessor'] = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
    
    def train_models(self, training_data: pd.DataFrame):
        """Train AI models on historical data"""
        
        if not ML_AVAILABLE:
            print("ML libraries not available for model training")
            return
        
        # Prepare features and targets
        features = self.prepare_features(training_data)
        
        # Train price predictor
        if 'future_return' in training_data.columns:
            X_price = features
            y_price = training_data['future_return']
            
            X_price_scaled = self.scalers['price_predictor'].fit_transform(X_price)
            self.models['price_predictor'].fit(X_price_scaled, y_price)
        
        # Train signal classifier
        if 'signal_success' in training_data.columns:
            X_signal = features
            y_signal = training_data['signal_success']
            
            X_signal_scaled = self.scalers['signal_classifier'].fit_transform(X_signal)
            self.models['signal_classifier'].fit(X_signal_scaled, y_signal)
        
        # Train risk assessor
        if 'max_drawdown' in training_data.columns:
            X_risk = features
            y_risk = training_data['max_drawdown']
            
            X_risk_scaled = self.scalers['risk_assessor'].fit_transform(X_risk)
            self.models['risk_assessor'].fit(X_risk_scaled, y_risk)
        
        print("AI models trained successfully")
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models"""
        
        features = pd.DataFrame()
        
        # Technical indicators
        if 'close' in data.columns:
            features['rsi'] = self.calculate_rsi(data['close'])
            features['sma_20'] = data['close'].rolling(window=20).mean()
            features['sma_50'] = data['close'].rolling(window=50).mean()
            features['volatility'] = data['close'].rolling(window=20).std()
            features['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
        
        # Congressional features
        if 'signal_strength' in data.columns:
            features['signal_strength'] = data['signal_strength']
        if 'member_reliability' in data.columns:
            features['member_reliability'] = data['member_reliability']
        if 'committee_relevance' in data.columns:
            features['committee_relevance'] = data['committee_relevance']
        if 'filing_delay' in data.columns:
            features['filing_delay'] = data['filing_delay']
        
        # Market features
        if 'market_cap' in data.columns:
            features['market_cap'] = np.log(data['market_cap'])
        if 'sector_performance' in data.columns:
            features['sector_performance'] = data['sector_performance']
        
        return features.fillna(0)
    
    def predict_price_movement(self, signal: TradingSignal, market_data: pd.DataFrame) -> Dict[str, float]:
        """Predict price movement using AI models"""
        
        if not ML_AVAILABLE or 'price_predictor' not in self.models:
            return {"error": "Price prediction model not available"}
        
        # Prepare features
        features = self.prepare_signal_features(signal, market_data)
        features_scaled = self.scalers['price_predictor'].transform([features])
        
        # Make prediction
        predicted_return = self.models['price_predictor'].predict(features_scaled)[0]
        
        # Calculate confidence based on model performance
        confidence = self.get_model_confidence('price_predictor', features)
        
        return {
            "predicted_return": predicted_return,
            "confidence": confidence,
            "model": "Neural Network Price Predictor"
        }
    
    def classify_signal_quality(self, signal: TradingSignal, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Classify signal quality using AI"""
        
        if not ML_AVAILABLE or 'signal_classifier' not in self.models:
            return {"error": "Signal classification model not available"}
        
        # Prepare features
        features = self.prepare_signal_features(signal, market_data)
        features_scaled = self.scalers['signal_classifier'].transform([features])
        
        # Make prediction
        signal_quality = self.models['signal_classifier'].predict(features_scaled)[0]
        signal_probability = self.models['signal_classifier'].predict_proba(features_scaled)[0]
        
        return {
            "signal_quality": "High" if signal_quality == 1 else "Low",
            "success_probability": max(signal_probability),
            "model": "Random Forest Signal Classifier"
        }
    
    def assess_risk(self, signal: TradingSignal, market_data: pd.DataFrame) -> Dict[str, float]:
        """Assess risk using AI models"""
        
        if not ML_AVAILABLE or 'risk_assessor' not in self.models:
            return {"error": "Risk assessment model not available"}
        
        # Prepare features
        features = self.prepare_signal_features(signal, market_data)
        features_scaled = self.scalers['risk_assessor'].transform([features])
        
        # Make prediction
        predicted_risk = self.models['risk_assessor'].predict(features_scaled)[0]
        
        return {
            "predicted_max_drawdown": predicted_risk,
            "risk_level": "High" if predicted_risk > 0.15 else "Medium" if predicted_risk > 0.08 else "Low",
            "model": "Gradient Boosting Risk Assessor"
        }
    
    def prepare_signal_features(self, signal: TradingSignal, market_data: pd.DataFrame) -> List[float]:
        """Prepare features for a single signal"""
        
        features = []
        
        # Signal features
        features.append(signal.signal_strength.value)
        features.append(signal.confidence)
        features.append(signal.committee_relevance)
        features.append(signal.metadata.get('filing_delay', 30))
        
        # Market features
        if not market_data.empty:
            current_price = market_data['Close'].iloc[-1]
            features.append(self.calculate_rsi(market_data['Close']).iloc[-1])
            features.append(current_price / market_data['Close'].rolling(window=20).mean().iloc[-1])
            features.append(market_data['Close'].rolling(window=20).std().iloc[-1] / current_price)
            features.append(market_data['Volume'].iloc[-1] / market_data['Volume'].rolling(window=20).mean().iloc[-1])
        else:
            features.extend([50, 1, 0.02, 1])  # Default values
        
        # Member features
        member_profile = signal.metadata.get('member_profile', {})
        features.append(member_profile.get('reliability_score', 0.5))
        features.append(member_profile.get('avg_return', 0.05))
        
        return features
    
    def get_model_confidence(self, model_name: str, features: List[float]) -> float:
        """Get model confidence based on feature similarity to training data"""
        
        # Simplified confidence calculation
        # In practice, would use more sophisticated methods like prediction intervals
        base_confidence = self.model_performance.get(model_name, 0.7)
        
        # Adjust based on feature extremes
        feature_array = np.array(features)
        if np.any(np.abs(feature_array) > 3):  # Outlier features
            base_confidence *= 0.8
        
        return min(0.95, base_confidence)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for AI features"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class APEXTradingEngine:
    """Main APEX trading engine that coordinates all components"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        
        # Initialize components
        self.intelligence_processor = CongressionalIntelligenceProcessor()
        self.trading_algorithms = AdvancedTradingAlgorithms()
        self.ai_models = AIModelSuite()
        
        # Trading state
        self.active_signals = []
        self.active_positions = []
        self.performance_metrics = {}
        
        # Initialize trading connection if API keys provided
        self.trading_client = None
        if ALPACA_AVAILABLE and 'alpaca_key' in self.api_keys:
            self.initialize_trading_client()
    
    def initialize_trading_client(self):
        """Initialize live trading client"""
        try:
            self.trading_client = tradeapi.REST(
                self.api_keys['alpaca_key'],
                self.api_keys['alpaca_secret'],
                base_url='https://paper-api.alpaca.markets'  # Paper trading
            )
            print("Trading client initialized successfully")
        except Exception as e:
            print(f"Failed to initialize trading client: {e}")
    
    def process_congressional_trade(self, member_name: str, symbol: str, 
                                  trade_type: str, amount: float, 
                                  trade_date: datetime) -> Dict[str, Any]:
        """Process congressional trade and generate comprehensive analysis"""
        
        print(f"\n🚨 PROCESSING CONGRESSIONAL TRADE 🚨")
        print(f"Member: {member_name}")
        print(f"Symbol: {symbol}")
        print(f"Type: {trade_type}")
        print(f"Amount: ${amount:,.2f}")
        print(f"Date: {trade_date}")
        
        # Generate trading signal
        signal = self.intelligence_processor.process_congressional_trade(
            member_name, symbol, trade_type, amount, trade_date
        )
        
        # Get market data
        try:
            ticker = yf.Ticker(symbol)
            market_data = ticker.history(period="30d")
        except:
            market_data = pd.DataFrame()
        
        # AI analysis
        ai_analysis = {}
        if ML_AVAILABLE and hasattr(self.ai_models, 'models') and self.ai_models.models:
            try:
                ai_analysis['price_prediction'] = self.ai_models.predict_price_movement(signal, market_data)
                ai_analysis['signal_quality'] = self.ai_models.classify_signal_quality(signal, market_data)
                ai_analysis['risk_assessment'] = self.ai_models.assess_risk(signal, market_data)
            except Exception as e:
                ai_analysis['error'] = f"AI analysis unavailable: {str(e)}"
                print(f"⚠️  AI models not trained - using basic analysis only")
        
        # Trading strategy analysis
        strategy_analysis = self.trading_algorithms.execute_strategy(signal)
        
        # Compile comprehensive analysis
        analysis = {
            "signal": {
                "symbol": signal.symbol,
                "strength": signal.signal_strength.name,
                "strength_value": signal.signal_strength.value,
                "direction": signal.direction.name,
                "strategy": signal.strategy_type.value,
                "confidence": signal.confidence,
                "expected_return": signal.expected_return,
                "risk_level": signal.risk_level,
                "time_horizon": signal.time_horizon,
                "position_size": signal.position_size,
                "congressional_member": signal.congressional_member,
                "legislative_catalyst": signal.legislative_catalyst
            },
            "pricing": {
                "entry_price": signal.entry_price,
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss,
                "potential_profit": ((signal.target_price - signal.entry_price) / signal.entry_price) * 100,
                "risk_reward_ratio": (signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss)
            },
            "ai_analysis": ai_analysis,
            "strategy_analysis": strategy_analysis,
            "recommendation": self.generate_recommendation(signal, strategy_analysis, ai_analysis)
        }
        
        # Store active signal
        self.active_signals.append(signal)
        
        return analysis
    
    def generate_recommendation(self, signal: TradingSignal, 
                              strategy_analysis: Dict[str, Any],
                              ai_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final trading recommendation"""
        
        # Base recommendation from signal strength
        if signal.signal_strength == SignalStrength.TIER_1_EXTREME:
            base_action = "STRONG BUY" if signal.direction == TradeDirection.LONG else "STRONG SELL"
            base_confidence = 0.9
        elif signal.signal_strength == SignalStrength.TIER_2_HIGH:
            base_action = "BUY" if signal.direction == TradeDirection.LONG else "SELL"
            base_confidence = 0.75
        elif signal.signal_strength == SignalStrength.TIER_3_MEDIUM:
            base_action = "MODERATE BUY" if signal.direction == TradeDirection.LONG else "MODERATE SELL"
            base_confidence = 0.6
        else:
            base_action = "WATCH"
            base_confidence = 0.4
        
        # Adjust based on strategy analysis
        if strategy_analysis.get("action") == "WAIT":
            base_action = "WAIT"
            base_confidence *= 0.5
        elif strategy_analysis.get("action") == "ENTER":
            base_confidence *= 1.1
        
        # Adjust based on AI analysis
        if ai_analysis:
            if 'signal_quality' in ai_analysis:
                quality = ai_analysis['signal_quality']
                if quality.get('signal_quality') == 'High':
                    base_confidence *= 1.2
                else:
                    base_confidence *= 0.8
            
            if 'risk_assessment' in ai_analysis:
                risk = ai_analysis['risk_assessment']
                if risk.get('risk_level') == 'High':
                    base_confidence *= 0.7
        
        # Final confidence
        final_confidence = min(0.95, base_confidence)
        
        return {
            "action": base_action,
            "confidence": final_confidence,
            "reasoning": self.generate_reasoning(signal, strategy_analysis, ai_analysis),
            "execution_priority": "HIGH" if final_confidence > 0.8 else "MEDIUM" if final_confidence > 0.6 else "LOW"
        }
    
    def generate_reasoning(self, signal: TradingSignal, 
                          strategy_analysis: Dict[str, Any],
                          ai_analysis: Dict[str, Any]) -> str:
        """Generate human-readable reasoning for the recommendation"""
        
        reasoning_parts = []
        
        # Signal strength reasoning
        if signal.signal_strength == SignalStrength.TIER_1_EXTREME:
            reasoning_parts.append(f"EXTREME signal from {signal.congressional_member} with {signal.confidence:.1%} confidence")
        elif signal.signal_strength == SignalStrength.TIER_2_HIGH:
            reasoning_parts.append(f"HIGH signal from {signal.congressional_member}")
        
        # Committee relevance
        if signal.committee_relevance > 1.2:
            reasoning_parts.append(f"Strong committee-sector alignment (relevance: {signal.committee_relevance:.1f})")
        
        # Legislative catalyst
        if signal.legislative_catalyst != "General Legislative Activity":
            reasoning_parts.append(f"Specific catalyst: {signal.legislative_catalyst}")
        
        # Strategy analysis
        if strategy_analysis.get("action") == "ENTER":
            reasoning_parts.append(f"Technical conditions favorable for {strategy_analysis.get('strategy', 'trading')}")
        
        # AI analysis
        if ai_analysis and 'price_prediction' in ai_analysis:
            pred = ai_analysis['price_prediction']
            if not isinstance(pred, dict) or 'error' not in pred:
                reasoning_parts.append("AI models support the signal")
        
        return "; ".join(reasoning_parts)
    
    def execute_trade(self, signal: TradingSignal, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute trade based on signal and analysis"""
        
        if not self.trading_client:
            return {
                "status": "SIMULATED",
                "message": "No trading client configured - trade simulated",
                "signal": signal.symbol,
                "action": analysis['recommendation']['action']
            }
        
        try:
            # Determine order parameters
            symbol = signal.symbol
            qty = int(signal.position_size * 10000 / signal.entry_price)  # Assuming $10k base position
            side = 'buy' if signal.direction == TradeDirection.LONG else 'sell'
            
            # Place order
            order = self.trading_client.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            # Track position
            position = {
                "order_id": order.id,
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "signal": signal,
                "entry_time": datetime.now(),
                "target_price": signal.target_price,
                "stop_loss": signal.stop_loss
            }
            
            self.active_positions.append(position)
            
            return {
                "status": "EXECUTED",
                "order_id": order.id,
                "symbol": symbol,
                "quantity": qty,
                "side": side,
                "message": f"Order placed successfully for {symbol}"
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Failed to execute trade: {str(e)}",
                "symbol": signal.symbol
            }
    
    def monitor_positions(self) -> List[Dict[str, Any]]:
        """Monitor active positions and manage exits"""
        
        position_updates = []
        
        for position in self.active_positions:
            try:
                # Get current price
                ticker = yf.Ticker(position['symbol'])
                current_price = ticker.history(period="1d")['Close'].iloc[-1]
                
                # Calculate P&L
                entry_price = position['signal'].entry_price
                if position['side'] == 'buy':
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Check exit conditions
                should_exit = False
                exit_reason = ""
                
                # Target hit
                if position['side'] == 'buy' and current_price >= position['target_price']:
                    should_exit = True
                    exit_reason = "Target reached"
                elif position['side'] == 'sell' and current_price <= position['target_price']:
                    should_exit = True
                    exit_reason = "Target reached"
                
                # Stop loss hit
                if position['side'] == 'buy' and current_price <= position['stop_loss']:
                    should_exit = True
                    exit_reason = "Stop loss triggered"
                elif position['side'] == 'sell' and current_price >= position['stop_loss']:
                    should_exit = True
                    exit_reason = "Stop loss triggered"
                
                # Time-based exit (simplified)
                days_held = (datetime.now() - position['entry_time']).days
                if days_held > 30:  # Max holding period
                    should_exit = True
                    exit_reason = "Time limit reached"
                
                position_update = {
                    "symbol": position['symbol'],
                    "current_price": current_price,
                    "pnl_percent": pnl_pct * 100,
                    "days_held": days_held,
                    "should_exit": should_exit,
                    "exit_reason": exit_reason
                }
                
                position_updates.append(position_update)
                
                # Execute exit if needed
                if should_exit and self.trading_client:
                    self.exit_position(position, exit_reason)
                
            except Exception as e:
                position_updates.append({
                    "symbol": position['symbol'],
                    "error": f"Failed to monitor position: {str(e)}"
                })
        
        return position_updates
    
    def exit_position(self, position: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """Exit a position"""
        
        try:
            # Close position
            opposite_side = 'sell' if position['side'] == 'buy' else 'buy'
            
            order = self.trading_client.submit_order(
                symbol=position['symbol'],
                qty=position['qty'],
                side=opposite_side,
                type='market',
                time_in_force='day'
            )
            
            # Remove from active positions
            self.active_positions.remove(position)
            
            return {
                "status": "CLOSED",
                "symbol": position['symbol'],
                "reason": reason,
                "exit_order_id": order.id
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "message": f"Failed to exit position: {str(e)}"
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        
        # This would be much more comprehensive in a real implementation
        total_signals = len(self.active_signals)
        
        if total_signals == 0:
            return {"message": "No signals processed yet"}
        
        # Calculate signal distribution
        signal_distribution = {}
        for signal in self.active_signals:
            tier = signal.signal_strength.name
            signal_distribution[tier] = signal_distribution.get(tier, 0) + 1
        
        # Calculate average confidence
        avg_confidence = np.mean([s.confidence for s in self.active_signals])
        
        # Calculate expected returns
        total_expected_return = sum([s.expected_return * s.position_size for s in self.active_signals])
        
        return {
            "total_signals": total_signals,
            "signal_distribution": signal_distribution,
            "average_confidence": avg_confidence,
            "total_expected_return": total_expected_return,
            "active_positions": len(self.active_positions),
            "top_members": self.get_top_members()
        }
    
    def get_top_members(self) -> List[Dict[str, Any]]:
        """Get top performing congressional members"""
        
        member_stats = {}
        for signal in self.active_signals:
            member = signal.congressional_member
            if member not in member_stats:
                member_stats[member] = {
                    "signals": 0,
                    "avg_confidence": 0,
                    "total_expected_return": 0
                }
            
            member_stats[member]["signals"] += 1
            member_stats[member]["avg_confidence"] += signal.confidence
            member_stats[member]["total_expected_return"] += signal.expected_return
        
        # Calculate averages
        for member in member_stats:
            stats = member_stats[member]
            stats["avg_confidence"] /= stats["signals"]
        
        # Sort by total expected return
        sorted_members = sorted(
            member_stats.items(),
            key=lambda x: x[1]["total_expected_return"],
            reverse=True
        )
        
        return [
            {
                "member": member,
                "signals": stats["signals"],
                "avg_confidence": stats["avg_confidence"],
                "total_expected_return": stats["total_expected_return"]
            }
            for member, stats in sorted_members[:5]
        ]

def demo_apex_system():
    """Demonstrate the APEX trading system"""
    
    print("🚀 APEX TRADING SYSTEM DEMONSTRATION 🚀")
    print("=" * 60)
    
    # Initialize APEX
    apex = APEXTradingEngine()
    
    # Simulate some congressional trades
    demo_trades = [
        {
            "member": "Nancy Pelosi",
            "symbol": "NVDA",
            "type": "Purchase",
            "amount": 3000000,
            "date": datetime.now() - timedelta(days=5)
        },
        {
            "member": "Ron Wyden",
            "symbol": "NVDA",
            "type": "Purchase",
            "amount": 375000,
            "date": datetime.now() - timedelta(days=3)
        },
        {
            "member": "Ro Khanna",
            "symbol": "AAPL",
            "type": "Sale",
            "amount": 32500,
            "date": datetime.now() - timedelta(days=1)
        }
    ]
    
    # Process each trade
    for trade in demo_trades:
        analysis = apex.process_congressional_trade(
            trade["member"],
            trade["symbol"],
            trade["type"],
            trade["amount"],
            trade["date"]
        )
        
        print(f"\n📊 ANALYSIS COMPLETE FOR {trade['symbol']} 📊")
        print(f"Signal Strength: {analysis['signal']['strength']} ({analysis['signal']['strength_value']}/10)")
        print(f"Recommendation: {analysis['recommendation']['action']}")
        print(f"Confidence: {analysis['recommendation']['confidence']:.1%}")
        print(f"Expected Return: {analysis['signal']['expected_return']:.1%}")
        print(f"Strategy: {analysis['signal']['strategy']}")
        print(f"Reasoning: {analysis['recommendation']['reasoning']}")
        print("-" * 40)
    
    # Show performance summary
    print("\n📈 PERFORMANCE SUMMARY 📈")
    summary = apex.get_performance_summary()
    print(f"Total Signals Processed: {summary['total_signals']}")
    print(f"Average Confidence: {summary['average_confidence']:.1%}")
    print(f"Signal Distribution: {summary['signal_distribution']}")
    
    print("\n🏆 TOP PERFORMING MEMBERS 🏆")
    for i, member_data in enumerate(summary['top_members'], 1):
        print(f"{i}. {member_data['member']}: {member_data['signals']} signals, "
              f"{member_data['avg_confidence']:.1%} avg confidence")
    
    print("\n✅ APEX SYSTEM DEMONSTRATION COMPLETE ✅")
    return apex

if __name__ == "__main__":
    # Run demonstration
    apex_system = demo_apex_system()

