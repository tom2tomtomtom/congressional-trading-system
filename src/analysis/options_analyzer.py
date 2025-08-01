#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Options and Derivatives Analysis
Advanced analysis of complex financial instruments in congressional trading.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import json

logger = logging.getLogger(__name__)

class OptionType(Enum):
    """Option types."""
    CALL = "call"
    PUT = "put"

class StrategyType(Enum):
    """Options strategy types."""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"
    SHORT_PUT = "short_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    COLLAR = "collar"
    BUTTERFLY = "butterfly"
    UNKNOWN = "unknown"

class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class OptionsContract:
    """Data model for options contracts."""
    underlying_symbol: str
    expiry_date: str
    strike_price: float
    option_type: OptionType
    premium: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

@dataclass
class TradeLeg:
    """Individual leg of a complex options strategy."""
    contract: OptionsContract
    quantity: int
    action: str  # buy, sell
    cost: float

@dataclass
class ComplexStrategy:
    """Complex options trading strategy."""
    strategy_id: str
    strategy_type: StrategyType
    member_id: str
    trade_date: str
    legs: List[TradeLeg]
    total_cost: float
    max_profit: Optional[float]
    max_loss: Optional[float]
    break_even_points: List[float]
    risk_level: RiskLevel
    market_outlook: str  # bullish, bearish, neutral
    probability_of_profit: float

@dataclass
class RiskMetrics:
    """Risk assessment metrics for options strategies."""
    strategy_id: str
    portfolio_beta: float
    var_95: float  # Value at Risk (95% confidence)
    expected_return: float
    sharpe_ratio: float
    maximum_drawdown: float
    leverage_ratio: float
    time_decay_risk: float
    volatility_risk: float

@dataclass
class TimingAnalysis:
    """Analysis of options trade timing relative to events."""
    trade_date: str
    days_to_expiry: int
    days_to_earnings: Optional[int]
    days_to_ex_dividend: Optional[int]
    implied_volatility_rank: float
    historical_volatility: float
    volatility_skew: float
    timing_score: float
    suspicious_indicators: List[str]

class BlackScholesCalculator:
    """Black-Scholes option pricing and Greeks calculation."""
    
    @staticmethod
    def calculate_option_price(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> float:
        """
        Calculate Black-Scholes option price.
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility
            option_type: Call or Put
            
        Returns:
            Option price
        """
        if T <= 0:
            # At expiration
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == OptionType.CALL:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(0, price)
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType) -> Dict[str, float]:
        """
        Calculate option Greeks.
        
        Returns:
            Dictionary with Delta, Gamma, Theta, Vega, Rho
        """
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type == OptionType.CALL:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Theta
        theta_common = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T)
        if option_type == OptionType.CALL:
            theta = (theta_common * norm.cdf(d2)) / 365  # Convert to daily
        else:
            theta = (theta_common * norm.cdf(-d2)) / 365  # Convert to daily
        
        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Convert to 1% vol change
        
        # Rho
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Convert to 1% rate change
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100  # Convert to 1% rate change
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    @staticmethod
    def calculate_implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: OptionType) -> float:
        """
        Calculate implied volatility using bisection method.
        
        Returns:
            Implied volatility (annualized)
        """
        if T <= 0:
            return 0.0
        
        def objective(sigma):
            theoretical_price = BlackScholesCalculator.calculate_option_price(S, K, T, r, sigma, option_type)
            return abs(theoretical_price - market_price)
        
        try:
            result = minimize_scalar(objective, bounds=(0.01, 5.0), method='bounded')
            return result.x
        except:
            return 0.2  # Default 20% volatility

class OptionsStrategyIdentifier:
    """Identifies complex options strategies from trading data."""
    
    def __init__(self):
        """Initialize strategy identifier."""
        self.strategy_patterns = self._load_strategy_patterns()
    
    def _load_strategy_patterns(self) -> Dict[StrategyType, Dict]:
        """Load strategy identification patterns."""
        return {
            StrategyType.COVERED_CALL: {
                'required_legs': 2,
                'pattern': [
                    {'action': 'buy', 'asset_type': 'stock', 'quantity': 100},
                    {'action': 'sell', 'asset_type': 'call', 'quantity': 1}
                ],
                'market_outlook': 'neutral_bullish',
                'risk_level': RiskLevel.MEDIUM
            },
            StrategyType.PROTECTIVE_PUT: {
                'required_legs': 2,
                'pattern': [
                    {'action': 'buy', 'asset_type': 'stock', 'quantity': 100},
                    {'action': 'buy', 'asset_type': 'put', 'quantity': 1}
                ],
                'market_outlook': 'bullish',
                'risk_level': RiskLevel.LOW
            },
            StrategyType.BULL_CALL_SPREAD: {
                'required_legs': 2,
                'pattern': [
                    {'action': 'buy', 'asset_type': 'call', 'strike': 'lower'},
                    {'action': 'sell', 'asset_type': 'call', 'strike': 'higher'}
                ],
                'market_outlook': 'bullish',
                'risk_level': RiskLevel.MEDIUM
            },
            StrategyType.IRON_CONDOR: {
                'required_legs': 4,
                'pattern': [
                    {'action': 'sell', 'asset_type': 'put', 'strike': 'low'},
                    {'action': 'buy', 'asset_type': 'put', 'strike': 'lower'},
                    {'action': 'sell', 'asset_type': 'call', 'strike': 'higher'},
                    {'action': 'buy', 'asset_type': 'call', 'strike': 'high'}
                ],
                'market_outlook': 'neutral',
                'risk_level': RiskLevel.MEDIUM
            },
            StrategyType.STRADDLE: {
                'required_legs': 2,
                'pattern': [
                    {'action': 'buy', 'asset_type': 'call', 'strike': 'atm'},
                    {'action': 'buy', 'asset_type': 'put', 'strike': 'atm'}
                ],
                'market_outlook': 'volatile',
                'risk_level': RiskLevel.HIGH
            }
        }
    
    def identify_strategy(self, trades: List[Dict[str, Any]], time_window_days: int = 1) -> StrategyType:
        """
        Identify options strategy from a group of trades.
        
        Args:
            trades: List of trading records
            time_window_days: Time window to group trades
            
        Returns:
            Identified strategy type
        """
        if not trades:
            return StrategyType.UNKNOWN
        
        # Group trades by date and symbol
        grouped_trades = self._group_trades(trades, time_window_days)
        
        for symbol, symbol_trades in grouped_trades.items():
            strategy = self._match_strategy_pattern(symbol_trades)
            if strategy != StrategyType.UNKNOWN:
                return strategy
        
        # If no complex strategy identified, check for simple strategies
        return self._identify_simple_strategy(trades)
    
    def _group_trades(self, trades: List[Dict], time_window_days: int) -> Dict[str, List[Dict]]:
        """Group trades by symbol and time window."""
        grouped = {}
        
        for trade in trades:
            symbol = trade.get('symbol', 'UNKNOWN')
            trade_date = pd.to_datetime(trade.get('transaction_date', datetime.now()))
            
            # Group by symbol and normalize date
            date_key = trade_date.date()
            group_key = f"{symbol}_{date_key}"
            
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(trade)
        
        return grouped
    
    def _match_strategy_pattern(self, trades: List[Dict]) -> StrategyType:
        """Match trades against known strategy patterns."""
        # Extract trade characteristics
        trade_legs = []
        for trade in trades:
            leg = {
                'action': 'buy' if trade.get('transaction_type') == 'Purchase' else 'sell',
                'asset_type': self._determine_asset_type(trade),
                'strike': trade.get('strike_price'),
                'quantity': trade.get('quantity', 1)
            }
            trade_legs.append(leg)
        
        # Match against patterns
        for strategy_type, pattern in self.strategy_patterns.items():
            if self._matches_pattern(trade_legs, pattern['pattern']):
                return strategy_type
        
        return StrategyType.UNKNOWN
    
    def _determine_asset_type(self, trade: Dict) -> str:
        """Determine asset type from trade data."""
        asset_type = trade.get('asset_type', '').lower()
        asset_name = trade.get('asset_name', '').lower()
        
        if 'call' in asset_type or 'call' in asset_name:
            return 'call'
        elif 'put' in asset_type or 'put' in asset_name:
            return 'put'
        elif 'stock' in asset_type or 'common' in asset_name:
            return 'stock'
        else:
            return 'unknown'
    
    def _matches_pattern(self, trade_legs: List[Dict], pattern: List[Dict]) -> bool:
        """Check if trade legs match a strategy pattern."""
        if len(trade_legs) != len(pattern):
            return False
        
        # Simple pattern matching (would be more sophisticated in production)
        matched_legs = 0
        
        for pattern_leg in pattern:
            for trade_leg in trade_legs:
                if (pattern_leg['action'] == trade_leg['action'] and
                    pattern_leg['asset_type'] == trade_leg['asset_type']):
                    matched_legs += 1
                    break
        
        return matched_legs == len(pattern)
    
    def _identify_simple_strategy(self, trades: List[Dict]) -> StrategyType:
        """Identify simple single-leg strategies."""
        if len(trades) == 1:
            trade = trades[0]
            action = 'buy' if trade.get('transaction_type') == 'Purchase' else 'sell'
            asset_type = self._determine_asset_type(trade)
            
            if asset_type == 'call':
                return StrategyType.LONG_CALL if action == 'buy' else StrategyType.SHORT_CALL
            elif asset_type == 'put':
                return StrategyType.LONG_PUT if action == 'buy' else StrategyType.SHORT_PUT
        
        return StrategyType.UNKNOWN

class OptionsRiskAnalyzer:
    """Analyzes risk metrics for options strategies."""
    
    def __init__(self):
        """Initialize risk analyzer."""
        self.risk_free_rate = 0.05  # 5% default risk-free rate
        self.bs_calculator = BlackScholesCalculator()
    
    def calculate_strategy_risk_metrics(self, strategy: ComplexStrategy, 
                                      market_data: Dict[str, Any]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for an options strategy.
        
        Args:
            strategy: Complex options strategy
            market_data: Current market data
            
        Returns:
            Risk metrics
        """
        logger.info(f"Calculating risk metrics for strategy {strategy.strategy_id}")
        
        # Get current stock price and volatility
        stock_price = market_data.get('current_price', 100)
        volatility = market_data.get('volatility', 0.2)
        
        # Calculate portfolio Greeks
        portfolio_delta = self._calculate_portfolio_delta(strategy, stock_price, volatility)
        portfolio_gamma = self._calculate_portfolio_gamma(strategy, stock_price, volatility)
        portfolio_theta = self._calculate_portfolio_theta(strategy, stock_price, volatility)
        portfolio_vega = self._calculate_portfolio_vega(strategy, stock_price, volatility)
        
        # Portfolio beta (simplified)
        portfolio_beta = abs(portfolio_delta) * (stock_price / 100)  # Normalized
        
        # Value at Risk (95% confidence)
        var_95 = self._calculate_var(strategy, stock_price, volatility, confidence=0.95)
        
        # Expected return (simplified)
        expected_return = self._estimate_expected_return(strategy, stock_price, volatility)
        
        # Sharpe ratio
        if strategy.max_loss and strategy.max_loss != 0:
            sharpe_ratio = expected_return / abs(strategy.max_loss or 1)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown (theoretical)
        max_drawdown = abs(strategy.max_loss or 0) / abs(strategy.total_cost) if strategy.total_cost != 0 else 0
        
        # Leverage ratio
        notional_value = sum(leg.contract.strike_price * abs(leg.quantity) * 100 for leg in strategy.legs)
        leverage_ratio = notional_value / abs(strategy.total_cost) if strategy.total_cost != 0 else 0
        
        # Time decay risk
        time_decay_risk = abs(portfolio_theta) / abs(strategy.total_cost) if strategy.total_cost != 0 else 0
        
        # Volatility risk
        volatility_risk = abs(portfolio_vega) / abs(strategy.total_cost) if strategy.total_cost != 0 else 0
        
        return RiskMetrics(
            strategy_id=strategy.strategy_id,
            portfolio_beta=portfolio_beta,
            var_95=var_95,
            expected_return=expected_return,
            sharpe_ratio=sharpe_ratio,
            maximum_drawdown=max_drawdown,
            leverage_ratio=leverage_ratio,
            time_decay_risk=time_decay_risk,
            volatility_risk=volatility_risk
        )
    
    def _calculate_portfolio_delta(self, strategy: ComplexStrategy, stock_price: float, volatility: float) -> float:
        """Calculate portfolio delta."""
        total_delta = 0.0
        
        for leg in strategy.legs:
            # Calculate time to expiration
            expiry_date = datetime.fromisoformat(leg.contract.expiry_date)
            trade_date = datetime.fromisoformat(strategy.trade_date)
            time_to_expiry = (expiry_date - trade_date).days / 365.25
            
            if time_to_expiry > 0:
                greeks = self.bs_calculator.calculate_greeks(
                    stock_price, leg.contract.strike_price, time_to_expiry,
                    self.risk_free_rate, volatility, leg.contract.option_type
                )
                
                leg_delta = greeks['delta'] * leg.quantity
                if leg.action == 'sell':
                    leg_delta *= -1
                
                total_delta += leg_delta
        
        return total_delta
    
    def _calculate_portfolio_gamma(self, strategy: ComplexStrategy, stock_price: float, volatility: float) -> float:
        """Calculate portfolio gamma."""
        total_gamma = 0.0
        
        for leg in strategy.legs:
            expiry_date = datetime.fromisoformat(leg.contract.expiry_date)
            trade_date = datetime.fromisoformat(strategy.trade_date)
            time_to_expiry = (expiry_date - trade_date).days / 365.25
            
            if time_to_expiry > 0:
                greeks = self.bs_calculator.calculate_greeks(
                    stock_price, leg.contract.strike_price, time_to_expiry,
                    self.risk_free_rate, volatility, leg.contract.option_type
                )
                
                leg_gamma = greeks['gamma'] * leg.quantity
                if leg.action == 'sell':
                    leg_gamma *= -1
                
                total_gamma += leg_gamma
        
        return total_gamma
    
    def _calculate_portfolio_theta(self, strategy: ComplexStrategy, stock_price: float, volatility: float) -> float:
        """Calculate portfolio theta (time decay)."""
        total_theta = 0.0
        
        for leg in strategy.legs:
            expiry_date = datetime.fromisoformat(leg.contract.expiry_date)
            trade_date = datetime.fromisoformat(strategy.trade_date)
            time_to_expiry = (expiry_date - trade_date).days / 365.25
            
            if time_to_expiry > 0:
                greeks = self.bs_calculator.calculate_greeks(
                    stock_price, leg.contract.strike_price, time_to_expiry,
                    self.risk_free_rate, volatility, leg.contract.option_type
                )
                
                leg_theta = greeks['theta'] * leg.quantity
                if leg.action == 'sell':
                    leg_theta *= -1
                
                total_theta += leg_theta
        
        return total_theta
    
    def _calculate_portfolio_vega(self, strategy: ComplexStrategy, stock_price: float, volatility: float) -> float:
        """Calculate portfolio vega (volatility sensitivity)."""
        total_vega = 0.0
        
        for leg in strategy.legs:
            expiry_date = datetime.fromisoformat(leg.contract.expiry_date)
            trade_date = datetime.fromisoformat(strategy.trade_date)
            time_to_expiry = (expiry_date - trade_date).days / 365.25
            
            if time_to_expiry > 0:
                greeks = self.bs_calculator.calculate_greeks(
                    stock_price, leg.contract.strike_price, time_to_expiry,
                    self.risk_free_rate, volatility, leg.contract.option_type
                )
                
                leg_vega = greeks['vega'] * leg.quantity
                if leg.action == 'sell':
                    leg_vega *= -1
                
                total_vega += leg_vega
        
        return total_vega
    
    def _calculate_var(self, strategy: ComplexStrategy, stock_price: float, 
                      volatility: float, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        # Simplified VaR calculation using delta-normal method
        portfolio_delta = self._calculate_portfolio_delta(strategy, stock_price, volatility)
        
        # Daily volatility
        daily_vol = volatility / np.sqrt(252)
        
        # Z-score for confidence level
        z_score = norm.ppf(confidence)
        
        # VaR calculation
        var = abs(portfolio_delta * stock_price * daily_vol * z_score)
        
        return var
    
    def _estimate_expected_return(self, strategy: ComplexStrategy, stock_price: float, volatility: float) -> float:
        """Estimate expected return of the strategy."""
        # Simplified expected return based on probability of profit and max profit/loss
        if strategy.max_profit and strategy.max_loss:
            expected_return = (strategy.probability_of_profit * strategy.max_profit + 
                             (1 - strategy.probability_of_profit) * strategy.max_loss)
            return expected_return / abs(strategy.total_cost) if strategy.total_cost != 0 else 0
        
        return 0.0

class OptionsTimingAnalyzer:
    """Analyzes timing of options trades relative to market events."""
    
    def __init__(self):
        """Initialize timing analyzer."""
        self.suspicious_thresholds = {
            'days_to_earnings': 7,
            'iv_rank_threshold': 80,
            'vol_skew_threshold': 0.1
        }
    
    def analyze_timing(self, trade_date: str, symbol: str, 
                      option_data: Dict[str, Any]) -> TimingAnalysis:
        """
        Analyze timing of options trade relative to events.
        
        Args:
            trade_date: Date of the trade
            symbol: Stock symbol
            option_data: Options contract data
            
        Returns:
            Timing analysis results
        """
        logger.info(f"Analyzing timing for {symbol} options trade on {trade_date}")
        
        trade_dt = datetime.fromisoformat(trade_date)
        
        # Days to expiry
        expiry_date = datetime.fromisoformat(option_data.get('expiry_date', trade_date))
        days_to_expiry = (expiry_date - trade_dt).days
        
        # Get market data for timing analysis
        market_data = self._get_market_data(symbol, trade_dt)
        
        # Days to earnings (simulated - would use actual earnings calendar)
        days_to_earnings = self._estimate_days_to_earnings(symbol, trade_dt)
        
        # Days to ex-dividend (simulated)
        days_to_ex_dividend = self._estimate_days_to_ex_dividend(symbol, trade_dt)
        
        # Implied volatility rank
        iv_rank = self._calculate_iv_rank(option_data.get('implied_volatility', 0.2), market_data)
        
        # Historical volatility
        historical_vol = market_data.get('historical_volatility', 0.2)
        
        # Volatility skew
        vol_skew = self._calculate_vol_skew(option_data, market_data)
        
        # Calculate timing score
        timing_score = self._calculate_timing_score(
            days_to_expiry, days_to_earnings, iv_rank, vol_skew
        )
        
        # Identify suspicious indicators
        suspicious_indicators = self._identify_suspicious_indicators(
            days_to_earnings, days_to_ex_dividend, iv_rank, vol_skew
        )
        
        return TimingAnalysis(
            trade_date=trade_date,
            days_to_expiry=days_to_expiry,
            days_to_earnings=days_to_earnings,
            days_to_ex_dividend=days_to_ex_dividend,
            implied_volatility_rank=iv_rank,
            historical_volatility=historical_vol,
            volatility_skew=vol_skew,
            timing_score=timing_score,
            suspicious_indicators=suspicious_indicators
        )
    
    def _get_market_data(self, symbol: str, trade_date: datetime) -> Dict[str, Any]:
        """Get market data around trade date (simulated)."""
        # In production, this would fetch actual market data
        return {
            'historical_volatility': np.random.normal(0.25, 0.05),
            'volume': np.random.randint(1000000, 10000000),
            'price_movement': np.random.normal(0, 0.02),
            'iv_rank_30d': np.random.uniform(0, 100)
        }
    
    def _estimate_days_to_earnings(self, symbol: str, trade_date: datetime) -> Optional[int]:
        """Estimate days until next earnings announcement."""
        # Simplified: assume earnings are quarterly, estimate next occurrence
        # In production, would use actual earnings calendar
        quarter_start_months = [1, 4, 7, 10]
        current_month = trade_date.month
        
        # Find next quarter
        next_quarter_month = min(m for m in quarter_start_months if m > current_month)
        if not next_quarter_month:
            next_quarter_month = quarter_start_months[0]  # Next year
        
        # Estimate earnings date (typically 2-3 weeks after quarter end)
        earnings_month = next_quarter_month + 3 if next_quarter_month < 10 else 1
        earnings_day = 15  # Mid-month approximation
        
        try:
            if earnings_month <= 12:
                earnings_date = datetime(trade_date.year, earnings_month, earnings_day)
            else:
                earnings_date = datetime(trade_date.year + 1, earnings_month - 12, earnings_day)
            
            return (earnings_date - trade_date).days
        except:
            return None
    
    def _estimate_days_to_ex_dividend(self, symbol: str, trade_date: datetime) -> Optional[int]:
        """Estimate days until ex-dividend date."""
        # Simplified: assume quarterly dividends
        # In production, would use actual dividend calendar
        quarterly_months = [3, 6, 9, 12]
        current_month = trade_date.month
        
        next_div_month = min((m for m in quarterly_months if m > current_month), default=quarterly_months[0])
        
        try:
            if next_div_month > current_month:
                div_date = datetime(trade_date.year, next_div_month, 15)
            else:
                div_date = datetime(trade_date.year + 1, next_div_month, 15)
            
            return (div_date - trade_date).days
        except:
            return None
    
    def _calculate_iv_rank(self, current_iv: float, market_data: Dict[str, Any]) -> float:
        """Calculate implied volatility rank."""
        # Simplified IV rank calculation
        # In production, would use 52-week IV history
        iv_30d_rank = market_data.get('iv_rank_30d', 50)
        return iv_30d_rank
    
    def _calculate_vol_skew(self, option_data: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate volatility skew."""
        # Simplified skew calculation
        # In production, would compare IVs across different strikes
        return np.random.uniform(-0.1, 0.1)  # Placeholder
    
    def _calculate_timing_score(self, days_to_expiry: int, days_to_earnings: Optional[int], 
                               iv_rank: float, vol_skew: float) -> float:
        """Calculate overall timing suspicion score."""
        score = 0.0
        
        # Short-term expiry before earnings
        if days_to_earnings and days_to_expiry < days_to_earnings and days_to_earnings <= 7:
            score += 0.4
        
        # High IV rank (expensive options)
        if iv_rank > 80:
            score += 0.3
        
        # Significant volatility skew
        if abs(vol_skew) > 0.1:
            score += 0.2
        
        # Very short expiry (weekly options)
        if days_to_expiry <= 7:
            score += 0.1
        
        return min(1.0, score)
    
    def _identify_suspicious_indicators(self, days_to_earnings: Optional[int], 
                                      days_to_ex_dividend: Optional[int],
                                      iv_rank: float, vol_skew: float) -> List[str]:
        """Identify suspicious timing indicators."""
        indicators = []
        
        if days_to_earnings and days_to_earnings <= self.suspicious_thresholds['days_to_earnings']:
            indicators.append(f"Trade within {days_to_earnings} days of earnings")
        
        if days_to_ex_dividend and days_to_ex_dividend <= 5:
            indicators.append(f"Trade within {days_to_ex_dividend} days of ex-dividend")
        
        if iv_rank > self.suspicious_thresholds['iv_rank_threshold']:
            indicators.append(f"High implied volatility rank ({iv_rank:.1f}%)")
        
        if abs(vol_skew) > self.suspicious_thresholds['vol_skew_threshold']:
            indicators.append(f"Significant volatility skew ({vol_skew:.3f})")
        
        return indicators

class OptionsAnalyzer:
    """Main class for comprehensive options and derivatives analysis."""
    
    def __init__(self):
        """Initialize options analyzer."""
        self.strategy_identifier = OptionsStrategyIdentifier()
        self.risk_analyzer = OptionsRiskAnalyzer()
        self.timing_analyzer = OptionsTimingAnalyzer()
        self.bs_calculator = BlackScholesCalculator()
    
    def analyze_options_trades(self, trades_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform comprehensive analysis of options trades.
        
        Args:
            trades_data: List of trading records
            
        Returns:
            List of analysis results
        """
        logger.info(f"Analyzing {len(trades_data)} options trades")
        
        results = []
        
        # Group trades by member and time period
        grouped_trades = self._group_trades_by_member(trades_data)
        
        for member_id, member_trades in grouped_trades.items():
            try:
                # Identify strategies
                strategy_type = self.strategy_identifier.identify_strategy(member_trades)
                
                if strategy_type != StrategyType.UNKNOWN:
                    # Create complex strategy object
                    strategy = self._create_strategy_object(member_id, member_trades, strategy_type)
                    
                    # Risk analysis
                    market_data = self._get_market_data_for_strategy(strategy)
                    risk_metrics = self.risk_analyzer.calculate_strategy_risk_metrics(strategy, market_data)
                    
                    # Timing analysis for each leg
                    timing_analyses = []
                    for leg in strategy.legs:
                        timing = self.timing_analyzer.analyze_timing(
                            strategy.trade_date, 
                            strategy.legs[0].contract.underlying_symbol,  # Use first leg's symbol
                            asdict(leg.contract)
                        )
                        timing_analyses.append(timing)
                    
                    result = {
                        'member_id': member_id,
                        'strategy': asdict(strategy),
                        'risk_metrics': asdict(risk_metrics),
                        'timing_analyses': [asdict(t) for t in timing_analyses],
                        'overall_suspicion_score': self._calculate_overall_suspicion_score(
                            strategy, risk_metrics, timing_analyses
                        )
                    }
                    
                    results.append(result)
                    
            except Exception as e:
                logger.error(f"Error analyzing trades for member {member_id}: {e}")
        
        logger.info(f"Completed analysis for {len(results)} strategies")
        return results
    
    def _group_trades_by_member(self, trades_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group trades by member ID."""
        grouped = {}
        
        for trade in trades_data:
            member_id = trade.get('bioguide_id', 'UNKNOWN')
            
            if member_id not in grouped:
                grouped[member_id] = []
            grouped[member_id].append(trade)
        
        return grouped
    
    def _create_strategy_object(self, member_id: str, trades: List[Dict], strategy_type: StrategyType) -> ComplexStrategy:
        """Create a complex strategy object from trades."""
        # Create strategy legs
        legs = []
        total_cost = 0.0
        
        for i, trade in enumerate(trades):
            # Create options contract (simplified)
            contract = OptionsContract(
                underlying_symbol=trade.get('symbol', 'UNKNOWN'),
                expiry_date=trade.get('expiry_date', (datetime.now() + timedelta(days=30)).isoformat()),
                strike_price=trade.get('strike_price', 100.0),
                option_type=OptionType.CALL if 'call' in trade.get('asset_name', '').lower() else OptionType.PUT,
                premium=trade.get('premium', trade.get('amount_mid', 0) / 100),
                volume=trade.get('volume', 100),
                open_interest=trade.get('open_interest', 1000),
                implied_volatility=0.25  # Default 25%
            )
            
            # Create trade leg
            leg = TradeLeg(
                contract=contract,
                quantity=trade.get('quantity', 1),
                action='buy' if trade.get('transaction_type') == 'Purchase' else 'sell',
                cost=trade.get('amount_mid', 0)
            )
            
            legs.append(leg)
            total_cost += leg.cost
        
        # Calculate strategy metrics (simplified)
        max_profit = self._estimate_max_profit(strategy_type, legs)
        max_loss = self._estimate_max_loss(strategy_type, legs, total_cost)
        break_even_points = self._calculate_break_even_points(strategy_type, legs)
        
        return ComplexStrategy(
            strategy_id=f"{member_id}_{strategy_type.value}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            strategy_type=strategy_type,
            member_id=member_id,
            trade_date=trades[0].get('transaction_date', datetime.now().isoformat()),
            legs=legs,
            total_cost=total_cost,
            max_profit=max_profit,
            max_loss=max_loss,
            break_even_points=break_even_points,
            risk_level=self._assess_risk_level(strategy_type, total_cost),
            market_outlook=self._determine_market_outlook(strategy_type),
            probability_of_profit=self._estimate_probability_of_profit(strategy_type)
        )
    
    def _estimate_max_profit(self, strategy_type: StrategyType, legs: List[TradeLeg]) -> Optional[float]:
        """Estimate maximum profit for strategy."""
        if strategy_type == StrategyType.LONG_CALL:
            return None  # Unlimited
        elif strategy_type == StrategyType.COVERED_CALL:
            if len(legs) >= 2:
                return (legs[1].contract.strike_price - legs[0].cost/100 + legs[1].contract.premium) * 100
        elif strategy_type == StrategyType.BULL_CALL_SPREAD:
            if len(legs) >= 2:
                return (legs[1].contract.strike_price - legs[0].contract.strike_price) * 100 - abs(sum(leg.cost for leg in legs))
        
        return 1000.0  # Default estimate
    
    def _estimate_max_loss(self, strategy_type: StrategyType, legs: List[TradeLeg], total_cost: float) -> Optional[float]:
        """Estimate maximum loss for strategy."""
        if strategy_type in [StrategyType.LONG_CALL, StrategyType.LONG_PUT]:
            return abs(total_cost)  # Limited to premium paid
        elif strategy_type == StrategyType.SHORT_CALL:
            return None  # Unlimited
        elif strategy_type == StrategyType.COVERED_CALL:
            return abs(total_cost)  # Stock can go to zero
        
        return abs(total_cost)  # Conservative estimate
    
    def _calculate_break_even_points(self, strategy_type: StrategyType, legs: List[TradeLeg]) -> List[float]:
        """Calculate break-even points."""
        # Simplified break-even calculation
        if strategy_type == StrategyType.LONG_CALL and legs:
            return [legs[0].contract.strike_price + legs[0].contract.premium]
        elif strategy_type == StrategyType.LONG_PUT and legs:
            return [legs[0].contract.strike_price - legs[0].contract.premium]
        
        return [100.0]  # Default
    
    def _assess_risk_level(self, strategy_type: StrategyType, total_cost: float) -> RiskLevel:
        """Assess risk level of strategy."""
        high_risk_strategies = [StrategyType.SHORT_CALL, StrategyType.SHORT_PUT, StrategyType.STRADDLE]
        
        if strategy_type in high_risk_strategies or abs(total_cost) > 50000:
            return RiskLevel.VERY_HIGH
        elif abs(total_cost) > 25000:
            return RiskLevel.HIGH
        elif abs(total_cost) > 10000:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _determine_market_outlook(self, strategy_type: StrategyType) -> str:
        """Determine market outlook implied by strategy."""
        outlook_map = {
            StrategyType.LONG_CALL: 'bullish',
            StrategyType.LONG_PUT: 'bearish',
            StrategyType.COVERED_CALL: 'neutral_bullish',
            StrategyType.PROTECTIVE_PUT: 'bullish',
            StrategyType.BULL_CALL_SPREAD: 'bullish',
            StrategyType.BEAR_PUT_SPREAD: 'bearish',
            StrategyType.IRON_CONDOR: 'neutral',
            StrategyType.STRADDLE: 'volatile'
        }
        
        return outlook_map.get(strategy_type, 'neutral')
    
    def _estimate_probability_of_profit(self, strategy_type: StrategyType) -> float:
        """Estimate probability of profit."""
        # Simplified probability estimates
        prob_map = {
            StrategyType.LONG_CALL: 0.4,
            StrategyType.LONG_PUT: 0.4,
            StrategyType.COVERED_CALL: 0.65,
            StrategyType.PROTECTIVE_PUT: 0.7,
            StrategyType.BULL_CALL_SPREAD: 0.45,
            StrategyType.IRON_CONDOR: 0.6,
            StrategyType.STRADDLE: 0.35
        }
        
        return prob_map.get(strategy_type, 0.5)
    
    def _get_market_data_for_strategy(self, strategy: ComplexStrategy) -> Dict[str, Any]:
        """Get market data for strategy analysis."""
        # Simplified market data (would fetch real data in production)
        return {
            'current_price': 150.0,
            'volatility': 0.25,
            'volume': 5000000,
            'beta': 1.2
        }
    
    def _calculate_overall_suspicion_score(self, strategy: ComplexStrategy, 
                                         risk_metrics: RiskMetrics,
                                         timing_analyses: List[TimingAnalysis]) -> float:
        """Calculate overall suspicion score for the strategy."""
        score = 0.0
        
        # Risk-based scoring
        if strategy.risk_level == RiskLevel.VERY_HIGH:
            score += 0.3
        elif strategy.risk_level == RiskLevel.HIGH:
            score += 0.2
        
        # Leverage scoring
        if risk_metrics.leverage_ratio > 10:
            score += 0.2
        elif risk_metrics.leverage_ratio > 5:
            score += 0.1
        
        # Timing scoring
        avg_timing_score = np.mean([t.timing_score for t in timing_analyses]) if timing_analyses else 0
        score += avg_timing_score * 0.3
        
        # Suspicious indicators
        total_indicators = sum(len(t.suspicious_indicators) for t in timing_analyses)
        if total_indicators > 2:
            score += 0.2
        
        return min(1.0, score)

def main():
    """Test function for options analyzer."""
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample options trades
    sample_trades = [
        {
            'bioguide_id': 'TEST001',
            'symbol': 'AAPL',
            'transaction_date': '2025-01-15',
            'transaction_type': 'Purchase',
            'asset_name': 'Apple Inc. Call Option',
            'asset_type': 'Option',
            'amount_mid': 50000,
            'strike_price': 150.0,
            'expiry_date': '2025-02-15',
            'quantity': 10
        },
        {
            'bioguide_id': 'TEST001',
            'symbol': 'AAPL',
            'transaction_date': '2025-01-15',
            'transaction_type': 'Sale',
            'asset_name': 'Apple Inc. Call Option',
            'asset_type': 'Option',
            'amount_mid': 30000,
            'strike_price': 160.0,
            'expiry_date': '2025-02-15',
            'quantity': 10
        }
    ]
    
    analyzer = OptionsAnalyzer()
    
    print("Analyzing sample options trades...")
    results = analyzer.analyze_options_trades(sample_trades)
    
    for result in results:
        print(f"\nMember: {result['member_id']}")
        print(f"Strategy: {result['strategy']['strategy_type']}")
        print(f"Risk Level: {result['strategy']['risk_level']}")
        print(f"Market Outlook: {result['strategy']['market_outlook']}")
        print(f"Max Profit: ${result['strategy']['max_profit']:,.2f}" if result['strategy']['max_profit'] else "Unlimited")
        print(f"Max Loss: ${result['strategy']['max_loss']:,.2f}" if result['strategy']['max_loss'] else "Unlimited")
        print(f"Overall Suspicion Score: {result['overall_suspicion_score']:.3f}")
        
        if result['timing_analyses']:
            print(f"Suspicious Timing Indicators:")
            for timing in result['timing_analyses']:
                for indicator in timing['suspicious_indicators']:
                    print(f"  - {indicator}")

if __name__ == "__main__":
    main()