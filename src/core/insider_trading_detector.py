"""
Congressional Insider Trading Real-Time Detection System
Core detection algorithms and alert generation system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass
from enum import Enum
try:
    import smtplib
    from email.mime.text import MIMEText as MimeText
    from email.mime.multipart import MIMEMultipart as MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    logger.warning("Email functionality not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

class TradeType(Enum):
    PURCHASE = "Purchase"
    SALE = "Sale"
    OPTION = "Option"

@dataclass
class CongressionalTrade:
    """Data structure for congressional trade information"""
    member_name: str
    stock_symbol: str
    trade_date: datetime
    filing_date: datetime
    trade_type: TradeType
    amount_min: float
    amount_max: float
    owner_type: str  # Self, Spouse, Child, Trust
    committee_assignments: List[str]
    leadership_position: Optional[str]
    sector: str
    
    @property
    def avg_amount(self) -> float:
        return (self.amount_min + self.amount_max) / 2
    
    @property
    def filing_delay_days(self) -> int:
        return (self.filing_date - self.trade_date).days

@dataclass
class LegislativeEvent:
    """Data structure for legislative events"""
    event_date: datetime
    event_type: str  # Bill Introduction, Committee Hearing, Vote, etc.
    title: str
    description: str
    affected_sectors: List[str]
    affected_stocks: List[str]
    committees_involved: List[str]
    significance_score: int  # 1-10, higher = more market impact

@dataclass
class SuspicionAlert:
    """Data structure for generated alerts"""
    trade: CongressionalTrade
    suspicion_score: float
    alert_level: AlertLevel
    risk_factors: List[str]
    related_legislation: List[LegislativeEvent]
    financial_impact: float
    generated_at: datetime

class InsiderTradingDetector:
    """
    Core detection engine for congressional insider trading
    """
    
    def __init__(self):
        self.committee_access_scores = {
            # Leadership positions (highest access)
            "House Speaker": 10,
            "Senate Majority Leader": 10,
            "Senate Minority Leader": 9,
            "House Majority Leader": 9,
            "House Minority Leader": 8,
            
            # Committee chairs (high access to specific sectors)
            "Finance Committee Chair": 9,
            "Banking Committee Chair": 8,
            "Energy Committee Chair": 8,
            "Judiciary Committee Chair": 7,
            "Intelligence Committee Chair": 10,
            "Appropriations Committee Chair": 8,
            
            # Regular committee members
            "Finance Committee": 6,
            "Banking Committee": 5,
            "Energy Committee": 5,
            "Technology Committee": 6,
            "Healthcare Committee": 5,
            "Defense Committee": 6,
        }
        
        self.sector_committee_mapping = {
            "Technology": ["Technology Committee", "Judiciary Committee"],
            "Financial": ["Banking Committee", "Finance Committee"],
            "Energy": ["Energy Committee", "Finance Committee"],
            "Healthcare": ["Healthcare Committee", "Finance Committee"],
            "Defense": ["Defense Committee", "Appropriations Committee"],
            "Agriculture": ["Agriculture Committee"],
        }
        
        # Historical performance benchmarks
        self.market_benchmark = 0.249  # S&P 500 2024 return (24.9%)
        self.hedge_fund_benchmark = 0.1275  # Average hedge fund return (12.75%)
        
    def calculate_suspicion_score(self, trade: CongressionalTrade, 
                                 recent_legislation: List[LegislativeEvent],
                                 member_performance: Optional[float] = None) -> Tuple[float, List[str]]:
        """
        Calculate comprehensive suspicion score for a trade
        Returns: (score, list of risk factors)
        """
        score = 0.0
        risk_factors = []
        
        # Factor 1: Trade size relative to typical congressional wealth (0-3 points)
        if trade.avg_amount > 5000000:  # >$5M
            score += 3
            risk_factors.append(f"Extremely large trade (${trade.avg_amount:,.0f})")
        elif trade.avg_amount > 1000000:  # >$1M
            score += 2.5
            risk_factors.append(f"Very large trade (${trade.avg_amount:,.0f})")
        elif trade.avg_amount > 500000:  # >$500K
            score += 2
            risk_factors.append(f"Large trade (${trade.avg_amount:,.0f})")
        elif trade.avg_amount > 100000:  # >$100K
            score += 1
            risk_factors.append(f"Moderate trade (${trade.avg_amount:,.0f})")
        
        # Factor 2: Committee access and position (0-3 points)
        max_access_score = 0
        for assignment in trade.committee_assignments:
            access_score = self.committee_access_scores.get(assignment, 0)
            max_access_score = max(max_access_score, access_score)
        
        if trade.leadership_position:
            leadership_score = self.committee_access_scores.get(trade.leadership_position, 0)
            max_access_score = max(max_access_score, leadership_score)
        
        if max_access_score >= 9:
            score += 3
            risk_factors.append("Extreme legislative access (Leadership/Chair)")
        elif max_access_score >= 7:
            score += 2.5
            risk_factors.append("High legislative access (Committee Chair)")
        elif max_access_score >= 5:
            score += 2
            risk_factors.append("Moderate legislative access (Committee Member)")
        
        # Factor 3: Sector-committee conflict (0-2 points)
        relevant_committees = self.sector_committee_mapping.get(trade.sector, [])
        committee_overlap = set(trade.committee_assignments) & set(relevant_committees)
        if committee_overlap:
            score += 2
            risk_factors.append(f"Trading in oversight sector ({trade.sector})")
        
        # Factor 4: Filing compliance (0-2 points)
        if trade.filing_delay_days > 90:
            score += 2
            risk_factors.append(f"Very late filing ({trade.filing_delay_days} days)")
        elif trade.filing_delay_days > 45:
            score += 1
            risk_factors.append(f"Late filing ({trade.filing_delay_days} days)")
        
        # Factor 5: Use of family/trust accounts (0-1 point)
        if trade.owner_type in ["Spouse", "Child", "Trust"]:
            score += 1
            risk_factors.append(f"Uses {trade.owner_type.lower()} account")
        
        # Factor 6: Legislative timing correlation (0-4 points)
        timing_score, timing_factors = self._analyze_legislative_timing(trade, recent_legislation)
        score += timing_score
        risk_factors.extend(timing_factors)
        
        # Factor 7: Historical performance (0-2 points)
        if member_performance and member_performance > self.market_benchmark * 2:
            score += 2
            risk_factors.append(f"Exceptional performance ({member_performance*100:.1f}% vs {self.market_benchmark*100:.1f}% market)")
        elif member_performance and member_performance > self.market_benchmark * 1.5:
            score += 1
            risk_factors.append(f"High performance ({member_performance*100:.1f}% vs {self.market_benchmark*100:.1f}% market)")
        
        # Cap score at 10
        score = min(score, 10.0)
        
        return score, risk_factors
    
    def _analyze_legislative_timing(self, trade: CongressionalTrade, 
                                  legislation: List[LegislativeEvent]) -> Tuple[float, List[str]]:
        """
        Analyze timing correlation between trade and legislative events
        """
        timing_score = 0.0
        timing_factors = []
        
        for event in legislation:
            # Check if trade is in relevant sector/stock
            if (trade.sector in event.affected_sectors or 
                trade.stock_symbol in event.affected_stocks):
                
                days_difference = (event.event_date - trade.trade_date).days
                
                # Trade before event (most suspicious)
                if 0 < days_difference <= 7:
                    timing_score += 3 * (event.significance_score / 10)
                    timing_factors.append(f"Traded {days_difference} days before {event.title}")
                elif 7 < days_difference <= 30:
                    timing_score += 2 * (event.significance_score / 10)
                    timing_factors.append(f"Traded {days_difference} days before {event.title}")
                elif 30 < days_difference <= 90:
                    timing_score += 1 * (event.significance_score / 10)
                    timing_factors.append(f"Traded {days_difference} days before {event.title}")
                
                # Trade during active legislative period
                elif -7 <= days_difference <= 0:
                    timing_score += 1.5 * (event.significance_score / 10)
                    timing_factors.append(f"Traded during active period for {event.title}")
        
        return min(timing_score, 4.0), timing_factors
    
    def generate_alert(self, trade: CongressionalTrade, 
                      suspicion_score: float, 
                      risk_factors: List[str],
                      related_legislation: List[LegislativeEvent]) -> SuspicionAlert:
        """
        Generate alert based on suspicion score and factors
        """
        # Determine alert level
        if suspicion_score >= 8.5:
            alert_level = AlertLevel.EXTREME
        elif suspicion_score >= 7.0:
            alert_level = AlertLevel.HIGH
        elif suspicion_score >= 5.0:
            alert_level = AlertLevel.MEDIUM
        else:
            alert_level = AlertLevel.LOW
        
        # Estimate financial impact (simplified calculation)
        financial_impact = trade.avg_amount * 0.2  # Assume 20% average gain from insider info
        
        return SuspicionAlert(
            trade=trade,
            suspicion_score=suspicion_score,
            alert_level=alert_level,
            risk_factors=risk_factors,
            related_legislation=related_legislation,
            financial_impact=financial_impact,
            generated_at=datetime.now()
        )

class AlertSystem:
    """
    Alert generation and notification system
    """
    
    def __init__(self, email_config: Optional[Dict] = None):
        self.email_config = email_config or {}
        self.alert_history = []
        
    def process_alert(self, alert: SuspicionAlert) -> bool:
        """
        Process and distribute alert based on level
        """
        try:
            # Log alert
            self._log_alert(alert)
            
            # Store in history
            self.alert_history.append(alert)
            
            # Send notifications based on alert level
            if alert.alert_level in [AlertLevel.HIGH, AlertLevel.EXTREME]:
                self._send_high_priority_notification(alert)
            elif alert.alert_level == AlertLevel.MEDIUM:
                self._send_medium_priority_notification(alert)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
            return False
    
    def _log_alert(self, alert: SuspicionAlert):
        """Log alert details"""
        logger.info(f"ALERT GENERATED - {alert.alert_level.value}")
        logger.info(f"Member: {alert.trade.member_name}")
        logger.info(f"Stock: {alert.trade.stock_symbol}")
        logger.info(f"Amount: ${alert.trade.avg_amount:,.0f}")
        logger.info(f"Suspicion Score: {alert.suspicion_score:.1f}/10")
        logger.info(f"Risk Factors: {', '.join(alert.risk_factors)}")
    
    def _send_high_priority_notification(self, alert: SuspicionAlert):
        """Send high priority alert notifications"""
        subject = f"🚨 EXTREME INSIDER TRADING ALERT - {alert.trade.member_name}"
        message = self._format_alert_message(alert)
        
        if self.email_config:
            self._send_email(subject, message, priority="high")
        
        # Could also send SMS, Slack, etc.
        logger.warning(f"HIGH PRIORITY ALERT: {alert.trade.member_name} - Score: {alert.suspicion_score:.1f}")
    
    def _send_medium_priority_notification(self, alert: SuspicionAlert):
        """Send medium priority alert notifications"""
        subject = f"⚠️ Insider Trading Alert - {alert.trade.member_name}"
        message = self._format_alert_message(alert)
        
        if self.email_config:
            self._send_email(subject, message, priority="medium")
    
    def _format_alert_message(self, alert: SuspicionAlert) -> str:
        """Format alert message for notifications"""
        message = f"""
CONGRESSIONAL INSIDER TRADING ALERT
{'='*50}

MEMBER: {alert.trade.member_name}
STOCK: {alert.trade.stock_symbol}
TRADE DATE: {alert.trade.trade_date.strftime('%Y-%m-%d')}
AMOUNT: ${alert.trade.avg_amount:,.0f}
TRADE TYPE: {alert.trade.trade_type.value}

SUSPICION SCORE: {alert.suspicion_score:.1f}/10 ({alert.alert_level.value} RISK)

RISK FACTORS:
{chr(10).join(f'• {factor}' for factor in alert.risk_factors)}

ESTIMATED FINANCIAL IMPACT: ${alert.financial_impact:,.0f}

RELATED LEGISLATION:
{chr(10).join(f'• {event.title} ({event.event_date.strftime("%Y-%m-%d")})' for event in alert.related_legislation)}

COMMITTEE ACCESS:
{', '.join(alert.trade.committee_assignments)}
{f'Leadership: {alert.trade.leadership_position}' if alert.trade.leadership_position else ''}

Generated at: {alert.generated_at.strftime('%Y-%m-%d %H:%M:%S')}
        """
        return message
    
    def _send_email(self, subject: str, message: str, priority: str = "medium"):
        """Send email notification (requires email configuration)"""
        if not EMAIL_AVAILABLE:
            logger.warning("Email functionality not available, skipping email notification")
            return
            
        if not self.email_config.get('smtp_server'):
            logger.warning("Email configuration not provided, skipping email notification")
            return
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = subject
            
            if priority == "high":
                msg['X-Priority'] = '1'
            
            msg.attach(MimeText(message, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

class RealTimeMonitor:
    """
    Main monitoring system that coordinates detection and alerting
    """
    
    def __init__(self, email_config: Optional[Dict] = None):
        self.detector = InsiderTradingDetector()
        self.alert_system = AlertSystem(email_config)
        self.monitored_members = set()
        self.recent_legislation = []
        
    def add_monitored_member(self, member_name: str):
        """Add member to monitoring list"""
        self.monitored_members.add(member_name)
        logger.info(f"Added {member_name} to monitoring list")
    
    def update_legislation(self, legislation: List[LegislativeEvent]):
        """Update recent legislation for correlation analysis"""
        self.recent_legislation = legislation
        logger.info(f"Updated legislation database with {len(legislation)} events")
    
    def process_trade(self, trade: CongressionalTrade, 
                     member_performance: Optional[float] = None) -> Optional[SuspicionAlert]:
        """
        Process a new trade and generate alerts if necessary
        """
        # Calculate suspicion score
        score, risk_factors = self.detector.calculate_suspicion_score(
            trade, self.recent_legislation, member_performance
        )
        
        # Generate alert if score is significant
        if score >= 4.0:  # Minimum threshold for alerts
            related_legislation = [
                event for event in self.recent_legislation
                if (trade.sector in event.affected_sectors or 
                    trade.stock_symbol in event.affected_stocks)
            ]
            
            alert = self.detector.generate_alert(trade, score, risk_factors, related_legislation)
            
            # Process alert
            success = self.alert_system.process_alert(alert)
            
            if success:
                return alert
        
        return None
    
    def get_alert_summary(self, days: int = 30) -> Dict:
        """Get summary of recent alerts"""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_alerts = [
            alert for alert in self.alert_system.alert_history
            if alert.generated_at >= cutoff_date
        ]
        
        return {
            "total_alerts": len(recent_alerts),
            "extreme_alerts": len([a for a in recent_alerts if a.alert_level == AlertLevel.EXTREME]),
            "high_alerts": len([a for a in recent_alerts if a.alert_level == AlertLevel.HIGH]),
            "medium_alerts": len([a for a in recent_alerts if a.alert_level == AlertLevel.MEDIUM]),
            "low_alerts": len([a for a in recent_alerts if a.alert_level == AlertLevel.LOW]),
            "total_financial_impact": sum(a.financial_impact for a in recent_alerts),
            "most_suspicious_member": max(recent_alerts, key=lambda x: x.suspicion_score).trade.member_name if recent_alerts else None
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize monitoring system
    monitor = RealTimeMonitor()
    
    # Add high-risk members to monitoring
    high_risk_members = [
        "Nancy Pelosi", "Ron Wyden", "Ro Khanna", 
        "Josh Gottheimer", "Debbie Wasserman Schultz"
    ]
    
    for member in high_risk_members:
        monitor.add_monitored_member(member)
    
    # Example legislation events
    sample_legislation = [
        LegislativeEvent(
            event_date=datetime(2025, 7, 15),
            event_type="Committee Hearing",
            title="AI Regulation Hearing",
            description="House Oversight Committee hearing on AI safety regulations",
            affected_sectors=["Technology"],
            affected_stocks=["NVDA", "GOOGL", "MSFT"],
            committees_involved=["House Oversight"],
            significance_score=8
        ),
        LegislativeEvent(
            event_date=datetime(2025, 8, 1),
            event_type="Bill Introduction",
            title="CHIPS Act Extension",
            description="Extension of semiconductor manufacturing incentives",
            affected_sectors=["Technology"],
            affected_stocks=["NVDA", "AMD", "INTC"],
            committees_involved=["Finance Committee"],
            significance_score=9
        )
    ]
    
    monitor.update_legislation(sample_legislation)
    
    # Example suspicious trade
    suspicious_trade = CongressionalTrade(
        member_name="Nancy Pelosi",
        stock_symbol="NVDA",
        trade_date=datetime(2025, 7, 10),  # 5 days before AI hearing
        filing_date=datetime(2025, 8, 25),  # Late filing
        trade_type=TradeType.PURCHASE,
        amount_min=1000000,
        amount_max=5000000,
        owner_type="Spouse",
        committee_assignments=["House Leadership"],
        leadership_position="House Speaker",
        sector="Technology"
    )
    
    # Process the trade
    alert = monitor.process_trade(suspicious_trade, member_performance=0.65)  # 65% return
    
    if alert:
        print(f"\n🚨 ALERT GENERATED!")
        print(f"Member: {alert.trade.member_name}")
        print(f"Score: {alert.suspicion_score:.1f}/10")
        print(f"Level: {alert.alert_level.value}")
        print(f"Factors: {', '.join(alert.risk_factors)}")
    
    # Get monitoring summary
    summary = monitor.get_alert_summary()
    print(f"\nMONITORING SUMMARY:")
    print(f"Total alerts: {summary['total_alerts']}")
    print(f"Extreme alerts: {summary['extreme_alerts']}")
    print(f"High alerts: {summary['high_alerts']}")
    print(f"Estimated financial impact: ${summary['total_financial_impact']:,.0f}")

