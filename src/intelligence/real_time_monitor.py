#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Phase 2
Real-Time Alert Monitoring System

This module implements real-time monitoring of congressional trading activity
with immediate alert generation and notification capabilities.
"""

import os
import sys
import logging
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path

# Async and Real-time Processing
import websockets
import redis
from celery import Celery

# Database and Data Processing
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import yaml

# Notification and Communication
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Import our suspicious trading detector
from .suspicious_trading_detector import SuspiciousTradingDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingAlert:
    """Data class for trading alerts."""
    alert_id: str
    member_name: str
    bioguide_id: str
    symbol: str
    transaction_type: str
    amount: float
    transaction_date: str
    filing_date: str
    suspicion_score: float
    risk_category: str
    alert_reasons: List[str]
    generated_at: str
    priority: str

@dataclass
class MonitoringConfig:
    """Configuration for real-time monitoring."""
    check_interval: int = 300  # 5 minutes
    alert_threshold: float = 7.0
    email_alerts: bool = False
    slack_alerts: bool = False
    websocket_alerts: bool = True
    redis_alerts: bool = True

class RealTimeMonitor:
    """
    Real-time monitoring system for congressional trading intelligence.
    Continuously monitors for new trades and generates instant alerts.
    """
    
    def __init__(self, config_path: str = "config/database.yml"):
        """Initialize the real-time monitor."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.db_config = self.config.get('development', {}).get('database', {})
        
        # Initialize monitoring config
        self.monitor_config = MonitoringConfig()
        
        # Initialize suspicious trading detector
        self.detector = SuspiciousTradingDetector(config_path)
        
        # Initialize Redis for real-time messaging
        self.redis_client = None
        self._init_redis()
        
        # Initialize Celery for background tasks
        self.celery_app = None
        self._init_celery()
        
        # WebSocket connections for real-time updates
        self.websocket_clients = set()
        
        # Alert history
        self.alert_history = []
        self.last_check_time = datetime.now() - timedelta(hours=1)
        
        # Notification handlers
        self.notification_handlers = {
            'email': self._send_email_alert,
            'slack': self._send_slack_alert,
            'websocket': self._send_websocket_alert,
            'redis': self._send_redis_alert
        }
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_celery(self):
        """Initialize Celery for background tasks."""
        try:
            self.celery_app = Celery(
                'congressional_monitor',
                broker='redis://localhost:6379/0',
                backend='redis://localhost:6379/0'
            )
            logger.info("Celery initialized")
        except Exception as e:
            logger.warning(f"Celery initialization failed: {e}")
            self.celery_app = None
    
    def _get_db_connection(self):
        """Get database connection."""
        return psycopg2.connect(
            host=self.db_config.get('host', 'localhost'),
            port=self.db_config.get('port', 5432),
            database=self.db_config.get('name', 'congressional_trading_dev'),
            user=self.db_config.get('user', 'postgres'),
            password=self.db_config.get('password', 'password')
        )
    
    def check_for_new_trades(self) -> List[Dict]:
        """
        Check for new trades since last monitoring cycle.
        
        Returns:
            List of new trade records
        """
        logger.info(f"Checking for new trades since {self.last_check_time}")
        
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Query for new trades
            query = """
            SELECT 
                t.id,
                t.bioguide_id,
                m.full_name,
                m.party,
                m.state,
                m.chamber,
                t.symbol,
                t.transaction_type,
                t.transaction_date,
                t.filing_date,
                t.amount_min,
                t.amount_max,
                t.amount_mid,
                t.created_at,
                t.filing_delay_days
            FROM trades t
            JOIN members m ON t.bioguide_id = m.bioguide_id
            WHERE t.created_at > %s
            ORDER BY t.created_at DESC
            """
            
            cursor.execute(query, (self.last_check_time,))
            new_trades = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            logger.info(f"Found {len(new_trades)} new trades")
            return [dict(trade) for trade in new_trades]
            
        except Exception as e:
            logger.error(f"Error checking for new trades: {e}")
            return []
    
    def analyze_new_trades(self, trades: List[Dict]) -> List[TradingAlert]:
        """
        Analyze new trades for suspicious patterns.
        
        Args:
            trades: List of new trade records
            
        Returns:
            List of generated alerts
        """
        if not trades:
            return []
        
        logger.info(f"Analyzing {len(trades)} new trades for suspicious patterns")
        
        alerts = []
        
        for trade in trades:
            try:
                # Quick suspicion analysis for real-time processing
                suspicion_score = self._calculate_quick_suspicion_score(trade)
                
                if suspicion_score >= self.monitor_config.alert_threshold:
                    alert = self._create_trading_alert(trade, suspicion_score)
                    alerts.append(alert)
                    
            except Exception as e:
                logger.error(f"Error analyzing trade {trade.get('id', 'unknown')}: {e}")
        
        logger.info(f"Generated {len(alerts)} alerts from new trades")
        return alerts
    
    def _calculate_quick_suspicion_score(self, trade: Dict) -> float:
        """
        Calculate a quick suspicion score for real-time processing.
        Uses simplified heuristics for speed.
        
        Args:
            trade: Trade record dictionary
            
        Returns:
            Suspicion score (0-10)
        """
        score = 0.0
        
        # Amount-based scoring
        amount = trade.get('amount_mid', 0)
        if amount > 1000000:  # > $1M
            score += 3.0
        elif amount > 500000:  # > $500K
            score += 2.0
        elif amount > 100000:  # > $100K
            score += 1.0
        
        # Filing delay scoring
        filing_delay = trade.get('filing_delay_days', 0)
        if filing_delay > 45:  # Late filing
            score += 4.0
        elif filing_delay > 30:
            score += 3.0
        elif filing_delay > 14:
            score += 2.0
        
        # Recent trade frequency (simplified)
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            # Count recent trades by same member
            cursor.execute("""
                SELECT COUNT(*) FROM trades 
                WHERE bioguide_id = %s 
                AND transaction_date >= %s
            """, (trade['bioguide_id'], datetime.now() - timedelta(days=30)))
            
            recent_trades = cursor.fetchone()[0]
            
            if recent_trades > 10:
                score += 2.0
            elif recent_trades > 5:
                score += 1.0
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not calculate trade frequency: {e}")
        
        # Committee-related scoring (simplified)
        member_name = trade.get('full_name', '').lower()
        if any(keyword in member_name for keyword in ['chair', 'leader', 'speaker']):
            score += 1.0
        
        return min(score, 10.0)  # Cap at 10
    
    def _create_trading_alert(self, trade: Dict, suspicion_score: float) -> TradingAlert:
        """Create a TradingAlert from trade data."""
        # Determine risk category
        if suspicion_score >= 9.0:
            risk_category = "EXTREME"
            priority = "CRITICAL"
        elif suspicion_score >= 7.0:
            risk_category = "HIGH"
            priority = "HIGH"
        elif suspicion_score >= 5.0:
            risk_category = "MEDIUM"
            priority = "MEDIUM"
        else:
            risk_category = "LOW"
            priority = "LOW"
        
        # Generate alert reasons
        alert_reasons = []
        amount = trade.get('amount_mid', 0)
        filing_delay = trade.get('filing_delay_days', 0)
        
        if amount > 500000:
            alert_reasons.append(f"Large trade amount: ${amount:,.0f}")
        
        if filing_delay > 30:
            alert_reasons.append(f"Late filing: {filing_delay} days")
        
        if suspicion_score >= 8.0:
            alert_reasons.append("Multiple risk factors detected")
        
        # Create alert
        alert = TradingAlert(
            alert_id=f"ALERT_{trade['id']}_{int(time.time())}",
            member_name=trade['full_name'],
            bioguide_id=trade['bioguide_id'],
            symbol=trade['symbol'],
            transaction_type=trade['transaction_type'],
            amount=trade.get('amount_mid', 0),
            transaction_date=str(trade['transaction_date']),
            filing_date=str(trade['filing_date']),
            suspicion_score=suspicion_score,
            risk_category=risk_category,
            alert_reasons=alert_reasons,
            generated_at=datetime.now().isoformat(),
            priority=priority
        )
        
        return alert
    
    async def process_alerts(self, alerts: List[TradingAlert]):
        """Process and send alerts through all configured channels."""
        if not alerts:
            return
        
        logger.info(f"Processing {len(alerts)} alerts")
        
        for alert in alerts:
            # Add to history
            self.alert_history.append(alert)
            
            # Send through all configured notification channels
            if self.monitor_config.email_alerts:
                await self._send_email_alert(alert)
            
            if self.monitor_config.slack_alerts:
                await self._send_slack_alert(alert)
            
            if self.monitor_config.websocket_alerts:
                await self._send_websocket_alert(alert)
            
            if self.monitor_config.redis_alerts and self.redis_client:
                await self._send_redis_alert(alert)
            
            logger.info(f"Alert processed: {alert.member_name} - {alert.symbol} "
                       f"({alert.priority}: {alert.suspicion_score:.1f})")
    
    async def _send_email_alert(self, alert: TradingAlert):
        """Send alert via email."""
        # Email implementation would go here
        logger.info(f"Email alert sent for {alert.alert_id}")
    
    async def _send_slack_alert(self, alert: TradingAlert):
        """Send alert to Slack."""
        # Slack implementation would go here
        logger.info(f"Slack alert sent for {alert.alert_id}")
    
    async def _send_websocket_alert(self, alert: TradingAlert):
        """Send alert via WebSocket to connected clients."""
        if not self.websocket_clients:
            return
        
        alert_data = json.dumps(asdict(alert), default=str)
        
        # Send to all connected WebSocket clients
        disconnected_clients = set()
        for client in self.websocket_clients:
            try:
                await client.send(alert_data)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
        
        logger.info(f"WebSocket alert sent to {len(self.websocket_clients)} clients")
    
    async def _send_redis_alert(self, alert: TradingAlert):
        """Send alert to Redis pub/sub channel."""
        if not self.redis_client:
            return
        
        try:
            alert_data = json.dumps(asdict(alert), default=str)
            self.redis_client.publish('congressional_alerts', alert_data)
            logger.info(f"Redis alert published for {alert.alert_id}")
        except Exception as e:
            logger.error(f"Failed to send Redis alert: {e}")
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time alerts."""
        logger.info(f"New WebSocket connection from {websocket.remote_address}")
        self.websocket_clients.add(websocket)
        
        try:
            # Send recent alerts to new client
            recent_alerts = self.alert_history[-10:]  # Last 10 alerts
            for alert in recent_alerts:
                alert_data = json.dumps(asdict(alert), default=str)
                await websocket.send(alert_data)
            
            # Keep connection alive
            await websocket.wait_closed()
        finally:
            self.websocket_clients.discard(websocket)
            logger.info(f"WebSocket connection closed")
    
    async def monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting real-time monitoring loop...")
        
        while True:
            try:
                start_time = time.time()
                
                # Check for new trades
                new_trades = self.check_for_new_trades()
                
                if new_trades:
                    # Analyze for suspicious patterns
                    alerts = self.analyze_new_trades(new_trades)
                    
                    # Process alerts
                    if alerts:
                        await self.process_alerts(alerts)
                
                # Update last check time
                self.last_check_time = datetime.now()
                
                # Calculate sleep time to maintain check interval
                elapsed_time = time.time() - start_time
                sleep_time = max(0, self.monitor_config.check_interval - elapsed_time)
                
                logger.info(f"Monitoring cycle completed in {elapsed_time:.2f}s, "
                           f"sleeping for {sleep_time:.2f}s")
                
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """Get summary of alerts from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert.generated_at) > cutoff_time
        ]
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'by_priority': {},
                'by_member': {},
                'avg_suspicion_score': 0.0
            }
        
        # Aggregate statistics
        priority_counts = {}
        member_counts = {}
        total_score = 0.0
        
        for alert in recent_alerts:
            # Priority counts
            priority_counts[alert.priority] = priority_counts.get(alert.priority, 0) + 1
            
            # Member counts
            member_counts[alert.member_name] = member_counts.get(alert.member_name, 0) + 1
            
            # Total score
            total_score += alert.suspicion_score
        
        return {
            'total_alerts': len(recent_alerts),
            'by_priority': priority_counts,
            'by_member': dict(sorted(member_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'avg_suspicion_score': total_score / len(recent_alerts),
            'time_range_hours': hours
        }
    
    async def start_monitoring(self):
        """Start the complete monitoring system."""
        logger.info("Starting Congressional Trading Real-Time Monitor...")
        
        # Start WebSocket server for real-time updates
        websocket_server = websockets.serve(
            self.websocket_handler,
            "localhost",
            8765
        )
        
        # Start monitoring loop
        monitoring_task = asyncio.create_task(self.monitoring_loop())
        
        # Run both concurrently
        await asyncio.gather(
            websocket_server,
            monitoring_task
        )

def main():
    """Main execution function."""
    logger.info("Initializing Congressional Trading Real-Time Monitor...")
    
    monitor = RealTimeMonitor()
    
    # Configure monitoring settings
    monitor.monitor_config.check_interval = 300  # 5 minutes
    monitor.monitor_config.alert_threshold = 6.0  # Lower threshold for demo
    monitor.monitor_config.websocket_alerts = True
    monitor.monitor_config.redis_alerts = True
    
    # Start monitoring
    try:
        asyncio.run(monitor.start_monitoring())
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")

if __name__ == "__main__":
    main()