"""
APEX Real-Time Streaming Intelligence Engine
Microsecond-latency congressional trading signal processing
"""

import asyncio
import json
import redis
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import websockets
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import aiohttp
import time

@dataclass
class TradingSignal:
    """Real-time trading signal structure"""
    symbol: str
    member_id: int
    member_name: str
    signal_type: str  # 'buy', 'sell', 'watch'
    confidence: float
    magnitude: float
    timestamp: datetime
    source: str
    committee_relevance: float
    risk_score: float
    
    def to_dict(self):
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat()
        }

class RealTimeStreamingEngine:
    """
    Ultra-fast streaming engine for congressional trading intelligence
    Target latency: <100ms from data to prediction
    """
    
    def __init__(self, redis_host='localhost', redis_port=6379, kafka_bootstrap_servers=['localhost:9092']):
        # Redis for ultra-fast caching
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1
        )
        
        # Kafka for event streaming
        self.kafka_producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            batch_size=16384,
            linger_ms=10,
            compression_type='snappy'
        )
        
        self.kafka_consumer = KafkaConsumer(
            'congressional-trades',
            'market-data',
            'news-events',
            'committee-activities',
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            enable_auto_commit=True,
            auto_offset_reset='latest'
        )
        
        # WebSocket connections for real-time dashboard updates
        self.websocket_clients = set()
        
        # Performance monitoring
        self.processing_times = []
        self.signal_count = 0
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def start_streaming(self):
        """Start the real-time streaming engine"""
        self.logger.info("🚀 Starting APEX Real-Time Streaming Engine")
        
        # Start parallel tasks
        tasks = [
            asyncio.create_task(self.process_kafka_stream()),
            asyncio.create_task(self.start_websocket_server()),
            asyncio.create_task(self.monitor_performance()),
            asyncio.create_task(self.collect_market_data()),
            asyncio.create_task(self.monitor_committee_activities())
        ]
        
        await asyncio.gather(*tasks)
    
    async def process_kafka_stream(self):
        """Process incoming Kafka messages in real-time"""
        for message in self.kafka_consumer:
            start_time = time.time()
            
            try:
                # Process different message types
                if message.topic == 'congressional-trades':
                    await self.process_congressional_trade(message.value)
                elif message.topic == 'market-data':
                    await self.process_market_data(message.value)
                elif message.topic == 'news-events':
                    await self.process_news_event(message.value)
                elif message.topic == 'committee-activities':
                    await self.process_committee_activity(message.value)
                
                # Track processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                self.processing_times.append(processing_time)
                
                # Keep only last 1000 measurements
                if len(self.processing_times) > 1000:
                    self.processing_times = self.processing_times[-1000:]
                
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def process_congressional_trade(self, trade_data):
        """Process congressional trading disclosure in real-time"""
        # Cache the trade data
        cache_key = f"trade:{trade_data['member_id']}:{trade_data['symbol']}:{trade_data['timestamp']}"
        await self.redis_set(cache_key, json.dumps(trade_data), ex=3600)
        
        # Generate immediate trading signal
        signal = await self.generate_trading_signal(trade_data)
        
        if signal:
            # Broadcast to all clients
            await self.broadcast_signal(signal)
            
            # Store in high-speed cache
            await self.cache_signal(signal)
            
            # Trigger any automated trading actions
            await self.handle_automated_trading(signal)
    
    async def generate_trading_signal(self, trade_data) -> Optional[TradingSignal]:
        """Generate trading signal from congressional trade data"""
        # Load member profile from cache
        member_profile = await self.get_cached_member_profile(trade_data['member_id'])
        
        if not member_profile:
            return None
        
        # Calculate signal confidence based on historical performance
        confidence = self.calculate_signal_confidence(member_profile, trade_data)
        
        # Determine signal type and magnitude
        signal_type = self.determine_signal_type(trade_data, member_profile)
        magnitude = self.calculate_magnitude(trade_data, member_profile)
        
        # Check committee relevance
        committee_relevance = await self.get_committee_relevance(
            trade_data['symbol'], 
            member_profile['committees']
        )
        
        # Calculate risk score
        risk_score = self.calculate_risk_score(trade_data, member_profile)
        
        return TradingSignal(
            symbol=trade_data['symbol'],
            member_id=trade_data['member_id'],
            member_name=member_profile['name'],
            signal_type=signal_type,
            confidence=confidence,
            magnitude=magnitude,
            timestamp=datetime.now(),
            source='congressional_trade',
            committee_relevance=committee_relevance,
            risk_score=risk_score
        )
    
    def calculate_signal_confidence(self, member_profile, trade_data) -> float:
        """Calculate confidence score for the trading signal"""
        # Base confidence from historical accuracy
        base_confidence = member_profile.get('historical_accuracy', 0.5)
        
        # Adjust for position size
        position_modifier = min(trade_data.get('amount', 0) / 1000000, 1.0) * 0.2
        
        # Adjust for committee relevance
        committee_modifier = 0.0
        if trade_data['symbol'] in member_profile.get('sector_expertise', []):
            committee_modifier = 0.3
        
        # Adjust for timing patterns
        timing_modifier = 0.0
        if self.is_optimal_timing(trade_data, member_profile):
            timing_modifier = 0.2
        
        return min(base_confidence + position_modifier + committee_modifier + timing_modifier, 1.0)
    
    def determine_signal_type(self, trade_data, member_profile) -> str:
        """Determine the type of trading signal"""
        if trade_data['transaction_type'].lower() in ['purchase', 'buy']:
            return 'buy'
        elif trade_data['transaction_type'].lower() in ['sale', 'sell']:
            return 'sell'
        else:
            return 'watch'
    
    def calculate_magnitude(self, trade_data, member_profile) -> float:
        """Calculate expected magnitude of price movement"""
        # Base magnitude from historical average
        base_magnitude = member_profile.get('avg_magnitude', 0.05)
        
        # Adjust for position size
        size_modifier = min(trade_data.get('amount', 0) / 500000, 2.0)
        
        # Adjust for member's historical performance
        performance_modifier = member_profile.get('historical_accuracy', 0.5)
        
        return base_magnitude * size_modifier * performance_modifier
    
    async def get_cached_member_profile(self, member_id: int) -> Optional[Dict]:
        """Get member profile from Redis cache"""
        cache_key = f"member_profile:{member_id}"
        cached_data = await self.redis_get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        
        # If not cached, load from database and cache
        profile = await self.load_member_profile_from_db(member_id)
        if profile:
            await self.redis_set(cache_key, json.dumps(profile), ex=7200)  # 2 hour cache
        
        return profile
    
    async def broadcast_signal(self, signal: TradingSignal):
        """Broadcast trading signal to all connected WebSocket clients"""
        if self.websocket_clients:
            message = {
                'type': 'trading_signal',
                'data': signal.to_dict()
            }
            
            # Broadcast to all clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
            
            self.signal_count += 1
    
    async def cache_signal(self, signal: TradingSignal):
        """Cache trading signal for fast retrieval"""
        # Cache by symbol
        symbol_key = f"signals:symbol:{signal.symbol}"
        await self.redis_lpush(symbol_key, json.dumps(signal.to_dict()))
        await self.redis_ltrim(symbol_key, 0, 99)  # Keep last 100 signals
        
        # Cache by member
        member_key = f"signals:member:{signal.member_id}"
        await self.redis_lpush(member_key, json.dumps(signal.to_dict()))
        await self.redis_ltrim(member_key, 0, 99)
        
        # Cache recent signals
        recent_key = "signals:recent"
        await self.redis_lpush(recent_key, json.dumps(signal.to_dict()))
        await self.redis_ltrim(recent_key, 0, 999)  # Keep last 1000 signals
    
    async def start_websocket_server(self):
        """Start WebSocket server for real-time client connections"""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            self.logger.info(f"New WebSocket client connected. Total: {len(self.websocket_clients)}")
            
            try:
                # Send recent signals to new client
                recent_signals = await self.get_recent_signals(limit=10)
                for signal_data in recent_signals:
                    await websocket.send(json.dumps({
                        'type': 'historical_signal',
                        'data': signal_data
                    }))
                
                # Keep connection alive
                await websocket.wait_closed()
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_clients.discard(websocket)
                self.logger.info(f"Client disconnected. Total: {len(self.websocket_clients)}")
        
        # Start WebSocket server
        start_server = websockets.serve(handle_client, "localhost", 8765)
        await start_server
        self.logger.info("🌐 WebSocket server started on ws://localhost:8765")
    
    async def monitor_performance(self):
        """Monitor and log performance metrics"""
        while True:
            await asyncio.sleep(30)  # Report every 30 seconds
            
            if self.processing_times:
                avg_latency = np.mean(self.processing_times)
                p95_latency = np.percentile(self.processing_times, 95)
                
                self.logger.info(f"📊 Performance Report:")
                self.logger.info(f"   Average Latency: {avg_latency:.2f}ms")
                self.logger.info(f"   P95 Latency: {p95_latency:.2f}ms")
                self.logger.info(f"   Signals Generated: {self.signal_count}")
                self.logger.info(f"   Connected Clients: {len(self.websocket_clients)}")
                
                # Reset counters
                self.processing_times = []
                self.signal_count = 0
    
    # Redis async wrapper methods
    async def redis_get(self, key):
        return self.redis_client.get(key)
    
    async def redis_set(self, key, value, ex=None):
        return self.redis_client.set(key, value, ex=ex)
    
    async def redis_lpush(self, key, value):
        return self.redis_client.lpush(key, value)
    
    async def redis_ltrim(self, key, start, end):
        return self.redis_client.ltrim(key, start, end)
    
    async def get_recent_signals(self, limit=100):
        """Get recent trading signals from cache"""
        recent_data = self.redis_client.lrange("signals:recent", 0, limit-1)
        return [json.loads(data) for data in recent_data]

class StreamingDataCollector:
    """Collect real-time data from multiple sources"""
    
    def __init__(self, streaming_engine: RealTimeStreamingEngine):
        self.engine = streaming_engine
        self.session = None
    
    async def start_collection(self):
        """Start collecting data from all sources"""
        self.session = aiohttp.ClientSession()
        
        # Start collection tasks
        tasks = [
            asyncio.create_task(self.collect_congressional_data()),
            asyncio.create_task(self.collect_market_data()),
            asyncio.create_task(self.collect_news_data()),
            asyncio.create_task(self.monitor_committees())
        ]
        
        await asyncio.gather(*tasks)
    
    async def collect_congressional_data(self):
        """Monitor congressional trading disclosures"""
        while True:
            try:
                # Check for new congressional trades
                # This would integrate with real data sources
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logging.error(f"Error collecting congressional data: {e}")
                await asyncio.sleep(30)

# Production deployment script
async def main():
    """Start the APEX Real-Time Streaming Engine"""
    print("🚀 Initializing APEX Real-Time Streaming Intelligence Engine")
    print("⚡ Target latency: <100ms")
    print("📡 Starting multi-source data collection...")
    
    # Initialize streaming engine
    engine = RealTimeStreamingEngine()
    
    # Start streaming
    await engine.start_streaming()

if __name__ == "__main__":
    asyncio.run(main())
