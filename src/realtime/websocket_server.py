#!/usr/bin/env python3
"""
Real-Time WebSocket Server for Congressional Trading Intelligence
Advanced WebSocket implementation with Redis pub/sub and authentication
"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict

import websockets
from websockets.server import WebSocketServerProtocol
import redis.asyncio as redis
import jwt
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import structlog

from src.models.trading import Trade, TradeAlert
from src.models.member import Member
from src.intelligence.suspicious_trading_detector import SuspiciousTradingDetector
from src.intelligence.network_analyzer import NetworkAnalyzer

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: str
    data: Any
    timestamp: str = None
    client_id: str = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class ClientConnection:
    """Client connection information"""
    websocket: WebSocketServerProtocol
    user_id: int
    username: str
    subscriptions: Set[str]
    connected_at: datetime
    last_ping: datetime
    
    def __post_init__(self):
        if not self.connected_at:
            self.connected_at = datetime.utcnow()
        if not self.last_ping:
            self.last_ping = datetime.utcnow()


class WebSocketServer:
    """
    Advanced WebSocket server with Redis pub/sub integration
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6789,
        redis_url: str = "redis://localhost:6379/0",
        database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/congressional_trading_dev",
        jwt_secret: str = "jwt-secret-change-in-production"
    ):
        self.host = host
        self.port = port
        self.redis_url = redis_url
        self.database_url = database_url
        self.jwt_secret = jwt_secret
        
        # Connection management
        self.clients: Dict[str, ClientConnection] = {}
        self.subscriptions: Dict[str, Set[str]] = {
            'trades': set(),
            'alerts': set(),
            'analysis': set(),
            'members': set(),
            'system': set()
        }
        
        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        
        # Database connection
        self.db_engine = None
        self.db_session_factory = None
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Analytics components
        self.detector = None
        self.network_analyzer = None
        
    async def start_server(self):
        """Start the WebSocket server and background services"""
        logger.info("Starting Congressional Trading WebSocket Server", 
                   host=self.host, port=self.port)
        
        # Initialize connections
        await self.initialize_redis()
        await self.initialize_database()
        await self.initialize_analytics()
        
        # Start background tasks
        await self.start_background_tasks()
        
        # Start WebSocket server
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
            close_timeout=10
        ):
            logger.info("WebSocket server started successfully")
            await asyncio.Future()  # Run forever
    
    async def initialize_redis(self):
        """Initialize Redis connection and pub/sub"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe(
                'trades_updates',
                'alerts_updates', 
                'analysis_updates',
                'system_updates'
            )
            
            logger.info("Redis connection established")
        except Exception as e:
            logger.error("Failed to initialize Redis", error=str(e))
            raise
    
    async def initialize_database(self):
        """Initialize async database connection"""
        try:
            self.db_engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.db_session_factory = sessionmaker(
                self.db_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database connection established")
        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise
    
    async def initialize_analytics(self):
        """Initialize analytics components"""
        try:
            self.detector = SuspiciousTradingDetector()
            self.network_analyzer = NetworkAnalyzer()
            logger.info("Analytics components initialized")
        except Exception as e:
            logger.error("Failed to initialize analytics", error=str(e))
            raise
    
    async def start_background_tasks(self):
        """Start background monitoring tasks"""
        tasks = [
            asyncio.create_task(self.redis_message_handler()),
            asyncio.create_task(self.health_monitor()),
            asyncio.create_task(self.data_sync_monitor()),
            asyncio.create_task(self.client_ping_monitor())
        ]
        
        for task in tasks:
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        logger.info(f"Started {len(tasks)} background tasks")
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle new client connection"""
        client_id = None
        
        try:
            # Authenticate client
            auth_result = await self.authenticate_client(websocket)
            if not auth_result:
                await websocket.close(code=4001, reason="Authentication failed")
                return
            
            user_id, username = auth_result
            client_id = f"{user_id}_{int(time.time())}"
            
            # Create client connection
            client = ClientConnection(
                websocket=websocket,
                user_id=user_id,
                username=username,
                subscriptions=set(),
                connected_at=datetime.utcnow(),
                last_ping=datetime.utcnow()
            )
            
            self.clients[client_id] = client
            
            logger.info("Client connected", 
                       client_id=client_id, user_id=user_id, username=username)
            
            # Send welcome message
            await self.send_to_client(client_id, WebSocketMessage(
                type="connection_established",
                data={
                    "client_id": client_id,
                    "server_time": datetime.utcnow().isoformat(),
                    "available_subscriptions": list(self.subscriptions.keys())
                }
            ))
            
            # Handle client messages
            async for message in websocket:
                await self.handle_client_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected", client_id=client_id)
        except Exception as e:
            logger.error("Error handling client", 
                        client_id=client_id, error=str(e), traceback=traceback.format_exc())
        finally:
            if client_id and client_id in self.clients:
                await self.disconnect_client(client_id)
    
    async def authenticate_client(self, websocket: WebSocketServerProtocol) -> Optional[tuple]:
        """Authenticate WebSocket client using JWT token"""
        try:
            # Wait for authentication message
            auth_message = await asyncio.wait_for(websocket.recv(), timeout=10.0)
            auth_data = json.loads(auth_message)
            
            if auth_data.get('type') != 'authenticate':
                return None
            
            token = auth_data.get('token')
            if not token:
                return None
            
            # Verify JWT token
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            user_id = payload.get('sub')
            username = payload.get('username')
            
            if not user_id or not username:
                return None
            
            return int(user_id), username
            
        except (asyncio.TimeoutError, json.JSONDecodeError, jwt.InvalidTokenError):
            return None
    
    async def handle_client_message(self, client_id: str, message: str):
        """Handle incoming client message"""
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            client = self.clients.get(client_id)
            if not client:
                return
            
            # Update last ping
            client.last_ping = datetime.utcnow()
            
            if message_type == 'subscribe':
                await self.handle_subscription(client_id, data.get('channels', []))
            elif message_type == 'unsubscribe':
                await self.handle_unsubscription(client_id, data.get('channels', []))
            elif message_type == 'ping':
                await self.send_to_client(client_id, WebSocketMessage(
                    type="pong",
                    data={"timestamp": datetime.utcnow().isoformat()}
                ))
            elif message_type == 'request_data':
                await self.handle_data_request(client_id, data)
            else:
                logger.warning("Unknown message type", 
                             client_id=client_id, message_type=message_type)
                
        except json.JSONDecodeError:
            logger.error("Invalid JSON message", client_id=client_id)
        except Exception as e:
            logger.error("Error handling client message", 
                        client_id=client_id, error=str(e))
    
    async def handle_subscription(self, client_id: str, channels: List[str]):
        """Handle client subscription to channels"""
        client = self.clients.get(client_id)
        if not client:
            return
        
        valid_channels = set(channels) & set(self.subscriptions.keys())
        
        for channel in valid_channels:
            if channel not in client.subscriptions:
                client.subscriptions.add(channel)
                self.subscriptions[channel].add(client_id)
        
        await self.send_to_client(client_id, WebSocketMessage(
            type="subscription_confirmed",
            data={
                "subscribed_channels": list(valid_channels),
                "total_subscriptions": len(client.subscriptions)
            }
        ))
        
        logger.info("Client subscribed to channels",
                   client_id=client_id, channels=list(valid_channels))
    
    async def handle_unsubscription(self, client_id: str, channels: List[str]):
        """Handle client unsubscription from channels"""
        client = self.clients.get(client_id)
        if not client:
            return
        
        for channel in channels:
            if channel in client.subscriptions:
                client.subscriptions.remove(channel)
                self.subscriptions[channel].discard(client_id)
        
        await self.send_to_client(client_id, WebSocketMessage(
            type="unsubscription_confirmed",
            data={
                "unsubscribed_channels": channels,
                "remaining_subscriptions": list(client.subscriptions)
            }
        ))
    
    async def handle_data_request(self, client_id: str, request_data: dict):
        """Handle client data request"""
        request_type = request_data.get('request_type')
        
        try:
            if request_type == 'recent_trades':
                data = await self.get_recent_trades(request_data.get('limit', 10))
            elif request_type == 'active_alerts':
                data = await self.get_active_alerts(request_data.get('level'))
            elif request_type == 'member_summary':
                data = await self.get_member_summary(request_data.get('member_id'))
            elif request_type == 'system_status':
                data = await self.get_system_status()
            else:
                data = {"error": "Unknown request type"}
            
            await self.send_to_client(client_id, WebSocketMessage(
                type="data_response",
                data={
                    "request_id": request_data.get('request_id'),
                    "request_type": request_type,
                    "data": data
                }
            ))
            
        except Exception as e:
            await self.send_to_client(client_id, WebSocketMessage(
                type="data_error",
                data={
                    "request_id": request_data.get('request_id'),
                    "error": str(e)
                }
            ))
    
    async def send_to_client(self, client_id: str, message: WebSocketMessage):
        """Send message to specific client"""
        client = self.clients.get(client_id)
        if not client:
            return
        
        try:
            message.client_id = client_id
            message_json = json.dumps(asdict(message))
            await client.websocket.send(message_json)
        except websockets.exceptions.ConnectionClosed:
            await self.disconnect_client(client_id)
        except Exception as e:
            logger.error("Error sending message to client", 
                        client_id=client_id, error=str(e))
    
    async def broadcast_to_channel(self, channel: str, message: WebSocketMessage):
        """Broadcast message to all clients subscribed to channel"""
        if channel not in self.subscriptions:
            return
        
        clients_to_notify = list(self.subscriptions[channel])
        
        for client_id in clients_to_notify:
            await self.send_to_client(client_id, message)
        
        logger.debug("Broadcasted message to channel",
                    channel=channel, client_count=len(clients_to_notify))
    
    async def disconnect_client(self, client_id: str):
        """Clean up client connection"""
        client = self.clients.pop(client_id, None)
        if not client:
            return
        
        # Remove from all subscriptions
        for channel in client.subscriptions:
            self.subscriptions[channel].discard(client_id)
        
        logger.info("Client cleaned up", client_id=client_id)
    
    async def redis_message_handler(self):
        """Handle messages from Redis pub/sub"""
        if not self.pubsub:
            return
        
        logger.info("Starting Redis message handler")
        
        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    await self.process_redis_message(message)
        except Exception as e:
            logger.error("Redis message handler error", error=str(e))
    
    async def process_redis_message(self, redis_message):
        """Process incoming Redis message"""
        try:
            channel = redis_message['channel'].decode('utf-8')
            data = json.loads(redis_message['data'].decode('utf-8'))
            
            # Map Redis channels to WebSocket channels
            channel_mapping = {
                'trades_updates': 'trades',
                'alerts_updates': 'alerts',
                'analysis_updates': 'analysis',
                'system_updates': 'system'
            }
            
            ws_channel = channel_mapping.get(channel)
            if ws_channel:
                message = WebSocketMessage(
                    type=data.get('type', 'update'),
                    data=data.get('data', {})
                )
                
                await self.broadcast_to_channel(ws_channel, message)
                
        except Exception as e:
            logger.error("Error processing Redis message", error=str(e))
    
    async def health_monitor(self):
        """Monitor system health and broadcast status"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                health_status = await self.get_system_status()
                
                message = WebSocketMessage(
                    type="health_update",
                    data=health_status
                )
                
                await self.broadcast_to_channel('system', message)
                
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
    
    async def data_sync_monitor(self):
        """Monitor for new data and trigger real-time updates"""
        last_check = datetime.utcnow()
        
        while True:
            try:
                await asyncio.sleep(30)  # Every 30 seconds
                
                # Check for new trades
                new_trades = await self.check_new_trades(last_check)
                if new_trades:
                    message = WebSocketMessage(
                        type="new_trades",
                        data={"trades": new_trades, "count": len(new_trades)}
                    )
                    await self.broadcast_to_channel('trades', message)
                
                # Check for new alerts
                new_alerts = await self.check_new_alerts(last_check)
                if new_alerts:
                    message = WebSocketMessage(
                        type="new_alerts",
                        data={"alerts": new_alerts, "count": len(new_alerts)}
                    )
                    await self.broadcast_to_channel('alerts', message)
                
                last_check = datetime.utcnow()
                
            except Exception as e:
                logger.error("Data sync monitor error", error=str(e))
    
    async def client_ping_monitor(self):
        """Monitor client connections and clean up stale ones"""
        while True:
            try:
                await asyncio.sleep(60)  # Every minute
                
                current_time = datetime.utcnow()
                stale_clients = []
                
                for client_id, client in self.clients.items():
                    time_since_ping = (current_time - client.last_ping).total_seconds()
                    
                    if time_since_ping > 120:  # 2 minutes timeout
                        stale_clients.append(client_id)
                
                for client_id in stale_clients:
                    logger.info("Removing stale client", client_id=client_id)
                    await self.disconnect_client(client_id)
                
            except Exception as e:
                logger.error("Client ping monitor error", error=str(e))
    
    # Data retrieval methods
    async def get_recent_trades(self, limit: int = 10) -> List[dict]:
        """Get recent trades data"""
        # Implementation would query database
        return []
    
    async def get_active_alerts(self, level: Optional[str] = None) -> List[dict]:
        """Get active alerts"""
        # Implementation would query database
        return []
    
    async def get_member_summary(self, member_id: Optional[int] = None) -> dict:
        """Get member summary data"""
        # Implementation would query database
        return {}
    
    async def get_system_status(self) -> dict:
        """Get system health status"""
        return {
            "status": "healthy",
            "connected_clients": len(self.clients),
            "uptime": time.time(),
            "memory_usage": "normal",
            "database_status": "connected",
            "redis_status": "connected"
        }
    
    async def check_new_trades(self, since: datetime) -> List[dict]:
        """Check for new trades since timestamp"""
        # Implementation would query database
        return []
    
    async def check_new_alerts(self, since: datetime) -> List[dict]:
        """Check for new alerts since timestamp"""
        # Implementation would query database
        return []


async def main():
    """Main entry point"""
    import os
    
    server = WebSocketServer(
        host=os.getenv('WS_HOST', 'localhost'),
        port=int(os.getenv('WS_PORT', 6789)),
        redis_url=os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        database_url=os.getenv('DATABASE_URL', 'postgresql+asyncpg://postgres:password@localhost:5432/congressional_trading_dev'),
        jwt_secret=os.getenv('JWT_SECRET_KEY', 'jwt-secret-change-in-production')
    )
    
    await server.start_server()


if __name__ == "__main__":
    asyncio.run(main())