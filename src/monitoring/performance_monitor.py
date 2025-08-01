#!/usr/bin/env python3
"""
Advanced Performance Monitoring System for Congressional Trading Intelligence
Real-time metrics collection, alerting, and performance optimization
"""

import time
import psutil
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import json
import os

import redis
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST
from flask import request, g, current_app
import structlog
from sqlalchemy import text
from sqlalchemy.pool import Pool

from src.database import db

# Setup structured logging
logger = structlog.get_logger()


@dataclass
class PerformanceMetric:
    """Performance metric data structure"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = None
    unit: str = ""
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


@dataclass
class SystemHealthStatus:
    """System health status"""
    status: str  # healthy, warning, critical
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    database_connections: int
    redis_connected: bool
    active_requests: int
    error_rate: float
    response_time_p95: float
    uptime_seconds: float
    timestamp: datetime


class MetricsCollector:
    """Prometheus metrics collector"""
    
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint']
        )
        
        self.request_size = Histogram(
            'http_request_size_bytes',
            'HTTP request size',
            ['method', 'endpoint']
        )
        
        self.response_size = Histogram(
            'http_response_size_bytes',
            'HTTP response size',
            ['method', 'endpoint']
        )
        
        # System metrics
        self.cpu_usage = Gauge('system_cpu_percent', 'System CPU usage percentage')
        self.memory_usage = Gauge('system_memory_percent', 'System memory usage percentage')
        self.disk_usage = Gauge('system_disk_percent', 'System disk usage percentage')
        
        # Database metrics
        self.db_connections = Gauge('database_connections_active', 'Active database connections')
        self.db_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration',
            ['query_type']
        )
        
        # Application metrics
        self.active_users = Gauge('active_users_total', 'Number of active users')
        self.cache_hits = Counter('cache_hits_total', 'Cache hits', ['cache_type'])
        self.cache_misses = Counter('cache_misses_total', 'Cache misses', ['cache_type'])
        
        # ML/Analysis metrics
        self.analysis_duration = Histogram(
            'analysis_duration_seconds',
            'Analysis execution duration',
            ['analysis_type']
        )
        self.alerts_generated = Counter(
            'alerts_generated_total',
            'Alerts generated',
            ['alert_type', 'severity']
        )
        
    def record_request(self, method: str, endpoint: str, status: int, duration: float, 
                      request_size: int = 0, response_size: int = 0):
        """Record HTTP request metrics"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
        
        if request_size > 0:
            self.request_size.labels(method=method, endpoint=endpoint).observe(request_size)
        if response_size > 0:
            self.response_size.labels(method=method, endpoint=endpoint).observe(response_size)
    
    def record_system_metrics(self):
        """Record system resource metrics"""
        self.cpu_usage.set(psutil.cpu_percent())
        self.memory_usage.set(psutil.virtual_memory().percent)
        self.disk_usage.set(psutil.disk_usage('/').percent)
    
    def record_database_metrics(self, pool: Pool):
        """Record database metrics"""
        if pool:
            self.db_connections.set(pool.checkedout())


class PerformanceMonitor:
    """Advanced performance monitoring system"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url)
        self.metrics_collector = MetricsCollector()
        self.start_time = time.time()
        
        # Performance tracking
        self.request_times: List[float] = []
        self.error_count = 0
        self.total_requests = 0
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Alert thresholds
        self.thresholds = {
            'cpu_warning': 70.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'disk_warning': 85.0,
            'disk_critical': 95.0,
            'response_time_warning': 2.0,
            'response_time_critical': 5.0,
            'error_rate_warning': 5.0,
            'error_rate_critical': 10.0
        }
    
    def start_monitoring(self):
        """Start background monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                self.metrics_collector.record_system_metrics()
                
                # Check health status
                health_status = self.get_health_status()
                self._check_alerts(health_status)
                
                # Store metrics in Redis
                self._store_metrics(health_status)
                
                time.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                time.sleep(60)  # Wait longer on error
    
    def _check_alerts(self, health_status: SystemHealthStatus):
        """Check for alert conditions"""
        alerts = []
        
        # CPU alerts
        if health_status.cpu_percent >= self.thresholds['cpu_critical']:
            alerts.append({
                'type': 'cpu_critical',
                'message': f'CPU usage critical: {health_status.cpu_percent:.1f}%',
                'severity': 'critical'
            })
        elif health_status.cpu_percent >= self.thresholds['cpu_warning']:
            alerts.append({
                'type': 'cpu_warning',
                'message': f'CPU usage high: {health_status.cpu_percent:.1f}%',
                'severity': 'warning'
            })
        
        # Memory alerts
        if health_status.memory_percent >= self.thresholds['memory_critical']:
            alerts.append({
                'type': 'memory_critical',
                'message': f'Memory usage critical: {health_status.memory_percent:.1f}%',
                'severity': 'critical'
            })
        elif health_status.memory_percent >= self.thresholds['memory_warning']:
            alerts.append({
                'type': 'memory_warning',
                'message': f'Memory usage high: {health_status.memory_percent:.1f}%',
                'severity': 'warning'
            })
        
        # Response time alerts
        if health_status.response_time_p95 >= self.thresholds['response_time_critical']:
            alerts.append({
                'type': 'response_time_critical',
                'message': f'Response time critical: {health_status.response_time_p95:.2f}s',
                'severity': 'critical'
            })
        elif health_status.response_time_p95 >= self.thresholds['response_time_warning']:
            alerts.append({
                'type': 'response_time_warning',
                'message': f'Response time high: {health_status.response_time_p95:.2f}s',
                'severity': 'warning'
            })
        
        # Error rate alerts
        if health_status.error_rate >= self.thresholds['error_rate_critical']:
            alerts.append({
                'type': 'error_rate_critical',
                'message': f'Error rate critical: {health_status.error_rate:.1f}%',
                'severity': 'critical'
            })
        elif health_status.error_rate >= self.thresholds['error_rate_warning']:
            alerts.append({
                'type': 'error_rate_warning',
                'message': f'Error rate high: {health_status.error_rate:.1f}%',
                'severity': 'warning'
            })
        
        # Send alerts
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send performance alert"""
        try:
            # Store alert in Redis
            alert_key = f"performance_alert:{int(time.time())}"
            alert['timestamp'] = datetime.utcnow().isoformat()
            
            self.redis_client.setex(alert_key, 86400, json.dumps(alert))  # 24 hours TTL
            
            # Publish to alert channel
            self.redis_client.publish('performance_alerts', json.dumps(alert))
            
            logger.warning("Performance alert generated", 
                          alert_type=alert['type'], 
                          severity=alert['severity'],
                          message=alert['message'])
            
        except Exception as e:
            logger.error("Failed to send performance alert", error=str(e))
    
    def _store_metrics(self, health_status: SystemHealthStatus):
        """Store metrics in Redis for historical tracking"""
        try:
            # Store current health status
            health_key = "health_status:current"
            self.redis_client.setex(health_key, 300, json.dumps(asdict(health_status), default=str))
            
            # Store in time series (keep last 24 hours)
            ts_key = f"health_status:ts:{int(time.time())}"
            self.redis_client.setex(ts_key, 86400, json.dumps(asdict(health_status), default=str))
            
            # Clean up old time series data
            cutoff_time = int(time.time()) - 86400  # 24 hours ago
            for key in self.redis_client.scan_iter(match="health_status:ts:*"):
                timestamp = int(key.decode().split(':')[-1])
                if timestamp < cutoff_time:
                    self.redis_client.delete(key)
                    
        except Exception as e:
            logger.error("Failed to store metrics", error=str(e))
    
    def get_health_status(self) -> SystemHealthStatus:
        """Get current system health status"""
        try:
            # System resources
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Database connections
            db_connections = 0
            try:
                if hasattr(db.engine, 'pool'):
                    db_connections = db.engine.pool.checkedout()
            except:
                pass
            
            # Redis connection
            redis_connected = True
            try:
                self.redis_client.ping()
            except:
                redis_connected = False
            
            # Request metrics
            active_requests = len(self.request_times)
            error_rate = (self.error_count / max(self.total_requests, 1)) * 100
            
            # Response time percentile
            response_time_p95 = 0.0
            if self.request_times:
                sorted_times = sorted(self.request_times)
                p95_index = int(len(sorted_times) * 0.95)
                response_time_p95 = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            
            # Determine overall status
            status = "healthy"
            if (cpu_percent >= self.thresholds['cpu_critical'] or
                memory_percent >= self.thresholds['memory_critical'] or
                disk_percent >= self.thresholds['disk_critical'] or
                response_time_p95 >= self.thresholds['response_time_critical'] or
                error_rate >= self.thresholds['error_rate_critical']):
                status = "critical"
            elif (cpu_percent >= self.thresholds['cpu_warning'] or
                  memory_percent >= self.thresholds['memory_warning'] or
                  disk_percent >= self.thresholds['disk_warning'] or
                  response_time_p95 >= self.thresholds['response_time_warning'] or
                  error_rate >= self.thresholds['error_rate_warning']):
                status = "warning"
            
            return SystemHealthStatus(
                status=status,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                database_connections=db_connections,
                redis_connected=redis_connected,
                active_requests=active_requests,
                error_rate=error_rate,
                response_time_p95=response_time_p95,
                uptime_seconds=time.time() - self.start_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error("Failed to get health status", error=str(e))
            return SystemHealthStatus(
                status="unknown",
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                database_connections=0,
                redis_connected=False,
                active_requests=0,
                error_rate=0.0,
                response_time_p95=0.0,
                uptime_seconds=0.0,
                timestamp=datetime.utcnow()
            )
    
    @contextmanager
    def track_request(self, method: str, endpoint: str):
        """Context manager to track request performance"""
        start_time = time.time()
        request_size = 0
        response_size = 0
        status = 200
        
        try:
            # Get request size if available
            if hasattr(request, 'content_length') and request.content_length:
                request_size = request.content_length
                
            yield
            
        except Exception as e:
            status = 500
            self.error_count += 1
            logger.error("Request error", method=method, endpoint=endpoint, error=str(e))
            raise
            
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Track request time
            self.request_times.append(duration)
            self.total_requests += 1
            
            # Keep only recent request times (last 1000)
            if len(self.request_times) > 1000:
                self.request_times = self.request_times[-1000:]
            
            # Record metrics
            self.metrics_collector.record_request(
                method=method,
                endpoint=endpoint,
                status=status,
                duration=duration,
                request_size=request_size,
                response_size=response_size
            )
    
    @contextmanager
    def track_database_query(self, query_type: str):
        """Context manager to track database query performance"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics_collector.db_query_duration.labels(query_type=query_type).observe(duration)
    
    @contextmanager
    def track_analysis(self, analysis_type: str):
        """Context manager to track analysis performance"""
        start_time = time.time()
        
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics_collector.analysis_duration.labels(analysis_type=analysis_type).observe(duration)
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit"""
        self.metrics_collector.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss"""
        self.metrics_collector.cache_misses.labels(cache_type=cache_type).inc()
    
    def record_alert_generated(self, alert_type: str, severity: str):
        """Record alert generation"""
        self.metrics_collector.alerts_generated.labels(alert_type=alert_type, severity=severity).inc()
    
    def get_metrics_data(self) -> str:
        """Get Prometheus metrics data"""
        return generate_latest()
    
    def get_historical_metrics(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics from Redis"""
        try:
            cutoff_time = int(time.time()) - (hours * 3600)
            metrics = []
            
            for key in self.redis_client.scan_iter(match="health_status:ts:*"):
                timestamp = int(key.decode().split(':')[-1])
                if timestamp >= cutoff_time:
                    data = self.redis_client.get(key)
                    if data:
                        metrics.append(json.loads(data))
            
            # Sort by timestamp
            metrics.sort(key=lambda x: x.get('timestamp', ''))
            return metrics
            
        except Exception as e:
            logger.error("Failed to get historical metrics", error=str(e))
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        try:
            health_status = self.get_health_status()
            historical_metrics = self.get_historical_metrics(1)  # Last hour
            
            # Calculate averages from historical data
            avg_cpu = 0.0
            avg_memory = 0.0
            avg_response_time = 0.0
            
            if historical_metrics:
                avg_cpu = sum(m.get('cpu_percent', 0) for m in historical_metrics) / len(historical_metrics)
                avg_memory = sum(m.get('memory_percent', 0) for m in historical_metrics) / len(historical_metrics)
                avg_response_time = sum(m.get('response_time_p95', 0) for m in historical_metrics) / len(historical_metrics)
            
            return {
                'current_status': health_status.status,
                'uptime_hours': health_status.uptime_seconds / 3600,
                'total_requests': self.total_requests,
                'error_rate': health_status.error_rate,
                'current_response_time_p95': health_status.response_time_p95,
                'avg_response_time_1h': avg_response_time,
                'current_cpu': health_status.cpu_percent,
                'avg_cpu_1h': avg_cpu,
                'current_memory': health_status.memory_percent,
                'avg_memory_1h': avg_memory,
                'database_connections': health_status.database_connections,
                'redis_connected': health_status.redis_connected,
                'active_requests': health_status.active_requests,
                'timestamp': health_status.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error("Failed to get performance summary", error=str(e))
            return {'error': 'Failed to get performance summary'}


# Global monitor instance
performance_monitor = PerformanceMonitor()


def init_performance_monitoring(app):
    """Initialize performance monitoring for Flask app"""
    
    @app.before_request
    def before_request():
        g.start_time = time.time()
        g.method = request.method
        g.endpoint = request.endpoint or 'unknown'
    
    @app.after_request
    def after_request(response):
        if hasattr(g, 'start_time'):
            with performance_monitor.track_request(g.method, g.endpoint):
                pass  # Request tracking is handled in the context manager
        return response
    
    @app.route('/metrics')
    def metrics():
        """Prometheus metrics endpoint"""
        return performance_monitor.get_metrics_data(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    
    @app.route('/health')
    def health():
        """Health check endpoint"""
        health_status = performance_monitor.get_health_status()
        return {
            'status': health_status.status,
            'timestamp': health_status.timestamp.isoformat(),
            'uptime': health_status.uptime_seconds,
            'details': {
                'cpu_percent': health_status.cpu_percent,
                'memory_percent': health_status.memory_percent,
                'database_connected': health_status.database_connections > 0,
                'redis_connected': health_status.redis_connected
            }
        }
    
    @app.route('/performance')
    def performance_summary():
        """Performance summary endpoint"""
        return performance_monitor.get_performance_summary()
    
    # Start monitoring
    performance_monitor.start_monitoring()
    
    # Cleanup on app teardown
    @app.teardown_appcontext
    def cleanup_monitoring(error):
        if error:
            performance_monitor.error_count += 1


if __name__ == "__main__":
    # Standalone monitoring for testing
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        while True:
            health = monitor.get_health_status()
            print(f"Status: {health.status}, CPU: {health.cpu_percent:.1f}%, "
                  f"Memory: {health.memory_percent:.1f}%, "
                  f"Response Time P95: {health.response_time_p95:.2f}s")
            time.sleep(30)
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("Monitoring stopped")