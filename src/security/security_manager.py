#!/usr/bin/env python3
"""
Advanced Security Manager for Congressional Trading Intelligence System
Comprehensive security hardening, threat detection, and compliance monitoring
"""

import hashlib
import hmac
import secrets
import time
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from functools import wraps
from ipaddress import ip_address, ip_network
import base64

from flask import request, jsonify, current_app, g
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import redis
import structlog
from sqlalchemy import text

from src.database import db
from src.models.user import User, UserSession
from src.models.audit import AuditLog

logger = structlog.get_logger()


@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: str
    severity: str  # low, medium, high, critical
    user_id: Optional[int]
    ip_address: str
    user_agent: str
    details: Dict[str, Any]
    timestamp: datetime
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow()


@dataclass
class ThreatIntelligence:
    """Threat intelligence data"""
    threat_type: str
    indicators: List[str]
    confidence: float  # 0.0 to 1.0
    severity: str
    source: str
    last_updated: datetime


class SecurityValidation:
    """Security validation utilities"""
    
    # Password policy
    MIN_PASSWORD_LENGTH = 12
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL_CHARS = True
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    # Common weak passwords (subset for demo)
    WEAK_PASSWORDS = {
        'password123', 'admin123', 'letmein', 'welcome123',
        'congressional', 'trading123', 'password!',
        'Password123', 'Admin123!'
    }
    
    @classmethod
    def validate_password(cls, password: str) -> Tuple[bool, List[str]]:
        """Validate password against security policy"""
        errors = []
        
        if len(password) < cls.MIN_PASSWORD_LENGTH:
            errors.append(f"Password must be at least {cls.MIN_PASSWORD_LENGTH} characters")
        
        if cls.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        if cls.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        if cls.REQUIRE_DIGITS and not re.search(r'\d', password):
            errors.append("Password must contain at least one digit")
        
        if cls.REQUIRE_SPECIAL_CHARS and not any(c in cls.SPECIAL_CHARS for c in password):
            errors.append(f"Password must contain at least one special character: {cls.SPECIAL_CHARS}")
        
        if password.lower() in cls.WEAK_PASSWORDS:
            errors.append("Password is too common and easily guessed")
        
        # Check for common patterns
        if re.search(r'(.)\1{2,}', password):  # Repeated characters
            errors.append("Password cannot have more than 2 consecutive identical characters")
        
        if re.search(r'(012|123|234|345|456|567|678|789|890)', password):
            errors.append("Password cannot contain sequential numbers")
        
        if re.search(r'(abc|bcd|cde|def|efg|fgh|ghi|hij|ijk|jkl|klm|lmn|mno|nop|opq|pqr|qrs|rst|stu|tuv|uvw|vwx|wxy|xyz)', password.lower()):
            errors.append("Password cannot contain sequential letters")
        
        return len(errors) == 0, errors
    
    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @classmethod
    def validate_input_sanitization(cls, input_str: str) -> Tuple[bool, str]:
        """Validate and sanitize user input"""
        if not input_str:
            return True, input_str
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', input_str)
        
        # Check for SQL injection patterns
        sql_patterns = [
            r'(union|select|insert|update|delete|drop|create|alter|exec|execute)',
            r'(script|javascript|vbscript|onload|onerror|onclick)',
            r'(\||\|\||&&|;|--|/\*|\*/)'
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, input_str.lower()):
                return False, "Input contains potentially malicious content"
        
        return True, sanitized


class EncryptionManager:
    """Advanced encryption and key management"""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            self.master_key = self._generate_master_key()
        
        self.fernet = self._create_fernet()
    
    def _generate_master_key(self) -> bytes:
        """Generate a new master key"""
        return secrets.token_bytes(32)
    
    def _create_fernet(self) -> Fernet:
        """Create Fernet instance from master key"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'congressional_trading_salt',  # In production, use random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Hash sensitive data with salt"""
        if not salt:
            salt = secrets.token_hex(16)
        
        hash_value = hashlib.pbkdf2_hmac(
            'sha256',
            data.encode(),
            salt.encode(),
            100000  # iterations
        )
        
        return hash_value.hex(), salt
    
    def verify_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """Verify hashed data"""
        computed_hash, _ = self.hash_sensitive_data(data, salt)
        return hmac.compare_digest(computed_hash, hash_value)


class ThreatDetection:
    """Advanced threat detection and prevention"""
    
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.failed_attempts = {}  # IP -> attempts count
        self.blocked_ips = set()
        
        # Load threat intelligence
        self.threat_indicators = self._load_threat_intelligence()
    
    def _load_threat_intelligence(self) -> Dict[str, ThreatIntelligence]:
        """Load threat intelligence data"""
        # In production, this would load from external threat feeds
        return {
            'malicious_ips': ThreatIntelligence(
                threat_type='malicious_ip',
                indicators=['192.168.1.100', '10.0.0.50'],  # Example IPs
                confidence=0.9,
                severity='high',
                source='internal_blacklist',
                last_updated=datetime.utcnow()
            ),
            'suspicious_user_agents': ThreatIntelligence(
                threat_type='suspicious_user_agent',
                indicators=['bot', 'crawler', 'scanner', 'sqlmap'],
                confidence=0.8,
                severity='medium',
                source='user_agent_analysis',
                last_updated=datetime.utcnow()
            )
        }
    
    def detect_brute_force(self, ip_address: str, user_id: Optional[int] = None) -> bool:
        """Detect brute force attacks"""
        key = f"failed_login:{ip_address}"
        
        try:
            # Get current failed attempts
            attempts = self.redis_client.get(key)
            attempts = int(attempts) if attempts else 0
            
            # Check if IP is already blocked
            if attempts >= 5:  # Block after 5 failed attempts
                self._block_ip(ip_address, reason="brute_force_detected")
                return True
            
            return False
            
        except Exception as e:
            logger.error("Error in brute force detection", error=str(e))
            return False
    
    def record_failed_login(self, ip_address: str, user_id: Optional[int] = None):
        """Record failed login attempt"""
        key = f"failed_login:{ip_address}"
        
        try:
            # Increment failed attempts
            self.redis_client.incr(key)
            self.redis_client.expire(key, 3600)  # Reset after 1 hour
            
            # Log security event
            event = SecurityEvent(
                event_type="failed_login",
                severity="medium",
                user_id=user_id,
                ip_address=ip_address,
                user_agent=request.headers.get('User-Agent', ''),
                details={'attempts': self.redis_client.get(key).decode()},
                timestamp=datetime.utcnow()
            )
            
            self._log_security_event(event)
            
        except Exception as e:
            logger.error("Error recording failed login", error=str(e))
    
    def clear_failed_attempts(self, ip_address: str):
        """Clear failed login attempts for IP"""
        key = f"failed_login:{ip_address}"
        self.redis_client.delete(key)
    
    def detect_anomalous_behavior(self, user_id: int, request_data: Dict[str, Any]) -> List[str]:
        """Detect anomalous user behavior"""
        anomalies = []
        
        try:
            # Check for unusual request patterns
            request_key = f"user_requests:{user_id}"
            current_requests = self.redis_client.get(request_key)
            current_requests = int(current_requests) if current_requests else 0
            
            # Rate limiting check
            if current_requests > 100:  # More than 100 requests per minute
                anomalies.append("excessive_request_rate")
            
            # Check for unusual geographic locations (simplified)
            ip_address = request.remote_addr
            if self._is_suspicious_location(ip_address):
                anomalies.append("suspicious_geographic_location")
            
            # Check for unusual access patterns
            if self._is_unusual_access_time():
                anomalies.append("unusual_access_time")
            
            # Update request counter
            self.redis_client.incr(request_key)
            self.redis_client.expire(request_key, 60)  # 1 minute window
            
        except Exception as e:
            logger.error("Error in anomaly detection", error=str(e))
        
        return anomalies
    
    def _is_suspicious_location(self, ip_address: str) -> bool:
        """Check if IP address is from suspicious location"""
        # In production, integrate with geolocation and threat intelligence
        try:
            ip = ip_address(ip_address)
            
            # Check against known malicious IP ranges
            malicious_ranges = [
                ip_network('192.168.100.0/24'),  # Example suspicious range
                ip_network('10.0.100.0/24')      # Another example
            ]
            
            for network in malicious_ranges:
                if ip in network:
                    return True
                    
        except Exception:
            pass
        
        return False
    
    def _is_unusual_access_time(self) -> bool:
        """Check if access time is unusual"""
        current_hour = datetime.utcnow().hour
        
        # Flag access outside business hours (simplified)
        if current_hour < 6 or current_hour > 22:  # Outside 6 AM - 10 PM UTC
            return True
        
        return False
    
    def _block_ip(self, ip_address: str, reason: str, duration: int = 3600):
        """Block IP address"""
        try:
            block_key = f"blocked_ip:{ip_address}"
            self.redis_client.setex(block_key, duration, json.dumps({
                'reason': reason,
                'blocked_at': datetime.utcnow().isoformat(),
                'duration': duration
            }))
            
            self.blocked_ips.add(ip_address)
            
            # Log security event
            event = SecurityEvent(
                event_type="ip_blocked",
                severity="high",
                user_id=None,
                ip_address=ip_address,
                user_agent="",
                details={'reason': reason, 'duration': duration},
                timestamp=datetime.utcnow()
            )
            
            self._log_security_event(event)
            
            logger.warning("IP address blocked", ip=ip_address, reason=reason)
            
        except Exception as e:
            logger.error("Error blocking IP", ip=ip_address, error=str(e))
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        try:
            block_key = f"blocked_ip:{ip_address}"
            return self.redis_client.exists(block_key)
        except Exception:
            return False
    
    def _log_security_event(self, event: SecurityEvent):
        """Log security event to database and Redis"""
        try:
            # Store in database
            audit_log = AuditLog(
                user_id=event.user_id,
                action=event.event_type,
                resource_type="security",
                resource_id=None,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                details=event.details,
                severity=event.severity
            )
            
            db.session.add(audit_log)
            db.session.commit()
            
            # Store in Redis for real-time monitoring
            event_key = f"security_event:{int(time.time())}"
            self.redis_client.setex(event_key, 86400, json.dumps(asdict(event), default=str))
            
            # Publish to security channel
            self.redis_client.publish('security_events', json.dumps(asdict(event), default=str))
            
        except Exception as e:
            logger.error("Error logging security event", error=str(e))


class SecurityManager:
    """Main security manager class"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url)
        self.encryption_manager = EncryptionManager()
        self.threat_detection = ThreatDetection(self.redis_client)
        
        # Security configuration
        self.config = {
            'session_timeout': 3600,  # 1 hour
            'max_failed_attempts': 5,
            'lockout_duration': 3600,  # 1 hour
            'password_expiry_days': 90,
            'require_2fa': False,  # Can be enabled for high-privilege users
            'audit_all_actions': True
        }
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str) -> Tuple[Optional[User], List[str]]:
        """Authenticate user with security checks"""
        errors = []
        
        try:
            # Check if IP is blocked
            if self.threat_detection.is_ip_blocked(ip_address):
                errors.append("Access denied from this IP address")
                return None, errors
            
            # Check for brute force attempts
            if self.threat_detection.detect_brute_force(ip_address):
                errors.append("Too many failed attempts. IP blocked.")
                return None, errors
            
            # Find user
            user = User.query.filter(
                (User.username == username) | (User.email == username)
            ).first()
            
            if not user:
                self.threat_detection.record_failed_login(ip_address)
                errors.append("Invalid credentials")
                return None, errors
            
            # Check if user is active
            if not user.is_active:
                errors.append("Account is disabled")
                return None, errors
            
            # Verify password
            if not check_password_hash(user.password_hash, password):
                self.threat_detection.record_failed_login(ip_address, user.id)
                errors.append("Invalid credentials")
                return None, errors
            
            # Check password expiry
            if self._is_password_expired(user):
                errors.append("Password has expired. Please change your password.")
                return user, errors  # Return user but with error
            
            # Clear failed attempts on successful login
            self.threat_detection.clear_failed_attempts(ip_address)
            
            # Update user login info
            user.last_login = datetime.utcnow()
            user.login_count += 1
            user.failed_login_attempts = 0
            db.session.commit()
            
            # Log successful login
            self._log_security_event(SecurityEvent(
                event_type="successful_login",
                severity="low",
                user_id=user.id,
                ip_address=ip_address,
                user_agent=request.headers.get('User-Agent', ''),
                details={'username': username},
                timestamp=datetime.utcnow()
            ))
            
            return user, []
            
        except Exception as e:
            logger.error("Authentication error", error=str(e))
            errors.append("Authentication service temporarily unavailable")
            return None, errors
    
    def create_secure_session(self, user: User, ip_address: str) -> str:
        """Create secure user session"""
        try:
            # Generate secure session token
            session_token = secrets.token_urlsafe(32)
            
            # Create session record
            session = UserSession(
                user_id=user.id,
                session_token=hashlib.sha256(session_token.encode()).hexdigest(),
                ip_address=ip_address,
                user_agent=request.headers.get('User-Agent', ''),
                expires_at=datetime.utcnow() + timedelta(seconds=self.config['session_timeout'])
            )
            
            db.session.add(session)
            db.session.commit()
            
            # Store in Redis for fast lookup
            session_key = f"session:{session_token}"
            session_data = {
                'user_id': user.id,
                'username': user.username,
                'ip_address': ip_address,
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.redis_client.setex(
                session_key, 
                self.config['session_timeout'], 
                json.dumps(session_data)
            )
            
            return session_token
            
        except Exception as e:
            logger.error("Error creating session", error=str(e))
            raise
    
    def validate_session(self, session_token: str, 
                        ip_address: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        """Validate user session"""
        errors = []
        
        try:
            if not session_token:
                errors.append("Session token required")
                return None, errors
            
            # Check Redis first (fast lookup)
            session_key = f"session:{session_token}"
            session_data = self.redis_client.get(session_key)
            
            if not session_data:
                errors.append("Invalid or expired session")
                return None, errors
            
            session_data = json.loads(session_data)
            
            # Verify IP address (basic session hijacking protection)
            if session_data.get('ip_address') != ip_address:
                logger.warning("Session IP mismatch", 
                             original_ip=session_data.get('ip_address'),
                             current_ip=ip_address,
                             user_id=session_data.get('user_id'))
                
                # In production, you might want to invalidate the session
                # For now, just log and continue
            
            # Detect anomalous behavior
            user_id = session_data.get('user_id')
            if user_id:
                anomalies = self.threat_detection.detect_anomalous_behavior(
                    user_id, {'ip_address': ip_address}
                )
                
                if anomalies:
                    self._log_security_event(SecurityEvent(
                        event_type="anomalous_behavior",
                        severity="medium",
                        user_id=user_id,
                        ip_address=ip_address,
                        user_agent=request.headers.get('User-Agent', ''),
                        details={'anomalies': anomalies},
                        timestamp=datetime.utcnow()
                    ))
            
            return session_data, []
            
        except Exception as e:
            logger.error("Session validation error", error=str(e))
            errors.append("Session validation failed")
            return None, errors
    
    def invalidate_session(self, session_token: str):
        """Invalidate user session"""
        try:
            # Remove from Redis
            session_key = f"session:{session_token}"
            self.redis_client.delete(session_key)
            
            # Update database record
            session_hash = hashlib.sha256(session_token.encode()).hexdigest()
            session = UserSession.query.filter_by(
                session_token=session_hash,
                is_active=True
            ).first()
            
            if session:
                session.is_active = False
                session.logged_out_at = datetime.utcnow()
                db.session.commit()
                
        except Exception as e:
            logger.error("Error invalidating session", error=str(e))
    
    def change_password(self, user: User, current_password: str, 
                       new_password: str) -> Tuple[bool, List[str]]:
        """Change user password with security validation"""
        errors = []
        
        try:
            # Verify current password
            if not check_password_hash(user.password_hash, current_password):
                errors.append("Current password is incorrect")
                return False, errors
            
            # Validate new password
            is_valid, validation_errors = SecurityValidation.validate_password(new_password)
            if not is_valid:
                errors.extend(validation_errors)
                return False, errors
            
            # Check password history (prevent reuse)
            if self._is_password_reused(user, new_password):
                errors.append("Cannot reuse recent passwords")
                return False, errors
            
            # Update password
            user.password_hash = generate_password_hash(new_password)
            user.password_changed_at = datetime.utcnow()
            
            # Store password history
            self._store_password_history(user, current_password)
            
            db.session.commit()
            
            # Log password change
            self._log_security_event(SecurityEvent(
                event_type="password_changed",
                severity="low",
                user_id=user.id,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent', ''),
                details={},
                timestamp=datetime.utcnow()
            ))
            
            return True, []
            
        except Exception as e:
            logger.error("Password change error", error=str(e))
            errors.append("Password change failed")
            return False, errors
    
    def _is_password_expired(self, user: User) -> bool:
        """Check if user password has expired"""
        if not user.password_changed_at:
            return False
        
        expiry_date = user.password_changed_at + timedelta(days=self.config['password_expiry_days'])
        return datetime.utcnow() > expiry_date
    
    def _is_password_reused(self, user: User, new_password: str) -> bool:
        """Check if password was recently used"""
        try:
            # Get password history from Redis
            history_key = f"password_history:{user.id}"
            history = self.redis_client.lrange(history_key, 0, 4)  # Last 5 passwords
            
            for old_password_hash in history:
                if check_password_hash(old_password_hash.decode(), new_password):
                    return True
            
            return False
            
        except Exception as e:
            logger.error("Error checking password history", error=str(e))
            return False
    
    def _store_password_history(self, user: User, old_password: str):
        """Store password in history"""
        try:
            history_key = f"password_history:{user.id}"
            password_hash = generate_password_hash(old_password)
            
            # Add to history
            self.redis_client.lpush(history_key, password_hash)
            
            # Keep only last 5 passwords
            self.redis_client.ltrim(history_key, 0, 4)
            
            # Set expiry
            self.redis_client.expire(history_key, 86400 * 365)  # 1 year
            
        except Exception as e:
            logger.error("Error storing password history", error=str(e))
    
    def _log_security_event(self, event: SecurityEvent):
        """Log security event"""
        self.threat_detection._log_security_event(event)
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics and statistics"""
        try:
            # Get recent security events
            event_keys = self.redis_client.keys('security_event:*')
            events = []
            
            for key in event_keys:
                event_data = self.redis_client.get(key)
                if event_data:
                    events.append(json.loads(event_data))
            
            # Calculate metrics
            total_events = len(events)
            failed_logins = len([e for e in events if e.get('event_type') == 'failed_login'])
            blocked_ips = len([e for e in events if e.get('event_type') == 'ip_blocked'])
            anomalies = len([e for e in events if e.get('event_type') == 'anomalous_behavior'])
            
            return {
                'total_security_events': total_events,
                'failed_login_attempts': failed_logins,
                'blocked_ip_addresses': blocked_ips,
                'anomalous_behaviors': anomalies,
                'active_sessions': len(self.redis_client.keys('session:*')),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Error getting security metrics", error=str(e))
            return {'error': 'Failed to get security metrics'}


# Global security manager instance
security_manager = SecurityManager()


def init_security(app):
    """Initialize security for Flask app"""
    
    @app.before_request
    def security_check():
        """Perform security checks on each request"""
        
        # Skip security checks for health and metrics endpoints
        if request.endpoint in ['health', 'metrics']:
            return
        
        # Check if IP is blocked
        ip_address = request.remote_addr
        if security_manager.threat_detection.is_ip_blocked(ip_address):
            return jsonify({'error': 'Access denied'}), 403
        
        # Validate input for POST/PUT requests
        if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
            try:
                data = request.get_json()
                if data:
                    for key, value in data.items():
                        if isinstance(value, str):
                            is_valid, cleaned = SecurityValidation.validate_input_sanitization(value)
                            if not is_valid:
                                return jsonify({'error': f'Invalid input in field: {key}'}), 400
            except Exception:
                pass  # Continue if JSON parsing fails
    
    @app.after_request
    def security_headers(response):
        """Add security headers to all responses"""
        
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'
        
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Strict Transport Security (HTTPS only)
        if request.is_secure:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Content Security Policy
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' https:; "
            "connect-src 'self' wss: ws:;"
        )
        
        # Referrer Policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Permissions Policy
        response.headers['Permissions-Policy'] = (
            'geolocation=(), microphone=(), camera=()'
        )
        
        return response


def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        
        token = auth_header.split(' ')[1]
        session_data, errors = security_manager.validate_session(token, request.remote_addr)
        
        if errors:
            return jsonify({'error': errors[0]}), 401
        
        g.current_user_id = session_data.get('user_id')
        g.current_username = session_data.get('username')
        
        return f(*args, **kwargs)
    
    return decorated_function


def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # This would check user permissions
            # Implementation depends on your permission system
            return f(*args, **kwargs)
        return decorated_function
    return decorator


if __name__ == "__main__":
    # Test security features
    sm = SecurityManager()
    
    # Test password validation
    test_passwords = [
        "weak",
        "password123",
        "StrongPassword123!",
        "VerySecurePassword2024!"
    ]
    
    for pwd in test_passwords:
        is_valid, errors = SecurityValidation.validate_password(pwd)
        print(f"Password '{pwd}': {'Valid' if is_valid else 'Invalid'}")
        if errors:
            for error in errors:
                print(f"  - {error}")
        print()