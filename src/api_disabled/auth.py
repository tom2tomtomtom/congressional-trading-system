"""
Authentication and authorization endpoints
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import (
    create_access_token, create_refresh_token, jwt_required,
    get_jwt_identity, get_jwt, verify_jwt_in_request
)
from werkzeug.security import check_password_hash, generate_password_hash
import structlog

from src.database import db
from src.models.user import User, UserSession, APIKey
from .validators import validate_json, validate_fields
from .decorators import limiter_decorator

logger = structlog.get_logger()

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
@limiter_decorator("5 per minute")
@validate_json
@validate_fields(['username', 'email', 'password'])
def register():
    """
    Register a new user account
    
    Required fields:
    - username: Unique username
    - email: Valid email address
    - password: Minimum 8 characters
    """
    data = request.get_json()
    
    # Validate input
    if len(data['password']) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already exists'}), 409
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    try:
        # Create new user
        user = User(
            username=data['username'],
            email=data['email'],
            password_hash=generate_password_hash(data['password']),
            full_name=data.get('full_name'),
            organization=data.get('organization'),
            is_active=True
        )
        
        db.session.add(user)
        db.session.commit()
        
        logger.info("User registered successfully", 
                   user_id=user.id, username=user.username)
        
        return jsonify({
            'message': 'User registered successfully',
            'user_id': user.id,
            'username': user.username
        }), 201
        
    except Exception as e:
        db.session.rollback()
        logger.error("User registration failed", error=str(e))
        return jsonify({'error': 'Registration failed'}), 500


@auth_bp.route('/login', methods=['POST'])
@limiter_decorator("10 per minute")
@validate_json
@validate_fields(['username', 'password'])
def login():
    """
    Authenticate user and return JWT tokens
    
    Required fields:
    - username: Username or email
    - password: User password
    """
    data = request.get_json()
    
    # Find user by username or email
    user = User.query.filter(
        (User.username == data['username']) | 
        (User.email == data['username'])
    ).first()
    
    if not user or not user.is_active:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not check_password_hash(user.password_hash, data['password']):
        # Update failed login attempts
        user.failed_login_attempts += 1
        user.last_failed_login = datetime.utcnow()
        
        # Lock account after 5 failed attempts
        if user.failed_login_attempts >= 5:
            user.is_active = False
            logger.warning("User account locked due to failed login attempts",
                         user_id=user.id, username=user.username)
        
        db.session.commit()
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Reset failed login attempts on successful login
    user.failed_login_attempts = 0
    user.last_login = datetime.utcnow()
    user.login_count += 1
    
    # Create JWT tokens
    access_token = create_access_token(
        identity=user.id,
        additional_claims={
            'username': user.username,
            'role': user.role.value,
            'permissions': user.get_permissions()
        }
    )
    
    refresh_token = create_refresh_token(identity=user.id)
    
    # Create user session record
    session = UserSession(
        user_id=user.id,
        access_token_jti=get_jwt()['jti'] if hasattr(get_jwt, '__call__') else None,
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent', ''),
        expires_at=datetime.utcnow() + timedelta(seconds=current_app.config['JWT_ACCESS_TOKEN_EXPIRES'])
    )
    
    db.session.add(session)
    db.session.commit()
    
    logger.info("User logged in successfully",
               user_id=user.id, username=user.username, ip=request.remote_addr)
    
    return jsonify({
        'message': 'Login successful',
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
            'role': user.role.value,
            'permissions': user.get_permissions()
        }
    }), 200


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token using refresh token"""
    current_user_id = get_jwt_identity()
    
    user = User.query.get(current_user_id)
    if not user or not user.is_active:
        return jsonify({'error': 'User not found or inactive'}), 404
    
    # Create new access token
    access_token = create_access_token(
        identity=user.id,
        additional_claims={
            'username': user.username,
            'role': user.role.value,
            'permissions': user.get_permissions()
        }
    )
    
    return jsonify({
        'access_token': access_token
    }), 200


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """Logout user and invalidate session"""
    current_user_id = get_jwt_identity()
    jti = get_jwt()['jti']
    
    # Find and deactivate session
    session = UserSession.query.filter_by(
        user_id=current_user_id,
        access_token_jti=jti,
        is_active=True
    ).first()
    
    if session:
        session.is_active = False
        session.logged_out_at = datetime.utcnow()
        db.session.commit()
    
    logger.info("User logged out successfully", user_id=current_user_id)
    
    return jsonify({'message': 'Logout successful'}), 200


@auth_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get current user profile"""
    current_user_id = get_jwt_identity()
    
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'full_name': user.full_name,
            'organization': user.organization,
            'role': user.role.value,
            'is_active': user.is_active,
            'created_at': user.created_at.isoformat(),
            'last_login': user.last_login.isoformat() if user.last_login else None,
            'login_count': user.login_count,
            'permissions': user.get_permissions()
        }
    }), 200


@auth_bp.route('/profile', methods=['PUT'])
@jwt_required()
@validate_json
def update_profile():
    """Update current user profile"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Update allowed fields
    allowed_fields = ['full_name', 'organization']
    for field in allowed_fields:
        if field in data:
            setattr(user, field, data[field])
    
    db.session.commit()
    
    logger.info("User profile updated", user_id=user.id)
    
    return jsonify({'message': 'Profile updated successfully'}), 200


@auth_bp.route('/change-password', methods=['PUT'])
@jwt_required()
@limiter_decorator("3 per minute")
@validate_json
@validate_fields(['current_password', 'new_password'])
def change_password():
    """Change user password"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    if len(data['new_password']) < 8:
        return jsonify({'error': 'New password must be at least 8 characters'}), 400
    
    user = User.query.get(current_user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if not check_password_hash(user.password_hash, data['current_password']):
        return jsonify({'error': 'Current password is incorrect'}), 401
    
    # Update password
    user.password_hash = generate_password_hash(data['new_password'])
    user.password_changed_at = datetime.utcnow()
    
    db.session.commit()
    
    logger.info("User password changed", user_id=user.id)
    
    return jsonify({'message': 'Password changed successfully'}), 200


@auth_bp.route('/api-keys', methods=['GET'])
@jwt_required()
def list_api_keys():
    """List user's API keys"""
    current_user_id = get_jwt_identity()
    
    keys = APIKey.query.filter_by(
        user_id=current_user_id,
        is_active=True
    ).all()
    
    return jsonify({
        'api_keys': [{
            'id': key.id,
            'name': key.name,
            'key_prefix': key.key_hash[:8] + '...',  # Show only prefix
            'permissions': key.permissions,
            'last_used': key.last_used.isoformat() if key.last_used else None,
            'created_at': key.created_at.isoformat(),
            'expires_at': key.expires_at.isoformat() if key.expires_at else None
        } for key in keys]
    }), 200


@auth_bp.route('/api-keys', methods=['POST'])
@jwt_required()
@limiter_decorator("5 per hour")
@validate_json
@validate_fields(['name'])
def create_api_key():
    """Create new API key"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    # Generate API key
    import secrets
    api_key = secrets.token_urlsafe(32)
    
    # Create API key record
    key_record = APIKey(
        user_id=current_user_id,
        name=data['name'],
        key_hash=generate_password_hash(api_key),
        permissions=data.get('permissions', ['read']),
        expires_at=datetime.utcnow() + timedelta(days=365) if data.get('expires_days') else None
    )
    
    db.session.add(key_record)
    db.session.commit()
    
    logger.info("API key created", user_id=current_user_id, key_id=key_record.id)
    
    return jsonify({
        'message': 'API key created successfully',
        'api_key': api_key,  # Only returned once
        'key_id': key_record.id,
        'warning': 'Store this API key securely. It will not be shown again.'
    }), 201


@auth_bp.route('/api-keys/<int:key_id>', methods=['DELETE'])
@jwt_required()
def revoke_api_key(key_id: int):
    """Revoke API key"""
    current_user_id = get_jwt_identity()
    
    key = APIKey.query.filter_by(
        id=key_id,
        user_id=current_user_id
    ).first()
    
    if not key:
        return jsonify({'error': 'API key not found'}), 404
    
    key.is_active = False
    key.revoked_at = datetime.utcnow()
    
    db.session.commit()
    
    logger.info("API key revoked", user_id=current_user_id, key_id=key_id)
    
    return jsonify({'message': 'API key revoked successfully'}), 200


@auth_bp.route('/sessions', methods=['GET'])
@jwt_required()
def list_sessions():
    """List user's active sessions"""
    current_user_id = get_jwt_identity()
    
    sessions = UserSession.query.filter_by(
        user_id=current_user_id,
        is_active=True
    ).order_by(UserSession.created_at.desc()).limit(10).all()
    
    return jsonify({
        'sessions': [{
            'id': session.id,
            'ip_address': session.ip_address,
            'user_agent': session.user_agent,
            'created_at': session.created_at.isoformat(),
            'last_activity': session.last_activity.isoformat() if session.last_activity else None,
            'expires_at': session.expires_at.isoformat()
        } for session in sessions]
    }), 200