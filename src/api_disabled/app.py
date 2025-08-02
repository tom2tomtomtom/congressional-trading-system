"""
Main Flask application factory for the Congressional Trading Intelligence API
"""

import os
from typing import Optional

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
import structlog
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from src.database import db, init_db
from src.models import *
from .auth import auth_bp
from .members import members_bp
from .trades import trades_bp
from .analysis import analysis_bp
from .alerts import alerts_bp
from .admin import admin_bp
from .middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from .exceptions import register_error_handlers


# Configure structured logging
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


def create_app(config_name: Optional[str] = None) -> Flask:
    """
    Application factory pattern for creating Flask app
    
    Args:
        config_name: Configuration environment (development, testing, production)
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    # Load configuration
    config_name = config_name or os.getenv('FLASK_ENV', 'development')
    configure_app(app, config_name)
    
    # Initialize extensions
    initialize_extensions(app)
    
    # Register blueprints
    register_blueprints(app)
    
    # Setup middleware
    setup_middleware(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Setup logging
    setup_logging(app)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Simple health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'congressional-trading-api',
            'version': '2.0.0'
        })
    
    # API info endpoint
    @app.route('/api/v1/info')
    def api_info():
        """API information endpoint"""
        return jsonify({
            'name': 'Congressional Trading Intelligence API',
            'version': '2.0.0',
            'description': 'Advanced REST API for congressional trading analysis',
            'endpoints': {
                'auth': '/api/v1/auth',
                'members': '/api/v1/members',
                'trades': '/api/v1/trades',
                'analysis': '/api/v1/analysis',
                'alerts': '/api/v1/alerts',
                'admin': '/api/v1/admin'
            },
            'documentation': '/api/v1/docs'
        })
    
    return app


def configure_app(app: Flask, config_name: str) -> None:
    """Configure Flask application"""
    
    # Basic configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-change-in-production')
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-change-in-production')
    
    # Database configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'DATABASE_URL',
        'postgresql://postgres:password@localhost:5432/congressional_trading_dev'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
        'pool_size': 10,
        'max_overflow': 20
    }
    
    # Redis configuration
    app.config['REDIS_URL'] = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # Rate limiting
    app.config['RATELIMIT_STORAGE_URL'] = os.getenv(
        'RATE_LIMIT_STORAGE_URL',
        'redis://localhost:6379/1'
    )
    app.config['RATELIMIT_DEFAULT'] = os.getenv('DEFAULT_RATE_LIMIT', '100 per hour')
    
    # CORS configuration
    app.config['CORS_ORIGINS'] = os.getenv(
        'CORS_ORIGINS',
        'http://localhost:3000,http://localhost:8080'
    ).split(',')
    
    # Caching
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_URL'] = app.config['REDIS_URL']
    app.config['CACHE_DEFAULT_TIMEOUT'] = int(os.getenv('CACHE_TTL', '300'))
    
    # JWT configuration
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # 1 hour
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = 86400 * 30  # 30 days
    
    # Request limits
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', '16777216'))  # 16MB
    
    # Environment specific settings
    if config_name == 'production':
        app.config['DEBUG'] = False
        app.config['TESTING'] = False
        # Setup Sentry for production
        sentry_dsn = os.getenv('SENTRY_DSN')
        if sentry_dsn:
            sentry_sdk.init(
                dsn=sentry_dsn,
                integrations=[
                    FlaskIntegration(),
                    SqlalchemyIntegration(),
                ],
                traces_sample_rate=0.1,
                environment='production'
            )
    elif config_name == 'testing':
        app.config['DEBUG'] = False
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    else:  # development
        app.config['DEBUG'] = True
        app.config['TESTING'] = False


def initialize_extensions(app: Flask) -> None:
    """Initialize Flask extensions"""
    
    # Database
    db.init_app(app)
    
    # JWT
    jwt = JWTManager(app)
    
    # CORS
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Rate limiting  
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=[app.config['RATELIMIT_DEFAULT']]
    )
    
    # Caching
    cache = Cache(app)
    
    # Store extensions in app for access in blueprints
    app.extensions['limiter'] = limiter
    app.extensions['cache'] = cache
    app.extensions['jwt'] = jwt


def register_blueprints(app: Flask) -> None:
    """Register API blueprints"""
    
    # API version prefix
    api_prefix = '/api/v1'
    
    app.register_blueprint(auth_bp, url_prefix=f'{api_prefix}/auth')
    app.register_blueprint(members_bp, url_prefix=f'{api_prefix}/members')
    app.register_blueprint(trades_bp, url_prefix=f'{api_prefix}/trades')
    app.register_blueprint(analysis_bp, url_prefix=f'{api_prefix}/analysis')
    app.register_blueprint(alerts_bp, url_prefix=f'{api_prefix}/alerts')
    app.register_blueprint(admin_bp, url_prefix=f'{api_prefix}/admin')


def setup_middleware(app: Flask) -> None:
    """Setup custom middleware"""
    
    # Request logging
    app.wsgi_app = RequestLoggingMiddleware(app.wsgi_app)
    
    # Security headers
    app.wsgi_app = SecurityHeadersMiddleware(app.wsgi_app)


def setup_logging(app: Flask) -> None:
    """Setup application logging"""
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Configure Flask's logger
    app.logger.setLevel(log_level)
    
    if not app.debug:
        # Production logging setup
        import logging
        from logging.handlers import RotatingFileHandler
        
        # File handler
        if not os.path.exists('logs'):
            os.mkdir('logs')
        
        file_handler = RotatingFileHandler(
            'logs/congressional_trading_api.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Congressional Trading API startup')


def main():
    """Main entry point for running the API server"""
    app = create_app()
    
    # Initialize database in application context
    with app.app_context():
        init_db()
    
    # Run development server
    port = int(os.getenv('PORT', 5000))
    host = os.getenv('HOST', '0.0.0.0')
    
    logger.info("Starting Congressional Trading Intelligence API", 
                port=port, host=host, debug=app.debug)
    
    app.run(host=host, port=port, debug=app.debug)


if __name__ == '__main__':
    main()