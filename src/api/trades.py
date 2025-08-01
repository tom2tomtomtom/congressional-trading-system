"""
Trading data and analysis endpoints
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import and_, or_, desc, asc, func, text
from sqlalchemy.orm import joinedload
import structlog

from src.database import db
from src.models.trading import Trade, TradeAlert, TradingPattern, TransactionType, AssetType
from src.models.member import Member
from src.intelligence.suspicious_trading_detector import SuspiciousTradingDetector
from .validators import validate_json, validate_fields, validate_date_range
from .decorators import limiter_decorator, cache_response, require_permission
from .serializers import TradeSerializer, TradeAlertSerializer

logger = structlog.get_logger()

trades_bp = Blueprint('trades', __name__)
trade_serializer = TradeSerializer()
alert_serializer = TradeAlertSerializer()


@trades_bp.route('', methods=['GET'])
@jwt_required()
@limiter_decorator("100 per minute")
@cache_response(timeout=300)  # 5 minutes
def list_trades():
    """
    List trades with filtering and pagination
    
    Query Parameters:
    - page: Page number (default: 1)
    - per_page: Items per page (default: 20, max: 100)
    - member_id: Filter by member ID
    - bioguide_id: Filter by bioguide ID
    - symbol: Filter by stock symbol
    - transaction_type: Filter by transaction type (purchase, sale, exchange)
    - asset_type: Filter by asset type
    - date_from: Start date (YYYY-MM-DD)
    - date_to: End date (YYYY-MM-DD)
    - amount_min: Minimum trade amount
    - amount_max: Maximum trade amount
    - sort: Sort field (transaction_date, amount_mid, filing_delay_days)
    - order: Sort order (asc, desc)
    """
    
    # Pagination parameters
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    
    # Build query
    query = Trade.query.options(
        joinedload(Trade.member)
    )
    
    # Apply filters
    if member_id := request.args.get('member_id', type=int):
        query = query.filter(Trade.member_id == member_id)
    
    if bioguide_id := request.args.get('bioguide_id'):
        query = query.join(Member).filter(Member.bioguide_id == bioguide_id)
    
    if symbol := request.args.get('symbol'):
        query = query.filter(Trade.symbol.ilike(f'%{symbol}%'))
    
    if transaction_type := request.args.get('transaction_type'):
        try:
            trans_type = TransactionType(transaction_type)
            query = query.filter(Trade.transaction_type == trans_type)
        except ValueError:
            return jsonify({'error': 'Invalid transaction type'}), 400
    
    if asset_type := request.args.get('asset_type'):
        try:
            asset_enum = AssetType(asset_type)
            query = query.filter(Trade.asset_type == asset_enum)
        except ValueError:
            return jsonify({'error': 'Invalid asset type'}), 400
    
    # Date range filtering
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    
    if date_from:
        try:
            from_date = datetime.strptime(date_from, '%Y-%m-%d').date()
            query = query.filter(Trade.transaction_date >= from_date)
        except ValueError:
            return jsonify({'error': 'Invalid date_from format. Use YYYY-MM-DD'}), 400
    
    if date_to:
        try:
            to_date = datetime.strptime(date_to, '%Y-%m-%d').date()
            query = query.filter(Trade.transaction_date <= to_date)
        except ValueError:
            return jsonify({'error': 'Invalid date_to format. Use YYYY-MM-DD'}), 400
    
    # Amount filtering
    if amount_min := request.args.get('amount_min', type=float):
        query = query.filter(Trade.amount_mid >= amount_min)
    
    if amount_max := request.args.get('amount_max', type=float):
        query = query.filter(Trade.amount_mid <= amount_max)
    
    # Sorting
    sort_field = request.args.get('sort', 'transaction_date')
    sort_order = request.args.get('order', 'desc')
    
    valid_sort_fields = ['transaction_date', 'amount_mid', 'filing_delay_days', 'created_at']
    if sort_field not in valid_sort_fields:
        return jsonify({'error': f'Invalid sort field. Valid options: {valid_sort_fields}'}), 400
    
    sort_column = getattr(Trade, sort_field)
    if sort_order == 'asc':
        query = query.order_by(asc(sort_column))
    else:
        query = query.order_by(desc(sort_column))
    
    # Execute query with pagination
    try:
        paginated = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        # Serialize results
        trades_data = []
        for trade in paginated.items:
            trade_dict = trade_serializer.serialize(trade)
            trades_data.append(trade_dict)
        
        return jsonify({
            'trades': trades_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': paginated.total,
                'pages': paginated.pages,
                'has_next': paginated.has_next,
                'has_prev': paginated.has_prev
            },
            'filters_applied': {
                'member_id': member_id,
                'bioguide_id': bioguide_id,
                'symbol': symbol,
                'transaction_type': transaction_type,
                'asset_type': asset_type,
                'date_from': date_from,
                'date_to': date_to,
                'amount_min': amount_min,
                'amount_max': amount_max
            }
        }), 200
        
    except Exception as e:
        logger.error("Error fetching trades", error=str(e))
        return jsonify({'error': 'Failed to fetch trades'}), 500


@trades_bp.route('/<int:trade_id>', methods=['GET'])
@jwt_required()
@limiter_decorator("200 per minute")
@cache_response(timeout=600)  # 10 minutes
def get_trade(trade_id: int):
    """Get specific trade by ID with detailed information"""
    
    trade = Trade.query.options(
        joinedload(Trade.member),
        joinedload(Trade.alerts),
        joinedload(Trade.patterns)
    ).get(trade_id)
    
    if not trade:
        return jsonify({'error': 'Trade not found'}), 404
    
    # Serialize trade with full details
    trade_data = trade_serializer.serialize(trade, include_relations=True)
    
    return jsonify({'trade': trade_data}), 200


@trades_bp.route('/analyze', methods=['POST'])
@jwt_required()
@require_permission('analysis')
@limiter_decorator("10 per minute")
@validate_json
def analyze_trades():
    """
    Run suspicious trading analysis on specified trades or date range
    
    Request body:
    - trade_ids: List of trade IDs to analyze (optional)
    - date_from: Start date for analysis (optional)
    - date_to: End date for analysis (optional)
    - member_ids: List of member IDs to analyze (optional)
    - threshold: Suspicion score threshold for alerts (default: 7.0)
    """
    data = request.get_json()
    
    try:
        # Initialize detector
        detector = SuspiciousTradingDetector()
        
        # Build trade query based on parameters
        query = Trade.query
        
        if trade_ids := data.get('trade_ids'):
            query = query.filter(Trade.id.in_(trade_ids))
        
        if member_ids := data.get('member_ids'):
            query = query.filter(Trade.member_id.in_(member_ids))
        
        if date_from := data.get('date_from'):
            from_date = datetime.strptime(date_from, '%Y-%m-%d').date()
            query = query.filter(Trade.transaction_date >= from_date)
        
        if date_to := data.get('date_to'):
            to_date = datetime.strptime(date_to, '%Y-%m-%d').date()
            query = query.filter(Trade.transaction_date <= to_date)
        
        # Get trades to analyze
        trades = query.all()
        
        if not trades:
            return jsonify({'error': 'No trades found for analysis'}), 404
        
        # Run analysis
        analysis_df, alerts_df = detector.run_full_analysis()
        
        if analysis_df.empty:
            return jsonify({
                'message': 'Analysis completed',
                'trades_analyzed': 0,
                'alerts_generated': 0
            }), 200
        
        # Store alerts in database
        threshold = data.get('threshold', 7.0)
        alerts_created = 0
        
        for _, alert_row in alerts_df.iterrows():
            if alert_row['suspicion_score'] >= threshold:
                # Find corresponding trade
                trade = Trade.query.filter_by(id=alert_row['trade_id']).first()
                if trade:
                    # Create alert record
                    alert = TradeAlert(
                        trade_id=trade.id,
                        alert_type='suspicious_pattern',
                        level=_determine_alert_level(alert_row['suspicion_score']),
                        suspicion_score=alert_row['suspicion_score'],
                        title=f"Suspicious trading pattern detected",
                        description=f"Trade flagged by ML analysis with score {alert_row['suspicion_score']:.2f}",
                        reason=alert_row.get('alert_reason', 'Multiple risk factors detected'),
                        generated_by='ml_analysis_engine'
                    )
                    
                    db.session.add(alert)
                    alerts_created += 1
        
        db.session.commit()
        
        # Update trade processing status
        Trade.query.filter(Trade.id.in_([t.id for t in trades])).update({
            'is_processed': True,
            'processing_date': datetime.utcnow()
        }, synchronize_session=False)
        
        db.session.commit()
        
        logger.info("Trade analysis completed",
                   trades_analyzed=len(trades),
                   alerts_generated=alerts_created,
                   threshold=threshold)
        
        return jsonify({
            'message': 'Analysis completed successfully',
            'trades_analyzed': len(trades),
            'alerts_generated': alerts_created,
            'average_suspicion_score': float(analysis_df['suspicion_score'].mean()),
            'high_risk_trades': len(analysis_df[analysis_df['suspicion_score'] >= 7.0]),
            'extreme_risk_trades': len(analysis_df[analysis_df['suspicion_score'] >= 9.0])
        }), 200
        
    except Exception as e:
        logger.error("Trade analysis failed", error=str(e))
        db.session.rollback()
        return jsonify({'error': 'Analysis failed'}), 500


@trades_bp.route('/alerts', methods=['GET'])
@jwt_required()
@limiter_decorator("100 per minute")
@cache_response(timeout=120)  # 2 minutes
def list_alerts():
    """
    List trade alerts with filtering
    
    Query Parameters:
    - page: Page number
    - per_page: Items per page
    - level: Alert level (low, medium, high, critical, extreme)
    - status: Alert status (pending, reviewed, escalated, resolved)
    - member_id: Filter by member
    - date_from: Start date
    - date_to: End date
    - min_score: Minimum suspicion score
    """
    
    page = request.args.get('page', 1, type=int)
    per_page = min(request.args.get('per_page', 20, type=int), 100)
    
    # Build query
    query = TradeAlert.query.options(
        joinedload(TradeAlert.trade).joinedload(Trade.member)
    )
    
    # Apply filters
    if level := request.args.get('level'):
        query = query.filter(TradeAlert.level == level)
    
    if status := request.args.get('status'):
        query = query.filter(TradeAlert.status == status)
    
    if member_id := request.args.get('member_id', type=int):
        query = query.join(Trade).filter(Trade.member_id == member_id)
    
    if min_score := request.args.get('min_score', type=float):
        query = query.filter(TradeAlert.suspicion_score >= min_score)
    
    # Date filtering
    if date_from := request.args.get('date_from'):
        from_date = datetime.strptime(date_from, '%Y-%m-%d')
        query = query.filter(TradeAlert.created_at >= from_date)
    
    if date_to := request.args.get('date_to'):
        to_date = datetime.strptime(date_to, '%Y-%m-%d')
        query = query.filter(TradeAlert.created_at <= to_date)
    
    # Sort by creation date (newest first)
    query = query.order_by(desc(TradeAlert.created_at))
    
    # Paginate
    paginated = query.paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )
    
    # Serialize alerts
    alerts_data = []
    for alert in paginated.items:
        alert_dict = alert_serializer.serialize(alert)
        alerts_data.append(alert_dict)
    
    return jsonify({
        'alerts': alerts_data,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': paginated.total,
            'pages': paginated.pages,
            'has_next': paginated.has_next,
            'has_prev': paginated.has_prev
        }
    }), 200


@trades_bp.route('/statistics', methods=['GET'])
@jwt_required()
@limiter_decorator("30 per minute")
@cache_response(timeout=900)  # 15 minutes
def get_trade_statistics():
    """Get trading statistics and summary metrics"""
    
    try:
        # Basic trade statistics
        total_trades = db.session.query(func.count(Trade.id)).scalar()
        
        # Trade volume statistics
        volume_stats = db.session.query(
            func.sum(Trade.amount_mid).label('total_volume'),
            func.avg(Trade.amount_mid).label('avg_amount'),
            func.max(Trade.amount_mid).label('max_amount'),
            func.min(Trade.amount_mid).label('min_amount')
        ).filter(Trade.amount_mid.isnot(None)).first()
        
        # Recent activity (last 30 days)
        thirty_days_ago = datetime.now().date() - timedelta(days=30)
        recent_trades = db.session.query(func.count(Trade.id)).filter(
            Trade.transaction_date >= thirty_days_ago
        ).scalar()
        
        # Top traded symbols
        top_symbols = db.session.query(
            Trade.symbol,
            func.count(Trade.id).label('trade_count'),
            func.sum(Trade.amount_mid).label('total_volume')
        ).filter(
            Trade.amount_mid.isnot(None)
        ).group_by(Trade.symbol).order_by(
            desc('total_volume')
        ).limit(10).all()
        
        # Transaction type distribution
        transaction_types = db.session.query(
            Trade.transaction_type,
            func.count(Trade.id).label('count')
        ).group_by(Trade.transaction_type).all()
        
        # Filing delay statistics
        filing_stats = db.session.query(
            func.avg(Trade.filing_delay_days).label('avg_delay'),
            func.max(Trade.filing_delay_days).label('max_delay'),
            func.count(func.nullif(Trade.filing_delay_days > 45, False)).label('late_filings')
        ).filter(Trade.filing_delay_days.isnot(None)).first()
        
        # Alert statistics
        alert_stats = db.session.query(
            func.count(TradeAlert.id).label('total_alerts'),
            func.count(func.nullif(TradeAlert.status == 'pending', False)).label('pending_alerts'),
            func.avg(TradeAlert.suspicion_score).label('avg_suspicion_score')
        ).first()
        
        return jsonify({
            'overview': {
                'total_trades': total_trades,
                'recent_trades_30d': recent_trades,
                'total_volume': float(volume_stats.total_volume or 0),
                'average_trade_amount': float(volume_stats.avg_amount or 0),
                'largest_trade': float(volume_stats.max_amount or 0)
            },
            'filing_compliance': {
                'average_filing_delay_days': float(filing_stats.avg_delay or 0),
                'maximum_filing_delay_days': int(filing_stats.max_delay or 0),
                'late_filings_count': int(filing_stats.late_filings or 0),
                'late_filing_rate': float((filing_stats.late_filings or 0) / max(total_trades, 1) * 100)
            },
            'top_symbols': [
                {
                    'symbol': symbol,
                    'trade_count': int(count),
                    'total_volume': float(volume)
                }
                for symbol, count, volume in top_symbols
            ],
            'transaction_distribution': [
                {
                    'type': trans_type.value,
                    'count': int(count)
                }
                for trans_type, count in transaction_types
            ],
            'alerts': {
                'total_alerts': int(alert_stats.total_alerts or 0),
                'pending_alerts': int(alert_stats.pending_alerts or 0),
                'average_suspicion_score': float(alert_stats.avg_suspicion_score or 0)
            }
        }), 200
        
    except Exception as e:
        logger.error("Error generating trade statistics", error=str(e))
        return jsonify({'error': 'Failed to generate statistics'}), 500


def _determine_alert_level(suspicion_score: float) -> str:
    """Determine alert level based on suspicion score"""
    if suspicion_score >= 9.0:
        return 'extreme'
    elif suspicion_score >= 8.0:
        return 'critical'
    elif suspicion_score >= 7.0:
        return 'high'
    elif suspicion_score >= 5.0:
        return 'medium'
    else:
        return 'low'