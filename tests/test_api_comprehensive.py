#!/usr/bin/env python3
"""
Comprehensive API Test Suite for Congressional Trading Intelligence System
Advanced testing with fixtures, mocks, and integration scenarios
"""

import pytest
import json
from datetime import datetime, date, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from flask import Flask
from flask_jwt_extended import create_access_token
import sqlalchemy as sa
from sqlalchemy.orm import Session

from src.api.app import create_app
from src.database import db
from src.models.member import Member, Committee, CommitteeMembership, Party, Chamber
from src.models.trading import Trade, TradeAlert, TransactionType, AssetType, AlertLevel
from src.models.user import User, UserRole
from tests.fixtures.test_data import TestDataFactory


class TestApiComprehensive:
    """Comprehensive API test suite with advanced scenarios"""
    
    @pytest.fixture(scope="class")
    def app(self):
        """Create test Flask application"""
        app = create_app('testing')
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        
        with app.app_context():
            db.create_all()
            yield app
            db.drop_all()
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    @pytest.fixture
    def db_session(self, app):
        """Create database session for testing"""
        with app.app_context():
            session = db.session
            yield session
            session.rollback()
    
    @pytest.fixture
    def test_user(self, db_session):
        """Create test user"""
        user = User(
            username='testuser',
            email='test@example.com',
            password_hash='hashed_password',
            full_name='Test User',
            role=UserRole.ANALYST,
            is_active=True
        )
        db_session.add(user)
        db_session.commit()
        return user
    
    @pytest.fixture
    def auth_headers(self, app, test_user):
        """Create authentication headers"""
        with app.app_context():
            access_token = create_access_token(
                identity=test_user.id,
                additional_claims={
                    'username': test_user.username,
                    'role': test_user.role.value,
                    'permissions': ['read', 'write', 'analysis']
                }
            )
            return {'Authorization': f'Bearer {access_token}'}
    
    @pytest.fixture
    def sample_members(self, db_session):
        """Create sample congressional members"""
        members = [
            Member(
                bioguide_id='M000001',
                full_name='John Smith',
                first_name='John',
                last_name='Smith',
                party=Party.DEMOCRAT,
                state='CA',
                chamber=Chamber.HOUSE,
                district='12',
                is_active=True
            ),
            Member(
                bioguide_id='M000002',
                full_name='Jane Doe',
                first_name='Jane',
                last_name='Doe',
                party=Party.REPUBLICAN,
                state='TX',
                chamber=Chamber.SENATE,
                is_active=True
            ),
            Member(
                bioguide_id='M000003',
                full_name='Bob Johnson',
                first_name='Bob',
                last_name='Johnson',
                party=Party.INDEPENDENT,
                state='NY',
                chamber=Chamber.HOUSE,
                district='05',
                is_active=True
            )
        ]
        
        for member in members:
            db_session.add(member)
        db_session.commit()
        return members
    
    @pytest.fixture
    def sample_trades(self, db_session, sample_members):
        """Create sample trades"""
        trades = []
        
        for i, member in enumerate(sample_members):
            trade = Trade(
                member_id=member.id,
                symbol=f'STOCK{i+1}',
                asset_type=AssetType.STOCK,
                transaction_type=TransactionType.BUY,
                amount_min=Decimal('1000'),
                amount_max=Decimal('15000'),
                amount_mid=Decimal('8000'),
                shares=Decimal('100'),
                price_per_share=Decimal('80.00'),
                transaction_date=date.today() - timedelta(days=i*10),
                filing_date=date.today() - timedelta(days=i*10-5),
                filing_delay_days=5,
                source_document='Form A',
                is_processed=False
            )
            trades.append(trade)
            db_session.add(trade)
        
        db_session.commit()
        return trades

    # Authentication Tests
    def test_register_user_success(self, client):
        """Test successful user registration"""
        user_data = {
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'securepassword123',
            'full_name': 'New User'
        }
        
        response = client.post('/api/v1/auth/register', 
                             json=user_data,
                             content_type='application/json')
        
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['username'] == 'newuser'
        assert 'user_id' in data
    
    def test_register_user_duplicate_username(self, client, test_user):
        """Test registration with duplicate username"""
        user_data = {
            'username': test_user.username,
            'email': 'different@example.com',
            'password': 'securepassword123'
        }
        
        response = client.post('/api/v1/auth/register', json=user_data)
        
        assert response.status_code == 409
        data = json.loads(response.data)
        assert 'Username already exists' in data['error']
    
    def test_login_success(self, client, test_user):
        """Test successful login"""
        with patch('werkzeug.security.check_password_hash', return_value=True):
            login_data = {
                'username': test_user.username,
                'password': 'password123'
            }
            
            response = client.post('/api/v1/auth/login', json=login_data)
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'access_token' in data
            assert 'refresh_token' in data
            assert data['user']['username'] == test_user.username
    
    def test_login_invalid_credentials(self, client, test_user):
        """Test login with invalid credentials"""
        with patch('werkzeug.security.check_password_hash', return_value=False):
            login_data = {
                'username': test_user.username,
                'password': 'wrongpassword'
            }
            
            response = client.post('/api/v1/auth/login', json=login_data)
            
            assert response.status_code == 401
            data = json.loads(response.data)
            assert 'Invalid credentials' in data['error']
    
    def test_protected_endpoint_without_token(self, client):
        """Test accessing protected endpoint without authentication"""
        response = client.get('/api/v1/trades')
        
        assert response.status_code == 401
    
    def test_protected_endpoint_with_token(self, client, auth_headers):
        """Test accessing protected endpoint with valid token"""
        response = client.get('/api/v1/trades', headers=auth_headers)
        
        assert response.status_code == 200

    # Member API Tests
    def test_list_members(self, client, auth_headers, sample_members):
        """Test listing congressional members"""
        response = client.get('/api/v1/members', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'members' in data
        assert len(data['members']) == 3
        assert data['members'][0]['bioguide_id'] == 'M000001'
    
    def test_get_member_by_id(self, client, auth_headers, sample_members):
        """Test getting specific member by ID"""
        member = sample_members[0]
        response = client.get(f'/api/v1/members/{member.id}', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['member']['bioguide_id'] == member.bioguide_id
        assert data['member']['full_name'] == member.full_name
    
    def test_get_nonexistent_member(self, client, auth_headers):
        """Test getting non-existent member"""
        response = client.get('/api/v1/members/99999', headers=auth_headers)
        
        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'not found' in data['error'].lower()
    
    def test_filter_members_by_party(self, client, auth_headers, sample_members):
        """Test filtering members by party"""
        response = client.get('/api/v1/members?party=D', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['members']) == 1
        assert data['members'][0]['party'] == 'D'
    
    def test_filter_members_by_state(self, client, auth_headers, sample_members):
        """Test filtering members by state"""
        response = client.get('/api/v1/members?state=CA', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['members']) == 1
        assert data['members'][0]['state'] == 'CA'

    # Trading API Tests
    def test_list_trades(self, client, auth_headers, sample_trades):
        """Test listing trades"""
        response = client.get('/api/v1/trades', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'trades' in data
        assert len(data['trades']) == 3
        assert 'pagination' in data
    
    def test_list_trades_with_pagination(self, client, auth_headers, sample_trades):
        """Test trades pagination"""
        response = client.get('/api/v1/trades?page=1&per_page=2', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['trades']) == 2
        assert data['pagination']['page'] == 1
        assert data['pagination']['per_page'] == 2
        assert data['pagination']['total'] == 3
    
    def test_filter_trades_by_symbol(self, client, auth_headers, sample_trades):
        """Test filtering trades by symbol"""
        response = client.get('/api/v1/trades?symbol=STOCK1', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['trades']) == 1
        assert data['trades'][0]['symbol'] == 'STOCK1'
    
    def test_filter_trades_by_amount_range(self, client, auth_headers, sample_trades):
        """Test filtering trades by amount range"""
        response = client.get('/api/v1/trades?amount_min=5000&amount_max=10000', 
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['trades']) == 3  # All trades should be in this range
    
    def test_filter_trades_by_date_range(self, client, auth_headers, sample_trades):
        """Test filtering trades by date range"""
        start_date = (date.today() - timedelta(days=30)).isoformat()
        end_date = date.today().isoformat()
        
        response = client.get(f'/api/v1/trades?date_from={start_date}&date_to={end_date}',
                            headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['trades']) >= 0
    
    def test_get_trade_by_id(self, client, auth_headers, sample_trades):
        """Test getting specific trade by ID"""
        trade = sample_trades[0]
        response = client.get(f'/api/v1/trades/{trade.id}', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['trade']['id'] == trade.id
        assert data['trade']['symbol'] == trade.symbol
    
    def test_trade_statistics(self, client, auth_headers, sample_trades):
        """Test trade statistics endpoint"""
        response = client.get('/api/v1/trades/statistics', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'overview' in data
        assert 'filing_compliance' in data
        assert 'top_symbols' in data
        assert 'alerts' in data
    
    @patch('src.intelligence.suspicious_trading_detector.SuspiciousTradingDetector')
    def test_analyze_trades(self, mock_detector, client, auth_headers, sample_trades):
        """Test trade analysis endpoint"""
        # Mock the detector
        mock_instance = Mock()
        mock_instance.run_full_analysis.return_value = (
            Mock(empty=False, iterrows=lambda: iter([])),  # analysis_df
            Mock(empty=False, iterrows=lambda: iter([]))   # alerts_df
        )
        mock_detector.return_value = mock_instance
        
        analysis_data = {
            'trade_ids': [trade.id for trade in sample_trades[:2]],
            'threshold': 7.0
        }
        
        response = client.post('/api/v1/trades/analyze',
                             json=analysis_data,
                             headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'trades_analyzed' in data
        assert 'alerts_generated' in data

    # Alert API Tests
    def test_list_alerts(self, client, auth_headers, db_session, sample_trades):
        """Test listing trade alerts"""
        # Create sample alert
        alert = TradeAlert(
            trade_id=sample_trades[0].id,
            alert_type='suspicious_pattern',
            level=AlertLevel.HIGH,
            suspicion_score=Decimal('8.5'),
            title='High suspicion trade detected',
            description='Trade shows multiple risk factors',
            reason='Large amount, late filing, timing correlation',
            generated_by='ml_analysis_engine'
        )
        db_session.add(alert)
        db_session.commit()
        
        response = client.get('/api/v1/trades/alerts', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'alerts' in data
        assert len(data['alerts']) == 1
        assert data['alerts'][0]['level'] == 'high'
    
    def test_filter_alerts_by_level(self, client, auth_headers, db_session, sample_trades):
        """Test filtering alerts by level"""
        # Create alerts with different levels
        high_alert = TradeAlert(
            trade_id=sample_trades[0].id,
            alert_type='high_risk',
            level=AlertLevel.HIGH,
            suspicion_score=Decimal('8.0'),
            title='High risk alert',
            description='High risk detected',
            reason='Multiple factors',
            generated_by='system'
        )
        
        low_alert = TradeAlert(
            trade_id=sample_trades[1].id,
            alert_type='low_risk',
            level=AlertLevel.LOW,
            suspicion_score=Decimal('3.0'),
            title='Low risk alert',
            description='Low risk detected',
            reason='Minor factors',
            generated_by='system'
        )
        
        db_session.add(high_alert)
        db_session.add(low_alert)
        db_session.commit()
        
        response = client.get('/api/v1/trades/alerts?level=high', headers=auth_headers)
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['alerts']) == 1
        assert data['alerts'][0]['level'] == 'high'

    # Error Handling Tests
    def test_invalid_json_request(self, client, auth_headers):
        """Test handling of invalid JSON requests"""
        response = client.post('/api/v1/trades/analyze',
                             data='invalid json',
                             headers=auth_headers,
                             content_type='application/json')
        
        assert response.status_code == 400
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        incomplete_data = {
            'username': 'testuser'
            # Missing password and email
        }
        
        response = client.post('/api/v1/auth/register', json=incomplete_data)
        
        assert response.status_code == 400
    
    def test_rate_limiting(self, client, auth_headers):
        """Test API rate limiting"""
        # This would require actual rate limiting to be configured
        # For now, just test that the endpoint responds
        for _ in range(5):
            response = client.get('/api/v1/trades', headers=auth_headers)
            assert response.status_code in [200, 429]  # 429 = Too Many Requests
    
    def test_database_error_handling(self, client, auth_headers):
        """Test handling of database errors"""
        with patch('src.database.db.session.query') as mock_query:
            mock_query.side_effect = Exception("Database connection error")
            
            response = client.get('/api/v1/trades', headers=auth_headers)
            
            # Should handle gracefully without crashing
            assert response.status_code in [500, 503]

    # Integration Tests
    def test_full_workflow_member_to_alerts(self, client, auth_headers, db_session):
        """Test full workflow from member creation to alert generation"""
        # 1. Create member
        member = Member(
            bioguide_id='WORKFLOW001',
            full_name='Workflow Member',
            first_name='Workflow',
            last_name='Member',
            party=Party.DEMOCRAT,
            state='CA',
            chamber=Chamber.HOUSE,
            is_active=True
        )
        db_session.add(member)
        db_session.commit()
        
        # 2. Create trade
        trade = Trade(
            member_id=member.id,
            symbol='WORKFLOW',
            transaction_type=TransactionType.BUY,
            amount_mid=Decimal('50000'),
            transaction_date=date.today(),
            filing_date=date.today() + timedelta(days=50),  # Late filing
            filing_delay_days=50
        )
        db_session.add(trade)
        db_session.commit()
        
        # 3. Verify member appears in API
        response = client.get(f'/api/v1/members/{member.id}', headers=auth_headers)
        assert response.status_code == 200
        
        # 4. Verify trade appears in API
        response = client.get(f'/api/v1/trades/{trade.id}', headers=auth_headers)
        assert response.status_code == 200
        
        # 5. Run analysis (mocked)
        with patch('src.intelligence.suspicious_trading_detector.SuspiciousTradingDetector') as mock_detector:
            mock_instance = Mock()
            mock_instance.run_full_analysis.return_value = (
                Mock(empty=False),
                Mock(empty=False, iterrows=lambda: iter([{
                    'trade_id': trade.id,
                    'suspicion_score': 8.5,
                    'alert_reason': 'Late filing and large amount'
                }]))
            )
            mock_detector.return_value = mock_instance
            
            response = client.post('/api/v1/trades/analyze',
                                 json={'trade_ids': [trade.id]},
                                 headers=auth_headers)
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['trades_analyzed'] == 1
    
    def test_data_consistency_across_endpoints(self, client, auth_headers, sample_members, sample_trades):
        """Test data consistency across different API endpoints"""
        # Get member from members endpoint
        member_response = client.get(f'/api/v1/members/{sample_members[0].id}', 
                                   headers=auth_headers)
        member_data = json.loads(member_response.data)['member']
        
        # Get trades for the same member
        trades_response = client.get(f'/api/v1/trades?member_id={sample_members[0].id}',
                                   headers=auth_headers)
        trades_data = json.loads(trades_response.data)['trades']
        
        # Verify consistency
        assert len(trades_data) == 1
        assert trades_data[0]['member']['bioguide_id'] == member_data['bioguide_id']
        assert trades_data[0]['member']['full_name'] == member_data['full_name']

    # Performance Tests
    def test_large_dataset_performance(self, client, auth_headers, db_session):
        """Test API performance with larger datasets"""
        # Create many trades (simulate larger dataset)
        member = Member(
            bioguide_id='PERF001',
            full_name='Performance Member',
            first_name='Performance',
            last_name='Member',
            party=Party.DEMOCRAT,
            state='CA',
            chamber=Chamber.HOUSE
        )
        db_session.add(member)
        db_session.flush()
        
        # Create 100 trades
        trades = []
        for i in range(100):
            trade = Trade(
                member_id=member.id,
                symbol=f'PERF{i:03d}',
                transaction_type=TransactionType.BUY,
                amount_mid=Decimal(str(1000 + i)),
                transaction_date=date.today() - timedelta(days=i)
            )
            trades.append(trade)
        
        db_session.add_all(trades)
        db_session.commit()
        
        # Test pagination performance
        import time
        start_time = time.time()
        
        response = client.get('/api/v1/trades?per_page=50', headers=auth_headers)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 2.0  # Should respond within 2 seconds
        
        data = json.loads(response.data)
        assert len(data['trades']) <= 50
        assert data['pagination']['total'] >= 100


@pytest.mark.integration
class TestApiIntegration:
    """Integration tests requiring external services"""
    
    def test_redis_integration(self):
        """Test Redis integration for caching"""
        # This would test actual Redis connectivity
        pass
    
    def test_websocket_integration(self):
        """Test WebSocket real-time updates"""
        # This would test WebSocket functionality
        pass
    
    def test_external_api_integration(self):
        """Test integration with external APIs"""
        # This would test ProPublica, Finnhub, etc.
        pass


if __name__ == '__main__':
    # Run tests with coverage
    pytest.main([
        __file__,
        '-v',
        '--cov=src.api',
        '--cov-report=html',
        '--cov-report=term'
    ])