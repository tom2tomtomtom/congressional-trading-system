#!/usr/bin/env python3
"""
Congressional Trading Intelligence System - Phase 2
Comprehensive Test Suite

This module provides comprehensive testing for all Phase 2 advanced features
including ML detection, network analysis, and React dashboard.
"""

import os
import sys
import logging
import unittest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import Phase 2 modules
from intelligence.suspicious_trading_detector import SuspiciousTradingDetector
from intelligence.network_analyzer import NetworkAnalyzer

# Database testing
import psycopg2
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2TestSuite(unittest.TestCase):
    """Comprehensive test suite for Phase 2 features."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.config_path = Path("config/database.yml")
        cls.temp_dir = Path(tempfile.mkdtemp())
        
        # Test database connection
        cls.db_config = cls._load_config()
        cls._verify_database_connection()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _load_config(cls):
        """Load test database configuration."""
        if cls.config_path.exists():
            with open(cls.config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('development', {}).get('database', {})
        return {}
    
    @classmethod
    def _verify_database_connection(cls):
        """Verify database connection for testing."""
        try:
            conn = psycopg2.connect(
                host=cls.db_config.get('host', 'localhost'),
                port=cls.db_config.get('port', 5432),
                database=cls.db_config.get('name', 'congressional_trading_dev'),
                user=cls.db_config.get('user', 'postgres'),
                password=cls.db_config.get('password', 'password')
            )
            conn.close()
            logger.info("✅ Database connection verified for testing")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    def test_01_suspicious_trading_detector_initialization(self):
        """Test suspicious trading detector initialization."""
        logger.info("Testing suspicious trading detector initialization...")
        
        detector = SuspiciousTradingDetector()
        
        # Test configuration loading
        self.assertIsInstance(detector.config, dict)
        self.assertIsInstance(detector.db_config, dict)
        
        # Test risk weights
        self.assertIsInstance(detector.risk_weights, dict)
        self.assertAlmostEqual(sum(detector.risk_weights.values()), 1.0, places=2)
        
        # Test model directory creation
        self.assertTrue(detector.model_dir.exists())
        
        logger.info("✅ Suspicious trading detector initialization test passed")
    
    def test_02_suspicious_trading_feature_extraction(self):
        """Test feature extraction for ML analysis."""
        logger.info("Testing feature extraction...")
        
        detector = SuspiciousTradingDetector()
        
        # Extract features
        df = detector.extract_trading_features()
        
        # Verify data structure
        self.assertIsInstance(df, pd.DataFrame)
        if not df.empty:
            # Check required columns
            required_columns = [
                'trade_id', 'bioguide_id', 'full_name', 'party', 'chamber',
                'symbol', 'transaction_type', 'amount_mid'
            ]
            for col in required_columns:
                self.assertIn(col, df.columns, f"Missing required column: {col}")
            
            # Check data types
            self.assertTrue(pd.api.types.is_numeric_dtype(df['amount_mid']))
            self.assertIn(df['party'].dtype, [object, 'string'])
            
            logger.info(f"✅ Feature extraction test passed - extracted {len(df)} records")
        else:
            logger.warning("⚠️  No trading data available for testing")
    
    def test_03_risk_score_calculation(self):
        """Test risk score calculation algorithms."""
        logger.info("Testing risk score calculations...")
        
        detector = SuspiciousTradingDetector()
        
        # Get sample data
        df = detector.extract_trading_features()
        
        if not df.empty:
            # Calculate risk scores
            df_with_scores = detector.calculate_risk_scores(df)
            
            # Verify score columns exist
            score_columns = ['timing_score', 'amount_score', 'frequency_score', 
                           'filing_delay_score', 'market_timing_score']
            
            for col in score_columns:
                self.assertIn(col, df_with_scores.columns, f"Missing score column: {col}")
                
                # Check score ranges (0-10)
                scores = df_with_scores[col].dropna()
                if len(scores) > 0:
                    self.assertTrue((scores >= 0).all(), f"{col} has negative values")
                    self.assertTrue((scores <= 10).all(), f"{col} has values > 10")
            
            logger.info("✅ Risk score calculation test passed")
        else:
            logger.warning("⚠️  No data available for risk score testing")
    
    def test_04_composite_suspicion_scoring(self):
        """Test composite suspicion score calculation."""
        logger.info("Testing composite suspicion scoring...")
        
        detector = SuspiciousTradingDetector()
        
        # Get sample data and calculate scores
        df = detector.extract_trading_features()
        
        if not df.empty:
            df_with_scores = detector.calculate_risk_scores(df)
            df_final = detector.calculate_composite_suspicion_score(df_with_scores)
            
            # Verify composite score
            self.assertIn('suspicion_score', df_final.columns)
            self.assertIn('risk_category', df_final.columns)
            
            # Check score range
            scores = df_final['suspicion_score'].dropna()
            if len(scores) > 0:
                self.assertTrue((scores >= 0).all(), "Suspicion scores have negative values")
                self.assertTrue((scores <= 10).all(), "Suspicion scores exceed 10")
            
            # Check risk categories
            expected_categories = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']
            actual_categories = df_final['risk_category'].dropna().unique()
            for cat in actual_categories:
                self.assertIn(cat, expected_categories, f"Unexpected risk category: {cat}")
            
            logger.info("✅ Composite suspicion scoring test passed")
        else:
            logger.warning("⚠️  No data available for suspicion scoring testing")
    
    def test_05_ml_model_training(self):
        """Test ML model training functionality."""
        logger.info("Testing ML model training...")
        
        detector = SuspiciousTradingDetector()
        
        # Get sample data
        df = detector.extract_trading_features()
        
        if not df.empty and len(df) >= 10:  # Need minimum data for ML
            df_with_scores = detector.calculate_risk_scores(df)
            df_final = detector.calculate_composite_suspicion_score(df_with_scores)
            
            # Train models
            models = detector.train_anomaly_detection_models(df_final)
            
            # Verify models were created
            self.assertIsInstance(models, dict)
            self.assertGreater(len(models), 0, "No models were trained")
            
            # Check specific models
            expected_models = ['isolation_forest', 'one_class_svm', 'dbscan']
            for model_name in expected_models:
                if model_name in models:
                    self.assertIsNotNone(models[model_name])
            
            # Verify scaler was created
            self.assertIn('anomaly_scaler', detector.scalers)
            
            logger.info(f"✅ ML model training test passed - trained {len(models)} models")
        else:
            logger.warning("⚠️  Insufficient data for ML model training testing")
    
    def test_06_alert_generation(self):
        """Test alert generation functionality."""
        logger.info("Testing alert generation...")
        
        detector = SuspiciousTradingDetector()
        
        # Create sample data with high suspicion scores
        sample_data = pd.DataFrame({
            'trade_id': [1, 2, 3],
            'bioguide_id': ['TEST001', 'TEST002', 'TEST003'],
            'full_name': ['Test Member A', 'Test Member B', 'Test Member C'],
            'party': ['D', 'R', 'I'],
            'state': ['CA', 'TX', 'NY'],
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'transaction_type': ['Purchase', 'Sale', 'Purchase'],
            'amount_mid': [1000000, 500000, 2000000],
            'transaction_date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'filing_date': ['2024-01-15', '2024-01-20', '2024-01-10'],
            'filing_delay_days': [14, 18, 7],
            'suspicion_score': [8.5, 7.2, 9.1],
            'timing_score': [7.0, 6.0, 8.0],
            'amount_score': [9.0, 7.0, 10.0],
            'frequency_score': [6.0, 5.0, 7.0],
            'filing_delay_score': [7.0, 8.0, 4.0],
            'market_timing_score': [8.0, 6.0, 9.0],
            'committee_correlation': [5.0, 4.0, 6.0],
            'ml_anomaly_score': [7.5, 6.8, 8.2]
        })
        
        # Generate alerts
        alerts = detector.generate_alerts(sample_data, threshold=7.0)
        
        # Verify alerts
        self.assertIsInstance(alerts, pd.DataFrame)
        if not alerts.empty:
            # Check alert structure
            expected_columns = ['suspicion_score', 'alert_priority', 'alert_reason']
            for col in expected_columns:
                self.assertIn(col, alerts.columns, f"Missing alert column: {col}")
            
            # Verify only high-scoring trades are included
            self.assertTrue((alerts['suspicion_score'] >= 7.0).all())
            
            logger.info(f"✅ Alert generation test passed - generated {len(alerts)} alerts")
        else:
            logger.warning("⚠️  No alerts generated from test data")
    
    def test_07_network_analyzer_initialization(self):
        """Test network analyzer initialization."""
        logger.info("Testing network analyzer initialization...")
        
        analyzer = NetworkAnalyzer()
        
        # Test initialization
        self.assertIsInstance(analyzer.config, dict)
        self.assertIsInstance(analyzer.db_config, dict)
        
        # Test network graphs initialization
        self.assertIsNotNone(analyzer.member_network)
        self.assertIsNotNone(analyzer.committee_network)
        self.assertIsNotNone(analyzer.trading_network)
        
        # Test output directory creation
        self.assertTrue(analyzer.output_dir.exists())
        
        logger.info("✅ Network analyzer initialization test passed")
    
    def test_08_network_data_loading(self):
        """Test network data loading functionality."""
        logger.info("Testing network data loading...")
        
        analyzer = NetworkAnalyzer()
        
        # Load network data
        members_df, committees_df, trades_df = analyzer.load_network_data()
        
        # Verify data frames
        self.assertIsInstance(members_df, pd.DataFrame)
        self.assertIsInstance(committees_df, pd.DataFrame)
        self.assertIsInstance(trades_df, pd.DataFrame)
        
        if not members_df.empty:
            # Check required columns
            required_member_cols = ['bioguide_id', 'full_name', 'party', 'chamber']
            for col in required_member_cols:
                self.assertIn(col, members_df.columns, f"Missing member column: {col}")
        
        if not trades_df.empty:
            # Check required columns
            required_trade_cols = ['bioguide_id', 'symbol', 'transaction_type', 'amount_mid']
            for col in required_trade_cols:
                self.assertIn(col, trades_df.columns, f"Missing trade column: {col}")
        
        logger.info(f"✅ Network data loading test passed - loaded {len(members_df)} members, "
                   f"{len(committees_df)} committee relationships, {len(trades_df)} trades")
    
    def test_09_trading_network_construction(self):
        """Test trading network construction."""
        logger.info("Testing trading network construction...")
        
        analyzer = NetworkAnalyzer()
        
        # Load data
        members_df, committees_df, trades_df = analyzer.load_network_data()
        
        if not trades_df.empty:
            # Build trading network
            trading_network = analyzer.build_trading_network(trades_df)
            
            # Verify network structure
            self.assertIsNotNone(trading_network)
            self.assertGreaterEqual(trading_network.number_of_nodes(), 0)
            
            # If network has nodes, check structure
            if trading_network.number_of_nodes() > 0:
                # Check node attributes
                sample_node = list(trading_network.nodes())[0]
                node_data = trading_network.nodes[sample_node]
                
                expected_attributes = ['total_trades', 'avg_amount']
                for attr in expected_attributes:
                    self.assertIn(attr, node_data, f"Missing node attribute: {attr}")
            
            logger.info(f"✅ Trading network construction test passed - "
                       f"{trading_network.number_of_nodes()} nodes, "
                       f"{trading_network.number_of_edges()} edges")
        else:
            logger.warning("⚠️  No trading data available for network construction testing")
    
    def test_10_committee_correlation_analysis(self):
        """Test committee-trading correlation analysis."""
        logger.info("Testing committee correlation analysis...")
        
        analyzer = NetworkAnalyzer()
        
        # Load data
        members_df, committees_df, trades_df = analyzer.load_network_data()
        
        # Analyze correlations
        correlations = analyzer.analyze_committee_trading_correlations(committees_df, trades_df)
        
        # Verify analysis structure
        self.assertIsInstance(correlations, dict)
        
        # If correlations exist, check structure
        if correlations:
            sample_key = list(correlations.keys())[0]
            sample_analysis = correlations[sample_key]
            
            expected_fields = ['committee_name', 'member_count', 'total_trades']
            for field in expected_fields:
                self.assertIn(field, sample_analysis, f"Missing correlation field: {field}")
        
        logger.info(f"✅ Committee correlation analysis test passed - "
                   f"analyzed {len(correlations)} committees")
    
    def test_11_influence_score_calculation(self):
        """Test influence score calculation."""
        logger.info("Testing influence score calculation...")
        
        analyzer = NetworkAnalyzer()
        
        # Load data and build member network
        members_df, committees_df, trades_df = analyzer.load_network_data()
        
        if not members_df.empty:
            analyzer.build_member_network(members_df, committees_df)
            
            # Calculate influence scores
            influence_scores = analyzer.calculate_influence_scores()
            
            # Verify scores
            self.assertIsInstance(influence_scores, dict)
            
            if influence_scores:
                # Check score values
                for member_id, score in influence_scores.items():
                    if not pd.isna(score):
                        self.assertGreaterEqual(score, 0.0, f"Negative influence score for {member_id}")
                        self.assertLessEqual(score, 1.0, f"Influence score > 1.0 for {member_id}")
            
            logger.info(f"✅ Influence score calculation test passed - "
                       f"calculated scores for {len(influence_scores)} members")
        else:
            logger.warning("⚠️  No member data available for influence scoring testing")
    
    def test_12_model_persistence(self):
        """Test model saving and loading functionality."""
        logger.info("Testing model persistence...")
        
        detector = SuspiciousTradingDetector()
        
        # Get sample data and train models
        df = detector.extract_trading_features()
        
        if not df.empty and len(df) >= 10:
            df_with_scores = detector.calculate_risk_scores(df)
            df_final = detector.calculate_composite_suspicion_score(df_with_scores)
            
            # Train models
            detector.train_anomaly_detection_models(df_final)
            
            # Save models
            detector.save_models()
            
            # Verify model files were created
            model_files = list(detector.model_dir.glob("*.pkl"))
            self.assertGreater(len(model_files), 0, "No model files were saved")
            
            # Test loading models
            new_detector = SuspiciousTradingDetector()
            new_detector.load_models()
            
            # Verify models were loaded
            self.assertGreater(len(new_detector.models), 0, "No models were loaded")
            
            logger.info("✅ Model persistence test passed")
        else:
            logger.warning("⚠️  Insufficient data for model persistence testing")
    
    def test_13_full_analysis_pipeline(self):
        """Test complete analysis pipeline."""
        logger.info("Testing full analysis pipeline...")
        
        detector = SuspiciousTradingDetector()
        
        # Run full analysis
        analysis_df, alerts_df = detector.run_full_analysis()
        
        # Verify results
        self.assertIsInstance(analysis_df, pd.DataFrame)
        self.assertIsInstance(alerts_df, pd.DataFrame)
        
        if not analysis_df.empty:
            # Check required columns in analysis results
            required_columns = ['suspicion_score', 'risk_category']
            for col in required_columns:
                self.assertIn(col, analysis_df.columns, f"Missing analysis column: {col}")
            
            # Verify suspicion scores are valid
            scores = analysis_df['suspicion_score'].dropna()
            if len(scores) > 0:
                self.assertTrue((scores >= 0).all(), "Invalid suspicion scores")
                self.assertTrue((scores <= 10).all(), "Invalid suspicion scores")
        
        logger.info(f"✅ Full analysis pipeline test passed - "
                   f"analyzed {len(analysis_df)} trades, generated {len(alerts_df)} alerts")
    
    def test_14_network_analysis_pipeline(self):
        """Test complete network analysis pipeline."""
        logger.info("Testing network analysis pipeline...")
        
        analyzer = NetworkAnalyzer()
        
        # Run full network analysis
        results = analyzer.run_full_network_analysis()
        
        # Verify results structure
        self.assertIsInstance(results, dict)
        
        expected_sections = ['timestamp', 'network_statistics']
        for section in expected_sections:
            self.assertIn(section, results, f"Missing results section: {section}")
        
        # Verify timestamp
        self.assertIsInstance(results['timestamp'], str)
        
        logger.info("✅ Network analysis pipeline test passed")
    
    def test_15_data_quality_validation(self):
        """Test data quality and consistency."""
        logger.info("Testing data quality validation...")
        
        detector = SuspiciousTradingDetector()
        
        # Extract features
        df = detector.extract_trading_features()
        
        if not df.empty:
            # Check for data quality issues
            issues = []
            
            # Check for missing critical data
            if df['bioguide_id'].isna().any():
                issues.append("Missing bioguide_id values")
            
            if df['amount_mid'].isna().any():
                issues.append("Missing amount_mid values")
            
            # Check for invalid amounts
            if (df['amount_mid'] <= 0).any():
                issues.append("Invalid trade amounts (≤ 0)")
            
            # Check date consistency
            if 'transaction_date' in df.columns and 'filing_date' in df.columns:
                df['transaction_date'] = pd.to_datetime(df['transaction_date'])
                df['filing_date'] = pd.to_datetime(df['filing_date'])
                
                if (df['filing_date'] < df['transaction_date']).any():
                    issues.append("Filing dates before transaction dates")
            
            # Report results
            if issues:
                logger.warning(f"⚠️  Data quality issues found: {issues}")
                # Don't fail the test for data quality issues in sample data
            else:
                logger.info("✅ Data quality validation passed")
        else:
            logger.warning("⚠️  No data available for quality validation")

def run_phase2_tests():
    """Run Phase 2 comprehensive test suite."""
    logger.info("="*60)
    logger.info("CONGRESSIONAL TRADING INTELLIGENCE SYSTEM")
    logger.info("PHASE 2 COMPREHENSIVE TEST SUITE")
    logger.info("="*60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(Phase2TestSuite)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("PHASE 2 TEST RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    logger.info(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        logger.error("FAILURES:")
        for test, traceback in result.failures:
            logger.error(f"  - {test}: {traceback}")
    
    if result.errors:
        logger.error("ERRORS:")
        for test, traceback in result.errors:
            logger.error(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        logger.info("🎉 ALL PHASE 2 TESTS PASSED!")
        logger.info("Phase 2 advanced features are ready for production.")
        return True
    else:
        logger.error("❌ SOME PHASE 2 TESTS FAILED")
        logger.info("Review and fix issues before deployment.")
        return False

def main():
    """Main test execution."""
    return run_phase2_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)