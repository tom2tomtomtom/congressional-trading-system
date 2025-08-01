# Phase 2 ML Models - Technical Specification

**Component**: Machine Learning Models  
**Phase**: 2 - Intelligence & Analytics  
**Status**: Implemented  
**Version**: 2.0  

## Overview

This specification details the machine learning models implemented in Phase 2 of the Congressional Trading Intelligence System. These models transform raw congressional trading data into predictive intelligence, anomaly detection, and pattern recognition capabilities.

## Model Architecture

### 1. Trade Prediction Model

#### Purpose
Predict the likelihood of congressional members making trades based on committee activities, news sentiment, historical patterns, and market conditions.

#### Technical Implementation
**File**: `src/ml_models/trade_predictor.py`

**Algorithm Stack**:
- **Primary**: XGBoost ensemble classifier
- **Secondary**: Random Forest classifier
- **Ensemble**: Voting classifier combining both models
- **Fallback**: Logistic regression for interpretability

**Model Class Structure**:
```python
class TradePredictionModel:
    def __init__(self, model_name: str = "trade_predictor_v1"):
        self.models = {
            'xgboost': XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2
            )
        }
        self.ensemble = VotingClassifier(
            estimators=list(self.models.items()),
            voting='soft'
        )
```

#### Feature Engineering
**Total Features**: 47 engineered features across 5 categories

1. **Committee Features (12 features)**:
   - Committee membership overlap with stock sectors
   - Committee leadership positions
   - Upcoming hearing schedules
   - Bill markup activities
   - Jurisdiction relevance scores

2. **Market Features (15 features)**:
   - Stock volatility (1d, 5d, 20d rolling)
   - Sector performance relative to market
   - Options volume and open interest
   - Earnings announcement proximity
   - Market regime indicators

3. **Temporal Features (8 features)**:
   - Days since last trade
   - Trade frequency patterns
   - Seasonal trading patterns
   - Congressional calendar alignment
   - Market timing indicators

4. **Historical Features (7 features)**:
   - Member trading frequency
   - Preferred stock sectors
   - Average trade size
   - Hold duration patterns
   - Performance tracking

5. **News Sentiment Features (5 features)**:
   - Member-specific news sentiment
   - Sector sentiment scores
   - News volume indicators
   - Sentiment momentum
   - Controversy detection

#### Training Process
```python
def train_model(self, training_data: pd.DataFrame) -> ModelPerformance:
    # Feature engineering pipeline
    features = self.feature_engineer(training_data)
    
    # Train-test split with temporal validation
    X_train, X_test, y_train, y_test = self.temporal_split(features)
    
    # Hyperparameter optimization
    best_params = self.optimize_hyperparameters(X_train, y_train)
    
    # Model training with cross-validation
    self.ensemble.fit(X_train, y_train)
    
    # Performance evaluation
    predictions = self.ensemble.predict_proba(X_test)
    performance = self.evaluate_performance(y_test, predictions)
    
    return performance
```

#### Performance Targets
- **Accuracy**: ≥87% on test set
- **Precision**: ≥85% for high-confidence predictions (>0.8 probability)
- **Recall**: ≥80% for actual trading events
- **AUC-ROC**: ≥0.90 for binary classification
- **Calibration**: Well-calibrated probabilities for confidence scoring

### 2. Anomaly Detection System

#### Purpose
Identify suspicious trading patterns that may indicate insider trading, coordination, or ethical violations using unsupervised learning techniques.

#### Technical Implementation
**File**: `src/ml_models/anomaly_detector.py`

**Multi-Algorithm Approach**:
```python
class SuspiciousPatternDetector:
    def __init__(self):
        self.timing_detector = TimingAnomalyDetector()
        self.volume_detector = VolumeAnomalyDetector()
        self.coordination_detector = CoordinationDetector()
        self.pattern_detector = PatternAnomalyDetector()
```

#### Algorithm Details

##### 2.1 Timing Anomaly Detection
**Algorithm**: Isolation Forest + Statistical outlier detection
**Purpose**: Detect trades with suspicious timing relative to committee activities

```python
class TimingAnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            n_estimators=200,
            max_samples=0.8
        )
        self.lstm_model = self._build_lstm_temporal_model()
    
    def detect_anomalies(self, trading_data: pd.DataFrame) -> pd.DataFrame:
        # Calculate timing features
        timing_features = self.extract_timing_features(trading_data)
        
        # Isolation forest detection
        anomaly_scores = self.isolation_forest.decision_function(timing_features)
        
        # LSTM temporal pattern detection
        temporal_scores = self.lstm_model.predict(timing_features)
        
        # Combined scoring
        combined_scores = self.combine_scores(anomaly_scores, temporal_scores)
        
        return self.format_results(trading_data, combined_scores)
```

**Key Timing Features**:
- Days between committee meeting and trade
- Trade timing relative to bill markup
- Hours before/after hearing announcements
- Weekend/holiday trading patterns
- After-hours trading frequency

##### 2.2 Volume Anomaly Detection
**Algorithm**: Statistical process control + Machine learning outlier detection
**Purpose**: Identify unusually large trades relative to member's historical patterns

```python
class VolumeAnomalyDetector:
    def __init__(self):
        self.scaler = RobustScaler()
        self.detector = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.15
        )
    
    def detect_volume_anomalies(self, member_trades: pd.DataFrame) -> pd.DataFrame:
        # Calculate volume metrics
        volume_features = self.calculate_volume_features(member_trades)
        
        # Statistical control limits (3-sigma rule)
        control_limits = self.calculate_control_limits(volume_features)
        
        # Machine learning anomaly detection
        anomaly_scores = self.detector.fit_predict(volume_features)
        
        # Combine statistical and ML approaches
        final_scores = self.combine_detection_methods(
            control_limits, anomaly_scores
        )
        
        return final_scores
```

##### 2.3 Coordination Detection
**Algorithm**: Network analysis + Time series clustering
**Purpose**: Detect multiple members trading the same stocks within suspicious timeframes

```python
class CoordinationDetector:
    def __init__(self):
        self.network_analyzer = NetworkX()
        self.time_clusterer = TimeSeriesKMeans(n_clusters=5)
    
    def detect_coordination(self, all_trades: pd.DataFrame) -> pd.DataFrame:
        # Build temporal trading networks
        trading_network = self.build_temporal_network(all_trades)
        
        # Identify suspicious clusters
        suspicious_clusters = self.find_coordinated_clusters(trading_network)
        
        # Calculate coordination scores
        coordination_scores = self.score_coordination_strength(suspicious_clusters)
        
        return coordination_scores
```

#### Scoring System
**Composite Suspicion Score**: 0-100 scale combining all anomaly types

```python
def calculate_composite_score(self, anomalies: dict) -> float:
    weights = {
        'timing': 0.35,      # Highest weight for timing anomalies
        'volume': 0.25,      # Volume spikes important
        'coordination': 0.25, # Cross-member coordination
        'pattern': 0.15      # Historical pattern deviations
    }
    
    composite_score = sum(
        weights[type_] * score 
        for type_, score in anomalies.items()
    )
    
    # Apply non-linear scaling for severity
    final_score = self.apply_severity_scaling(composite_score)
    
    return min(100, max(0, final_score))
```

**Alert Thresholds**:
- **High Severity**: Score >80 (immediate notification)
- **Medium Severity**: Score 60-80 (daily digest)
- **Low Severity**: Score 40-60 (weekly report)
- **Informational**: Score 20-40 (monthly summary)

### 3. Sentiment Analysis Pipeline

#### Purpose
Analyze news sentiment and social media discussions to correlate with congressional trading activities and predict market impact.

#### Technical Implementation
**File**: `src/intelligence/news_monitor.py`

**Model Architecture**:
```python
class NewsIntelligenceMonitor:
    def __init__(self):
        self.sentiment_model = AutoModel.from_pretrained(
            'nlptown/bert-base-multilingual-uncased-sentiment'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            'nlptown/bert-base-multilingual-uncased-sentiment'
        )
        self.market_correlator = MarketCorrelationAnalyzer()
```

#### News Processing Pipeline
1. **Source Aggregation**: 247+ news sources via RSS, APIs, and web scraping
2. **Content Extraction**: Article text extraction and cleaning
3. **Entity Recognition**: Congressional member and stock symbol identification
4. **Sentiment Analysis**: BERT-based sentiment scoring (-1 to +1)
5. **Market Correlation**: Statistical correlation with stock price movements

#### Advanced Features
```python
class MarketCorrelationAnalyzer:
    def analyze_sentiment_impact(self, news_data: pd.DataFrame, 
                                market_data: pd.DataFrame) -> dict:
        correlations = {}
        
        for member in self.get_tracked_members():
            member_news = self.filter_member_news(news_data, member)
            member_stocks = self.get_member_portfolio(member)
            
            # Calculate correlation between sentiment and stock performance
            correlation = self.calculate_correlation(
                member_news['sentiment_score'],
                market_data[member_stocks]['returns']
            )
            
            correlations[member] = {
                'correlation': correlation,
                'significance': self.test_significance(correlation),
                'sample_size': len(member_news)
            }
        
        return correlations
```

## Model Training & Deployment

### Training Infrastructure
**Compute Requirements**:
- **CPU**: 8+ cores for parallel processing
- **RAM**: 16GB+ for large dataset handling
- **Storage**: 500GB+ for training data and model artifacts
- **GPU**: Optional NVIDIA GPU for deep learning acceleration

### Training Schedule
- **Full Retraining**: Weekly with new STOCK Act disclosures
- **Incremental Updates**: Daily for sentiment models with new news data
- **Hyperparameter Tuning**: Monthly optimization cycles
- **Model Validation**: Continuous performance monitoring

### Model Versioning
```python
class ModelVersionManager:
    def __init__(self):
        self.model_registry = ModelRegistry()
        self.version_control = MLFlowTracker()
    
    def deploy_new_version(self, model, performance_metrics):
        # Version validation
        if self.validate_performance(performance_metrics):
            # A/B testing deployment
            self.deploy_ab_test(model)
            
            # Performance monitoring
            self.monitor_production_performance(model)
            
            # Automatic rollback if performance degrades
            self.setup_rollback_triggers(model)
```

## Performance Monitoring

### Real-time Metrics
- **Prediction Accuracy**: Continuously tracked against actual outcomes
- **Model Drift Detection**: Statistical tests for feature distribution changes
- **Latency Monitoring**: Response time tracking for real-time predictions
- **Resource Utilization**: CPU, memory, and storage usage monitoring

### Model Validation
```python
class ModelValidator:
    def validate_model_performance(self, model, test_data):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'calibration_error': self.calculate_calibration_error(y_true, y_pred_proba)
        }
        
        # Performance thresholds
        thresholds = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.80,
            'auc_roc': 0.88
        }
        
        # Validation checks
        validation_results = {
            metric: value >= thresholds[metric] 
            for metric, value in metrics.items() 
            if metric in thresholds
        }
        
        return metrics, validation_results
```

## Integration Points

### Database Integration
- **Feature Store**: Centralized storage for engineered features
- **Model Registry**: Version control and metadata management
- **Prediction Storage**: Historical predictions for backtesting
- **Performance Logs**: Model accuracy and drift metrics

### API Integration
```python
class MLModelAPI:
    @app.route('/api/v2/predict/trade', methods=['POST'])
    def predict_trade(self):
        member_id = request.json['member_id']
        features = self.extract_features(member_id)
        
        prediction = self.trade_model.predict_proba(features)[0][1]
        confidence = self.calculate_confidence(prediction)
        
        return {
            'member_id': member_id,
            'trade_probability': prediction,
            'confidence': confidence,
            'factors': self.explain_prediction(features),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    @app.route('/api/v2/detect/anomalies', methods=['POST'])
    def detect_anomalies(self):
        trade_data = request.json['trades']
        anomalies = self.anomaly_detector.detect_all_types(trade_data)
        
        return {
            'anomalies': anomalies,
            'total_suspicious': len(anomalies),
            'high_severity': len([a for a in anomalies if a['score'] > 80]),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
```

### Dashboard Integration
- **Real-time Predictions**: Live updates of trading probability scores
- **Anomaly Alerts**: Immediate notification system for suspicious activities
- **Model Performance**: Dashboard showing accuracy trends and model health
- **Explanation Interface**: User-friendly prediction explanations

## Security & Privacy

### Model Security
- **Input Validation**: Comprehensive validation of all model inputs
- **Output Sanitization**: Secure handling of model predictions
- **Access Control**: Role-based access to different model capabilities
- **Audit Logging**: Complete logging of all model queries and results

### Privacy Protection
- **Data Anonymization**: Member IDs anonymized in certain model outputs
- **Aggregation**: Individual predictions aggregated for privacy protection
- **Retention Policies**: Automatic deletion of detailed prediction logs
- **Compliance**: Adherence to data protection regulations

## Testing & Quality Assurance

### Model Testing
```python
class ModelTestSuite:
    def run_comprehensive_tests(self, model):
        test_results = {
            'unit_tests': self.run_unit_tests(model),
            'integration_tests': self.run_integration_tests(model),
            'performance_tests': self.run_performance_tests(model),
            'bias_tests': self.run_bias_tests(model),
            'robustness_tests': self.run_robustness_tests(model)
        }
        
        return test_results
    
    def run_bias_tests(self, model):
        # Test for bias across different demographics
        bias_metrics = {}
        
        for party in ['D', 'R', 'I']:
            party_data = self.filter_by_party(self.test_data, party)
            party_predictions = model.predict(party_data)
            bias_metrics[party] = self.calculate_bias_metrics(party_predictions)
        
        return bias_metrics
```

### Continuous Testing
- **Automated Test Suite**: Runs on every model update
- **Performance Regression Tests**: Ensures new versions don't degrade performance
- **Data Quality Tests**: Validates input data quality and completeness
- **End-to-end Integration Tests**: Full workflow validation from raw data to predictions

This specification provides the comprehensive technical details for all machine learning models implemented in Phase 2, ensuring maintainability, scalability, and continued performance excellence.