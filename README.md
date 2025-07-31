# Congressional Trading Intelligence System

A sophisticated financial intelligence platform that analyzes congressional trading patterns, market data, and legislative information to generate actionable trading insights.

## 🎯 Overview

This system combines advanced data analysis, machine learning, and real-time monitoring to track congressional trading disclosures and correlate them with market movements and legislative activities. The platform provides comprehensive intelligence for informed investment decisions.

## ✨ Key Features

### 📊 **Multi-Source Intelligence Fusion**
- Real-time news sentiment analysis with impact scoring
- Social media monitoring and volume anomaly detection  
- Options flow analysis for unusual market activity
- Legislative calendar tracking and bill monitoring
- Automated correlation between different intelligence sources

### 🧠 **Advanced Trading Engine**
- 10+ sophisticated trading algorithms (momentum, mean reversion, breakout)
- AI/ML models for price prediction and signal classification
- Multi-factor risk assessment with dynamic position sizing
- Real-time market analysis with technical indicators

### 🏛️ **Congressional Analysis**
- Comprehensive tracking of congressional trading disclosures
- Pattern recognition and performance analysis
- Committee assignment and sector correlation analysis
- Legislative timing and market impact assessment

### 🌐 **Professional Dashboard**
- Real-time monitoring interface with 4 functional tabs
- Interactive charts and performance visualizations
- Alert system with configurable risk thresholds
- Mobile-responsive design with professional styling

### 🤖 **Machine Learning Pipeline**
- Complete ML training framework for trading prediction
- Feature engineering and data preprocessing
- Multiple model types: Random Forest, XGBoost, Neural Networks
- Performance evaluation and model optimization

## 🏗️ Architecture

```
congressional-trading-system/
├── src/
│   ├── core/                 # Core system components
│   │   ├── apex_trading_engine.py      # Main trading algorithms
│   │   ├── intelligence_fusion_engine.py # Multi-source data fusion
│   │   ├── data_collector.py           # Automated data collection
│   │   ├── api_key_manager.py          # Secure API management
│   │   └── ai_model_trainer.py         # ML training pipeline
│   ├── analysis/             # Analysis and research tools
│   │   ├── congressional_analysis.py   # Congressional data analysis
│   │   └── enhanced_analysis.py        # Advanced pattern recognition
│   └── dashboard/            # Web interface
│       ├── App.jsx          # React dashboard component
│       ├── App.css          # Professional styling
│       └── index.html       # HTML template
├── docs/                    # Comprehensive documentation
├── data/                    # Analysis results and visualizations
└── config/                  # Configuration files
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Required API keys (see Configuration section)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/congressional-trading-system.git
cd congressional-trading-system
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up API keys**
```bash
python src/core/api_key_manager.py
```

4. **Run the intelligence system**
```bash
python src/core/intelligence_fusion_engine.py
```

5. **Launch the dashboard**
```bash
cd src/dashboard
npm install
npm run dev
```

## ⚙️ Configuration

### Required API Keys
- **Finnhub API** - Congressional trading data and market information
- **Financial Modeling Prep API** - Enhanced financial data and analysis
- **Alpha Vantage API** - Real-time market data and technical indicators
- **Alpaca API** - Trading execution and portfolio management

### Environment Setup
```bash
export FINNHUB_API_KEY="your_finnhub_key"
export FINANCIAL_MODELING_PREP_API_KEY="your_fmp_key"
export ALPHA_VANTAGE_API_KEY="your_av_key"
export ALPACA_API_KEY="your_alpaca_key"
export ALPACA_SECRET_KEY="your_alpaca_secret"
```

## 📈 Core Components

### Intelligence Fusion Engine
Collects and analyzes data from multiple sources:
- News sentiment analysis
- Social media monitoring
- Options flow tracking
- Legislative intelligence
- Market data integration

### Trading Engine
Advanced algorithmic trading system with:
- Multiple trading strategies
- Risk management protocols
- Position sizing algorithms
- Performance tracking

### Congressional Analysis
Comprehensive analysis of congressional trading patterns:
- Trading disclosure monitoring
- Performance benchmarking
- Sector and timing analysis
- Legislative correlation studies

## 📊 Data Sources

- **Congressional Trading Data**: Official STOCK Act disclosures
- **Market Data**: Real-time and historical price/volume data
- **News & Sentiment**: Financial news aggregation and analysis
- **Social Media**: Twitter, Reddit sentiment and volume tracking
- **Legislative Data**: Bill tracking, committee schedules, voting records
- **Options Data**: Unusual activity and flow analysis

## 🔧 Usage Examples

### Basic Intelligence Collection
```python
from src.core.intelligence_fusion_engine import IntelligenceFusionEngine

# Initialize the engine
fusion = IntelligenceFusionEngine()

# Collect intelligence for specific symbols
signals = fusion.collect_all_intelligence(['NVDA', 'AAPL', 'MSFT'])

# Analyze results
for signal in signals:
    print(f"Symbol: {signal.symbol}, Confidence: {signal.confidence}")
```

### Congressional Analysis
```python
from src.analysis.congressional_analysis import CongressionalAnalyzer

# Initialize analyzer
analyzer = CongressionalAnalyzer()

# Analyze recent trading patterns
results = analyzer.analyze_recent_trades(days=30)

# Generate performance report
report = analyzer.generate_performance_report()
```

### Trading Engine
```python
from src.core.apex_trading_engine import APEXTradingEngine

# Initialize trading engine
engine = APEXTradingEngine()

# Generate trading signals
signals = engine.generate_signals(['NVDA', 'AAPL'])

# Execute trades (paper trading mode)
engine.execute_signals(signals, paper_trading=True)
```

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **System Architecture** - Technical design and component overview
- **API Documentation** - Detailed API reference and examples  
- **Development Guide** - Setup, configuration, and development workflow
- **Analysis Reports** - Research findings and performance studies
- **User Manual** - Complete usage guide and tutorials

## 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/
```

Run integration tests:
```bash
python tests/integration_tests.py
```

## 📊 Performance Metrics

The system tracks various performance indicators:
- Signal accuracy and confidence levels
- Trading performance vs benchmarks
- Data collection reliability
- System uptime and response times

## 🔒 Security & Compliance

- Secure API key management with encryption
- Rate limiting and error handling
- Compliance with financial data regulations
- Audit logging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It provides analysis tools and should not be considered as financial advice. Users are responsible for their own investment decisions and should consult with qualified financial advisors.

## 🙏 Acknowledgments

- Congressional trading data sourced from official STOCK Act disclosures
- Market data provided by various financial data vendors
- Built with modern Python, React, and machine learning frameworks

---

**Built with ❤️ for financial intelligence and market analysis**

