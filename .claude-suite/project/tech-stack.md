# Congressional Trading Intelligence System - Tech Stack

> Created: January 31, 2025
> Environment: Development
> Architecture: Hybrid Python/React with Intelligence Fusion

## Backend & Analysis Engine

**Core Language**: Python 3.13+
- **pandas>=2.0.0** - Financial data manipulation and analysis
- **numpy>=1.24.0** - Numerical computing for trading algorithms
- **requests>=2.31.0** - API integrations and data fetching
- **sqlite3** (built-in) - Development database, transitioning to PostgreSQL

**Machine Learning & Analytics**
- **scikit-learn>=1.3.0** - Trading pattern recognition and classification
- **tensorflow>=2.13.0** - Deep learning models for prediction
- **xgboost>=1.7.0** - Gradient boosting for trading signal generation
- **scipy>=1.11.0** - Statistical analysis and hypothesis testing

**Data Visualization**
- **matplotlib>=3.7.0** - Statistical charts and analysis plots
- **seaborn>=0.12.0** - Enhanced statistical visualizations
- **plotly>=5.15.0** - Interactive charts for dashboard integration

## Financial Data Sources

**Market Data APIs**
- **yfinance>=0.2.0** - Real-time and historical stock prices
- **alpha-vantage>=2.3.0** - Technical indicators and market data
- **alpaca-trade-api>=3.0.0** - Trading execution capabilities (paper trading)

**Congressional Data** (Planned)
- **Congress.gov API** - Bill tracking and legislative information
- **ProPublica Congress API** - Member profiles and voting records
- **Finnhub API** - Congressional trading disclosures

## Frontend Dashboard

**Current Implementation**: Hybrid HTML/CSS/JavaScript
- **HTML5** - Semantic structure with responsive design
- **CSS3** - Professional styling with grid layouts and animations
- **Vanilla JavaScript** - Tab navigation and interactive features
- **Python HTTP Server** - Development serving (port 3000/8000)

**Planned Migration**: React Application
- **React 18.2.0** - Component-based UI framework
- **Vite 4.4.5** - Fast build tool and development server
- **Lucide React 0.263.1** - Professional icon system
- **Recharts 2.7.2** - Data visualization components

## Data Collection & Processing

**Web Scraping & APIs**
- **beautifulsoup4>=4.12.0** - HTML parsing for data extraction
- **selenium>=4.11.0** - Dynamic content scraping
- **schedule>=1.2.0** - Automated data collection workflows

**Natural Language Processing**
- **textblob>=0.17.0** - Sentiment analysis of congressional communications
- **tweepy>=4.14.0** - Social media monitoring and analysis
- **nltk>=3.8.0** - Advanced text processing
- **spacy>=3.6.0** - Named entity recognition

## Security & Infrastructure

**Data Security**
- **cryptography>=41.0.0** - Secure API key management and data encryption
- **python-dotenv>=1.0.0** - Environment variable management

**Development Tools**
- **pytest>=7.4.0** - Unit and integration testing framework
- **pytest-cov>=4.1.0** - Code coverage analysis
- **black>=23.7.0** - Code formatting and style consistency
- **flake8>=6.0.0** - Linting and code quality checks

## Architecture Patterns

**Intelligence Fusion Engine**
- Multi-source data aggregation and correlation
- Real-time pattern detection and alert generation
- ML-based suspicious activity scoring

**Modular Component Design**
```
src/
├── core/           # Intelligence fusion, trading engines, AI models
├── analysis/       # Congressional analysis, pattern recognition
└── dashboard/      # Frontend interface and visualization
```

**Data Flow Architecture**
1. **Collection Layer** - APIs, web scraping, real-time feeds
2. **Processing Layer** - Data cleaning, ML analysis, pattern detection
3. **Intelligence Layer** - Correlation analysis, scoring, alert generation
4. **Presentation Layer** - Dashboard, reports, API endpoints

## Deployment & Operations

**Current Setup**
- **Development**: Local Python server with file-based data
- **Version Control**: Git with GitHub remote
- **Branching**: Feature branches for major enhancements

**Planned Infrastructure**
- **Database**: PostgreSQL for production data storage
- **Caching**: Redis for real-time data and session management
- **API Gateway**: Rate limiting and authentication
- **Monitoring**: Application performance and uptime tracking

## Performance Characteristics

**Current Metrics** (Development)
- **Dashboard Load Time**: <2 seconds
- **Data Processing**: 15 members in <1 second
- **Memory Usage**: ~50MB for analysis engine
- **Concurrent Users**: Single-user development setup

**Target Metrics** (Production)
- **Response Time**: <500ms for API queries
- **Concurrent Users**: 1000+ simultaneous dashboard users
- **Data Processing**: Real-time ingestion of congressional filings
- **Uptime**: 99.9% availability