# Hobby Congressional Trading Agent: Personal Project Design

**Author:** Manus AI  
**Date:** June 26, 2025  
**Project Type:** Personal Hobby/Learning Project

## Executive Summary

This document presents a simplified, hobby-scale design for a congressional trading agent suitable for an individual developer working on a personal side project. Unlike enterprise-scale systems, this approach prioritizes learning, experimentation, and modest returns over institutional-grade infrastructure and massive capital deployment.

The "CongressWatch" system is designed to be built and operated by a single developer with a budget under $500/month, focusing on educational value and small-scale automated trading. The system will track congressional trades, generate simple signals, and execute small trades through retail brokerages while maintaining compliance with personal trading regulations.

## Project Philosophy and Scope

### Hobby Project Goals

This is fundamentally a learning and experimentation project rather than a serious investment strategy. The primary goals are to understand congressional trading patterns, learn about automated trading systems, and potentially generate modest supplemental income while gaining valuable technical experience.

The system should be simple enough to build and maintain as a weekend project, with total development time under 100 hours spread over 3-6 months. The focus is on rapid prototyping, learning, and iteration rather than production-grade reliability or institutional compliance.

Financial expectations should be modest - the goal is to beat basic index fund returns on a small portfolio ($1,000-$10,000) while learning about quantitative trading, data analysis, and financial markets. Any profits are secondary to the educational value and technical experience gained.

### Simplified Architecture Approach

The architecture follows a "good enough" philosophy, using simple, proven technologies and avoiding over-engineering. The system will run on a single cloud instance or even a home computer, with basic monitoring and minimal operational complexity.

Data storage will use simple SQLite or PostgreSQL databases without complex time-series optimizations. Processing will be batch-oriented rather than real-time, running daily or weekly analysis rather than millisecond-latency trading.

The system will integrate with retail trading platforms like Alpaca or Interactive Brokers' retail API, focusing on simple buy/hold strategies rather than complex algorithmic trading. Risk management will be basic but effective, with simple position sizing and stop-loss rules.

## Simplified System Design

### Core Components

The system consists of four main components that can be built incrementally: Data Collector, Signal Generator, Trade Executor, and Monitor Dashboard. Each component is designed to be simple, maintainable, and educational.

**Data Collector** fetches congressional trading data from free or low-cost APIs, stores it in a local database, and performs basic data cleaning and validation. This component runs on a daily schedule and handles API rate limiting and error recovery gracefully.

**Signal Generator** analyzes the collected data to identify potentially profitable trades using simple heuristics and basic machine learning. The focus is on interpretable signals that provide learning opportunities rather than complex black-box algorithms.

**Trade Executor** places small trades through retail brokerage APIs based on generated signals. The component includes basic risk management, position sizing, and order management suitable for personal trading accounts.

**Monitor Dashboard** provides a simple web interface to view portfolio performance, recent trades, and system status. The dashboard emphasizes learning and understanding over professional-grade analytics.

### Technology Stack

The technology stack prioritizes simplicity, low cost, and ease of development over performance and scalability. Python serves as the primary language for its extensive libraries and beginner-friendly ecosystem.

**Backend Framework**: Flask or FastAPI for simple web APIs and dashboard serving. These frameworks provide sufficient functionality without the complexity of enterprise frameworks.

**Database**: SQLite for development and small-scale deployment, with optional PostgreSQL for slightly larger datasets. No complex time-series databases or distributed systems are needed.

**Data Processing**: Pandas for data manipulation and analysis, with scikit-learn for basic machine learning. These libraries provide powerful capabilities while remaining accessible to hobby developers.

**Trading Integration**: Alpaca API for commission-free stock trading, with paper trading mode for safe testing and learning. The API is well-documented and designed for retail developers.

**Deployment**: Single DigitalOcean droplet or AWS EC2 instance, with simple cron jobs for scheduling. No container orchestration or complex deployment pipelines are needed.

**Monitoring**: Simple logging to files with basic email alerts for critical issues. Professional monitoring tools are unnecessary for hobby-scale operations.

### Data Sources and Integration

The system will use free or low-cost data sources appropriate for hobby projects, avoiding expensive enterprise data feeds. Congressional trading data will come from publicly available APIs with reasonable rate limits.

**Primary Data Source**: Finnhub's free tier provides 60 API calls per minute for congressional trading data, sufficient for daily batch processing. The free tier includes basic congressional trading information without premium features.

**Backup Data Source**: Financial Modeling Prep's free tier offers 250 API calls per day, providing redundancy and additional data validation. The service includes senate trading data and basic company information.

**Market Data**: Alpha Vantage's free tier provides 5 API calls per minute for stock prices and basic market data. This is sufficient for daily portfolio valuation and basic technical analysis.

**News Data**: Optional integration with free news APIs for basic sentiment analysis and context around congressional trades. This adds educational value without significant cost.

### Trading Strategy

The trading strategy emphasizes simplicity, interpretability, and risk management over sophisticated algorithms. The approach should be easy to understand, modify, and learn from while providing reasonable performance expectations.

**Core Strategy**: Direct replication of congressional trades with appropriate delays and position sizing. When a congress member buys a stock, the system considers buying a small position after the disclosure becomes public.

**Signal Filters**: Basic filters to identify potentially informed trades, such as large position changes, trades by committee members in relevant sectors, or unusual trading activity by multiple members.

**Position Sizing**: Simple fixed-dollar amounts ($100-$500 per trade) or percentage-based sizing (1-2% of portfolio per position). Risk management focuses on limiting individual position sizes rather than complex portfolio optimization.

**Exit Strategy**: Simple time-based exits (hold for 30-90 days) or basic profit/loss targets (±10-20%). The focus is on learning about trade timing and market behavior rather than optimizing returns.

**Risk Management**: Maximum portfolio allocation per stock (5-10%), maximum number of positions (10-20), and simple stop-loss rules. These limits prevent catastrophic losses while allowing for learning experiences.

## Implementation Plan

### Phase 1: Data Collection and Analysis (Weeks 1-4)

The first phase focuses on building the data collection system and performing basic analysis to understand congressional trading patterns. This phase provides immediate learning value while establishing the foundation for automated trading.

**Week 1-2: Data Collection Setup**
- Set up development environment with Python, pandas, and basic libraries
- Create accounts with Finnhub and Financial Modeling Prep for API access
- Build basic data collection scripts to fetch and store congressional trading data
- Set up SQLite database with simple schema for trades, members, and companies

**Week 3-4: Exploratory Data Analysis**
- Analyze historical congressional trading patterns and performance
- Identify interesting trends, such as which members trade most frequently or which sectors see the most activity
- Create basic visualizations to understand the data and identify potential signals
- Document findings and insights for future strategy development

**Deliverables**: Working data collection system, basic database with historical data, analysis notebook with insights and visualizations.

### Phase 2: Signal Generation and Backtesting (Weeks 5-8)

The second phase develops simple trading signals and tests them against historical data to understand their potential effectiveness. This phase emphasizes learning about quantitative analysis and strategy development.

**Week 5-6: Signal Development**
- Implement basic signal generation algorithms (trade replication, consensus signals, timing analysis)
- Create simple scoring systems to rank potential trades by attractiveness
- Develop basic feature engineering (trade size relative to portfolio, member committee assignments, etc.)

**Week 7-8: Backtesting and Validation**
- Build simple backtesting framework to test signals against historical data
- Analyze signal performance, including win rates, average returns, and risk metrics
- Iterate on signal generation based on backtesting results
- Document strategy performance and lessons learned

**Deliverables**: Signal generation system, backtesting framework, performance analysis showing strategy viability.

### Phase 3: Trading Integration and Paper Trading (Weeks 9-12)

The third phase integrates with trading APIs and implements paper trading to test the complete system without risking real money. This phase provides experience with trading APIs and order management.

**Week 9-10: Trading API Integration**
- Set up Alpaca account and API access for paper trading
- Implement basic order placement and portfolio management functions
- Create simple trade execution logic with basic risk checks
- Test API integration with small paper trades

**Week 11-12: Paper Trading System**
- Deploy complete system for paper trading with real-time data
- Monitor system performance and identify bugs or issues
- Refine signal generation and execution based on paper trading results
- Create basic monitoring and alerting for system health

**Deliverables**: Complete paper trading system, performance monitoring, documented lessons learned from live testing.

### Phase 4: Live Trading and Monitoring (Weeks 13-16)

The final phase transitions to live trading with small amounts and implements basic monitoring and reporting. This phase provides real trading experience while maintaining appropriate risk levels.

**Week 13-14: Live Trading Deployment**
- Transition to live trading with small position sizes ($100-$200 per trade)
- Implement enhanced monitoring and alerting for live trading
- Create basic dashboard for portfolio tracking and performance monitoring
- Establish procedures for manual intervention and system maintenance

**Week 15-16: Optimization and Documentation**
- Analyze live trading performance and identify improvement opportunities
- Optimize signal generation and execution based on real trading experience
- Create comprehensive documentation for system operation and maintenance
- Plan future enhancements and learning objectives

**Deliverables**: Live trading system, performance dashboard, comprehensive documentation, lessons learned report.

## Budget and Resource Requirements

### Development Costs

The total development cost should remain under $2,000 for the complete project, with most expenses being learning investments that provide long-term value.

**API Costs**: $0-$50/month for data feeds using free tiers of Finnhub, Financial Modeling Prep, and Alpha Vantage. Paid tiers may be considered later for enhanced data access.

**Cloud Infrastructure**: $20-$50/month for a basic DigitalOcean droplet or AWS EC2 instance. A $5/month droplet is sufficient for initial development and testing.

**Trading Platform**: $0 commission trading through Alpaca, with potential account minimums ($0-$500) depending on account type. Paper trading is completely free.

**Development Tools**: $0-$100 for optional paid tools like PyCharm Professional or premium GitHub features. Most development can be done with free tools.

**Learning Resources**: $50-$200 for books, courses, or documentation related to quantitative trading and financial analysis. This is an investment in knowledge rather than operational cost.

### Time Investment

The project is designed to fit into a hobby schedule with approximately 5-10 hours per week over 4 months. Total time investment should be 80-160 hours, making it a substantial but manageable side project.

**Development Time**: 60-120 hours for coding, testing, and debugging across all phases. This includes learning time for new technologies and concepts.

**Research and Analysis**: 20-40 hours for market research, strategy development, and performance analysis. This time provides significant educational value.

**Monitoring and Maintenance**: 2-4 hours per week once the system is operational, primarily for monitoring performance and making minor adjustments.

### Risk Management

Risk management for a hobby project focuses on limiting financial exposure while maximizing learning opportunities. The approach should prevent significant losses while allowing for meaningful experimentation.

**Financial Risk Limits**: Maximum portfolio size of $1,000-$10,000 depending on personal financial situation. Individual position limits of $100-$500 per trade to prevent concentration risk.

**Technical Risk Management**: Simple backup procedures, basic error handling, and manual override capabilities. The system should fail safely rather than catastrophically.

**Learning Risk Management**: Focus on understanding why trades succeed or fail rather than just maximizing returns. Document lessons learned and maintain detailed records for future analysis.

## Expected Outcomes and Learning Objectives

### Financial Expectations

Financial returns should be viewed as a secondary benefit rather than the primary goal. Realistic expectations help maintain focus on learning while avoiding disappointment from modest returns.

**Return Targets**: Beat S&P 500 returns by 2-5% annually on the managed portfolio. This is ambitious but achievable with good signal generation and risk management.

**Risk Targets**: Maximum drawdown of 15-20% to ensure the hobby remains enjoyable rather than stressful. Volatility should be comparable to or lower than broad market indices.

**Portfolio Size**: Start with $1,000-$2,000 and potentially scale to $5,000-$10,000 based on performance and comfort level. The focus is on percentage returns rather than absolute dollar amounts.

### Learning Objectives

The primary value of the project comes from the knowledge and skills gained rather than financial returns. The learning objectives span multiple domains relevant to technology and finance careers.

**Technical Skills**: Python programming, API integration, database design, web development, and basic machine learning. These skills are valuable for many technology roles.

**Financial Knowledge**: Understanding of market structure, trading mechanics, quantitative analysis, and risk management. This knowledge is valuable for finance and fintech careers.

**Data Analysis**: Experience with real-world data collection, cleaning, analysis, and visualization. These skills are applicable to many data-driven roles.

**System Design**: Experience designing, building, and operating automated systems with real-world constraints and requirements. This experience is valuable for software engineering roles.

### Success Metrics

Success should be measured across multiple dimensions rather than just financial returns. A successful hobby project provides learning, enjoyment, and modest financial returns while maintaining appropriate risk levels.

**Educational Success**: Demonstrated understanding of quantitative trading concepts, ability to analyze and improve the system, and documented lessons learned.

**Technical Success**: Working system that operates reliably with minimal maintenance, good code quality and documentation, and successful integration with external APIs.

**Financial Success**: Positive risk-adjusted returns over a 6-12 month period, with performance tracking and analysis to understand sources of returns.

**Personal Success**: Enjoyable and engaging hobby that provides satisfaction and motivation for continued learning and improvement.

## Conclusion

The hobby-scale congressional trading agent represents an excellent learning project that combines technology, finance, and data analysis in a practical application. The simplified design prioritizes education and experimentation over institutional-grade performance while maintaining appropriate risk management.

The project provides valuable experience with real-world systems, APIs, and financial markets while requiring only modest time and financial investment. The skills and knowledge gained are applicable to many technology and finance career paths.

Success should be measured primarily by learning outcomes and personal satisfaction rather than financial returns, though modest outperformance of market indices is a reasonable secondary goal. The project provides a foundation for more sophisticated trading systems and quantitative analysis as skills and interest develop.

The key to success is maintaining realistic expectations, focusing on learning over profits, and enjoying the process of building and operating an automated trading system. The project should remain a hobby that provides satisfaction and education rather than becoming a source of stress or financial pressure.

