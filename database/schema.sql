-- Congressional Trading Intelligence System - Database Schema
-- Version: 2.0 (Phase 1 - Core Data Infrastructure)
-- Created: January 31, 2025

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Core congressional members table
CREATE TABLE members (
    bioguide_id VARCHAR(10) PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    full_name VARCHAR(100) NOT NULL,
    party CHAR(1) NOT NULL CHECK (party IN ('D', 'R', 'I')),
    state CHAR(2) NOT NULL,
    district INTEGER,
    chamber VARCHAR(10) NOT NULL CHECK (chamber IN ('House', 'Senate')),
    served_from DATE NOT NULL,
    served_to DATE,
    birth_date DATE,
    occupation VARCHAR(200),
    education TEXT[],
    net_worth_estimate VARCHAR(50),
    leadership_position VARCHAR(100),
    official_full_name VARCHAR(100),
    nickname VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Congressional trading transactions
CREATE TABLE trades (
    id SERIAL PRIMARY KEY,
    bioguide_id VARCHAR(10) NOT NULL REFERENCES members(bioguide_id),
    transaction_date DATE NOT NULL,
    filing_date DATE NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    transaction_type VARCHAR(10) NOT NULL CHECK (transaction_type IN ('Purchase', 'Sale', 'Exchange')),
    amount_min INTEGER NOT NULL,
    amount_max INTEGER NOT NULL,
    amount_mid INTEGER GENERATED ALWAYS AS ((amount_min + amount_max) / 2) STORED,
    asset_name VARCHAR(200),
    asset_type VARCHAR(50) DEFAULT 'Stock',
    owner_type VARCHAR(20) NOT NULL CHECK (owner_type IN ('Self', 'Spouse', 'Dependent Child')),
    filing_id VARCHAR(100) UNIQUE,
    filing_delay_days INTEGER GENERATED ALWAYS AS (filing_date - transaction_date) STORED,
    source VARCHAR(50) DEFAULT 'Finnhub',
    raw_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Committee structure
CREATE TABLE committees (
    id SERIAL PRIMARY KEY,
    thomas_id VARCHAR(10) UNIQUE,
    name VARCHAR(200) NOT NULL,
    chamber VARCHAR(10) NOT NULL CHECK (chamber IN ('House', 'Senate', 'Joint')),
    committee_type VARCHAR(50) DEFAULT 'Standing',
    parent_committee_id INTEGER REFERENCES committees(id),
    jurisdiction TEXT,
    website_url VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Committee memberships with historical tracking
CREATE TABLE committee_memberships (
    id SERIAL PRIMARY KEY,
    bioguide_id VARCHAR(10) NOT NULL REFERENCES members(bioguide_id),
    committee_id INTEGER NOT NULL REFERENCES committees(id),
    role VARCHAR(50) DEFAULT 'Member' CHECK (role IN ('Chair', 'Ranking Member', 'Vice Chair', 'Member')),
    start_date DATE NOT NULL,
    end_date DATE,
    congress INTEGER NOT NULL,
    is_current BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Stock price data for performance analysis
CREATE TABLE stock_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4),
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4) NOT NULL,
    adjusted_close DECIMAL(12,4),
    volume BIGINT,
    source VARCHAR(50) DEFAULT 'yfinance',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, date)
);

-- Benchmark indices for performance comparison
CREATE TABLE benchmark_prices (
    id SERIAL PRIMARY KEY,
    benchmark VARCHAR(10) NOT NULL, -- SPY, QQQ, IWM, etc.
    date DATE NOT NULL,
    close_price DECIMAL(12,4) NOT NULL,
    return_1d DECIMAL(8,6),
    return_7d DECIMAL(8,6),
    return_30d DECIMAL(8,6),
    return_90d DECIMAL(8,6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(benchmark, date)
);

-- Trading performance analysis
CREATE TABLE trade_performance (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER NOT NULL REFERENCES trades(id),
    symbol VARCHAR(10) NOT NULL,
    transaction_date DATE NOT NULL,
    
    -- Pre-trade performance
    price_at_trade DECIMAL(12,4),
    pre_1d_return DECIMAL(8,6),
    pre_7d_return DECIMAL(8,6),
    pre_30d_return DECIMAL(8,6),
    
    -- Post-trade performance
    post_1d_return DECIMAL(8,6),
    post_7d_return DECIMAL(8,6),
    post_30d_return DECIMAL(8,6),
    post_90d_return DECIMAL(8,6),
    
    -- Benchmark comparison (vs S&P 500)
    benchmark_1d_return DECIMAL(8,6),
    benchmark_7d_return DECIMAL(8,6),
    benchmark_30d_return DECIMAL(8,6),
    benchmark_90d_return DECIMAL(8,6),
    
    -- Alpha calculation (excess return)
    alpha_1d DECIMAL(8,6) GENERATED ALWAYS AS (post_1d_return - benchmark_1d_return) STORED,
    alpha_7d DECIMAL(8,6) GENERATED ALWAYS AS (post_7d_return - benchmark_7d_return) STORED,
    alpha_30d DECIMAL(8,6) GENERATED ALWAYS AS (post_30d_return - benchmark_30d_return) STORED,
    alpha_90d DECIMAL(8,6) GENERATED ALWAYS AS (post_90d_return - benchmark_90d_return) STORED,
    
    -- Statistical significance
    t_statistic_30d DECIMAL(8,4),
    p_value_30d DECIMAL(8,6),
    
    calculated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bills and legislation tracking
CREATE TABLE bills (
    id SERIAL PRIMARY KEY,
    bill_id VARCHAR(20) NOT NULL UNIQUE, -- e.g., "hr1234-118"
    title TEXT NOT NULL,
    bill_type VARCHAR(10), -- HR, S, HJRES, SJRES
    number INTEGER,
    congress INTEGER NOT NULL,
    introduced_date DATE,
    latest_action_date DATE,
    latest_action TEXT,
    status VARCHAR(50),
    summary TEXT,
    policy_area VARCHAR(100),
    subjects TEXT[],
    sponsor_bioguide_id VARCHAR(10) REFERENCES members(bioguide_id),
    cosponsors_count INTEGER DEFAULT 0,
    committees INTEGER[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Bill cosponsorship tracking
CREATE TABLE bill_cosponsors (
    id SERIAL PRIMARY KEY,
    bill_id VARCHAR(20) NOT NULL REFERENCES bills(bill_id),
    bioguide_id VARCHAR(10) NOT NULL REFERENCES members(bioguide_id),
    cosponsor_date DATE,
    withdrawn_date DATE,
    is_original_cosponsor BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sector and industry mappings for correlation analysis
CREATE TABLE stock_sectors (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    sector VARCHAR(100),
    industry VARCHAR(150),
    market_cap VARCHAR(20), -- Large, Mid, Small
    exchange VARCHAR(10),
    is_etf BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol)
);

-- Committee jurisdiction to sector mapping
CREATE TABLE committee_sectors (
    id SERIAL PRIMARY KEY,
    committee_id INTEGER NOT NULL REFERENCES committees(id),
    sector VARCHAR(100) NOT NULL,
    oversight_strength DECIMAL(3,2) DEFAULT 1.0, -- 0.0 to 1.0 scale
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(committee_id, sector)
);

-- Data quality and audit logging
CREATE TABLE data_quality_metrics (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    expected_value DECIMAL(10,4),
    status VARCHAR(20) CHECK (status IN ('PASS', 'WARN', 'FAIL')),
    details TEXT,
    checked_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API call logging for rate limiting and monitoring
CREATE TABLE api_calls (
    id SERIAL PRIMARY KEY,
    api_source VARCHAR(50) NOT NULL,
    endpoint VARCHAR(200),
    status_code INTEGER,
    response_time_ms INTEGER,
    records_returned INTEGER,
    rate_limit_remaining INTEGER,
    called_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_members_party_state ON members(party, state);
CREATE INDEX idx_members_chamber ON members(chamber);
CREATE INDEX idx_trades_bioguide_date ON trades(bioguide_id, transaction_date DESC);
CREATE INDEX idx_trades_symbol_date ON trades(symbol, transaction_date DESC);
CREATE INDEX idx_trades_filing_delay ON trades(filing_delay_days);
CREATE INDEX idx_committee_memberships_current ON committee_memberships(bioguide_id) WHERE is_current = TRUE;
CREATE INDEX idx_stock_prices_symbol_date ON stock_prices(symbol, date DESC);
CREATE INDEX idx_benchmark_prices_date ON benchmark_prices(benchmark, date DESC);
CREATE INDEX idx_trade_performance_trade_id ON trade_performance(trade_id);
CREATE INDEX idx_bills_congress_status ON bills(congress, status);
CREATE INDEX idx_bills_policy_area ON bills(policy_area);

-- Full-text search indexes
CREATE INDEX idx_bills_title_search ON bills USING gin(to_tsvector('english', title));
CREATE INDEX idx_members_name_search ON members USING gin(to_tsvector('english', full_name));

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_members_updated_at BEFORE UPDATE ON members
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_committees_updated_at BEFORE UPDATE ON committees
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_committee_memberships_updated_at BEFORE UPDATE ON committee_memberships
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    
CREATE TRIGGER update_bills_updated_at BEFORE UPDATE ON bills
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert initial benchmark data
INSERT INTO stock_sectors (symbol, sector, industry, market_cap, exchange) VALUES
('SPY', 'Benchmark', 'S&P 500 ETF', 'Large', 'NYSE'),
('QQQ', 'Benchmark', 'NASDAQ 100 ETF', 'Large', 'NASDAQ'),
('IWM', 'Benchmark', 'Russell 2000 ETF', 'Small', 'NYSE'),
('VTI', 'Benchmark', 'Total Stock Market ETF', 'Large', 'NYSE');

-- Comments for documentation
COMMENT ON TABLE members IS 'Congressional members with biographical and political information';
COMMENT ON TABLE trades IS 'Congressional trading transactions from STOCK Act disclosures';
COMMENT ON TABLE committees IS 'Congressional committees and subcommittees';
COMMENT ON TABLE committee_memberships IS 'Historical and current committee membership records';
COMMENT ON TABLE stock_prices IS 'Daily stock price data for performance analysis';
COMMENT ON TABLE trade_performance IS 'Pre/post trade performance analysis with benchmarking';
COMMENT ON TABLE bills IS 'Congressional bills and legislation';
COMMENT ON TABLE data_quality_metrics IS 'Automated data quality monitoring and validation';

-- Grant permissions (adjust as needed for production)
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO app_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;