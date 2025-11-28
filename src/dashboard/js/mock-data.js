/**
 * Congressional Trading Intelligence System
 * Mock Data Module - Provides sample data matching expected API responses
 *
 * This data structure matches the backend intelligence engine outputs
 * and will be replaced with real API calls when backends are complete.
 */

// =============================================================================
// CONVICTION SCORE DATA (C1)
// =============================================================================

const ConvictionMockData = {
    // Risk level colors and thresholds
    riskLevels: {
        low: { min: 0, max: 30, color: '#10b981', label: 'Low Risk', bgColor: '#d1fae5' },
        moderate: { min: 30, max: 50, color: '#f59e0b', label: 'Moderate', bgColor: '#fef3c7' },
        elevated: { min: 50, max: 65, color: '#f97316', label: 'Elevated', bgColor: '#fed7aa' },
        high: { min: 65, max: 80, color: '#ef4444', label: 'High Risk', bgColor: '#fee2e2' },
        critical: { min: 80, max: 100, color: '#dc2626', label: 'Critical', bgColor: '#fecaca' }
    },

    // Sample trades with conviction scores
    trades: [
        {
            trade_id: "T001",
            member_id: "M001",
            member_name: "Nancy Pelosi",
            party: "D",
            state: "CA",
            symbol: "NVDA",
            asset_name: "NVIDIA Corporation",
            transaction_type: "Purchase",
            amount_from: 1000000,
            amount_to: 5000000,
            transaction_date: "2024-12-15",
            filing_date: "2025-01-20",
            score: 87,
            risk_level: "critical",
            factors: {
                committee_access: { score: 95, weight: 0.25, explanation: "Member serves on Intelligence Committee with AI/tech oversight" },
                timing_proximity: { score: 88, weight: 0.25, explanation: "Trade occurred 3 days before AI legislation announcement" },
                filing_delay: { score: 80, weight: 0.15, explanation: "Filed near deadline (36 days)" },
                trade_size_anomaly: { score: 92, weight: 0.15, explanation: "Extremely large trade ($3M avg) - 4.2 std deviations above normal" },
                historical_pattern: { score: 70, weight: 0.10, explanation: "First trade in NVDA (has traded in Technology sector before)" },
                sector_concentration: { score: 85, weight: 0.10, explanation: "High concentration in oversight sectors (72% of trades)" }
            },
            top_factors: ["committee_access", "trade_size_anomaly", "timing_proximity"],
            explanation: "This trade has a conviction score of 87/100 (critical risk) because: Member serves on Intelligence Committee with AI/tech oversight; Extremely large trade ($3M avg) - 4.2 std deviations above normal; Trade occurred 3 days before AI legislation announcement."
        },
        {
            trade_id: "T002",
            member_id: "M002",
            member_name: "Joe Manchin",
            party: "D",
            state: "WV",
            symbol: "XOM",
            asset_name: "Exxon Mobil Corporation",
            transaction_type: "Purchase",
            amount_from: 250000,
            amount_to: 500000,
            transaction_date: "2024-02-14",
            filing_date: "2024-06-15",
            score: 82,
            risk_level: "critical",
            factors: {
                committee_access: { score: 100, weight: 0.25, explanation: "Energy Committee Chair with direct oversight of oil sector" },
                timing_proximity: { score: 75, weight: 0.25, explanation: "Trade before Energy Security Act vote" },
                filing_delay: { score: 100, weight: 0.15, explanation: "Extremely late filing (121 days) - potential concealment" },
                trade_size_anomaly: { score: 65, weight: 0.15, explanation: "Large trade relative to net worth ($375K)" },
                historical_pattern: { score: 45, weight: 0.10, explanation: "Consistent with historical energy sector trading" },
                sector_concentration: { score: 90, weight: 0.10, explanation: "Very high concentration in oversight sectors (85% of trades)" }
            },
            top_factors: ["committee_access", "filing_delay", "sector_concentration"],
            explanation: "This trade has a conviction score of 82/100 (critical risk) because: Energy Committee Chair with direct oversight of oil sector; Extremely late filing (121 days) - potential concealment; Very high concentration in oversight sectors (85% of trades)."
        },
        {
            trade_id: "T003",
            member_id: "M003",
            member_name: "Dan Crenshaw",
            party: "R",
            state: "TX",
            symbol: "AMZN",
            asset_name: "Amazon.com Inc",
            transaction_type: "Purchase",
            amount_from: 15001,
            amount_to: 50000,
            transaction_date: "2020-03-20",
            filing_date: "2020-09-15",
            score: 74,
            risk_level: "high",
            factors: {
                committee_access: { score: 55, weight: 0.25, explanation: "Energy Committee - indirect tech oversight" },
                timing_proximity: { score: 95, weight: 0.25, explanation: "Trade during COVID-19 briefing period (March 2020)" },
                filing_delay: { score: 100, weight: 0.15, explanation: "Extremely late filing (179 days)" },
                trade_size_anomaly: { score: 40, weight: 0.15, explanation: "Moderate trade size ($32.5K)" },
                historical_pattern: { score: 60, weight: 0.10, explanation: "First trade in AMZN" },
                sector_concentration: { score: 50, weight: 0.10, explanation: "Moderate concentration in tech sector" }
            },
            top_factors: ["filing_delay", "timing_proximity", "historical_pattern"],
            explanation: "This trade has a conviction score of 74/100 (high risk) because: Trade during COVID-19 briefing period (March 2020); Extremely late filing (179 days); First trade in AMZN."
        },
        {
            trade_id: "T004",
            member_id: "M004",
            member_name: "Ron Wyden",
            party: "D",
            state: "OR",
            symbol: "NVDA",
            asset_name: "NVIDIA Corporation",
            transaction_type: "Purchase",
            amount_from: 250000,
            amount_to: 500000,
            transaction_date: "2024-06-16",
            filing_date: "2025-01-16",
            score: 79,
            risk_level: "high",
            factors: {
                committee_access: { score: 90, weight: 0.25, explanation: "Finance Committee with tech tax policy oversight" },
                timing_proximity: { score: 85, weight: 0.25, explanation: "Trade 190 days before CHIPS Act implementation" },
                filing_delay: { score: 100, weight: 0.15, explanation: "Extremely late filing (214 days)" },
                trade_size_anomaly: { score: 55, weight: 0.15, explanation: "Moderate to large trade ($375K)" },
                historical_pattern: { score: 65, weight: 0.10, explanation: "New position in semiconductor sector" },
                sector_concentration: { score: 60, weight: 0.10, explanation: "Moderate concentration in tech" }
            },
            top_factors: ["filing_delay", "committee_access", "timing_proximity"],
            explanation: "This trade has a conviction score of 79/100 (high risk) because: Extremely late filing (214 days); Finance Committee with tech tax policy oversight; Trade 190 days before CHIPS Act implementation."
        },
        {
            trade_id: "T005",
            member_id: "M005",
            member_name: "Pat Toomey",
            party: "R",
            state: "PA",
            symbol: "JPM",
            asset_name: "JPMorgan Chase & Co",
            transaction_type: "Purchase",
            amount_from: 100000,
            amount_to: 250000,
            transaction_date: "2023-11-10",
            filing_date: "2023-12-20",
            score: 68,
            risk_level: "high",
            factors: {
                committee_access: { score: 100, weight: 0.25, explanation: "Former Banking Committee Chair with direct oversight" },
                timing_proximity: { score: 50, weight: 0.25, explanation: "Standard market timing" },
                filing_delay: { score: 35, weight: 0.15, explanation: "Filed on time (40 days)" },
                trade_size_anomaly: { score: 50, weight: 0.15, explanation: "Normal trade size for member ($175K)" },
                historical_pattern: { score: 30, weight: 0.10, explanation: "Consistent with historical banking trades" },
                sector_concentration: { score: 95, weight: 0.10, explanation: "Extremely high concentration in banking (78%)" }
            },
            top_factors: ["committee_access", "sector_concentration", "timing_proximity"],
            explanation: "This trade has a conviction score of 68/100 (high risk) because: Former Banking Committee Chair with direct oversight; Extremely high concentration in banking (78%); Standard market timing."
        },
        {
            trade_id: "T006",
            member_id: "M006",
            member_name: "Sherrod Brown",
            party: "D",
            state: "OH",
            symbol: "BAC",
            asset_name: "Bank of America Corp",
            transaction_type: "Purchase",
            amount_from: 50001,
            amount_to: 100000,
            transaction_date: "2024-01-15",
            filing_date: "2024-02-28",
            score: 72,
            risk_level: "high",
            factors: {
                committee_access: { score: 100, weight: 0.25, explanation: "Senate Banking Chair with direct oversight" },
                timing_proximity: { score: 65, weight: 0.25, explanation: "Trade before banking regulation hearing" },
                filing_delay: { score: 25, weight: 0.15, explanation: "Filed well within deadline (44 days)" },
                trade_size_anomaly: { score: 45, weight: 0.15, explanation: "Moderate trade size ($75K)" },
                historical_pattern: { score: 50, weight: 0.10, explanation: "Consistent with banking sector focus" },
                sector_concentration: { score: 88, weight: 0.10, explanation: "Very high banking sector concentration" }
            },
            top_factors: ["committee_access", "sector_concentration", "timing_proximity"],
            explanation: "This trade has a conviction score of 72/100 (high risk) because: Senate Banking Chair with direct oversight; Very high banking sector concentration; Trade before banking regulation hearing."
        },
        {
            trade_id: "T007",
            member_id: "M007",
            member_name: "Josh Gottheimer",
            party: "D",
            state: "NJ",
            symbol: "MSFT",
            asset_name: "Microsoft Corporation",
            transaction_type: "Purchase",
            amount_from: 50001,
            amount_to: 100000,
            transaction_date: "2024-01-15",
            filing_date: "2024-02-28",
            score: 55,
            risk_level: "elevated",
            factors: {
                committee_access: { score: 45, weight: 0.25, explanation: "Financial Services - indirect tech oversight" },
                timing_proximity: { score: 55, weight: 0.25, explanation: "Standard timing" },
                filing_delay: { score: 30, weight: 0.15, explanation: "Filed within deadline (44 days)" },
                trade_size_anomaly: { score: 60, weight: 0.15, explanation: "Average trade size ($75K)" },
                historical_pattern: { score: 65, weight: 0.10, explanation: "First MSFT trade" },
                sector_concentration: { score: 55, weight: 0.10, explanation: "Moderate tech concentration" }
            },
            top_factors: ["historical_pattern", "trade_size_anomaly", "timing_proximity"],
            explanation: "This trade has a conviction score of 55/100 (elevated risk) because: First MSFT trade; Average trade size ($75K); Standard timing."
        },
        {
            trade_id: "T008",
            member_id: "M008",
            member_name: "Ted Cruz",
            party: "R",
            state: "TX",
            symbol: "COIN",
            asset_name: "Coinbase Global Inc",
            transaction_type: "Purchase",
            amount_from: 50001,
            amount_to: 100000,
            transaction_date: "2024-01-08",
            filing_date: "2024-02-15",
            score: 63,
            risk_level: "elevated",
            factors: {
                committee_access: { score: 65, weight: 0.25, explanation: "Commerce Committee with crypto oversight" },
                timing_proximity: { score: 70, weight: 0.25, explanation: "Trade before crypto regulation hearing" },
                filing_delay: { score: 30, weight: 0.15, explanation: "Filed within deadline (38 days)" },
                trade_size_anomaly: { score: 55, weight: 0.15, explanation: "Average trade size ($75K)" },
                historical_pattern: { score: 75, weight: 0.10, explanation: "First crypto trade" },
                sector_concentration: { score: 45, weight: 0.10, explanation: "Low fintech concentration" }
            },
            top_factors: ["historical_pattern", "timing_proximity", "committee_access"],
            explanation: "This trade has a conviction score of 63/100 (elevated risk) because: First crypto trade; Trade before crypto regulation hearing; Commerce Committee with crypto oversight."
        },
        {
            trade_id: "T009",
            member_id: "M009",
            member_name: "Mark Warner",
            party: "D",
            state: "VA",
            symbol: "PANW",
            asset_name: "Palo Alto Networks Inc",
            transaction_type: "Purchase",
            amount_from: 15001,
            amount_to: 50000,
            transaction_date: "2024-03-10",
            filing_date: "2024-04-15",
            score: 71,
            risk_level: "high",
            factors: {
                committee_access: { score: 95, weight: 0.25, explanation: "Intelligence Committee Chair with cybersecurity oversight" },
                timing_proximity: { score: 70, weight: 0.25, explanation: "Trade before cybersecurity legislation" },
                filing_delay: { score: 25, weight: 0.15, explanation: "Filed on time (36 days)" },
                trade_size_anomaly: { score: 40, weight: 0.15, explanation: "Small trade size ($32.5K)" },
                historical_pattern: { score: 80, weight: 0.10, explanation: "New position in cybersecurity" },
                sector_concentration: { score: 75, weight: 0.10, explanation: "High concentration in tech/cyber" }
            },
            top_factors: ["committee_access", "historical_pattern", "sector_concentration"],
            explanation: "This trade has a conviction score of 71/100 (high risk) because: Intelligence Committee Chair with cybersecurity oversight; New position in cybersecurity; High concentration in tech/cyber."
        },
        {
            trade_id: "T010",
            member_id: "M010",
            member_name: "Alexandria Ocasio-Cortez",
            party: "D",
            state: "NY",
            symbol: "TSLA",
            asset_name: "Tesla Inc",
            transaction_type: "Sale",
            amount_from: 1001,
            amount_to: 15000,
            transaction_date: "2024-02-20",
            filing_date: "2024-03-25",
            score: 28,
            risk_level: "low",
            factors: {
                committee_access: { score: 25, weight: 0.25, explanation: "Financial Services - limited auto sector oversight" },
                timing_proximity: { score: 30, weight: 0.25, explanation: "No correlated events" },
                filing_delay: { score: 25, weight: 0.15, explanation: "Filed early (33 days)" },
                trade_size_anomaly: { score: 20, weight: 0.15, explanation: "Small trade size ($8K)" },
                historical_pattern: { score: 35, weight: 0.10, explanation: "Consistent small trading pattern" },
                sector_concentration: { score: 30, weight: 0.10, explanation: "Low oversight sector concentration" }
            },
            top_factors: ["historical_pattern", "timing_proximity", "sector_concentration"],
            explanation: "This trade has a conviction score of 28/100 (low risk) because: Consistent small trading pattern; No correlated events; Low oversight sector concentration."
        },
        {
            trade_id: "T011",
            member_id: "M011",
            member_name: "Ro Khanna",
            party: "D",
            state: "CA",
            symbol: "AAPL",
            asset_name: "Apple Inc",
            transaction_type: "Sale",
            amount_from: 15001,
            amount_to: 50000,
            transaction_date: "2024-01-05",
            filing_date: "2024-02-10",
            score: 61,
            risk_level: "elevated",
            factors: {
                committee_access: { score: 60, weight: 0.25, explanation: "Oversight Committee with tech hearings" },
                timing_proximity: { score: 75, weight: 0.25, explanation: "Sale before tariff announcement affecting Apple" },
                filing_delay: { score: 30, weight: 0.15, explanation: "Filed within deadline (36 days)" },
                trade_size_anomaly: { score: 45, weight: 0.15, explanation: "Average trade size ($32.5K)" },
                historical_pattern: { score: 55, weight: 0.10, explanation: "Has traded AAPL before" },
                sector_concentration: { score: 65, weight: 0.10, explanation: "Moderate tech concentration" }
            },
            top_factors: ["timing_proximity", "sector_concentration", "committee_access"],
            explanation: "This trade has a conviction score of 61/100 (elevated risk) because: Sale before tariff announcement affecting Apple; Moderate tech concentration; Oversight Committee with tech hearings."
        },
        {
            trade_id: "T012",
            member_id: "M012",
            member_name: "Tommy Tuberville",
            party: "R",
            state: "AL",
            symbol: "RTX",
            asset_name: "RTX Corporation",
            transaction_type: "Purchase",
            amount_from: 50001,
            amount_to: 100000,
            transaction_date: "2024-04-01",
            filing_date: "2024-05-10",
            score: 76,
            risk_level: "high",
            factors: {
                committee_access: { score: 100, weight: 0.25, explanation: "Armed Services Committee with direct defense oversight" },
                timing_proximity: { score: 70, weight: 0.25, explanation: "Trade before defense budget markup" },
                filing_delay: { score: 28, weight: 0.15, explanation: "Filed within deadline (39 days)" },
                trade_size_anomaly: { score: 55, weight: 0.15, explanation: "Average trade size ($75K)" },
                historical_pattern: { score: 70, weight: 0.10, explanation: "Frequent defense sector trading" },
                sector_concentration: { score: 95, weight: 0.10, explanation: "Extremely high defense concentration (82%)" }
            },
            top_factors: ["committee_access", "sector_concentration", "timing_proximity"],
            explanation: "This trade has a conviction score of 76/100 (high risk) because: Armed Services Committee with direct defense oversight; Extremely high defense concentration (82%); Trade before defense budget markup."
        }
    ],

    // Get trade by ID
    getTradeById(tradeId) {
        return this.trades.find(t => t.trade_id === tradeId);
    },

    // Get trades filtered by criteria
    getFilteredTrades({ minScore, maxScore, riskLevel, party, sortBy, sortOrder }) {
        let filtered = [...this.trades];

        if (minScore !== undefined) {
            filtered = filtered.filter(t => t.score >= minScore);
        }
        if (maxScore !== undefined) {
            filtered = filtered.filter(t => t.score <= maxScore);
        }
        if (riskLevel) {
            filtered = filtered.filter(t => t.risk_level === riskLevel);
        }
        if (party) {
            filtered = filtered.filter(t => t.party === party);
        }

        // Sort
        if (sortBy) {
            filtered.sort((a, b) => {
                let aVal = a[sortBy];
                let bVal = b[sortBy];
                if (typeof aVal === 'string') {
                    return sortOrder === 'desc' ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
                }
                return sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
            });
        }

        return filtered;
    },

    // Get risk level config from score
    getRiskLevel(score) {
        for (const [level, config] of Object.entries(this.riskLevels)) {
            if (score >= config.min && score < config.max) {
                return { level, ...config };
            }
        }
        return { level: 'critical', ...this.riskLevels.critical };
    },

    // Get summary statistics
    getStatistics() {
        const scores = this.trades.map(t => t.score);
        const riskCounts = {};
        for (const level of Object.keys(this.riskLevels)) {
            riskCounts[level] = this.trades.filter(t => t.risk_level === level).length;
        }

        return {
            totalTrades: this.trades.length,
            averageScore: Math.round(scores.reduce((a, b) => a + b, 0) / scores.length),
            highestScore: Math.max(...scores),
            lowestScore: Math.min(...scores),
            riskDistribution: riskCounts
        };
    }
};


// =============================================================================
// COMMITTEE CONNECTION DATA (C2)
// =============================================================================

const CommitteeConnectionMockData = {
    // Network graph nodes
    nodes: [
        // Members
        { id: "m_pelosi", type: "member", label: "Nancy Pelosi", party: "D", oversightPct: 72 },
        { id: "m_manchin", type: "member", label: "Joe Manchin", party: "D", oversightPct: 85 },
        { id: "m_toomey", type: "member", label: "Pat Toomey", party: "R", oversightPct: 78 },
        { id: "m_brown", type: "member", label: "Sherrod Brown", party: "D", oversightPct: 81 },
        { id: "m_warner", type: "member", label: "Mark Warner", party: "D", oversightPct: 76 },
        { id: "m_cruz", type: "member", label: "Ted Cruz", party: "R", oversightPct: 58 },
        { id: "m_crenshaw", type: "member", label: "Dan Crenshaw", party: "R", oversightPct: 62 },
        { id: "m_tuberville", type: "member", label: "Tommy Tuberville", party: "R", oversightPct: 82 },

        // Committees
        { id: "c_intelligence", type: "committee", label: "Intelligence" },
        { id: "c_banking", type: "committee", label: "Banking" },
        { id: "c_energy", type: "committee", label: "Energy" },
        { id: "c_finance", type: "committee", label: "Financial Services" },
        { id: "c_armed", type: "committee", label: "Armed Services" },
        { id: "c_commerce", type: "committee", label: "Commerce" },

        // Sectors
        { id: "s_tech", type: "sector", label: "Technology" },
        { id: "s_energy", type: "sector", label: "Energy" },
        { id: "s_finance", type: "sector", label: "Financials" },
        { id: "s_defense", type: "sector", label: "Defense" },
        { id: "s_cyber", type: "sector", label: "Cybersecurity" }
    ],

    // Network graph edges
    edges: [
        // Member -> Committee
        { source: "m_pelosi", target: "c_intelligence" },
        { source: "m_pelosi", target: "c_finance" },
        { source: "m_manchin", target: "c_energy" },
        { source: "m_toomey", target: "c_banking" },
        { source: "m_brown", target: "c_banking" },
        { source: "m_warner", target: "c_intelligence" },
        { source: "m_cruz", target: "c_commerce" },
        { source: "m_crenshaw", target: "c_energy" },
        { source: "m_tuberville", target: "c_armed" },

        // Committee -> Sector
        { source: "c_intelligence", target: "s_tech" },
        { source: "c_intelligence", target: "s_cyber" },
        { source: "c_intelligence", target: "s_defense" },
        { source: "c_banking", target: "s_finance" },
        { source: "c_energy", target: "s_energy" },
        { source: "c_finance", target: "s_finance" },
        { source: "c_finance", target: "s_tech" },
        { source: "c_armed", target: "s_defense" },
        { source: "c_commerce", target: "s_tech" }
    ],

    // Sankey diagram data
    sankey: {
        nodes: [
            { name: "Financial Services" },
            { name: "Intelligence" },
            { name: "Energy" },
            { name: "Banking" },
            { name: "Armed Services" },
            { name: "Technology" },
            { name: "Financials" },
            { name: "Energy Sector" },
            { name: "Defense" },
            { name: "Cybersecurity" },
            { name: "NVDA" },
            { name: "MSFT" },
            { name: "JPM" },
            { name: "BAC" },
            { name: "XOM" },
            { name: "RTX" },
            { name: "PANW" }
        ],
        links: [
            // Committee -> Sector flows
            { source: 0, target: 5, value: 15 },  // Financial Services -> Technology
            { source: 0, target: 6, value: 25 },  // Financial Services -> Financials
            { source: 1, target: 5, value: 20 },  // Intelligence -> Technology
            { source: 1, target: 9, value: 12 },  // Intelligence -> Cybersecurity
            { source: 2, target: 7, value: 18 },  // Energy -> Energy Sector
            { source: 3, target: 6, value: 22 },  // Banking -> Financials
            { source: 4, target: 8, value: 16 },  // Armed Services -> Defense

            // Sector -> Stock flows
            { source: 5, target: 10, value: 12 },  // Technology -> NVDA
            { source: 5, target: 11, value: 8 },   // Technology -> MSFT
            { source: 6, target: 12, value: 14 },  // Financials -> JPM
            { source: 6, target: 13, value: 10 },  // Financials -> BAC
            { source: 7, target: 14, value: 15 },  // Energy Sector -> XOM
            { source: 8, target: 15, value: 11 },  // Defense -> RTX
            { source: 9, target: 16, value: 9 }    // Cybersecurity -> PANW
        ]
    },

    // Heat map data: Committee vs Sector trading volume
    heatMap: {
        committees: ["Financial Services", "Intelligence", "Energy", "Banking", "Armed Services", "Commerce"],
        sectors: ["Technology", "Financials", "Energy", "Defense", "Healthcare", "Consumer"],
        data: [
            // Financial Services
            [45, 87, 12, 8, 15, 23],
            // Intelligence
            [78, 25, 5, 45, 8, 12],
            // Energy
            [15, 8, 92, 10, 5, 8],
            // Banking
            [22, 95, 6, 4, 3, 15],
            // Armed Services
            [18, 5, 8, 88, 12, 6],
            // Commerce
            [65, 35, 12, 8, 25, 45]
        ]
    },

    // Leaderboard by oversight trading %
    leaderboard: [
        { rank: 1, member: "Joe Manchin", party: "D", state: "WV", oversightPct: 85, trades: 12, committees: ["Energy"] },
        { rank: 2, member: "Tommy Tuberville", party: "R", state: "AL", oversightPct: 82, trades: 45, committees: ["Armed Services"] },
        { rank: 3, member: "Sherrod Brown", party: "D", state: "OH", oversightPct: 81, trades: 8, committees: ["Banking"] },
        { rank: 4, member: "Pat Toomey", party: "R", state: "PA", oversightPct: 78, trades: 15, committees: ["Banking", "Finance"] },
        { rank: 5, member: "Mark Warner", party: "D", state: "VA", oversightPct: 76, trades: 11, committees: ["Intelligence"] },
        { rank: 6, member: "Nancy Pelosi", party: "D", state: "CA", oversightPct: 72, trades: 23, committees: ["Intelligence", "Financial Services"] },
        { rank: 7, member: "Dan Crenshaw", party: "R", state: "TX", oversightPct: 62, trades: 16, committees: ["Energy"] },
        { rank: 8, member: "Ted Cruz", party: "R", state: "TX", oversightPct: 58, trades: 14, committees: ["Commerce"] },
        { rank: 9, member: "Josh Gottheimer", party: "D", state: "NJ", oversightPct: 52, trades: 19, committees: ["Financial Services"] },
        { rank: 10, member: "Ro Khanna", party: "D", state: "CA", oversightPct: 45, trades: 8, committees: ["Oversight"] }
    ]
};


// =============================================================================
// COPY CONGRESS SIMULATOR DATA (C3)
// =============================================================================

const CopyCongressMockData = {
    // Member performance data
    memberPerformance: [
        {
            member_id: "M001",
            member_name: "Nancy Pelosi",
            party: "D",
            state: "CA",
            total_return_pct: 65.4,
            benchmark_return_pct: 24.2,
            alpha: 41.2,
            sharpe_ratio: 2.1,
            win_rate: 78,
            avg_trade_return: 12.3,
            max_drawdown: -15.2,
            trades_analyzed: 23,
            portfolio_value_10k: 16540,
            top_trades: [
                { symbol: "NVDA", return: 145.2 },
                { symbol: "GOOGL", return: 52.3 },
                { symbol: "RBLX", return: 38.7 }
            ],
            monthly_returns: [
                { month: "2024-01", return: 5.2 },
                { month: "2024-02", return: 8.1 },
                { month: "2024-03", return: -2.3 },
                { month: "2024-04", return: 12.5 },
                { month: "2024-05", return: 6.8 },
                { month: "2024-06", return: 15.4 },
                { month: "2024-07", return: -4.2 },
                { month: "2024-08", return: 7.3 },
                { month: "2024-09", return: 9.1 },
                { month: "2024-10", return: 3.5 },
                { month: "2024-11", return: 8.9 },
                { month: "2024-12", return: -4.9 }
            ]
        },
        {
            member_id: "M002",
            member_name: "Joe Manchin",
            party: "D",
            state: "WV",
            total_return_pct: 42.8,
            benchmark_return_pct: 24.2,
            alpha: 18.6,
            sharpe_ratio: 1.5,
            win_rate: 72,
            avg_trade_return: 8.7,
            max_drawdown: -18.5,
            trades_analyzed: 12,
            portfolio_value_10k: 14280,
            top_trades: [
                { symbol: "XOM", return: 35.2 },
                { symbol: "CVX", return: 28.4 },
                { symbol: "SLB", return: 22.1 }
            ],
            monthly_returns: [
                { month: "2024-01", return: 3.2 },
                { month: "2024-02", return: 5.1 },
                { month: "2024-03", return: 8.3 },
                { month: "2024-04", return: -2.5 },
                { month: "2024-05", return: 4.8 },
                { month: "2024-06", return: 6.4 },
                { month: "2024-07", return: 3.2 },
                { month: "2024-08", return: -1.3 },
                { month: "2024-09", return: 5.1 },
                { month: "2024-10", return: 4.5 },
                { month: "2024-11", return: 2.9 },
                { month: "2024-12", return: 3.1 }
            ]
        },
        {
            member_id: "M003",
            member_name: "Dan Crenshaw",
            party: "R",
            state: "TX",
            total_return_pct: 38.5,
            benchmark_return_pct: 24.2,
            alpha: 14.3,
            sharpe_ratio: 1.3,
            win_rate: 68,
            avg_trade_return: 7.2,
            max_drawdown: -22.1,
            trades_analyzed: 16,
            portfolio_value_10k: 13850,
            top_trades: [
                { symbol: "AMZN", return: 42.1 },
                { symbol: "MSFT", return: 28.5 },
                { symbol: "TSLA", return: 18.3 }
            ],
            monthly_returns: [
                { month: "2024-01", return: 4.2 },
                { month: "2024-02", return: 2.1 },
                { month: "2024-03", return: 6.3 },
                { month: "2024-04", return: -1.5 },
                { month: "2024-05", return: 5.8 },
                { month: "2024-06", return: 4.4 },
                { month: "2024-07", return: 2.2 },
                { month: "2024-08", return: 3.3 },
                { month: "2024-09", return: -2.1 },
                { month: "2024-10", return: 6.5 },
                { month: "2024-11", return: 4.9 },
                { month: "2024-12", return: 2.4 }
            ]
        },
        {
            member_id: "M012",
            member_name: "Tommy Tuberville",
            party: "R",
            state: "AL",
            total_return_pct: 52.1,
            benchmark_return_pct: 24.2,
            alpha: 27.9,
            sharpe_ratio: 1.8,
            win_rate: 74,
            avg_trade_return: 9.8,
            max_drawdown: -16.8,
            trades_analyzed: 45,
            portfolio_value_10k: 15210,
            top_trades: [
                { symbol: "RTX", return: 45.2 },
                { symbol: "LMT", return: 38.4 },
                { symbol: "NOC", return: 32.1 }
            ],
            monthly_returns: [
                { month: "2024-01", return: 5.2 },
                { month: "2024-02", return: 4.1 },
                { month: "2024-03", return: 3.3 },
                { month: "2024-04", return: 6.5 },
                { month: "2024-05", return: 2.8 },
                { month: "2024-06", return: 5.4 },
                { month: "2024-07", return: -2.2 },
                { month: "2024-08", return: 4.3 },
                { month: "2024-09", return: 7.1 },
                { month: "2024-10", return: 3.5 },
                { month: "2024-11", return: 6.9 },
                { month: "2024-12", return: 5.2 }
            ]
        },
        {
            member_id: "M005",
            member_name: "Pat Toomey",
            party: "R",
            state: "PA",
            total_return_pct: 28.4,
            benchmark_return_pct: 24.2,
            alpha: 4.2,
            sharpe_ratio: 1.1,
            win_rate: 65,
            avg_trade_return: 5.8,
            max_drawdown: -12.3,
            trades_analyzed: 15,
            portfolio_value_10k: 12840,
            top_trades: [
                { symbol: "JPM", return: 22.1 },
                { symbol: "BAC", return: 18.4 },
                { symbol: "GS", return: 15.2 }
            ],
            monthly_returns: [
                { month: "2024-01", return: 2.2 },
                { month: "2024-02", return: 3.1 },
                { month: "2024-03", return: 1.3 },
                { month: "2024-04", return: 2.5 },
                { month: "2024-05", return: 3.8 },
                { month: "2024-06", return: 2.4 },
                { month: "2024-07", return: -1.2 },
                { month: "2024-08", return: 4.3 },
                { month: "2024-09", return: 2.1 },
                { month: "2024-10", return: 3.5 },
                { month: "2024-11", return: 2.9 },
                { month: "2024-12", return: 1.5 }
            ]
        },
        {
            member_id: "M006",
            member_name: "Sherrod Brown",
            party: "D",
            state: "OH",
            total_return_pct: 22.1,
            benchmark_return_pct: 24.2,
            alpha: -2.1,
            sharpe_ratio: 0.9,
            win_rate: 58,
            avg_trade_return: 4.2,
            max_drawdown: -8.5,
            trades_analyzed: 8,
            portfolio_value_10k: 12210,
            top_trades: [
                { symbol: "BAC", return: 15.2 },
                { symbol: "WFC", return: 12.4 },
                { symbol: "C", return: 8.1 }
            ],
            monthly_returns: [
                { month: "2024-01", return: 2.2 },
                { month: "2024-02", return: 1.1 },
                { month: "2024-03", return: 2.3 },
                { month: "2024-04", return: 1.5 },
                { month: "2024-05", return: 2.8 },
                { month: "2024-06", return: 1.4 },
                { month: "2024-07", return: 2.2 },
                { month: "2024-08", return: 1.3 },
                { month: "2024-09", return: 2.1 },
                { month: "2024-10", return: 1.5 },
                { month: "2024-11", return: 1.9 },
                { month: "2024-12", return: 1.8 }
            ]
        },
        {
            member_id: "M009",
            member_name: "Mark Warner",
            party: "D",
            state: "VA",
            total_return_pct: 35.2,
            benchmark_return_pct: 24.2,
            alpha: 11.0,
            sharpe_ratio: 1.4,
            win_rate: 70,
            avg_trade_return: 6.8,
            max_drawdown: -14.2,
            trades_analyzed: 11,
            portfolio_value_10k: 13520,
            top_trades: [
                { symbol: "PANW", return: 38.2 },
                { symbol: "CRWD", return: 32.4 },
                { symbol: "MSFT", return: 22.1 }
            ],
            monthly_returns: [
                { month: "2024-01", return: 3.2 },
                { month: "2024-02", return: 2.1 },
                { month: "2024-03", return: 4.3 },
                { month: "2024-04", return: -1.5 },
                { month: "2024-05", return: 3.8 },
                { month: "2024-06", return: 5.4 },
                { month: "2024-07", return: 2.2 },
                { month: "2024-08", return: 3.3 },
                { month: "2024-09", return: 4.1 },
                { month: "2024-10", return: 2.5 },
                { month: "2024-11", return: 3.9 },
                { month: "2024-12", return: 1.9 }
            ]
        },
        {
            member_id: "M008",
            member_name: "Ted Cruz",
            party: "R",
            state: "TX",
            total_return_pct: 31.5,
            benchmark_return_pct: 24.2,
            alpha: 7.3,
            sharpe_ratio: 1.2,
            win_rate: 64,
            avg_trade_return: 5.5,
            max_drawdown: -18.9,
            trades_analyzed: 14,
            portfolio_value_10k: 13150,
            top_trades: [
                { symbol: "COIN", return: 42.5 },
                { symbol: "TSLA", return: 25.2 },
                { symbol: "XOM", return: 18.1 }
            ],
            monthly_returns: [
                { month: "2024-01", return: 4.2 },
                { month: "2024-02", return: 3.1 },
                { month: "2024-03", return: -2.3 },
                { month: "2024-04", return: 5.5 },
                { month: "2024-05", return: 2.8 },
                { month: "2024-06", return: 3.4 },
                { month: "2024-07", return: -1.2 },
                { month: "2024-08", return: 4.3 },
                { month: "2024-09", return: 2.1 },
                { month: "2024-10", return: 3.5 },
                { month: "2024-11", return: 2.9 },
                { month: "2024-12", return: 3.2 }
            ]
        }
    ],

    // S&P 500 benchmark monthly returns
    benchmarkReturns: [
        { month: "2024-01", return: 1.6 },
        { month: "2024-02", return: 5.2 },
        { month: "2024-03", return: 3.1 },
        { month: "2024-04", return: -4.2 },
        { month: "2024-05", return: 4.8 },
        { month: "2024-06", return: 3.5 },
        { month: "2024-07", return: 1.1 },
        { month: "2024-08", return: 2.3 },
        { month: "2024-09", return: 2.0 },
        { month: "2024-10", return: -0.9 },
        { month: "2024-11", return: 5.7 },
        { month: "2024-12", return: -2.5 }
    ],

    // Party comparison
    partyComparison: {
        democrat: {
            avgReturn: 38.5,
            avgAlpha: 14.2,
            avgWinRate: 68,
            members: 5
        },
        republican: {
            avgReturn: 37.6,
            avgAlpha: 13.4,
            avgWinRate: 67,
            members: 4
        }
    },

    // Get member performance by ID
    getMemberById(memberId) {
        return this.memberPerformance.find(m => m.member_id === memberId);
    },

    // Get leaderboard sorted by return
    getLeaderboard(sortBy = 'total_return_pct') {
        return [...this.memberPerformance].sort((a, b) => b[sortBy] - a[sortBy]);
    },

    // Calculate portfolio value for a given investment
    calculatePortfolioValue(memberId, initialInvestment, months = 12) {
        const member = this.getMemberById(memberId);
        if (!member) return null;

        let value = initialInvestment;
        const history = [{ month: "Start", value }];

        for (let i = 0; i < Math.min(months, member.monthly_returns.length); i++) {
            const monthReturn = member.monthly_returns[i].return / 100;
            value = value * (1 + monthReturn);
            history.push({
                month: member.monthly_returns[i].month,
                value: Math.round(value)
            });
        }

        return {
            finalValue: Math.round(value),
            totalReturn: ((value - initialInvestment) / initialInvestment * 100).toFixed(1),
            history
        };
    }
};


// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ConvictionMockData,
        CommitteeConnectionMockData,
        CopyCongressMockData
    };
}
