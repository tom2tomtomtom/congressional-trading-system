// Congressional Trading Intelligence System - TypeScript Type Definitions

export interface CongressionalMember {
  bioguide_id: string;
  first_name: string;
  last_name: string;
  full_name: string;
  party: 'D' | 'R' | 'I';
  state: string;
  district?: number;
  chamber: 'House' | 'Senate';
  served_from: string;
  served_to?: string;
  birth_date?: string;
  leadership_position?: string;
  net_worth_estimate?: string;
  education?: string[];
  occupation?: string;
  created_at: string;
  updated_at: string;
}

export interface Trade {
  id: number;
  bioguide_id: string;
  transaction_date: string;
  filing_date: string;
  symbol: string;
  transaction_type: 'Purchase' | 'Sale' | 'Exchange';
  amount_min: number;
  amount_max: number;
  amount_mid: number;
  asset_name?: string;
  asset_type: string;
  owner_type: 'Self' | 'Spouse' | 'Dependent Child';
  filing_id?: string;
  filing_delay_days: number;
  source: string;
  raw_data?: any;
  created_at: string;
  updated_at: string;
}

export interface Committee {
  id: number;
  thomas_id: string;
  name: string;
  chamber: 'House' | 'Senate' | 'Joint';
  committee_type: string;
  parent_committee_id?: number;
  jurisdiction?: string;
  website_url?: string;
  created_at: string;
  updated_at: string;
}

export interface CommitteeMembership {
  id: number;
  bioguide_id: string;
  committee_id: number;
  role: 'Chair' | 'Ranking Member' | 'Vice Chair' | 'Member';
  start_date: string;
  end_date?: string;
  congress: number;
  is_current: boolean;
  created_at: string;
  updated_at: string;
}

export interface StockPrice {
  id: number;
  symbol: string;
  date: string;
  open_price: number;
  high_price: number;
  low_price: number;
  close_price: number;
  adjusted_close?: number;
  volume: number;
  source: string;
  created_at: string;
}

export interface Bill {
  id: number;
  bill_id: string;
  title: string;
  bill_type: string;
  number: number;
  congress: number;
  introduced_date?: string;
  latest_action_date?: string;
  latest_action?: string;
  status?: string;
  summary?: string;
  policy_area?: string;
  subjects: string[];
  sponsor_bioguide_id?: string;
  cosponsors_count: number;
  committees: number[];
  created_at: string;
  updated_at: string;
}

// Analytics Types
export interface TradingAnalytics {
  member_id: string;
  total_trades: number;
  total_volume: number;
  avg_trade_size: number;
  performance_metrics: {
    total_return: number;
    annualized_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    alpha: number;
    beta: number;
  };
  top_holdings: Array<{
    symbol: string;
    total_volume: number;
    trade_count: number;
    avg_return: number;
  }>;
}

export interface MarketTrend {
  date: string;
  congressional_volume: number;
  market_return: number;
  congressional_return: number;
  sentiment_score?: number;
}

// ML Model Types
export interface TradePrediction {
  member_id: string;
  symbol: string;
  prediction_date: string;
  trade_probability: number;
  predicted_action: 'Buy' | 'Sell' | 'Hold';
  confidence_score: number;
  key_factors: string[];
  model_version: string;
}

export interface AnomalyDetection {
  anomaly_id: string;
  anomaly_type: 'timing_anomaly' | 'volume_anomaly' | 'coordination_pattern' | 'insider_timing';
  severity: 'low' | 'medium' | 'high' | 'critical';
  member_id: string;
  symbol: string;
  detection_date: string;
  trade_date: string;
  description: string;
  confidence_score: number;
  evidence: Record<string, any>;
  related_members: string[];
  committee_context?: string;
}

// Network Analysis Types
export interface NetworkNode {
  id: string;
  label: string;
  node_type: 'member' | 'committee' | 'stock' | 'bill';
  size: number;
  color: string;
  data: Record<string, any>;
}

export interface NetworkEdge {
  source: string;
  target: string;
  weight: number;
  edge_type: 'trade' | 'membership' | 'cosponsorship' | 'coordination';
  label: string;
  data: Record<string, any>;
}

export interface NetworkAnalysis {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  communities: Record<string, string[]>;
  centrality_metrics: Record<string, {
    degree: number;
    betweenness: number;
    closeness: number;
    pagerank: number;
  }>;
}

// News & Intelligence Types
export interface NewsArticle {
  article_id: string;
  title: string;
  content: string;
  url: string;
  source: string;
  author?: string;
  published_date: string;
  collected_date: string;
  entities: string[];
  keywords: string[];
  sentiment_score: number;
  sentiment_label: 'very_negative' | 'negative' | 'neutral' | 'positive' | 'very_positive';
  relevance_score: number;
  category: string;
}

export interface SentimentAnalysis {
  overall_sentiment: number;
  compound_score: number;
  positive_score: number;
  negative_score: number;
  neutral_score: number;
  confidence: number;
  sentiment_label: 'very_negative' | 'negative' | 'neutral' | 'positive' | 'very_positive';
  key_phrases: string[];
  emotions: Record<string, number>;
}

export interface MarketCorrelation {
  symbol: string;
  correlation_coefficient: number;
  p_value: number;
  time_lag_hours: number;
  sentiment_impact: number;
  statistical_significance: boolean;
}

// Dashboard State Types
export interface DashboardFilters {
  dateRange: {
    start: string;
    end: string;
  };
  members: string[];
  parties: Array<'D' | 'R' | 'I'>;
  chambers: Array<'House' | 'Senate'>;
  committees: string[];
  symbols: string[];
  transactionTypes: Array<'Purchase' | 'Sale' | 'Exchange'>;
  amountRange: {
    min: number;
    max: number;
  };
}

export interface AppState {
  user: {
    preferences: {
      theme: 'light' | 'dark';
      notifications: boolean;
      defaultView: string;
    };
  };
  filters: DashboardFilters;
  loading: {
    members: boolean;
    trades: boolean;
    analytics: boolean;
    network: boolean;
    intelligence: boolean;
  };
  errors: {
    members?: string;
    trades?: string;
    analytics?: string;
    network?: string;
    intelligence?: string;
  };
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  total?: number;
  page?: number;
  pageSize?: number;
}

export interface ApiError {
  message: string;
  code?: string;
  details?: any;
}

// Chart Data Types
export interface ChartDataPoint {
  x: string | number;
  y: number;
  label?: string;
  color?: string;
  size?: number;
}

export interface TimeSeriesData {
  date: string;
  value: number;
  series?: string;
}

export interface HeatmapData {
  x: string;
  y: string;
  value: number;
  label?: string;
}

// Table Types
export interface TableColumn<T = any> {
  id: keyof T;
  label: string;
  minWidth?: number;
  align?: 'left' | 'right' | 'center';
  format?: (value: any) => string;
  sortable?: boolean;
}

export interface TableState {
  page: number;
  pageSize: number;
  sortBy?: string;
  sortDirection: 'asc' | 'desc';
  filters: Record<string, any>;
}

// Form Types
export interface SearchParams {
  query?: string;
  filters?: Record<string, any>;
  sort?: string;
  order?: 'asc' | 'desc';
  page?: number;
  limit?: number;
}

// Utility Types
export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export type DateRange = {
  start: Date | string;
  end: Date | string;
};

export type SortDirection = 'asc' | 'desc';

export type ViewMode = 'table' | 'cards' | 'chart' | 'network';

// Event Types
export interface FilterChangeEvent {
  type: 'filter_change';
  payload: Partial<DashboardFilters>;
}

export interface DataUpdateEvent {
  type: 'data_update';
  payload: {
    dataType: string;
    data: any;
  };
}

export interface ErrorEvent {
  type: 'error';
  payload: {
    message: string;
    code?: string;
    context?: string;
  };
}