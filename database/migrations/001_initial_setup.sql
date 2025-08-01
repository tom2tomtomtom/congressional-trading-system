-- Congressional Trading Intelligence System - Initial Database Setup
-- Migration 001: Create core database structure
-- Created: January 31, 2025

-- This migration creates the foundational database structure for Phase 1

BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create enum types for data validation
CREATE TYPE member_party AS ENUM ('D', 'R', 'I');
CREATE TYPE member_chamber AS ENUM ('House', 'Senate');
CREATE TYPE transaction_type AS ENUM ('Purchase', 'Sale', 'Exchange');
CREATE TYPE owner_type AS ENUM ('Self', 'Spouse', 'Dependent Child');
CREATE TYPE committee_chamber AS ENUM ('House', 'Senate', 'Joint');
CREATE TYPE committee_role AS ENUM ('Chair', 'Ranking Member', 'Vice Chair', 'Member');
CREATE TYPE data_quality_status AS ENUM ('PASS', 'WARN', 'FAIL');

-- Create core tables by importing main schema
\i '/Users/thomasdowuona-hyde/congressional-trading-system/database/schema.sql'

-- Insert initial configuration data
INSERT INTO data_quality_metrics (table_name, metric_name, metric_value, expected_value, status, details)
VALUES 
    ('members', 'total_count', 0, 535, 'WARN', 'Initial setup - members to be loaded'),
    ('trades', 'total_count', 0, 1000, 'WARN', 'Initial setup - historical trades to be loaded'),
    ('committees', 'total_count', 0, 25, 'WARN', 'Initial setup - committees to be loaded');

-- Create indexes for migration tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    description TEXT
);

-- Record this migration
INSERT INTO schema_migrations (version, description) 
VALUES ('001', 'Initial database setup with core congressional trading schema');

COMMIT;