-- AI Sales Call Evaluator Database Schema for Supabase
-- Run this in your Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table (admin accounts only)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'admin',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Closers table (sales team members)
CREATE TABLE IF NOT EXISTS closers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    hire_date DATE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Calls table (main call records)
CREATE TABLE IF NOT EXISTS calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closer_name VARCHAR(255),
    lead_name VARCHAR(255),
    closer_email VARCHAR(255),
    transcript_link TEXT,
    transcript_text TEXT,
    transcript_length INT,
    call_date DATE,
    status VARCHAR(50),
    overall_score FLOAT,
    letter_grade VARCHAR(5),
    analysis_timestamp TIMESTAMP WITH TIME ZONE,
    total_chunks INT,
    total_reference_files_used INT,
    is_read BOOLEAN DEFAULT false
);

-- Admin call reads tracking table
CREATE TABLE IF NOT EXISTS admin_call_reads (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    admin_email VARCHAR(255) NOT NULL,
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    read_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(admin_email, call_id)
);

-- Call analyses table (detailed analysis results)
CREATE TABLE IF NOT EXISTS call_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    chunk_number INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    chunk_text_preview TEXT,
    analysis_data JSONB NOT NULL,
    reference_files_used JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Final analysis table (aggregated results)
CREATE TABLE IF NOT EXISTS final_analyses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    analysis_data JSONB NOT NULL,
    report_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics table (for analytics)
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    overall_score FLOAT,
    letter_grade VARCHAR(5),
    rapport_building_score FLOAT,
    discovery_score FLOAT,
    objection_handling_score FLOAT,
    pitch_delivery_score FLOAT,
    closing_effectiveness_score FLOAT,
    total_objections INT,
    total_questions INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Custom evaluation criteria table (for configurable business rules)
CREATE TABLE IF NOT EXISTS evaluation_criteria (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    criteria_name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    violation_text VARCHAR(500) NOT NULL,
    correct_text VARCHAR(500),
    score_penalty INTEGER DEFAULT 0,
    feedback_message TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    category VARCHAR(100) DEFAULT 'general',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Criteria violations tracking table
CREATE TABLE IF NOT EXISTS criteria_violations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    criteria_id UUID REFERENCES evaluation_criteria(id) ON DELETE CASCADE,
    violation_text_found TEXT NOT NULL,
    context TEXT,
    chunk_number INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(status);
CREATE INDEX IF NOT EXISTS idx_calls_date ON calls(call_date);
CREATE INDEX IF NOT EXISTS idx_calls_score ON calls(overall_score);
CREATE INDEX IF NOT EXISTS idx_admin_call_reads_admin ON admin_call_reads(admin_email);
CREATE INDEX IF NOT EXISTS idx_admin_call_reads_call ON admin_call_reads(call_id);
CREATE INDEX IF NOT EXISTS idx_call_analyses_call_id ON call_analyses(call_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_date ON performance_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_calls_closer_email ON calls(closer_email);
CREATE INDEX IF NOT EXISTS idx_evaluation_criteria_active ON evaluation_criteria(is_active);
CREATE INDEX IF NOT EXISTS idx_criteria_violations_call_id ON criteria_violations(call_id);

-- Triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_closers_updated_at BEFORE UPDATE ON closers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_evaluation_criteria_updated_at BEFORE UPDATE ON evaluation_criteria
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sample data for testing
INSERT INTO closers (name, email, hire_date) VALUES 
    ('John Doe', 'john@company.com', '2024-01-15'),
    ('Jane Smith', 'jane@company.com', '2024-02-01'),
    ('Mike Johnson', 'mike@company.com', '2024-03-10')
ON CONFLICT DO NOTHING;

-- Sample evaluation criteria for testing
INSERT INTO evaluation_criteria (criteria_name, description, violation_text, correct_text, score_penalty, feedback_message, category) VALUES 
    ('currency_violation', 'Must use USD, not pounds', 'pounds', 'dollars', -2, 'Used incorrect currency - all transactions must be in USD', 'currency'),
    ('currency_violation_euro', 'Must use USD, not euros', 'euro', 'dollars', -2, 'Used incorrect currency - all transactions must be in USD', 'currency'),
    ('currency_violation_quid', 'Must use USD, not quid', 'quid', 'dollars', -2, 'Used incorrect currency - all transactions must be in USD', 'currency')
ON CONFLICT DO NOTHING;

-- Enable Row Level Security (RLS) for better security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE closers ENABLE ROW LEVEL SECURITY;
ALTER TABLE calls ENABLE ROW LEVEL SECURITY;
ALTER TABLE call_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE final_analyses ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE evaluation_criteria ENABLE ROW LEVEL SECURITY;
ALTER TABLE criteria_violations ENABLE ROW LEVEL SECURITY;

-- Create policies for public access (you can modify these later for better security)
CREATE POLICY "Enable read access for all users" ON users FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON users FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON users FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON users FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON closers FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON closers FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON closers FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON closers FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON calls FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON calls FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON calls FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON calls FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON admin_call_reads FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON admin_call_reads FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON admin_call_reads FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON admin_call_reads FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON call_analyses FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON call_analyses FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable delete access for all users" ON call_analyses FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON final_analyses FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON final_analyses FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable delete access for all users" ON final_analyses FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON performance_metrics FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON performance_metrics FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable delete access for all users" ON performance_metrics FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON evaluation_criteria FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON evaluation_criteria FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable update access for all users" ON evaluation_criteria FOR UPDATE USING (true);
CREATE POLICY "Enable delete access for all users" ON evaluation_criteria FOR DELETE USING (true);

CREATE POLICY "Enable read access for all users" ON criteria_violations FOR SELECT USING (true);
CREATE POLICY "Enable insert access for all users" ON criteria_violations FOR INSERT WITH CHECK (true);
CREATE POLICY "Enable delete access for all users" ON criteria_violations FOR DELETE USING (true);

-- Success message
SELECT 'Database schema created successfully!' as status; 