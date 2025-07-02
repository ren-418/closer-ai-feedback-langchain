-- AI Sales Call Evaluator Database Schema
-- Supabase/PostgreSQL compatible

-- Users table (admin accounts only)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now()),
    closer_name VARCHAR(255),
    closer_email VARCHAR(255) REFERENCES closers(email) ON DELETE SET NULL,
    transcript_text TEXT,
    transcript_length INT,
    call_date DATE,
    status VARCHAR(50),
    overall_score FLOAT,
    letter_grade VARCHAR(5),
    analysis_timestamp TIMESTAMP WITH TIME ZONE,
    total_chunks INT,
    total_reference_files_used INT
);

-- Call analyses table (detailed analysis results)

CREATE TABLE IF NOT EXISTS call_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    chunk_number INTEGER NOT NULL,
    total_chunks INTEGER NOT NULL,
    chunk_text_preview TEXT,
    analysis_data JSONB NOT NULL, -- Store the full analysis JSON
    reference_files_used JSONB, -- Store reference file tracking
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Final analysis table (aggregated results)
CREATE TABLE IF NOT EXISTS final_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id UUID REFERENCES calls(id) ON DELETE CASCADE,
    analysis_data JSONB NOT NULL, -- Store the complete final analysis
    report_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance metrics table (for analytics)
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
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
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc', now())
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_calls_status ON calls(status);
CREATE INDEX IF NOT EXISTS idx_calls_date ON calls(call_date);
CREATE INDEX IF NOT EXISTS idx_calls_score ON calls(overall_score);
CREATE INDEX IF NOT EXISTS idx_call_analyses_call_id ON call_analyses(call_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_date ON performance_metrics(created_at);
CREATE INDEX IF NOT EXISTS idx_calls_closer_email ON calls(closer_email);

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

CREATE TRIGGER update_calls_updated_at BEFORE UPDATE ON calls
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sample data for testing
INSERT INTO closers (name, email, hire_date) VALUES 
    ('John Doe', 'john@company.com', '2024-01-15'),
    ('Jane Smith', 'jane@company.com', '2024-02-01'),
    ('Mike Johnson', 'mike@company.com', '2024-03-10')
ON CONFLICT DO NOTHING; 