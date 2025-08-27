-- Initialize database schema for content generation agent

-- Table to store generated articles/posts
CREATE TABLE generated_content (
    id SERIAL PRIMARY KEY,
    platform VARCHAR(20) NOT NULL CHECK (platform IN ('reddit', 'linkedin', 'twitter')),
    topic_source VARCHAR(255) NOT NULL,
    topic_title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

-- Index for efficient querying
CREATE INDEX idx_generated_content_platform ON generated_content(platform);
CREATE INDEX idx_generated_content_created_at ON generated_content(created_at);
CREATE INDEX idx_generated_content_topic ON generated_content(topic_title);

-- Table to track inspiration sources used
CREATE TABLE inspiration_sources (
    id SERIAL PRIMARY KEY,
    platform VARCHAR(20) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(platform, file_path)
);