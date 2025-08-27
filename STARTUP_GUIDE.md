# 🚀 Social Media Content Generation Agent - Startup Guide

## Overview

Your project has been transformed from a restaurant review chatbot into a powerful social media content generation agent that creates high-impact posts for Reddit, LinkedIn, and Twitter using AI and inspiration from successful content.

## 🏗️ Architecture Overview

- **Qdrant Vector Database**: Stores inspiration content for similarity search
- **PostgreSQL Database**: Stores generated articles and metadata
- **Ollama LLM**: Generates platform-specific content
- **Inspiration Base**: Your existing data files (LinkedIn, Reddit, Twitter JSON/TXT)
- **Topic Processing**: Handles PDF, TXT, and MD files from the topic directory

## 🔧 Prerequisites

1. **Docker & Docker Compose** (for Qdrant and PostgreSQL)
2. **Python 3.8+**
3. **Ollama** with `gpt-oss` and `nomic-embed-text-v2-moe` models

### Install Ollama Models
```bash
ollama pull gpt-oss
ollama pull nomic-embed-text-v2-moe
```

## 🚀 Getting Started

### Step 1: Start the Infrastructure
```bash
# Start Qdrant and PostgreSQL containers
docker-compose up -d

# Verify containers are running
docker ps
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Add Topic Files
Add your topic files to the `topic/` directory:
- **PDF files**: Research papers, articles, reports
- **TXT files**: Plain text content, notes, ideas  
- **MD files**: Markdown documents, formatted content

Examples provided:
- `topic/artificial_intelligence_trends.md`
- `topic/remote_work_productivity.txt`

### Step 4: Run the Agent
```bash
python main.py
```

## 📋 How It Works

### 1. Inspiration Base Loading
- Loads high-performing posts from your `data/` directory
- JSON files contain structured post data with metadata
- TXT files contain additional inspiration content
- Creates embeddings and stores in Qdrant for similarity search

### 2. Topic Processing
- Reads PDF, TXT, and MD files from `topic/` directory
- Extracts titles and content automatically
- Handles various document formats

### 3. Content Generation
- Uses topic content to generate platform-specific posts
- Retrieves relevant inspiration posts using vector similarity
- Applies platform-specific templates and best practices
- Saves generated content to PostgreSQL database

### 4. Platform Optimization

#### LinkedIn
- Professional tone with actionable insights
- Bullet points and structured formatting
- Engagement-driving questions
- 1300-3000 character optimization

#### Reddit
- Conversational and authentic tone
- Community-focused approach
- Story-telling elements
- Anti-promotional language

#### Twitter
- Concise, punchy content
- Thread format for complex topics  
- Emoji and hashtag optimization
- 280-character tweet optimization

## 🎯 Usage Examples

### Generate Content for Specific Topic
1. Launch the agent: `python main.py`
2. Select option 1: "Generate content for a specific topic"
3. Choose your topic from the list
4. Select platform(s) (LinkedIn, Reddit, Twitter, or All)
5. Review and save generated content

### Batch Generate for All Topics
1. Select option 2: "Generate content for all topics"
2. Agent processes all files in `topic/` directory
3. Generates content for all platforms
4. Saves everything to database

### View Generated Content
- Option 3: View recent generated content
- Option 4: View content statistics and metrics

## 📊 Database Schema

### Generated Content Table
- `platform`: Target social media platform
- `topic_source`: Source file path
- `topic_title`: Extracted or derived title
- `content`: Generated social media content
- `metadata`: Word count, character count, etc.
- `created_at`: Timestamp

### Inspiration Sources Table
- Tracks which inspiration files have been indexed
- Prevents duplicate processing
- Maintains content hashes for change detection

## 🔍 Troubleshooting

### "Could not connect to database"
- Ensure Docker containers are running: `docker ps`
- Check container logs: `docker logs content_agent_postgres`

### "Failed to initialize Qdrant"  
- Verify Qdrant container is running on port 6333
- Check container logs: `docker logs content_agent_qdrant`

### "No inspiration documents found"
- Ensure data files exist in `data/` directory
- Check file permissions and formats
- Verify JSON file structure matches expected format

### "Error with Ollama models"
- Confirm Ollama is running: `ollama list`
- Pull required models: `ollama pull gpt-oss` and `ollama pull nomic-embed-text-v2-moe`

## 🔧 Configuration

### Database Settings
- PostgreSQL runs on port **5433** (non-standard as requested)
- Default credentials: `agent_user` / `agent_pass`
- Database name: `content_agent`

### Qdrant Settings  
- Runs on port 6333 (HTTP) and 6334 (gRPC)
- Collection name: `inspiration_content`
- Vector dimensions: 768 (for nomic-embed-text-v2-moe)

## 📈 Advanced Usage

### Custom Prompts
Modify templates in `content_generators.py` to customize:
- Platform-specific writing styles
- Industry-specific terminology
- Brand voice and tone
- Content structure preferences

### Adding New Platforms
Extend the system by creating new generator classes:
1. Inherit from `ContentGenerator`
2. Define platform-specific templates
3. Add to `ContentGenerationAgent.generators`

### Inspiration Data Management
- Add new inspiration files to `data/` directory
- System automatically detects and indexes new content
- Supports both JSON (structured) and TXT (unstructured) formats

## 🎉 Success Metrics

The system tracks:
- Content generation success rates
- Platform-specific performance metrics
- Topic processing statistics
- Database storage efficiency
- Inspiration source utilization

Your social media content generation agent is now ready to create high-impact posts based on your topics and inspired by successful content patterns!