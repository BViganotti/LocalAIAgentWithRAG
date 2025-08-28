#!/usr/bin/env python3
"""
Web interface for viewing generated articles from the Content Agent pipeline.
Provides a clean, readable interface for browsing and reading articles.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
from datetime import datetime
from typing import List, Dict, Any
from database import get_database
import os

app = Flask(__name__)

# Initialize database
db = get_database()

def format_content(content: str, platform: str) -> str:
    """Format content for better display based on platform"""
    if platform == 'newsletter':
        # Newsletter content often has markdown-style formatting
        content = content.replace('**', '<strong>').replace('**', '</strong>')
        content = content.replace('*', '<em>').replace('*', '</em>')
        content = content.replace('\n\n', '</p><p>')
        content = content.replace('\n', '<br>')
        if not content.startswith('<p>'):
            content = '<p>' + content
        if not content.endswith('</p>'):
            content = content + '</p>'
    elif platform == 'linkedin':
        # LinkedIn content - preserve line breaks and formatting
        content = content.replace('\n\n', '</p><p>')
        content = content.replace('\n', '<br>')
        content = '<p>' + content + '</p>'
    elif platform == 'twitter':
        # Twitter - preserve hashtags and mentions
        content = content.replace('#', '<span class="hashtag">#')
        content = content.replace(' ', '</span> ', 1) if '<span class="hashtag">' in content else content
        content = '<p>' + content + '</p>'
    else:
        # Default formatting
        content = content.replace('\n\n', '</p><p>')
        content = content.replace('\n', '<br>')
        content = '<p>' + content + '</p>'
    
    return content

def get_platform_icon(platform: str) -> str:
    """Get platform icon"""
    icons = {
        'linkedin': '💼',
        'twitter': '🐦',
        'reddit': '🔴',
        'newsletter': '📧'
    }
    return icons.get(platform, '📄')

def format_metadata(metadata_json: str) -> Dict[str, Any]:
    """Format metadata for display"""
    if not metadata_json:
        return {}
    
    try:
        metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
        return metadata
    except:
        return {}

@app.route('/')
def index():
    """Main page showing recent articles"""
    try:
        if not db.connect():
            return render_template('error.html', 
                                 error="Could not connect to database. Please ensure PostgreSQL is running.")
        
        # Get recent articles
        recent_articles = db.get_recent_content(hours=168, limit=50)  # Last week
        
        # Format articles for display
        formatted_articles = []
        for article in recent_articles:
            formatted_article = {
                'id': article['id'],
                'platform': article['platform'],
                'platform_icon': get_platform_icon(article['platform']),
                'topic_title': article['topic_title'],
                'topic_source': article['topic_source'],
                'content_preview': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                'content_full': format_content(article['content'], article['platform']),
                'created_at': article['created_at'].strftime('%Y-%m-%d %H:%M') if article['created_at'] else 'Unknown',
                'metadata': format_metadata(article.get('metadata')),
                'word_count': len(article['content'].split()),
                'char_count': len(article['content'])
            }
            formatted_articles.append(formatted_article)
        
        # Get platform stats
        stats = db.get_content_stats()
        
        return render_template('index.html', 
                             articles=formatted_articles,
                             stats=stats)
    
    except Exception as e:
        return render_template('error.html', error=f"Error loading articles: {e}")
    
    finally:
        db.disconnect()

@app.route('/article/<int:article_id>')
def article_detail(article_id: int):
    """Show detailed view of a specific article"""
    try:
        if not db.connect():
            return render_template('error.html', 
                                 error="Could not connect to database. Please ensure PostgreSQL is running.")
        
        # Get specific article
        query = "SELECT * FROM generated_content WHERE id = %s"
        article = db.execute_query(query, (article_id,), fetch_one=True, commit=False)
        
        if not article:
            return render_template('error.html', error="Article not found.")
        
        # Format article for display
        formatted_article = {
            'id': article['id'],
            'platform': article['platform'],
            'platform_icon': get_platform_icon(article['platform']),
            'topic_title': article['topic_title'],
            'topic_source': article['topic_source'],
            'content': format_content(article['content'], article['platform']),
            'content_raw': article['content'],
            'created_at': article['created_at'].strftime('%Y-%m-%d %H:%M:%S') if article['created_at'] else 'Unknown',
            'metadata': format_metadata(article.get('metadata')),
            'word_count': len(article['content'].split()),
            'char_count': len(article['content'])
        }
        
        return render_template('article.html', article=formatted_article)
    
    except Exception as e:
        return render_template('error.html', error=f"Error loading article: {e}")
    
    finally:
        db.disconnect()

@app.route('/platform/<platform>')
def platform_articles(platform: str):
    """Show articles for a specific platform"""
    try:
        if not db.connect():
            return render_template('error.html', 
                                 error="Could not connect to database. Please ensure PostgreSQL is running.")
        
        # Get articles for platform
        articles = db.get_content_by_platform(platform, limit=100)
        
        # Format articles for display
        formatted_articles = []
        for article in articles:
            formatted_article = {
                'id': article['id'],
                'platform': article['platform'],
                'platform_icon': get_platform_icon(article['platform']),
                'topic_title': article['topic_title'],
                'topic_source': article['topic_source'],
                'content_preview': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                'created_at': article['created_at'].strftime('%Y-%m-%d %H:%M') if article['created_at'] else 'Unknown',
                'metadata': format_metadata(article.get('metadata')),
                'word_count': len(article['content'].split()),
                'char_count': len(article['content'])
            }
            formatted_articles.append(formatted_article)
        
        return render_template('platform.html', 
                             articles=formatted_articles,
                             platform=platform,
                             platform_icon=get_platform_icon(platform))
    
    except Exception as e:
        return render_template('error.html', error=f"Error loading platform articles: {e}")
    
    finally:
        db.disconnect()

@app.route('/search')
def search():
    """Search articles by topic"""
    query = request.args.get('q', '').strip()
    
    if not query:
        return redirect(url_for('index'))
    
    try:
        if not db.connect():
            return render_template('error.html', 
                                 error="Could not connect to database. Please ensure PostgreSQL is running.")
        
        # Search articles
        articles = db.get_content_by_topic(query, limit=50)
        
        # Format articles for display
        formatted_articles = []
        for article in articles:
            formatted_article = {
                'id': article['id'],
                'platform': article['platform'],
                'platform_icon': get_platform_icon(article['platform']),
                'topic_title': article['topic_title'],
                'topic_source': article['topic_source'],
                'content_preview': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                'created_at': article['created_at'].strftime('%Y-%m-%d %H:%M') if article['created_at'] else 'Unknown',
                'metadata': format_metadata(article.get('metadata')),
                'word_count': len(article['content'].split()),
                'char_count': len(article['content'])
            }
            formatted_articles.append(formatted_article)
        
        return render_template('search.html', 
                             articles=formatted_articles,
                             query=query,
                             count=len(formatted_articles))
    
    except Exception as e:
        return render_template('error.html', error=f"Error searching articles: {e}")
    
    finally:
        db.disconnect()

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    try:
        if not db.connect():
            return jsonify({'error': 'Database connection failed'}), 500
        
        stats = db.get_content_stats()
        return jsonify(stats)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        db.disconnect()

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    # Create static directory if it doesn't exist
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
    
    print("🌐 Starting Content Agent Web Interface...")
    print("📡 Server will run on http://localhost:3600")
    print("📚 Database connection will be established on first request")
    
    app.run(host='localhost', port=3600, debug=True)