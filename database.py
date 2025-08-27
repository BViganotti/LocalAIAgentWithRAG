import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

class DatabaseManager:
    """Manages PostgreSQL database connections and operations"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5433,  # Non-standard port as requested
                 database: str = "content_agent",
                 user: str = "agent_user",
                 password: str = "agent_pass"):
        
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            print("Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            print("Disconnected from database")
    
    def execute_query(self, query: str, params: tuple = None, fetch_one: bool = False, fetch_all: bool = False, commit: bool = True):
        """Execute a database query"""
        if not self.connection:
            if not self.connect():
                raise Exception("Could not connect to database")
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                
                if fetch_one:
                    result = cursor.fetchone()
                    if commit:
                        self.connection.commit()
                    return dict(result) if result else None
                elif fetch_all:
                    results = cursor.fetchall()
                    if commit:
                        self.connection.commit()
                    return [dict(row) for row in results]
                else:
                    if commit:
                        self.connection.commit()
                    return cursor.rowcount
        except Exception as e:
            self.connection.rollback()
            print(f"Database error: {e}")
            raise
    
    def save_generated_content(self, 
                             platform: str,
                             topic_source: str,
                             topic_title: str,
                             content: str,
                             metadata: Dict[str, Any] = None) -> int:
        """Save generated content to database"""
        
        query = """
        INSERT INTO generated_content (platform, topic_source, topic_title, content, metadata)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """
        
        metadata_json = json.dumps(metadata) if metadata else None
        
        result = self.execute_query(
            query, 
            (platform, topic_source, topic_title, content, metadata_json),
            fetch_one=True
        )
        
        return result['id'] if result else None
    
    def get_content_by_platform(self, platform: str, limit: int = 50) -> List[Dict]:
        """Retrieve generated content for a specific platform"""
        
        query = """
        SELECT * FROM generated_content 
        WHERE platform = %s 
        ORDER BY created_at DESC 
        LIMIT %s
        """
        
        return self.execute_query(query, (platform, limit), fetch_all=True, commit=False)
    
    def get_content_by_topic(self, topic_title: str, limit: int = 10) -> List[Dict]:
        """Retrieve generated content for a specific topic"""
        
        query = """
        SELECT * FROM generated_content 
        WHERE topic_title ILIKE %s 
        ORDER BY created_at DESC 
        LIMIT %s
        """
        
        return self.execute_query(query, (f"%{topic_title}%", limit), fetch_all=True, commit=False)
    
    def get_recent_content(self, hours: int = 24, limit: int = 50) -> List[Dict]:
        """Get recently generated content"""
        
        query = """
        SELECT * FROM generated_content 
        WHERE created_at >= NOW() - INTERVAL %s
        ORDER BY created_at DESC 
        LIMIT %s
        """
        
        return self.execute_query(query, (f"{hours} hours", limit), fetch_all=True, commit=False)
    
    def track_inspiration_source(self, platform: str, file_path: str, content_hash: str):
        """Track inspiration sources used"""
        
        query = """
        INSERT INTO inspiration_sources (platform, file_path, content_hash)
        VALUES (%s, %s, %s)
        ON CONFLICT (platform, file_path) DO NOTHING
        """
        
        self.execute_query(query, (platform, file_path, content_hash))
    
    def get_content_stats(self) -> Dict[str, Any]:
        """Get statistics about generated content"""
        
        stats_query = """
        SELECT 
            platform,
            COUNT(*) as count,
            MIN(created_at) as first_generated,
            MAX(created_at) as last_generated
        FROM generated_content 
        GROUP BY platform
        """
        
        platform_stats = self.execute_query(stats_query, fetch_all=True, commit=False)
        
        total_query = "SELECT COUNT(*) as total FROM generated_content"
        total_result = self.execute_query(total_query, fetch_one=True, commit=False)
        
        return {
            'total_content': total_result['total'] if total_result else 0,
            'platform_breakdown': platform_stats
        }

# Global database instance
db_manager = DatabaseManager()

def get_database():
    """Get the global database manager instance"""
    return db_manager