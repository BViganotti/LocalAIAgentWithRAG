#!/usr/bin/env python3
"""Debug script to check database content"""

from database import get_database
import sys

def debug_database():
    db = get_database()
    
    try:
        # Connect to database
        if not db.connect():
            print("Failed to connect to database")
            return
        
        # Check total count
        count_query = "SELECT COUNT(*) as total FROM generated_content"
        result = db.execute_query(count_query, fetch_one=True, commit=False)
        print(f"Total records in generated_content: {result['total'] if result else 0}")
        
        # Check all records
        all_query = "SELECT id, platform, topic_title, created_at FROM generated_content ORDER BY created_at DESC"
        all_records = db.execute_query(all_query, fetch_all=True, commit=False)
        
        if all_records:
            print(f"\nAll records ({len(all_records)}):")
            for record in all_records:
                print(f"  ID: {record['id']}, Platform: {record['platform']}, Topic: {record['topic_title'][:50]}..., Created: {record['created_at']}")
        else:
            print("No records found")
        
        # Test recent content query with debug info
        print(f"\nTesting recent content query...")
        recent_query = """
        SELECT id, platform, topic_title, created_at,
               NOW() as current_time,
               NOW() - INTERVAL '24 hours' as cutoff_time,
               (created_at >= NOW() - INTERVAL '24 hours') as is_recent
        FROM generated_content 
        ORDER BY created_at DESC 
        LIMIT 10
        """
        recent_records = db.execute_query(recent_query, fetch_all=True, commit=False)
        
        if recent_records:
            print(f"Recent query debug results ({len(recent_records)}):")
            for record in recent_records:
                print(f"  ID: {record['id']}, Created: {record['created_at']}, Current: {record['current_time']}, Cutoff: {record['cutoff_time']}, Is Recent: {record['is_recent']}")
        else:
            print("No records in recent query debug")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.disconnect()

if __name__ == "__main__":
    debug_database()