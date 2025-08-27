#!/usr/bin/env python3
"""Test content generation and saving"""

from content_generators import ContentGenerationAgent
from topic_reader import TopicReader

def test_content_generation():
    # Initialize components
    agent = ContentGenerationAgent()
    reader = TopicReader()
    
    # Load a topic
    topics = reader.get_all_topics()
    if not topics:
        print("No topics found")
        return
    
    topic_data = topics[0]  # Use first topic
    print(f"Testing with topic: {topic_data['title']}")
    
    # Generate content for LinkedIn only
    print("\nGenerating content for LinkedIn...")
    results = agent.generate_and_save_content(topic_data, ['linkedin'])
    
    print(f"\nResults: {results}")

if __name__ == "__main__":
    test_content_generation()