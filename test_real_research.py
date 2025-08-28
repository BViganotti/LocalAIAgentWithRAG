#!/usr/bin/env python3
"""
Test script for the enhanced research engine with real API integration.
Demonstrates the difference between fake research data and real research data.
"""

import sys
from advanced_content_pipeline import EnhancedResearchEngine

def test_research_capabilities():
    """Test both fake and real research capabilities"""
    
    print("🔬 Testing Enhanced Research Engine")
    print("=" * 60)
    
    # Test topic
    topic_data = {
        'title': 'Artificial Intelligence in Healthcare',
        'content': 'AI is transforming healthcare through machine learning algorithms that can diagnose diseases, predict patient outcomes, and assist in treatment planning. Recent studies show significant improvements in diagnostic accuracy.'
    }
    
    print("\n1️⃣  Testing with FAKE research (old behavior):")
    print("-" * 50)
    fake_engine = EnhancedResearchEngine(enable_real_research=False)
    fake_results = fake_engine.enhance_research(topic_data)
    
    print(f"✅ Factual Claims Found: {len(fake_results.factual_claims)}")
    print(f"✅ Verified Sources: {len(fake_results.verified_sources)}")
    print(f"✅ Expert Quotes: {len(fake_results.expert_quotes)}")
    print(f"✅ Academic References: {len(fake_results.academic_references)}")
    
    print("\n2️⃣  Testing with REAL research (new behavior):")
    print("-" * 50)
    real_engine = EnhancedResearchEngine(enable_real_research=True)
    real_results = real_engine.enhance_research(topic_data)
    
    print(f"📚 Factual Claims Found: {len(real_results.factual_claims)}")
    print(f"📚 Verified Sources: {len(real_results.verified_sources)}")
    print(f"📚 Expert Quotes: {len(real_results.expert_quotes)}")
    print(f"📚 Academic References: {len(real_results.academic_references)}")
    
    print("\n🔍 Sample Real Research Results:")
    print("-" * 50)
    
    if real_results.verified_sources:
        print("\n📖 Sample Credible Sources:")
        for i, source in enumerate(real_results.verified_sources[:2], 1):
            print(f"  {i}. {source.get('title', 'Unknown Title')[:60]}...")
            print(f"     Source: {source.get('source', 'Unknown')}")
            print(f"     URL: {source.get('url', 'No URL')}")
    
    if real_results.academic_references:
        print("\n🎓 Sample Academic Papers:")
        for i, paper in enumerate(real_results.academic_references[:2], 1):
            print(f"  {i}. {paper.get('title', 'Unknown Title')[:60]}...")
            print(f"     Authors: {paper.get('authors', 'Unknown')}")
            print(f"     Year: {paper.get('year', 'Unknown')}")
            print(f"     Source: {paper.get('source', 'Unknown')}")
    
    if real_results.expert_quotes:
        print("\n💬 Sample Expert Perspectives:")
        for i, quote in enumerate(real_results.expert_quotes[:1], 1):
            print(f"  {i}. \"{quote.get('quote', 'No quote')[:100]}...\"")
            print(f"     Expert: {quote.get('expert', 'Unknown')}")
            print(f"     Relevance: {quote.get('relevance', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("✅ Research engine testing completed!")
    print("\n📋 Summary:")
    print(f"   • Real research provides {len(real_results.verified_sources)} credible sources")
    print(f"   • Real research provides {len(real_results.academic_references)} academic papers")
    print(f"   • Integration with Semantic Scholar, arXiv, and Wikipedia")
    print(f"   • Proper fact-checking with multiple verification sources")

def test_individual_apis():
    """Test individual API endpoints"""
    print("\n🧪 Testing Individual API Endpoints:")
    print("-" * 50)
    
    engine = EnhancedResearchEngine(enable_real_research=True)
    
    # Test Semantic Scholar
    print("\n1. Testing Semantic Scholar API...")
    semantic_results = engine._search_semantic_scholar("artificial intelligence healthcare", limit=2)
    print(f"   Found {len(semantic_results)} papers")
    
    # Test arXiv
    print("\n2. Testing arXiv API...")
    arxiv_results = engine._search_arxiv("artificial intelligence healthcare", max_results=2)
    print(f"   Found {len(arxiv_results)} preprints")
    
    # Test Wikipedia
    print("\n3. Testing Wikipedia API...")
    wiki_result = engine._search_wikipedia("Artificial intelligence in healthcare")
    if wiki_result:
        print(f"   Found Wikipedia article: {wiki_result.get('title', 'Unknown')}")
    else:
        print("   No Wikipedia article found")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Research Engine Tests")
    test_research_capabilities()
    test_individual_apis()
    print("\n🎉 All tests completed!")