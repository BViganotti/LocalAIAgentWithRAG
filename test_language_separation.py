#!/usr/bin/env python3
"""
Test script for language separation functionality.

This script tests the language detection, data ingestion, and content generation
with language-specific settings.
"""

import os
import sys
from language_utils import Language, LanguageDetector, LanguageManager
from data_ingestion import LanguageSeparatedDataIngestion
from advanced_content_pipeline import get_advanced_content_pipeline

def test_language_detection():
    """Test language detection functionality"""
    print("🔍 Testing Language Detection")
    print("=" * 40)
    
    test_cases = [
        ("This is a great business opportunity for entrepreneurs", Language.ENGLISH),
        ("To jest świetna okazja biznesowa dla przedsiębiorców", Language.POLISH),
        ("How to build a successful startup from scratch", Language.ENGLISH),
        ("Jak zbudować udany startup od podstaw", Language.POLISH),
        ("The future of technology and innovation", Language.ENGLISH),
        ("Przyszłość technologii i innowacji", Language.POLISH),
    ]
    
    correct = 0
    for text, expected in test_cases:
        detected = LanguageDetector.detect_language(text)
        is_correct = detected == expected
        correct += is_correct
        
        print(f"Text: {text[:50]}...")
        print(f"Expected: {expected.value}, Detected: {detected.value} {'✅' if is_correct else '❌'}")
        print()
    
    accuracy = (correct / len(test_cases)) * 100
    print(f"Language Detection Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")

def test_language_manager():
    """Test language manager configurations"""
    print("\n🌍 Testing Language Manager")
    print("=" * 40)
    
    for language in [Language.ENGLISH, Language.POLISH]:
        config = LanguageManager.get_config(language)
        collection_name = LanguageManager.get_collection_name(language)
        data_dir = LanguageManager.get_data_directory(language)
        prompt = LanguageManager.get_prompt(language, 'generation_system')
        
        print(f"Language: {LanguageManager.get_language_display_name(language)}")
        print(f"  Collection: {collection_name}")
        print(f"  Data Dir: {data_dir}")
        print(f"  System Prompt: {prompt[:100]}...")
        print()

def test_data_ingestion_status():
    """Test data ingestion status"""
    print("📊 Testing Data Ingestion Status")
    print("=" * 40)
    
    ingestion = LanguageSeparatedDataIngestion()
    status = ingestion.get_ingestion_status()
    
    print(f"Data Hash: {status['data_hash'][:16]}...")
    print(f"Last Processed: {status['last_processed']}")
    
    print("\nEnglish Collections:")
    for platform, info in status['english_collections'].items():
        if 'error' in info:
            print(f"  {platform}: {info['error']}")
        else:
            print(f"  {platform}: {info['posts_count']} posts")
    
    print("\nPolish Collections:")  
    for platform, info in status['polish_collections'].items():
        if 'error' in info:
            print(f"  {platform}: {info['error']}")
        else:
            print(f"  {platform}: {info['posts_count']} posts")

def test_pipeline_creation():
    """Test pipeline creation with different languages"""
    print("\n🚀 Testing Pipeline Creation")
    print("=" * 40)
    
    for language_code in ['en', 'pl']:
        try:
            pipeline = get_advanced_content_pipeline(language=language_code)
            language_obj = Language(language_code)
            
            print(f"✅ Created pipeline for {LanguageManager.get_language_display_name(language_obj)}")
            print(f"   Collection: {pipeline.collection_name}")
            print(f"   Language: {pipeline.language}")
            print()
            
        except Exception as e:
            print(f"❌ Failed to create pipeline for {language_code}: {e}")

def test_topic_with_language_detection():
    """Test topic processing with language detection"""
    print("📝 Testing Topic Language Detection")
    print("=" * 40)
    
    # Test with the ReturnEase topic (Polish)
    topic_file = "topic/returnease.txt"
    
    if os.path.exists(topic_file):
        with open(topic_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        detected_lang = LanguageDetector.detect_language(content)
        print(f"ReturnEase topic detected language: {detected_lang.value} ({LanguageManager.get_language_display_name(detected_lang)})")
        
        # Test content generation with detected language
        try:
            pipeline = get_advanced_content_pipeline(language=detected_lang.value)
            print(f"✅ Pipeline created for detected language: {detected_lang.value}")
            
        except Exception as e:
            print(f"❌ Failed to create pipeline for detected language: {e}")
    else:
        print(f"❌ Topic file not found: {topic_file}")

def main():
    """Main test function"""
    print("🧪 Language Separation System Tests")
    print("=" * 60)
    
    try:
        test_language_detection()
        test_language_manager()
        test_data_ingestion_status()
        test_pipeline_creation()
        test_topic_with_language_detection()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()