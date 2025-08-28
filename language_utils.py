"""
Language Detection and Management Utilities

This module provides language detection, validation, and management
capabilities for the content generation pipeline.
"""

from typing import Dict, List, Optional
from enum import Enum
import re


class Language(Enum):
    """Supported languages for content generation"""
    ENGLISH = "en"
    POLISH = "pl"


class LanguageDetector:
    """Simple language detection based on common words and patterns"""
    
    # Common Polish words that rarely appear in English
    POLISH_INDICATORS = {
        'words': [
            'że', 'i', 'w', 'na', 'z', 'do', 'nie', 'się', 'od', 'po', 'o', 'przez',
            'ale', 'tak', 'jak', 'może', 'może', 'będzie', 'tylko', 'bardzo', 'już',
            'wszystko', 'wszystkie', 'wszystkich', 'Polski', 'Polska', 'polskich',
            'które', 'która', 'który', 'naszych', 'nasza', 'nasz', 'tych', 'tej', 'ten',
            'biznes', 'firma', 'firmy', 'przedsiębiorstwo', 'rynek', 'klient', 'klienci'
        ],
        'patterns': [
            r'\b\w+ować\b',  # Polish verb endings
            r'\b\w+acja\b',  # Polish noun endings
            r'\b\w+ński\b',  # Polish adjective endings
            r'\b\w+ość\b',   # Polish noun endings
        ]
    }
    
    # Common English words that rarely appear in Polish
    ENGLISH_INDICATORS = {
        'words': [
            'the', 'and', 'that', 'have', 'for', 'not', 'with', 'you', 'this', 'but',
            'his', 'from', 'they', 'she', 'her', 'been', 'than', 'its', 'who', 'did',
            'business', 'company', 'market', 'customer', 'customers', 'growth', 'success',
            'strategy', 'management', 'team', 'product', 'service', 'technology'
        ],
        'patterns': [
            r'\b\w+ing\b',   # English gerunds
            r'\b\w+ed\b',    # English past tense
            r'\b\w+tion\b',  # English noun endings
            r'\b\w+ly\b',    # English adverbs
        ]
    }
    
    @classmethod
    def detect_language(cls, text: str) -> Language:
        """
        Detect language of the given text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Detected language (English or Polish)
        """
        if not text or not text.strip():
            return Language.ENGLISH  # Default to English
        
        text_lower = text.lower()
        words = text_lower.split()
        
        polish_score = 0
        english_score = 0
        
        # Count word-based indicators with higher weights for key words
        for word in cls.POLISH_INDICATORS['words']:
            word_lower = word.lower()
            if word_lower in words:  # Check for whole words
                polish_score += words.count(word_lower) * 2  # Higher weight for exact matches
            elif word_lower in text_lower:  # Check for partial matches
                polish_score += text_lower.count(word_lower)
        
        for word in cls.ENGLISH_INDICATORS['words']:
            word_lower = word.lower()
            if word_lower in words:  # Check for whole words
                english_score += words.count(word_lower) * 2  # Higher weight for exact matches
            elif word_lower in text_lower:  # Check for partial matches
                english_score += text_lower.count(word_lower)
        
        # Check for Polish patterns
        for pattern in cls.POLISH_INDICATORS['patterns']:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            polish_score += len(matches)
        
        # Check for English patterns
        for pattern in cls.ENGLISH_INDICATORS['patterns']:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            english_score += len(matches)
        
        # Special check: if text contains a lot of English articles/prepositions, bias towards English
        english_articles = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with']
        english_article_count = sum(words.count(article) for article in english_articles)
        english_score += english_article_count * 1.5
        
        # Special check: if text contains Polish diacritics, strongly bias towards Polish
        polish_chars = ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż']
        polish_char_count = sum(text_lower.count(char) for char in polish_chars)
        polish_score += polish_char_count * 3  # Strong weight for Polish characters
        
        # Debug information for testing
        # print(f"Debug - English score: {english_score}, Polish score: {polish_score}")
        
        # Return the language with higher score, with a bias towards English for ties
        if polish_score > english_score * 1.2:  # Polish needs to be significantly higher
            return Language.POLISH
        else:
            return Language.ENGLISH
    
    @classmethod
    def validate_language(cls, language: str) -> Language:
        """
        Validate and convert language string to Language enum.
        
        Args:
            language: Language code string ('en', 'pl', 'english', 'polish')
            
        Returns:
            Validated Language enum
            
        Raises:
            ValueError: If language is not supported
        """
        language = language.lower().strip()
        
        if language in ['en', 'english', 'eng']:
            return Language.ENGLISH
        elif language in ['pl', 'polish', 'pol', 'polski']:
            return Language.POLISH
        else:
            raise ValueError(f"Unsupported language: {language}. Supported languages: 'en', 'pl'")


class LanguageManager:
    """Manages language-specific configurations and resources"""
    
    LANGUAGE_CONFIGS = {
        Language.ENGLISH: {
            'name': 'English',
            'code': 'en',
            'collection_suffix': 'en',
            'data_dir': 'data/en',
            'prompts': {
                'generation_system': "You are an expert content creator. Generate high-quality, engaging social media content in English.",
                'quality_assessment': "Assess the quality of this English content for social media.",
                'optimization': "Optimize this English content to improve engagement and clarity."
            }
        },
        Language.POLISH: {
            'name': 'Polish',
            'code': 'pl',
            'collection_suffix': 'pl',
            'data_dir': 'data/pl',
            'prompts': {
                'generation_system': "Jesteś ekspertem od tworzenia treści. Generuj wysokiej jakości, angażujące treści mediów społecznościowych w języku polskim.",
                'quality_assessment': "Oceń jakość tej polskiej treści dla mediów społecznościowych.",
                'optimization': "Zoptymalizuj tę polską treść, aby poprawić zaangażowanie i przejrzystość."
            }
        }
    }
    
    @classmethod
    def get_config(cls, language: Language) -> Dict:
        """
        Get configuration for a specific language.
        
        Args:
            language: Target language
            
        Returns:
            Language configuration dictionary
        """
        return cls.LANGUAGE_CONFIGS.get(language, cls.LANGUAGE_CONFIGS[Language.ENGLISH])
    
    @classmethod
    def get_collection_name(cls, language: Language, base_name: str = "inspiration_content") -> str:
        """
        Get language-specific collection name for Qdrant.
        
        Args:
            language: Target language
            base_name: Base collection name
            
        Returns:
            Language-specific collection name
        """
        config = cls.get_config(language)
        return f"{base_name}_{config['collection_suffix']}"
    
    @classmethod
    def get_data_directory(cls, language: Language) -> str:
        """
        Get language-specific data directory path.
        
        Args:
            language: Target language
            
        Returns:
            Path to language-specific data directory
        """
        config = cls.get_config(language)
        return config['data_dir']
    
    @classmethod
    def get_prompt(cls, language: Language, prompt_type: str) -> str:
        """
        Get language-specific prompt template.
        
        Args:
            language: Target language
            prompt_type: Type of prompt ('generation_system', 'quality_assessment', 'optimization')
            
        Returns:
            Language-specific prompt string
        """
        config = cls.get_config(language)
        return config['prompts'].get(prompt_type, config['prompts']['generation_system'])
    
    @classmethod
    def get_supported_languages(cls) -> List[Language]:
        """
        Get list of supported languages.
        
        Returns:
            List of supported Language enums
        """
        return list(cls.LANGUAGE_CONFIGS.keys())
    
    @classmethod
    def get_language_display_name(cls, language: Language) -> str:
        """
        Get human-readable language name.
        
        Args:
            language: Target language
            
        Returns:
            Human-readable language name
        """
        config = cls.get_config(language)
        return config['name']


def detect_content_language(content: str) -> Language:
    """
    Convenience function to detect language of content.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Detected language
    """
    return LanguageDetector.detect_language(content)


def validate_language_code(language_code: str) -> Language:
    """
    Convenience function to validate language code.
    
    Args:
        language_code: Language code to validate
        
    Returns:
        Validated Language enum
    """
    return LanguageDetector.validate_language(language_code)