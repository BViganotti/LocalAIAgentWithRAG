"""
Enhanced RAG System with Hybrid Search and Content Categorization

Provides improved search capabilities combining keyword and semantic search,
automatic content categorization, and better inspiration matching.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from typing import Dict, List, Optional, Any, Union
import re
from collections import Counter
from enum import Enum


class ContentCategory(Enum):
    """Content categories for automatic tagging"""
    TECHNOLOGY = "technology"
    BUSINESS = "business"
    MARKETING = "marketing"
    PRODUCTIVITY = "productivity"
    CAREER = "career"
    EDUCATION = "education"
    HEALTH = "health"
    FINANCE = "finance"
    LIFESTYLE = "lifestyle"
    SCIENCE = "science"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    OTHER = "other"


class ContentCategorizer:
    """Automatically categorize content based on keywords and patterns"""
    
    def __init__(self):
        self.category_keywords = {
            ContentCategory.TECHNOLOGY: [
                'ai', 'artificial intelligence', 'machine learning', 'software', 'programming',
                'coding', 'development', 'tech', 'digital', 'algorithm', 'data science',
                'cybersecurity', 'blockchain', 'cloud', 'api', 'automation', 'innovation'
            ],
            ContentCategory.BUSINESS: [
                'business', 'startup', 'entrepreneur', 'company', 'corporate', 'strategy',
                'management', 'leadership', 'revenue', 'profit', 'growth', 'scaling',
                'investment', 'venture', 'market', 'industry', 'enterprise'
            ],
            ContentCategory.MARKETING: [
                'marketing', 'advertising', 'branding', 'content', 'social media',
                'seo', 'campaign', 'audience', 'engagement', 'conversion', 'leads',
                'customer', 'brand', 'promotion', 'viral', 'influencer'
            ],
            ContentCategory.PRODUCTIVITY: [
                'productivity', 'efficiency', 'time management', 'workflow', 'organization',
                'planning', 'focus', 'goals', 'habits', 'optimization', 'tools',
                'system', 'process', 'automation', 'scheduling'
            ],
            ContentCategory.CAREER: [
                'career', 'job', 'work', 'professional', 'resume', 'interview',
                'networking', 'skills', 'promotion', 'salary', 'workplace',
                'remote work', 'freelance', 'employment', 'industry'
            ],
            ContentCategory.EDUCATION: [
                'education', 'learning', 'teaching', 'course', 'training', 'skill',
                'knowledge', 'study', 'research', 'academic', 'university',
                'school', 'certification', 'development', 'growth'
            ],
            ContentCategory.HEALTH: [
                'health', 'wellness', 'fitness', 'medical', 'mental health',
                'nutrition', 'exercise', 'diet', 'wellbeing', 'healthcare',
                'medicine', 'therapy', 'mindfulness', 'stress'
            ],
            ContentCategory.FINANCE: [
                'finance', 'money', 'investment', 'trading', 'stocks', 'crypto',
                'savings', 'budget', 'financial', 'economy', 'banking',
                'retirement', 'wealth', 'income', 'debt'
            ]
        }
    
    def categorize_content(self, content: str, title: str = "") -> List[ContentCategory]:
        """Categorize content based on keywords and context"""
        text = (content + " " + title).lower()
        
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences with word boundaries
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = len(re.findall(pattern, text))
                score += matches
                
                # Bonus for title matches
                if keyword.lower() in title.lower():
                    score += 2
            
            if score > 0:
                category_scores[category] = score
        
        # Return categories sorted by relevance score
        if not category_scores:
            return [ContentCategory.OTHER]
        
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top categories (at least 1, up to 3)
        result = [cat for cat, score in sorted_categories[:3] if score >= 1]
        return result if result else [ContentCategory.OTHER]
    
    def get_category_keywords(self, category: ContentCategory) -> List[str]:
        """Get keywords for a specific category"""
        return self.category_keywords.get(category, [])


class HybridSearchRetriever:
    """Combines semantic and keyword search for better results"""
    
    def __init__(self, qdrant_client: QdrantClient, embeddings: OllamaEmbeddings, 
                 collection_name: str):
        self.client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.categorizer = ContentCategorizer()
    
    def hybrid_search(self, query: str, platform: str = None, category: ContentCategory = None,
                     k: int = 5, keyword_weight: float = 0.3, semantic_weight: float = 0.7) -> List[Any]:
        """Perform hybrid search combining semantic and keyword approaches"""
        
        # Semantic search using vector similarity
        semantic_results = self._semantic_search(query, platform, category, k * 2)  # Get more for reranking
        
        # Keyword search using text matching
        keyword_results = self._keyword_search(query, platform, category, k * 2)
        
        # Combine and rerank results
        combined_results = self._combine_and_rerank(
            semantic_results, keyword_results, query, 
            keyword_weight, semantic_weight, k
        )
        
        return combined_results
    
    def _semantic_search(self, query: str, platform: str = None, category: ContentCategory = None, k: int = 10) -> List[Any]:
        """Perform semantic search using embeddings"""
        try:
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings
            )
            
            # Build filter
            search_filter = self._build_filter(platform, category)
            
            search_kwargs = {"k": k}
            if search_filter:
                search_kwargs["filter"] = search_filter
            
            retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            return retriever.invoke(query)
        
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def _keyword_search(self, query: str, platform: str = None, category: ContentCategory = None, k: int = 10) -> List[Any]:
        """Perform keyword-based search"""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            # Use Qdrant's payload-based search for keyword matching
            # This is a simplified approach - in production you might use a proper text search engine
            search_filter = self._build_filter(platform, category)
            
            # For now, fall back to semantic search with keyword-enhanced query
            enhanced_query = " ".join(keywords) + " " + query
            return self._semantic_search(enhanced_query, platform, category, k)
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Simple keyword extraction - remove stop words and get important terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _build_filter(self, platform: str = None, category: ContentCategory = None) -> Optional[Filter]:
        """Build Qdrant filter for platform and category"""
        conditions = []
        
        if platform:
            conditions.append(FieldCondition(
                key="platform",
                match=MatchValue(value=platform)
            ))
        
        if category:
            conditions.append(FieldCondition(
                key="category",
                match=MatchValue(value=category.value)
            ))
        
        if conditions:
            return Filter(must=conditions)
        
        return None
    
    def _combine_and_rerank(self, semantic_results: List[Any], keyword_results: List[Any], 
                           query: str, keyword_weight: float, semantic_weight: float, k: int) -> List[Any]:
        """Combine and rerank results from both search methods"""
        
        # Create a scoring system
        result_scores = {}
        query_lower = query.lower()
        
        # Score semantic results
        for i, result in enumerate(semantic_results):
            content = result.page_content.lower()
            # Distance-based scoring (closer = higher score)
            semantic_score = 1.0 / (i + 1)  # Simple ranking-based score
            
            # Boost for exact keyword matches in content
            keyword_bonus = 0
            for word in self._extract_keywords(query):
                if word in content:
                    keyword_bonus += 0.1
            
            total_score = (semantic_score * semantic_weight) + keyword_bonus
            result_scores[result.page_content] = {
                'score': total_score,
                'result': result
            }
        
        # Score keyword results (boost if not already in semantic results)
        for i, result in enumerate(keyword_results):
            content_key = result.page_content
            if content_key not in result_scores:
                keyword_score = 1.0 / (i + 1)
                result_scores[content_key] = {
                    'score': keyword_score * keyword_weight,
                    'result': result
                }
            else:
                # Boost existing results that appear in both
                result_scores[content_key]['score'] += (keyword_weight * 0.5)
        
        # Sort by score and return top k
        sorted_results = sorted(result_scores.values(), key=lambda x: x['score'], reverse=True)
        return [item['result'] for item in sorted_results[:k]]


class EnhancedRAGSystem:
    """Enhanced RAG system with all improvements"""
    
    def __init__(self, qdrant_client: QdrantClient, embeddings: OllamaEmbeddings, 
                 collection_name: str):
        self.client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.hybrid_retriever = HybridSearchRetriever(qdrant_client, embeddings, collection_name)
        self.categorizer = ContentCategorizer()
    
    def search_inspiration(self, query: str, platform: str = None, 
                          content_category: ContentCategory = None, k: int = 5,
                          use_hybrid: bool = True) -> List[Any]:
        """Search for inspiration content with enhanced capabilities"""
        
        if use_hybrid:
            # Use hybrid search
            return self.hybrid_retriever.hybrid_search(query, platform, content_category, k)
        else:
            # Use traditional semantic search
            return self.hybrid_retriever._semantic_search(query, platform, content_category, k)
    
    def auto_categorize_and_search(self, query: str, platform: str = None, k: int = 5) -> Dict[str, Any]:
        """Automatically categorize query and search for relevant content"""
        
        # Auto-categorize the query
        categories = self.categorizer.categorize_content(query)
        primary_category = categories[0] if categories else ContentCategory.OTHER
        
        # Search with the primary category
        results = self.search_inspiration(query, platform, primary_category, k)
        
        return {
            'results': results,
            'detected_categories': [cat.value for cat in categories],
            'primary_category': primary_category.value,
            'search_metadata': {
                'platform_filter': platform,
                'category_filter': primary_category.value,
                'results_count': len(results)
            }
        }
    
    def get_category_specific_inspiration(self, category: ContentCategory, platform: str = None, k: int = 5) -> List[Any]:
        """Get inspiration specifically from a content category"""
        
        # Use category keywords to build a search query
        keywords = self.categorizer.get_category_keywords(category)
        query = " ".join(keywords[:5])  # Use top 5 keywords
        
        return self.search_inspiration(query, platform, category, k)


# Global enhanced RAG system instance
enhanced_rag = None

def get_enhanced_rag_system(qdrant_client: QdrantClient, embeddings: OllamaEmbeddings, 
                           collection_name: str) -> EnhancedRAGSystem:
    """Get or create the enhanced RAG system"""
    global enhanced_rag
    if enhanced_rag is None:
        enhanced_rag = EnhancedRAGSystem(qdrant_client, embeddings, collection_name)
    return enhanced_rag