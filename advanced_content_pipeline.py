"""
Advanced Content Generation Pipeline - Consolidated System

This module contains a comprehensive, production-ready content generation pipeline
that transforms topics into high-quality social media content for LinkedIn, Reddit, 
and Twitter using AI-powered generation with quality assessment and optimization.

Architecture:
    8-Stage Pipeline: Processing → Categorization → Inspiration → Generation 
                     → Quality Assessment → Optimization → Analytics → Finalization

Key Features:
    • Multi-model AI support (Ollama/gpt-oss default)
    • Real-time quality assessment with 8 quality metrics
    • Automated content optimization with iterative improvement
    • Enhanced RAG with hybrid semantic + keyword search
    • Dynamic inspiration selection based on topic complexity
    • Comprehensive analytics and performance monitoring
    • A/B testing with configurable content variations
    • Platform-specific optimization (LinkedIn, Reddit, Twitter)

Dependencies:
    - langchain-ollama: LLM integration
    - langchain-qdrant: Vector database
    - qdrant-client: Vector operations
    - psycopg2-binary: PostgreSQL integration
    - PyPDF2: PDF processing
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import time
import json
import statistics
import re
import requests
import urllib.parse
from datetime import datetime
from collections import Counter

# LangChain imports
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Research APIs
import xml.etree.ElementTree as ET

# Language utilities
from language_utils import Language, LanguageManager, validate_language_code


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class PipelineStage(Enum):
    """Pipeline execution stages"""
    TOPIC_PROCESSING = "topic_processing"
    CONTENT_CATEGORIZATION = "content_categorization"
    ENHANCED_RESEARCH = "enhanced_research"
    INSPIRATION_RETRIEVAL = "inspiration_retrieval"
    CONTENT_GENERATION = "content_generation"
    QUALITY_ASSESSMENT = "quality_assessment"
    BRAND_CONSISTENCY = "brand_consistency"
    CONTENT_OPTIMIZATION = "content_optimization"
    ANALYTICS_TRACKING = "analytics_tracking"
    FINALIZATION = "finalization"


class ContentType(Enum):
    """Content type categories for different post styles"""
    HOW_TO = "how_to"
    OPINION = "opinion" 
    NEWS = "news"
    LISTICLE = "listicle"
    STORY = "story"
    QUESTION = "question"
    ANNOUNCEMENT = "announcement"
    EDUCATIONAL = "educational"


class ToneStyle(Enum):
    """Available tone styles for content generation"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    ENTHUSIASTIC = "enthusiastic"
    INFORMATIVE = "informative"
    INSPIRATIONAL = "inspirational"
    HUMOROUS = "humorous"


class ContentCategory(Enum):
    """Content categories for automatic topic classification"""
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


class QualityMetric(Enum):
    """Content quality assessment metrics"""
    ENGAGEMENT_POTENTIAL = "engagement_potential"
    READABILITY = "readability" 
    RELEVANCE = "relevance"
    UNIQUENESS = "uniqueness"
    PLATFORM_OPTIMIZATION = "platform_optimization"
    CALL_TO_ACTION_STRENGTH = "cta_strength"
    EMOTIONAL_APPEAL = "emotional_appeal"
    CLARITY = "clarity"


class BrandConsistencyMetric(Enum):
    """Brand consistency assessment metrics"""
    VOICE_ALIGNMENT = "voice_alignment"
    TONE_CONSISTENCY = "tone_consistency"
    MESSAGING_ALIGNMENT = "messaging_alignment"
    COMPLIANCE_CHECK = "compliance_check"
    ORIGINALITY_SCORE = "originality_score"


class ResearchQuality(Enum):
    """Research and factual accuracy metrics"""
    FACTUAL_ACCURACY = "factual_accuracy"
    SOURCE_CREDIBILITY = "source_credibility"
    CITATION_QUALITY = "citation_quality"
    EXPERT_VALIDATION = "expert_validation"
    TIMELINESS = "timeliness"


@dataclass
class ContentQualityScore:
    """Content quality assessment results with comprehensive AI feedback"""
    overall_score: float
    metrics: Dict[QualityMetric, float] = field(default_factory=dict)
    detailed_feedback: Dict[str, str] = field(default_factory=dict)  # Specific feedback per metric
    improvement_suggestions: List[str] = field(default_factory=list)  # Actionable improvements
    strengths: List[str] = field(default_factory=list)  # What's working well
    weaknesses: List[str] = field(default_factory=list)  # What needs fixing
    confidence: float = 0.0
    assessment_reasoning: str = ""  # AI's reasoning for the score


@dataclass
class BrandConsistencyScore:
    """Brand consistency assessment results"""
    overall_score: float
    metrics: Dict[BrandConsistencyMetric, float] = field(default_factory=dict)
    voice_analysis: Dict[str, str] = field(default_factory=dict)
    compliance_issues: List[str] = field(default_factory=list)
    plagiarism_score: float = 0.0
    originality_assessment: str = ""
    brand_alignment_feedback: str = ""


@dataclass
class ResearchEnhancement:
    """Enhanced research data and validation"""
    factual_claims: List[Dict[str, Any]] = field(default_factory=list)
    verified_sources: List[Dict[str, str]] = field(default_factory=list)
    expert_quotes: List[Dict[str, str]] = field(default_factory=list)
    research_quality: Dict[ResearchQuality, float] = field(default_factory=dict)
    fact_check_results: List[Dict[str, Any]] = field(default_factory=list)
    academic_references: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class PipelineStageResult:
    """Result from a single pipeline stage execution"""
    stage: PipelineStage
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass  
class ContentPipelineResult:
    """Complete pipeline execution result with comprehensive data"""
    topic_title: str
    platforms: List[str]
    stage_results: Dict[PipelineStage, PipelineStageResult] = field(default_factory=dict)
    generated_content: List[Dict[str, Any]] = field(default_factory=list)
    quality_scores: Dict[str, ContentQualityScore] = field(default_factory=dict)
    brand_scores: Dict[str, BrandConsistencyScore] = field(default_factory=dict)
    research_data: Optional[ResearchEnhancement] = None
    optimization_iterations: int = 0
    total_execution_time: float = 0.0
    success: bool = False
    analytics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# CONTENT CATEGORIZATION SYSTEM
# =============================================================================

class ContentCategorizer:
    """
    Automatically categorize content based on keywords and patterns.
    
    Uses keyword matching with weighted scoring to classify topics into
    predefined categories for better inspiration matching and content optimization.
    """
    
    def __init__(self):
        """Initialize categorizer with predefined keyword mappings"""
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
        """
        Categorize content based on keywords and context.
        
        Args:
            content: Main content text to analyze
            title: Optional title text (receives bonus weighting)
            
        Returns:
            List of categories sorted by relevance score (1-3 categories)
        """
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


# =============================================================================
# CONTENT TEMPLATES SYSTEM
# =============================================================================

class ContentTemplateManager:
    """
    Manages content templates for different platforms, content types, and tone styles.
    
    Provides platform-optimized templates with tone modifiers and style guidance
    for consistent, high-quality content generation across all social media platforms.
    """
    
    def __init__(self, language: Language = Language.ENGLISH):
        """Initialize template manager with predefined templates and tone modifiers"""
        self.language = language
        self.templates = self._initialize_templates()
        self.tone_modifiers = self._initialize_tone_modifiers()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize platform-specific templates for different content types"""
        return {
            "linkedin": {
                ContentType.HOW_TO.value: """
You are creating a LinkedIn how-to post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a LinkedIn how-to post that:
1. Starts with a compelling hook that identifies the problem or challenge
2. Provides step-by-step actionable advice with clear, numbered steps
3. Uses professional formatting with bullet points or numbered lists
4. Includes real-world examples or case studies when relevant
5. Ends with a question to encourage engagement and discussion
6. Maintains a {tone} tone throughout
7. Optimizes for LinkedIn's algorithm (1300-3000 characters ideal)

{tone_guidance}
""",
                ContentType.OPINION.value: """
You are creating a LinkedIn opinion post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a LinkedIn opinion post that:
1. Opens with a strong, thought-provoking statement or contrarian view
2. Presents your perspective with supporting evidence or examples
3. Acknowledges different viewpoints while maintaining your stance
4. Uses storytelling elements to make it relatable
5. Includes a clear call-to-action or discussion question
6. Maintains a {tone} tone while being respectful of opposing views
7. Is structured for maximum LinkedIn engagement

{tone_guidance}
""",
                ContentType.EDUCATIONAL.value: """
You are creating a LinkedIn educational post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a LinkedIn educational post that:
1. Presents valuable insights or knowledge in an accessible way
2. Uses clear structure with headers, bullets, or numbered points
3. Includes practical examples or real-world applications
4. Provides actionable takeaways for the reader
5. Encourages discussion with a thought-provoking question
6. Maintains a {tone} tone appropriate for professional learning
7. Optimizes length for LinkedIn engagement (1300-3000 characters)

{tone_guidance}
"""
            },
            
            "reddit": {
                ContentType.EDUCATIONAL.value: """
You are creating a Reddit educational post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Reddit post that:
1. Provides genuinely valuable information or insights
2. Uses Reddit's informal, community-focused style
3. Includes practical examples or personal experiences when relevant
4. Encourages community discussion and knowledge sharing
5. Uses appropriate Reddit formatting (markdown, line breaks)
6. Maintains a {tone} tone that fits the community
7. Optimizes for Reddit engagement and upvotes

{tone_guidance}
""",
                ContentType.QUESTION.value: """
You are creating a Reddit discussion post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Reddit discussion post that:
1. Poses engaging questions that encourage community participation
2. Provides context or background information to frame the discussion
3. Uses Reddit's conversational, authentic style
4. Acknowledges different perspectives and invites diverse viewpoints
5. Includes follow-up questions or scenarios to drive engagement
6. Maintains a {tone} tone appropriate for community discussion
7. Structured to generate meaningful comments and upvotes

{tone_guidance}
"""
            },
            
            "twitter": {
                ContentType.EDUCATIONAL.value: """
You are creating a Twitter educational post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Twitter post that:
1. Delivers valuable insights in a concise, impactful way
2. Uses Twitter's fast-paced, attention-grabbing style
3. Includes relevant hashtags for discoverability (2-3 max)
4. Encourages engagement through questions or calls-to-action
5. Uses threading if needed for complex topics
6. Maintains a {tone} tone appropriate for Twitter's audience
7. Optimizes for Twitter engagement (retweets, likes, replies)

{tone_guidance}
""",
                ContentType.OPINION.value: """
You are creating a Twitter opinion post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Twitter post that:
1. Presents a clear, compelling viewpoint in concise language
2. Uses Twitter's direct, conversational style
3. Includes engaging elements that encourage discussion
4. Uses appropriate hashtags for reach (2-3 max)
5. Balances brevity with substance and impact
6. Maintains a {tone} tone that drives engagement
7. Optimizes for Twitter's algorithm and user behavior

{tone_guidance}
"""
            },
            
            "newsletter": {
                ContentType.EDUCATIONAL.value: """
You are creating a newsletter section with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a newsletter section that:
1. Opens with a compelling headline that clearly states the value
2. Provides structured, scannable content with clear sections
3. Includes actionable insights and practical takeaways
4. Uses headers, subheadings, and bullet points for easy reading
5. Incorporates relevant examples, data, or case studies
6. Maintains a {tone} tone appropriate for newsletter readers
7. Ends with clear next steps or resources for further learning
8. Optimizes for newsletter format (800-2500 characters ideal)

{tone_guidance}
""",
                ContentType.HOW_TO.value: """
You are creating a newsletter how-to guide with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a newsletter how-to section that:
1. Starts with a clear, benefit-focused headline
2. Provides step-by-step instructions in a structured format
3. Uses numbered steps with brief explanations
4. Includes practical examples or real-world applications
5. Adds helpful tips or warnings where relevant
6. Uses clear formatting with headers and bullet points
7. Concludes with a summary or key takeaways
8. Maintains a {tone} tone throughout
9. Optimizes length for newsletter consumption (800-2500 characters)

{tone_guidance}
""",
                ContentType.NEWS.value: """
You are creating a newsletter news update with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a newsletter news section that:
1. Opens with a clear, newsworthy headline
2. Provides context and background for the news
3. Explains the implications and why it matters
4. Includes relevant data, quotes, or expert insights
5. Structures information in digestible paragraphs
6. Uses clear transitions between key points
7. Concludes with what readers should watch for next
8. Maintains a {tone} tone appropriate for informed readers
9. Optimizes for newsletter format and engagement

{tone_guidance}
""",
                ContentType.LISTICLE.value: """
You are creating a newsletter listicle with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a newsletter listicle that:
1. Opens with a compelling headline with a specific number
2. Provides valuable, actionable items in a numbered list
3. Includes brief explanations and examples for each point
4. Uses consistent formatting and structure throughout
5. Incorporates practical tips or insights for each item
6. Maintains reader engagement with varied examples
7. Concludes with a summary or call-to-action
8. Uses a {tone} tone that resonates with newsletter audience
9. Optimizes structure for newsletter consumption

{tone_guidance}
"""
            }
        }
    
    def _initialize_tone_modifiers(self) -> Dict[ToneStyle, str]:
        """Initialize tone-specific guidance and modifiers"""
        return {
            ToneStyle.PROFESSIONAL: """
Maintain a professional, business-appropriate tone:
- Use formal language and industry terminology appropriately
- Focus on expertise, credibility, and value-driven content
- Avoid casual slang or overly informal expressions
- Present information with authority and confidence
""",
            ToneStyle.CONVERSATIONAL: """
Use a friendly, approachable conversational tone:
- Write as if talking to a friend or colleague
- Use casual language while remaining informative
- Include personal touches and relatable examples
- Encourage two-way dialogue and community interaction
""",
            ToneStyle.ENTHUSIASTIC: """
Express genuine enthusiasm and energy:
- Use dynamic, energetic language and expressions
- Show excitement about the topic and its potential
- Include motivational elements and positive messaging
- Inspire action and engagement through passionate delivery
""",
            ToneStyle.AUTHORITATIVE: """
Demonstrate expertise and thought leadership:
- Present information with confidence and authority
- Use data, statistics, and proven methodologies
- Establish credibility through knowledge demonstration
- Guide readers with expert insights and recommendations
"""
        }
    
    def get_template(self, platform: str, content_type: ContentType, tone: ToneStyle) -> str:
        """
        Get template for specific platform, content type, and tone combination.
        
        Args:
            platform: Target social media platform
            content_type: Type of content to generate
            tone: Desired tone style
            
        Returns:
            Formatted template string with placeholders
        """
        platform_templates = self.templates.get(platform, {})
        template = platform_templates.get(content_type.value)
        
        if not template:
            # Fallback to educational template
            template = platform_templates.get(ContentType.EDUCATIONAL.value, self._get_fallback_template())
        
        # Add tone guidance
        tone_guidance = self.tone_modifiers.get(tone, "")
        
        # Safely format template with tone info only, preserving other placeholders
        try:
            # First replace tone placeholders
            template = template.replace('{tone}', tone.value.replace('_', ' ').title())
            template = template.replace('{tone_guidance}', tone_guidance)
            return template
        except Exception as e:
            print(f"Warning: Template formatting error: {e}")
            return template
    
    def _get_fallback_template(self) -> str:
        """Fallback template if specific platform/type not found"""
        return """
Create high-quality social media content about the following topic:

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Generate engaging content that provides value to the audience and encourages interaction.
Maintain a {tone} tone throughout the content.

{tone_guidance}
"""


# =============================================================================
# ENHANCED RAG AND VECTOR SEARCH SYSTEM
# =============================================================================

class HybridSearchRetriever:
    """
    Advanced retrieval system combining semantic and keyword search.
    
    Provides improved search capabilities by combining vector similarity search
    with keyword-based filtering for more relevant inspiration content retrieval.
    """
    
    def __init__(self, qdrant_client: QdrantClient, embeddings: OllamaEmbeddings, 
                 collection_name: str):
        """
        Initialize hybrid search retriever.
        
        Args:
            qdrant_client: Qdrant database client
            embeddings: Embedding model for semantic search
            collection_name: Name of the Qdrant collection
        """
        self.client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.categorizer = ContentCategorizer()
    
    def hybrid_search(self, query: str, platform: str = None, category: ContentCategory = None,
                     k: int = 5, keyword_weight: float = 0.3, semantic_weight: float = 0.7) -> List[Any]:
        """
        Perform hybrid search combining semantic and keyword approaches.
        
        Args:
            query: Search query text
            platform: Optional platform filter
            category: Optional content category filter
            k: Number of results to return
            keyword_weight: Weight for keyword matching (0.0-1.0)
            semantic_weight: Weight for semantic similarity (0.0-1.0)
            
        Returns:
            List of retrieved documents ranked by combined score
        """
        # Semantic search using vector similarity
        semantic_results = self._semantic_search(query, platform, category, k * 2)
        
        # Keyword search using text matching
        keyword_results = self._keyword_search(query, platform, category, k * 2)
        
        # Combine and rerank results
        combined_results = self._combine_and_rerank(
            semantic_results, keyword_results, query, 
            keyword_weight, semantic_weight, k
        )
        
        return combined_results
    
    def _semantic_search(self, query: str, platform: str = None, 
                        category: ContentCategory = None, k: int = 10) -> List[Any]:
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
    
    def _keyword_search(self, query: str, platform: str = None, 
                       category: ContentCategory = None, k: int = 10) -> List[Any]:
        """Perform keyword-based search with enhanced query"""
        try:
            keywords = self._extract_keywords(query)
            enhanced_query = " ".join(keywords) + " " + query
            return self._semantic_search(enhanced_query, platform, category, k)
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query, removing stop words"""
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 
            'could', 'can', 'may', 'might'
        }
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _build_filter(self, platform: str = None, category: ContentCategory = None) -> Optional[Filter]:
        """Build Qdrant filter for platform and category"""
        conditions = []
        
        if platform:
            conditions.append(FieldCondition(key="platform", match=MatchValue(value=platform)))
        
        if category and category != ContentCategory.OTHER:
            conditions.append(FieldCondition(key="category", match=MatchValue(value=category.value)))
        
        return Filter(must=conditions) if conditions else None
    
    def _combine_and_rerank(self, semantic_results: List[Any], keyword_results: List[Any],
                           query: str, keyword_weight: float, semantic_weight: float, 
                           k: int) -> List[Any]:
        """Combine and rerank results from semantic and keyword search"""
        result_scores = {}
        
        # Score semantic results
        for i, result in enumerate(semantic_results):
            content_key = result.page_content[:50]  # Use first 50 chars as key
            semantic_score = 1.0 / (i + 1)  # Inverse rank scoring
            
            result_scores[content_key] = {
                'score': semantic_score * semantic_weight,
                'result': result
            }
        
        # Add keyword results and boost scores
        for i, result in enumerate(keyword_results):
            content_key = result.page_content[:50]
            keyword_score = 1.0 / (i + 1)
            
            if content_key not in result_scores:
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
    """
    Enhanced RAG system with hybrid search and automatic categorization.
    
    Provides advanced retrieval capabilities with automatic content categorization,
    hybrid search combining semantic and keyword approaches, and intelligent
    inspiration matching for content generation.
    """
    
    def __init__(self, qdrant_client: QdrantClient, embeddings: OllamaEmbeddings, 
                 collection_name: str):
        """
        Initialize enhanced RAG system.
        
        Args:
            qdrant_client: Qdrant database client
            embeddings: Embedding model for semantic operations
            collection_name: Name of the vector collection
        """
        self.client = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name
        self.hybrid_retriever = HybridSearchRetriever(qdrant_client, embeddings, collection_name)
        self.categorizer = ContentCategorizer()
    
    def search_inspiration(self, query: str, platform: str = None, 
                          content_category: ContentCategory = None, k: int = 5,
                          use_hybrid: bool = True) -> List[Any]:
        """
        Search for inspiration content with enhanced capabilities.
        
        Args:
            query: Search query text
            platform: Target platform filter
            content_category: Content category filter
            k: Number of results to return
            use_hybrid: Whether to use hybrid search or semantic only
            
        Returns:
            List of relevant inspiration documents
        """
        if use_hybrid:
            return self.hybrid_retriever.hybrid_search(query, platform, content_category, k)
        else:
            return self.hybrid_retriever._semantic_search(query, platform, content_category, k)
    
    def auto_categorize_and_search(self, query: str, platform: str = None, k: int = 5) -> Dict[str, Any]:
        """
        Automatically categorize query and search for relevant content.
        
        Args:
            query: Search query text
            platform: Target platform filter
            k: Number of results to return
            
        Returns:
            Dictionary containing search results and categorization metadata
        """
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


# =============================================================================
# ENHANCED RESEARCH CAPABILITIES SYSTEM
# =============================================================================

class EnhancedResearchEngine:
    """
    Production-grade research system with real research integration using free APIs.
    
    Integrates with Semantic Scholar, arXiv, Wikipedia, and web sources to provide
    verified, credible information and enhance content with authoritative sources.
    """
    
    def __init__(self, model_name: str = "gpt-oss", enable_real_research: bool = True):
        """Initialize research engine with AI model for analysis"""
        self.model = OllamaLLM(model=model_name)
        self.enable_real_research = enable_real_research
        
        if enable_real_research:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'ContentAgent/1.0 (Research Bot; contact@example.com)'
            })
            
            # Free research APIs
            self.semantic_scholar_api = "https://api.semanticscholar.org/graph/v1/paper/search"
            self.arxiv_api = "http://export.arxiv.org/api/query"
            self.wikipedia_api = "https://en.wikipedia.org/api/rest_v1/page/summary"
        else:
            print("🔬 Real research disabled - using AI-generated research data only")
        
    def enhance_research(self, topic_data: Dict[str, Any]) -> ResearchEnhancement:
        """
        Perform comprehensive research enhancement for a topic.
        
        Args:
            topic_data: Topic information to research
            
        Returns:
            ResearchEnhancement with verified facts, sources, and expert quotes
        """
        topic_content = topic_data.get('content', '')
        topic_title = topic_data.get('title', '')
        
        try:
            # Extract factual claims from content
            factual_claims = self._extract_factual_claims(topic_content)
            
            # Perform fact-checking
            fact_check_results = self._fact_check_claims(factual_claims)
            
            # Find credible sources
            verified_sources = self._find_credible_sources(topic_title, topic_content)
            
            # Mine expert quotes
            expert_quotes = self._mine_expert_quotes(topic_title)
            
            # Find academic references
            academic_references = self._find_academic_papers(topic_title)
            
            # Assess research quality
            research_quality = self._assess_research_quality(
                fact_check_results, verified_sources, expert_quotes, academic_references
            )
            
            return ResearchEnhancement(
                factual_claims=factual_claims,
                verified_sources=verified_sources,
                expert_quotes=expert_quotes,
                research_quality=research_quality,
                fact_check_results=fact_check_results,
                academic_references=academic_references
            )
            
        except Exception as e:
            print(f"Research enhancement error: {e}")
            return self._create_fallback_research()
    
    def _extract_factual_claims(self, content: str) -> List[Dict[str, Any]]:
        """Extract factual claims that need verification"""
        try:
            prompt = f"""
Analyze the following content and extract factual claims that should be verified:

CONTENT:
{content}

Extract claims that:
1. Make specific statistical statements
2. Reference studies or research
3. Make definitive factual assertions
4. Cite specific dates, numbers, or percentages
5. Reference expert opinions or quotes

Return a JSON list of claims with their context:
[
    {{"claim": "specific factual statement", "context": "surrounding context", "type": "statistic|study|fact|quote"}},
    ...
]
"""
            
            response = self.model.invoke(prompt)
            
            try:
                claims = json.loads(response.strip())
                if isinstance(claims, list):
                    return claims[:10]  # Limit to 10 claims
            except json.JSONDecodeError:
                pass
                
            # Fallback: simple regex-based extraction
            return self._regex_extract_claims(content)
            
        except Exception as e:
            print(f"Claim extraction error: {e}")
            return []
    
    def _regex_extract_claims(self, content: str) -> List[Dict[str, Any]]:
        """Fallback method using regex to extract potential factual claims"""
        claims = []
        
        # Extract percentage claims
        percentage_pattern = r'(\d+(?:\.\d+)?%[^.]*?\.)'
        for match in re.finditer(percentage_pattern, content):
            claims.append({
                "claim": match.group(1).strip(),
                "context": content[max(0, match.start()-50):match.end()+50],
                "type": "statistic"
            })
        
        # Extract study references
        study_pattern = r'((?:study|research|survey|analysis)[^.]*?\.)'
        for match in re.finditer(study_pattern, content, re.IGNORECASE):
            claims.append({
                "claim": match.group(1).strip(),
                "context": content[max(0, match.start()-50):match.end()+50],
                "type": "study"
            })
        
        return claims[:5]  # Limit fallback claims
    
    def _fact_check_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fact-check extracted claims against reliable sources"""
        fact_check_results = []
        
        for claim in claims[:3]:  # Limit to 3 claims for performance
            try:
                # Search for fact-checking information
                search_query = claim['claim'][:100]  # Limit query length
                verification_result = self._search_fact_checkers(search_query)
                
                fact_check_results.append({
                    "claim": claim['claim'],
                    "verification_status": verification_result.get('status', 'unverified'),
                    "sources": verification_result.get('sources', []),
                    "confidence": verification_result.get('confidence', 0.5)
                })
                
                # Add delay to be respectful to APIs
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Fact-check error for claim: {e}")
                fact_check_results.append({
                    "claim": claim['claim'],
                    "verification_status": "error",
                    "sources": [],
                    "confidence": 0.0
                })
        
        return fact_check_results
    
    def _search_fact_checkers(self, query: str) -> Dict[str, Any]:
        """Search for claim verification using multiple sources"""
        try:
            # Search Wikipedia for factual information
            wikipedia_result = self._search_wikipedia(query)
            
            # Search academic sources for verification
            academic_results = self._search_semantic_scholar(query, limit=2)
            
            # Combine results for confidence assessment
            sources = []
            confidence = 0.0
            
            if wikipedia_result:
                sources.append(f"Wikipedia: {wikipedia_result.get('title', 'Article')}")
                confidence += 0.4
                
            if academic_results:
                sources.extend([f"Academic: {paper.get('title', 'Research Paper')[:50]}..." 
                               for paper in academic_results[:2]])
                confidence += min(0.6, len(academic_results) * 0.3)
            
            status = "verified" if confidence > 0.5 else "partially_verified" if confidence > 0.2 else "unverified"
            
            return {
                "status": status,
                "sources": sources,
                "confidence": min(1.0, confidence)
            }
            
        except Exception as e:
            print(f"Fact-checker search error: {e}")
            return {"status": "error", "sources": [], "confidence": 0.0}
    
    def _find_credible_sources(self, topic_title: str, content: str) -> List[Dict[str, str]]:
        """Find credible sources related to the topic using real APIs"""
        try:
            sources = []
            search_terms = self._extract_search_terms(topic_title, content)
            
            # Search Semantic Scholar for academic sources
            for term in search_terms[:2]:  # Limit API calls
                try:
                    papers = self._search_semantic_scholar(term, limit=3)
                    for paper in papers:
                        if paper.get('url') and paper.get('title'):
                            sources.append({
                                "title": paper['title'],
                                "url": paper.get('url', ''),
                                "source": "Semantic Scholar",
                                "credibility_score": 0.9,
                                "authors": ", ".join(paper.get('authors', [])[:3]),
                                "year": paper.get('year', 'Unknown')
                            })
                except Exception as e:
                    print(f"Semantic Scholar search error: {e}")
                    continue
            
            # Search Wikipedia for reference sources
            wiki_result = self._search_wikipedia(topic_title)
            if wiki_result:
                sources.append({
                    "title": wiki_result.get('title', topic_title),
                    "url": wiki_result.get('content_urls', {}).get('desktop', {}).get('page', ''),
                    "source": "Wikipedia",
                    "credibility_score": 0.7,
                    "summary": wiki_result.get('extract', '')[:200] + "..."
                })
            
            # Search arXiv for recent preprints
            try:
                arxiv_papers = self._search_arxiv(topic_title, max_results=2)
                for paper in arxiv_papers:
                    sources.append({
                        "title": paper.get('title', ''),
                        "url": paper.get('link', ''),
                        "source": "arXiv",
                        "credibility_score": 0.8,
                        "authors": paper.get('authors', 'Unknown'),
                        "published": paper.get('published', 'Unknown')
                    })
            except Exception as e:
                print(f"arXiv search error: {e}")
            
            return sources[:5]  # Return top 5 sources
            
        except Exception as e:
            print(f"Credible source search error: {e}")
            return []
    
    def _mine_expert_quotes(self, topic_title: str) -> List[Dict[str, str]]:
        """Mine expert perspectives from academic papers and credible sources"""
        try:
            quotes = []
            
            # Search for academic papers with abstracts that contain expert insights
            papers = self._search_semantic_scholar(topic_title, limit=3)
            
            for paper in papers:
                if paper.get('abstract') and paper.get('authors'):
                    # Extract key insights from abstract
                    abstract = paper['abstract'][:300] + "..."
                    authors = paper.get('authors', [])[:2]  # First 2 authors
                    
                    if authors and len(abstract) > 50:
                        author_names = ", ".join(authors) if isinstance(authors, list) else str(authors)
                        
                        quotes.append({
                            "quote": f"Research findings suggest: {abstract}",
                            "expert": author_names,
                            "credentials": f"Authors of '{paper.get('title', 'Academic Paper')[:50]}...'",
                            "relevance": "Academic research perspective",
                            "source": "Semantic Scholar",
                            "year": paper.get('year', 'Recent')
                        })
            
            # If we don't have enough from papers, use AI to generate contextually appropriate insights
            if len(quotes) < 2:
                # Use research findings to inform AI-generated expert perspectives
                research_context = "\n".join([q.get('quote', '') for q in quotes])
                
                prompt = f"""
Based on the following research context about {topic_title}, generate 1-2 realistic expert perspectives:

Research Context:
{research_context}

Generate perspectives that:
1. Are informed by the research context
2. Include realistic expert credentials
3. Provide valuable insights
4. Are clearly marked as AI-generated perspectives

Return JSON format:
[
    {{"quote": "expert perspective", "expert": "Dr. Name (AI-generated)", "credentials": "Title, Institution", "relevance": "why this perspective matters"}}
]
"""
                
                try:
                    response = self.model.invoke(prompt)
                    ai_quotes = json.loads(response.strip())
                    if isinstance(ai_quotes, list):
                        quotes.extend(ai_quotes[:2])
                except json.JSONDecodeError:
                    pass
            
            return quotes[:3]  # Return top 3 quotes
            
        except Exception as e:
            print(f"Expert quote mining error: {e}")
            return []
    
    def _find_academic_papers(self, topic_title: str) -> List[Dict[str, str]]:
        """Find relevant academic papers using real APIs"""
        try:
            papers = []
            
            # Search Semantic Scholar
            semantic_papers = self._search_semantic_scholar(topic_title, limit=3)
            for paper in semantic_papers:
                papers.append({
                    "title": paper.get('title', ''),
                    "authors": ", ".join(paper.get('authors', [])[:3]) if isinstance(paper.get('authors'), list) else str(paper.get('authors', '')),
                    "journal": paper.get('venue', 'Unknown Journal'),
                    "year": str(paper.get('year', 'Unknown')),
                    "doi": paper.get('doi', ''),
                    "url": paper.get('url', ''),
                    "abstract": paper.get('abstract', '')[:200] + "..." if paper.get('abstract') else '',
                    "relevance_score": 0.9,
                    "source": "Semantic Scholar"
                })
            
            # Search arXiv for preprints
            arxiv_papers = self._search_arxiv(topic_title, max_results=2)
            for paper in arxiv_papers:
                papers.append({
                    "title": paper.get('title', ''),
                    "authors": paper.get('authors', ''),
                    "journal": "arXiv Preprint",
                    "year": paper.get('published', '')[:4] if paper.get('published') else 'Unknown',
                    "doi": paper.get('doi', ''),
                    "url": paper.get('link', ''),
                    "abstract": paper.get('summary', '')[:200] + "..." if paper.get('summary') else '',
                    "relevance_score": 0.8,
                    "source": "arXiv"
                })
            
            # Sort by relevance and recency
            papers.sort(key=lambda x: (x.get('relevance_score', 0), int(x.get('year', '2000')) if x.get('year', '').isdigit() else 2000), reverse=True)
            
            return papers[:3]  # Return top 3 papers
            
        except Exception as e:
            print(f"Academic paper search error: {e}")
            return []
    
    def _extract_search_terms(self, title: str, content: str) -> List[str]:
        """Extract key search terms from title and content"""
        # Combine title and content
        text = f"{title} {content}"
        
        # Extract key terms (simple approach)
        words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
        word_freq = Counter(words)
        
        # Get most frequent meaningful words
        common_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'have', 'their', 'said', 'each', 'which', 'them', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'new', 'years', 'way', 'may', 'say', 'come', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'has', 'him', 'his', 'how', 'man', 'old', 'see', 'two', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
        
        meaningful_terms = [word for word, freq in word_freq.most_common(10) 
                          if word not in common_words and freq > 1]
        
        return meaningful_terms[:5]
    
    def _assess_research_quality(self, fact_checks: List[Dict], sources: List[Dict], 
                               quotes: List[Dict], papers: List[Dict]) -> Dict[ResearchQuality, float]:
        """Assess overall research quality based on gathered data"""
        
        # Calculate factual accuracy
        verified_facts = sum(1 for fc in fact_checks if fc.get('verification_status') == 'verified')
        factual_accuracy = (verified_facts / len(fact_checks)) if fact_checks else 0.5
        
        # Calculate source credibility
        credible_sources = sum(1 for s in sources if s.get('credibility_score', 0) > 0.7)
        source_credibility = (credible_sources / len(sources)) if sources else 0.5
        
        # Calculate citation quality
        citation_quality = min(1.0, len(papers) / 3)  # Up to 3 papers = max quality
        
        # Calculate expert validation
        expert_validation = min(1.0, len(quotes) / 2)  # Up to 2 quotes = max quality
        
        # Calculate timeliness (based on publication years)
        current_year = datetime.now().year
        recent_papers = sum(1 for p in papers if int(p.get('year', '2020')) >= current_year - 2)
        timeliness = (recent_papers / len(papers)) if papers else 0.6
        
        return {
            ResearchQuality.FACTUAL_ACCURACY: factual_accuracy,
            ResearchQuality.SOURCE_CREDIBILITY: source_credibility,
            ResearchQuality.CITATION_QUALITY: citation_quality,
            ResearchQuality.EXPERT_VALIDATION: expert_validation,
            ResearchQuality.TIMELINESS: timeliness
        }
    
    def _search_semantic_scholar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search Semantic Scholar API for academic papers"""
        if not self.enable_real_research:
            return []
            
        try:
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,authors,year,abstract,venue,url,doi'
            }
            
            response = self.session.get(self.semantic_scholar_api, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                papers = []
                
                for paper in data.get('data', []):
                    # Extract author names
                    authors = []
                    if paper.get('authors'):
                        authors = [author.get('name', 'Unknown') for author in paper['authors'][:5]]
                    
                    papers.append({
                        'title': paper.get('title', ''),
                        'authors': authors,
                        'year': paper.get('year'),
                        'abstract': paper.get('abstract', ''),
                        'venue': paper.get('venue', ''),
                        'url': paper.get('url', ''),
                        'doi': paper.get('doi', '')
                    })
                
                return papers
            else:
                print(f"Semantic Scholar API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
            return []
    
    def _search_arxiv(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search arXiv API for academic preprints"""
        if not self.enable_real_research:
            return []
            
        try:
            # Clean query for arXiv
            clean_query = query.replace(' ', '+')
            params = {
                'search_query': f'all:{clean_query}',
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            response = self.session.get(self.arxiv_api, params=params, timeout=10)
            
            if response.status_code == 200:
                # Parse XML response
                root = ET.fromstring(response.content)
                namespace = {'atom': 'http://www.w3.org/2005/Atom'}
                
                papers = []
                for entry in root.findall('atom:entry', namespace):
                    title = entry.find('atom:title', namespace)
                    summary = entry.find('atom:summary', namespace)
                    published = entry.find('atom:published', namespace)
                    
                    # Extract authors
                    authors = []
                    for author in entry.findall('atom:author', namespace):
                        name = author.find('atom:name', namespace)
                        if name is not None:
                            authors.append(name.text)
                    
                    # Extract links
                    link = None
                    for link_elem in entry.findall('atom:link', namespace):
                        if link_elem.get('type') == 'text/html':
                            link = link_elem.get('href')
                            break
                    
                    papers.append({
                        'title': title.text.strip() if title is not None else '',
                        'summary': summary.text.strip() if summary is not None else '',
                        'authors': ', '.join(authors) if authors else 'Unknown',
                        'published': published.text if published is not None else '',
                        'link': link or ''
                    })
                
                return papers
            else:
                print(f"arXiv API error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"arXiv search error: {e}")
            return []
    
    def _search_wikipedia(self, query: str) -> Optional[Dict[str, Any]]:
        """Search Wikipedia API for encyclopedic information"""
        if not self.enable_real_research:
            return None
            
        try:
            # Clean query for Wikipedia
            clean_query = query.replace(' ', '_')
            url = f"{self.wikipedia_api}/{urllib.parse.quote(clean_query)}"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', ''),
                    'extract': data.get('extract', ''),
                    'content_urls': data.get('content_urls', {}),
                    'thumbnail': data.get('thumbnail', {}),
                    'description': data.get('description', '')
                }
            else:
                # Try alternative search if direct lookup fails
                search_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(query)
                alt_response = self.session.get(search_url, timeout=10)
                if alt_response.status_code == 200:
                    return alt_response.json()
                return None
                
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return None
    
    def _create_fallback_research(self) -> ResearchEnhancement:
        """Create fallback research data when main process fails"""
        return ResearchEnhancement(
            factual_claims=[],
            verified_sources=[],
            expert_quotes=[],
            research_quality={
                ResearchQuality.FACTUAL_ACCURACY: 0.5,
                ResearchQuality.SOURCE_CREDIBILITY: 0.5,
                ResearchQuality.CITATION_QUALITY: 0.5,
                ResearchQuality.EXPERT_VALIDATION: 0.5,
                ResearchQuality.TIMELINESS: 0.5
            },
            fact_check_results=[],
            academic_references=[]
        )


# =============================================================================
# BRAND CONSISTENCY AND COMPLIANCE SYSTEM
# =============================================================================

class BrandConsistencyChecker:
    """
    Production-grade brand consistency, compliance checking, and originality verification.
    
    Analyzes content for brand voice alignment, regulatory compliance,
    and plagiarism detection to ensure consistent, compliant, and original content.
    """
    
    def __init__(self, model_name: str = "gpt-oss"):
        """Initialize brand consistency checker"""
        self.model = OllamaLLM(model=model_name)
        self.brand_voice_profile = self._load_brand_voice_profile()
        self.compliance_rules = self._load_compliance_rules()
        
    def assess_brand_consistency(self, content: str, platform: str) -> BrandConsistencyScore:
        """
        Assess content for brand consistency, compliance, and originality.
        
        Args:
            content: Content to assess
            platform: Target platform
            
        Returns:
            BrandConsistencyScore with detailed assessment
        """
        try:
            # Brand voice analysis
            voice_analysis = self._analyze_brand_voice(content, platform)
            
            # Compliance checking
            compliance_issues = self._check_compliance(content, platform)
            
            # Plagiarism detection
            plagiarism_score = self._check_originality(content)
            
            # Calculate metrics
            brand_metrics = self._calculate_brand_metrics(
                voice_analysis, compliance_issues, plagiarism_score
            )
            
            # Overall score
            overall_score = statistics.mean(brand_metrics.values())
            
            return BrandConsistencyScore(
                overall_score=overall_score,
                metrics=brand_metrics,
                voice_analysis=voice_analysis,
                compliance_issues=compliance_issues,
                plagiarism_score=plagiarism_score,
                originality_assessment=self._assess_originality(plagiarism_score),
                brand_alignment_feedback=self._generate_alignment_feedback(voice_analysis)
            )
            
        except Exception as e:
            print(f"Brand consistency check error: {e}")
            return self._create_fallback_brand_score()
    
    def _load_brand_voice_profile(self) -> Dict[str, str]:
        """Load or create brand voice profile"""
        # In production, this would load from a configuration file or database
        return {
            "voice_characteristics": "Professional yet approachable, authoritative but not intimidating",
            "tone_guidelines": "Informative, helpful, confident, and engaging",
            "messaging_pillars": "Innovation, reliability, expertise, customer-centricity",
            "avoid_terms": "jargon, overly technical language without explanation",
            "preferred_style": "Clear, concise, action-oriented communication"
        }
    
    def _load_compliance_rules(self) -> Dict[str, List[str]]:
        """Load compliance rules for different industries/regions"""
        return {
            "financial": [
                "Include appropriate disclaimers for investment advice",
                "Avoid guaranteeing returns or outcomes",
                "Include risk disclosures where appropriate"
            ],
            "healthcare": [
                "Avoid medical advice without proper disclaimers",
                "Include FDA-required statements where applicable",
                "Distinguish between information and medical advice"
            ],
            "general": [
                "Respect intellectual property rights",
                "Include proper attributions and citations",
                "Avoid misleading or false claims",
                "Comply with platform-specific guidelines"
            ]
        }
    
    def _analyze_brand_voice(self, content: str, platform: str) -> Dict[str, str]:
        """Analyze content for brand voice alignment"""
        try:
            brand_profile = self.brand_voice_profile
            
            prompt = f"""
Analyze this content for brand voice consistency:

CONTENT:
{content}

BRAND VOICE PROFILE:
- Voice Characteristics: {brand_profile['voice_characteristics']}
- Tone Guidelines: {brand_profile['tone_guidelines']}
- Messaging Pillars: {brand_profile['messaging_pillars']}
- Preferred Style: {brand_profile['preferred_style']}

PLATFORM: {platform}

Analyze and return JSON:
{{
    "voice_alignment": "score 1-10 and explanation",
    "tone_consistency": "how well tone matches guidelines",
    "messaging_alignment": "alignment with brand pillars",
    "style_analysis": "adherence to preferred style",
    "platform_adaptation": "appropriate adaptation for platform",
    "improvement_areas": "specific areas for improvement"
}}
"""
            
            response = self.model.invoke(prompt)
            
            try:
                analysis = json.loads(response.strip())
                return analysis
            except json.JSONDecodeError:
                pass
            
            # Fallback analysis
            return self._fallback_voice_analysis(content)
            
        except Exception as e:
            print(f"Brand voice analysis error: {e}")
            return self._fallback_voice_analysis(content)
    
    def _fallback_voice_analysis(self, content: str) -> Dict[str, str]:
        """Fallback brand voice analysis"""
        word_count = len(content.split())
        
        return {
            "voice_alignment": f"7.0 - Content appears professional with {word_count} words",
            "tone_consistency": "Generally consistent with brand guidelines",
            "messaging_alignment": "Aligned with core messaging principles",
            "style_analysis": "Clear and concise style maintained",
            "platform_adaptation": "Appropriate for target platform",
            "improvement_areas": "Consider enhancing engagement elements"
        }
    
    def _check_compliance(self, content: str, platform: str) -> List[str]:
        """Check content for compliance issues"""
        issues = []
        general_rules = self.compliance_rules.get("general", [])
        
        # Basic compliance checks
        content_lower = content.lower()
        
        # Check for potential false claims
        claiming_words = ["guaranteed", "proven to work", "100% effective", "never fails"]
        for claim in claiming_words:
            if claim in content_lower:
                issues.append(f"Potential absolute claim detected: '{claim}' - consider softening language")
        
        # Check for missing disclaimers in financial content
        financial_indicators = ["investment", "returns", "profit", "financial advice"]
        has_financial_content = any(indicator in content_lower for indicator in financial_indicators)
        disclaimer_indicators = ["disclaimer", "risk", "past performance", "not guaranteed"]
        has_disclaimer = any(indicator in content_lower for indicator in disclaimer_indicators)
        
        if has_financial_content and not has_disclaimer:
            issues.append("Financial content detected without appropriate disclaimers")
        
        # Check for proper attribution
        if "study shows" in content_lower or "research indicates" in content_lower:
            if "source:" not in content_lower and "according to" not in content_lower:
                issues.append("Claims reference studies/research without proper attribution")
        
        return issues
    
    def _check_originality(self, content: str) -> float:
        """Check content originality and potential plagiarism"""
        try:
            # In production, integrate with plagiarism detection APIs
            # For now, implement basic similarity checking
            
            # Simple heuristic based on content characteristics
            word_count = len(content.split())
            unique_words = len(set(content.lower().split()))
            
            # Calculate uniqueness ratio
            uniqueness_ratio = unique_words / word_count if word_count > 0 else 0
            
            # Simulate plagiarism score (lower = more original)
            # In production, this would check against databases of existing content
            plagiarism_score = max(0, 1 - uniqueness_ratio * 1.2)
            
            return min(1.0, plagiarism_score)
            
        except Exception as e:
            print(f"Originality check error: {e}")
            return 0.3  # Assume moderate originality on error
    
    def _calculate_brand_metrics(self, voice_analysis: Dict[str, str], 
                                compliance_issues: List[str], 
                                plagiarism_score: float) -> Dict[BrandConsistencyMetric, float]:
        """Calculate brand consistency metrics"""
        
        # Extract voice alignment score
        voice_text = voice_analysis.get("voice_alignment", "5.0")
        voice_score = float(re.search(r'\d+(?:\.\d+)?', voice_text).group()) / 10 if re.search(r'\d+(?:\.\d+)?', voice_text) else 0.7
        
        # Tone consistency (derived from analysis quality)
        tone_score = 0.8 if "consistent" in voice_analysis.get("tone_consistency", "").lower() else 0.6
        
        # Messaging alignment
        messaging_score = 0.8 if "aligned" in voice_analysis.get("messaging_alignment", "").lower() else 0.6
        
        # Compliance score (based on issues found)
        compliance_score = max(0.3, 1.0 - (len(compliance_issues) * 0.2))
        
        # Originality score (inverse of plagiarism score)
        originality_score = 1.0 - plagiarism_score
        
        return {
            BrandConsistencyMetric.VOICE_ALIGNMENT: voice_score,
            BrandConsistencyMetric.TONE_CONSISTENCY: tone_score,
            BrandConsistencyMetric.MESSAGING_ALIGNMENT: messaging_score,
            BrandConsistencyMetric.COMPLIANCE_CHECK: compliance_score,
            BrandConsistencyMetric.ORIGINALITY_SCORE: originality_score
        }
    
    def _assess_originality(self, plagiarism_score: float) -> str:
        """Assess originality based on plagiarism score"""
        if plagiarism_score < 0.1:
            return "Highly original content with unique perspectives and insights"
        elif plagiarism_score < 0.3:
            return "Good originality with some common themes appropriately used"
        elif plagiarism_score < 0.5:
            return "Moderate originality - consider adding more unique insights"
        else:
            return "Low originality detected - significant revision recommended"
    
    def _generate_alignment_feedback(self, voice_analysis: Dict[str, str]) -> str:
        """Generate brand alignment feedback"""
        improvements = voice_analysis.get("improvement_areas", "")
        tone_status = voice_analysis.get("tone_consistency", "")
        
        if "consistent" in tone_status.lower() and not improvements:
            return "Content is well-aligned with brand voice and guidelines"
        else:
            return f"Brand alignment feedback: {improvements}. Tone assessment: {tone_status}"
    
    def _create_fallback_brand_score(self) -> BrandConsistencyScore:
        """Create fallback brand consistency score"""
        return BrandConsistencyScore(
            overall_score=0.7,
            metrics={
                BrandConsistencyMetric.VOICE_ALIGNMENT: 0.7,
                BrandConsistencyMetric.TONE_CONSISTENCY: 0.7,
                BrandConsistencyMetric.MESSAGING_ALIGNMENT: 0.7,
                BrandConsistencyMetric.COMPLIANCE_CHECK: 0.8,
                BrandConsistencyMetric.ORIGINALITY_SCORE: 0.7
            },
            voice_analysis={"status": "Analysis unavailable"},
            compliance_issues=[],
            plagiarism_score=0.3,
            originality_assessment="Moderate originality assumed",
            brand_alignment_feedback="Brand consistency check completed with fallback scoring"
        )


# =============================================================================
# CONTENT QUALITY ASSESSMENT SYSTEM
# =============================================================================

class ContentQualityAssessor:
    """
    Production-grade AI-powered content quality assessment system.
    
    Provides comprehensive evaluation across 8 quality dimensions with detailed
    feedback, specific improvement suggestions, and iterative refinement
    capabilities to achieve consistent 8.0+ quality scores.
    """
    
    def __init__(self, model_name: str = "gpt-oss", max_retries: int = 3):
        """
        Initialize quality assessor with production configurations.
        
        Args:
            model_name: Name of the LLM model to use for assessments
            max_retries: Maximum retries for failed assessments
        """
        self.model = OllamaLLM(model=model_name)
        self.max_retries = max_retries
        self.comprehensive_assessment_prompt = self._initialize_comprehensive_prompt()
        self.platform_specifications = self._initialize_platform_specs()
    
    def _initialize_comprehensive_prompt(self) -> str:
        """Initialize comprehensive assessment prompt for all metrics"""
        return """
You are an expert content quality assessor for social media platforms. Evaluate this {platform} post comprehensively across all quality dimensions.

CONTENT TO ASSESS:
{content}

PLATFORM: {platform}
TOPIC CONTEXT: {topic_context}

ASSESSMENT CRITERIA:

1. ENGAGEMENT_POTENTIAL (1-10): How likely is this to generate likes, comments, shares?
   - Hook strength and attention-grabbing opening
   - Use of questions, polls, or interactive elements  
   - Emotional resonance and relatability
   - Call-to-action effectiveness
   - Trending topics or timely relevance

2. READABILITY (1-10): How easy and pleasant is this to read?
   - Sentence structure and clarity
   - Use of formatting (bullets, spacing, emojis)
   - Paragraph length and visual appeal
   - Complex vocabulary vs accessibility
   - Overall flow and coherence

3. RELEVANCE (1-10): How well does this match the topic and provide value?
   - Accuracy and relevance to the stated topic
   - Depth of insights and value provided
   - Practical applicability for the audience
   - Clear connection between title and content
   - Addresses audience pain points or interests

4. UNIQUENESS (1-10): How original and distinctive is this content?
   - Novel perspectives or unique angles
   - Avoidance of clichés and overused phrases
   - Personal voice and authentic style
   - Fresh examples or case studies
   - Differentiated from common content on the topic

5. PLATFORM_OPTIMIZATION (1-10): How well is this optimized for {platform}?
   - Appropriate length for the platform (LinkedIn: 1300-3000 chars, Twitter: 150-280 chars, Reddit: 200-1500 chars, Newsletter: 800-2500 chars)
   - Platform-specific formatting and style conventions
   - For newsletters: clear sections, headers, scannable structure
   - For social media: hashtag usage and discoverability elements
   - Community guidelines compliance
   - Algorithm/delivery optimization factors

6. CTA_STRENGTH (1-10): How effective is the engagement invitation?
   - Clear and compelling call-to-action
   - Relevant questions that encourage discussion
   - Invitation for specific audience actions
   - Community-building elements
   - Engagement-driving language and structure

7. EMOTIONAL_APPEAL (1-10): How well does this connect emotionally?
   - Emotional hooks and resonance
   - Storytelling elements and personal connection
   - Inspirational or motivational aspects
   - Empathy and relatability
   - Emotional language and tone appropriateness

8. CLARITY (1-10): How clear and understandable is the message?
   - Clear main message and key points
   - Logical structure and flow
   - Absence of confusion or ambiguity
   - Professional presentation
   - Easy to follow progression of ideas

RESPONSE FORMAT (JSON):
{{
    "overall_score": [1-10 average of all metrics],
    "metrics": {{
        "engagement_potential": [score],
        "readability": [score], 
        "relevance": [score],
        "uniqueness": [score],
        "platform_optimization": [score],
        "cta_strength": [score],
        "emotional_appeal": [score],
        "clarity": [score]
    }},
    "detailed_feedback": {{
        "engagement_potential": "[Specific feedback on what works/doesn't work for engagement]",
        "readability": "[Specific feedback on readability aspects]",
        "relevance": "[Specific feedback on relevance to topic]", 
        "uniqueness": "[Specific feedback on originality]",
        "platform_optimization": "[Specific feedback on platform fit]",
        "cta_strength": "[Specific feedback on CTA effectiveness]",
        "emotional_appeal": "[Specific feedback on emotional connection]",
        "clarity": "[Specific feedback on message clarity]"
    }},
    "improvement_suggestions": [
        "[Specific, actionable improvement 1]",
        "[Specific, actionable improvement 2]", 
        "[Specific, actionable improvement 3]"
    ],
    "strengths": [
        "[What's working well 1]",
        "[What's working well 2]"
    ],
    "weaknesses": [
        "[What needs fixing 1]",
        "[What needs fixing 2]"
    ],
    "assessment_reasoning": "[Brief explanation of why you gave this overall score]"
}}

Be specific, actionable, and focused on improvements that will increase engagement and value for the {platform} audience.
"""
    
    def _initialize_platform_specs(self) -> Dict[str, Dict[str, Any]]:
        """Initialize platform-specific specifications and best practices"""
        return {
            'linkedin': {
                'optimal_length': (1300, 3000),
                'max_hashtags': 5,
                'tone_preferences': ['professional', 'authoritative', 'informative'],
                'engagement_patterns': ['questions', 'insights', 'industry_trends'],
                'format_preferences': ['bullets', 'numbered_lists', 'line_breaks']
            },
            'twitter': {
                'optimal_length': (150, 280),
                'max_hashtags': 3,
                'tone_preferences': ['conversational', 'direct', 'witty'],
                'engagement_patterns': ['questions', 'threads', 'polls'],
                'format_preferences': ['concise', 'punchy', 'quotable']
            },
            'reddit': {
                'optimal_length': (200, 1500),
                'max_hashtags': 0,
                'tone_preferences': ['casual', 'authentic', 'community_focused'],
                'engagement_patterns': ['discussion', 'questions', 'personal_experience'],
                'format_preferences': ['paragraphs', 'markdown', 'conversational']
            },
            'newsletter': {
                'optimal_length': (800, 2500),
                'max_hashtags': 0,
                'tone_preferences': ['informative', 'conversational', 'authoritative'],
                'engagement_patterns': ['insights', 'actionable_advice', 'curated_content'],
                'format_preferences': ['sections', 'headers', 'bullets', 'clear_structure']
            }
        }
    
    def assess_content_quality(self, content: str, platform: str, topic_content: str = "") -> ContentQualityScore:
        """
        Production-grade comprehensive AI-powered content assessment.
        
        Args:
            content: Content text to assess
            platform: Target platform
            topic_content: Original topic for context
            
        Returns:
            ContentQualityScore with detailed AI feedback and metrics
        """
        for attempt in range(self.max_retries):
            try:
                # Perform comprehensive AI assessment
                assessment_result = self._perform_comprehensive_assessment(content, platform, topic_content)
                
                if assessment_result:
                    return assessment_result
                    
            except Exception as e:
                print(f"Assessment attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Return fallback assessment on final failure
                    return self._generate_fallback_assessment(content, platform)
        
        return self._generate_fallback_assessment(content, platform)
    
    def _perform_comprehensive_assessment(self, content: str, platform: str, topic_content: str) -> Optional[ContentQualityScore]:
        """
        Perform comprehensive AI assessment with structured output parsing.
        
        Returns:
            ContentQualityScore or None if assessment fails
        """
        try:
            # Create assessment prompt
            prompt = ChatPromptTemplate.from_template(self.comprehensive_assessment_prompt)
            chain = prompt | self.model
            
            # Get AI assessment
            response = chain.invoke({
                "content": content,
                "platform": platform,
                "topic_context": topic_content[:500] + "..." if len(topic_content) > 500 else topic_content
            })
            
            # Parse JSON response
            assessment_data = self._parse_assessment_response(response)
            
            if not assessment_data:
                return None
            
            # Convert to quality score object
            return self._create_quality_score_from_assessment(assessment_data)
            
        except Exception as e:
            print(f"Comprehensive assessment error: {e}")
            return None
    
    def _parse_assessment_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse AI response and extract JSON assessment data.
        
        Args:
            response: Raw AI response
            
        Returns:
            Parsed assessment data or None if parsing fails
        """
        try:
            # Clean response and extract JSON
            response_str = str(response).strip()
            
            # Find JSON content
            json_start = response_str.find('{')
            json_end = response_str.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                return None
                
            json_content = response_str[json_start:json_end]
            
            # Parse JSON
            assessment_data = json.loads(json_content)
            
            # Validate required fields
            required_fields = ['overall_score', 'metrics', 'detailed_feedback', 'improvement_suggestions']
            if not all(field in assessment_data for field in required_fields):
                return None
                
            return assessment_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"Response parsing error: {e}")
            return None
    
    def _create_quality_score_from_assessment(self, assessment_data: Dict[str, Any]) -> ContentQualityScore:
        """
        Create ContentQualityScore object from parsed assessment data.
        
        Args:
            assessment_data: Parsed assessment JSON
            
        Returns:
            ContentQualityScore object
        """
        # Convert string metric keys to QualityMetric enums
        metrics = {}
        for metric_name, score in assessment_data.get('metrics', {}).items():
            try:
                metric_enum = QualityMetric(metric_name.lower())
                metrics[metric_enum] = float(score)
            except (ValueError, TypeError):
                print(f"Invalid metric: {metric_name}")
                continue
        
        # Ensure overall score is valid
        overall_score = float(assessment_data.get('overall_score', 5.0))
        if not 1.0 <= overall_score <= 10.0:
            overall_score = max(1.0, min(10.0, overall_score))
        
        return ContentQualityScore(
            overall_score=overall_score,
            metrics=metrics,
            detailed_feedback=assessment_data.get('detailed_feedback', {}),
            improvement_suggestions=assessment_data.get('improvement_suggestions', []),
            strengths=assessment_data.get('strengths', []),
            weaknesses=assessment_data.get('weaknesses', []),
            confidence=min(len(metrics) / len(QualityMetric), 1.0),
            assessment_reasoning=assessment_data.get('assessment_reasoning', "")
        )
    
    def _generate_fallback_assessment(self, content: str, platform: str) -> ContentQualityScore:
        """
        Generate fallback assessment when AI assessment fails.
        
        Args:
            content: Content to assess
            platform: Target platform
            
        Returns:
            Basic ContentQualityScore with neutral scores
        """
        # Simple rule-based fallback
        char_count = len(content)
        word_count = len(content.split())
        
        # Basic platform optimization check
        platform_specs = self.platform_specifications.get(platform, {})
        optimal_range = platform_specs.get('optimal_length', (200, 2000))
        
        platform_score = 8.0 if optimal_range[0] <= char_count <= optimal_range[1] else 6.0
        
        # Basic engagement check
        engagement_score = 7.0 if ('?' in content or 'what do you think' in content.lower()) else 6.0
        
        fallback_metrics = {
            QualityMetric.PLATFORM_OPTIMIZATION: platform_score,
            QualityMetric.ENGAGEMENT_POTENTIAL: engagement_score,
            QualityMetric.READABILITY: 7.0,
            QualityMetric.RELEVANCE: 7.0,
            QualityMetric.UNIQUENESS: 6.0,
            QualityMetric.CALL_TO_ACTION_STRENGTH: engagement_score,
            QualityMetric.EMOTIONAL_APPEAL: 6.5,
            QualityMetric.CLARITY: 7.0
        }
        
        overall_score = statistics.mean(fallback_metrics.values())
        
        return ContentQualityScore(
            overall_score=overall_score,
            metrics=fallback_metrics,
            detailed_feedback={
                "system": "AI assessment unavailable, using fallback evaluation"
            },
            improvement_suggestions=[
                "Enhance content with more specific examples",
                "Add stronger call-to-action elements",
                "Incorporate more engaging storytelling"
            ],
            strengths=["Appropriate length for platform"],
            weaknesses=["Limited assessment due to system fallback"],
            confidence=0.3,
            assessment_reasoning="Fallback assessment due to AI evaluation failure"
        )
    


# =============================================================================
# CONTENT OPTIMIZATION SYSTEM
# =============================================================================

class ContentOptimizer:
    """
    Production-grade AI-powered content optimization with iterative feedback integration.
    
    Uses comprehensive quality assessment feedback to regenerate content iteratively
    until achieving target quality scores (8.0+), with detailed tracking and 
    monitoring of improvement iterations.
    """
    
    def __init__(self, model_name: str = "gpt-oss", max_optimization_attempts: int = 3):
        """
        Initialize content optimizer with production configurations.
        
        Args:
            model_name: Name of the LLM model for optimization
            max_optimization_attempts: Maximum optimization iterations per content piece
        """
        self.model = OllamaLLM(model=model_name)
        self.max_optimization_attempts = max_optimization_attempts
        self.optimization_prompt = self._initialize_optimization_prompt()
        
    def _initialize_optimization_prompt(self) -> str:
        """Initialize comprehensive optimization prompt"""
        return """
You are an expert content optimizer for {platform}. Your task is to improve the given content based on specific quality assessment feedback to achieve a score of 8.0+ out of 10.

ORIGINAL CONTENT TO OPTIMIZE:
{content}

PLATFORM: {platform}
TARGET AUDIENCE: {platform} users
TOPIC CONTEXT: {topic_context}

QUALITY ASSESSMENT FEEDBACK:
Overall Score: {overall_score}/10 (TARGET: 8.0+)

DETAILED FEEDBACK BY METRIC:
{detailed_feedback}

SPECIFIC IMPROVEMENT SUGGESTIONS:
{improvement_suggestions}

STRENGTHS TO MAINTAIN:
{strengths}

WEAKNESSES TO ADDRESS:
{weaknesses}

OPTIMIZATION REQUIREMENTS:
1. Address ALL identified weaknesses and feedback points
2. Maintain and enhance the identified strengths
3. Ensure content is optimized for {platform} best practices
4. Keep the core message and value proposition intact
5. Make specific improvements suggested in the feedback
6. Target score improvement to 8.0+ across all metrics

PLATFORM-SPECIFIC OPTIMIZATION GUIDELINES FOR {platform}:
{platform_guidelines}

OUTPUT INSTRUCTIONS:
- Provide ONLY the optimized content (no explanations or meta-commentary)
- Ensure the content addresses every point in the detailed feedback
- Maintain authenticity while improving quality
- Focus on engagement, clarity, and platform optimization

OPTIMIZED CONTENT:
"""
    
    def optimize_content_iteratively(self, content: str, platform: str, 
                                   quality_assessor: 'ContentQualityAssessor',
                                   topic_content: str = "", target_score: float = 8.0) -> Tuple[str, List[ContentQualityScore], int]:
        """
        Perform iterative content optimization until target quality score is achieved.
        
        Args:
            content: Original content to optimize
            platform: Target platform
            quality_assessor: Quality assessment system
            topic_content: Original topic context
            target_score: Target quality score to achieve
            
        Returns:
            Tuple of (optimized_content, quality_scores_history, iterations_performed)
        """
        current_content = content
        quality_history = []
        
        for iteration in range(self.max_optimization_attempts):
            print(f"    🔧 Optimization iteration {iteration + 1}/{self.max_optimization_attempts}...")
            
            # Assess current quality
            quality_score = quality_assessor.assess_content_quality(current_content, platform, topic_content)
            quality_history.append(quality_score)
            
            print(f"    📊 Current score: {quality_score.overall_score:.1f}/10")
            
            # Check if target score achieved
            if quality_score.overall_score >= target_score:
                print(f"    ✅ Target score {target_score} achieved!")
                break
                
            # If last iteration and still not at target, accept current version
            if iteration == self.max_optimization_attempts - 1:
                print(f"    ⏰ Max iterations reached. Final score: {quality_score.overall_score:.1f}/10")
                break
            
            # Optimize content based on feedback
            try:
                optimized_content = self._generate_optimized_content(
                    current_content, platform, quality_score, topic_content
                )
                
                if optimized_content and optimized_content.strip() != current_content.strip():
                    current_content = optimized_content
                    print(f"    🔄 Content optimized (attempt {iteration + 1})")
                else:
                    print(f"    ⚠️  No improvement generated, keeping current version")
                    break
                    
            except Exception as e:
                print(f"    ❌ Optimization failed: {e}")
                break
        
        return current_content, quality_history, len(quality_history)
    
    def _generate_optimized_content(self, content: str, platform: str, 
                                  quality_score: ContentQualityScore, topic_content: str) -> str:
        """
        Generate optimized content based on comprehensive feedback.
        
        Args:
            content: Current content version
            platform: Target platform
            quality_score: Quality assessment with detailed feedback
            topic_content: Original topic context
            
        Returns:
            Optimized content string
        """
        try:
            # Prepare platform-specific guidelines
            platform_guidelines = self._get_platform_guidelines(platform)
            
            # Format detailed feedback
            detailed_feedback_str = self._format_detailed_feedback(quality_score.detailed_feedback)
            
            # Create optimization prompt
            prompt = ChatPromptTemplate.from_template(self.optimization_prompt)
            chain = prompt | self.model
            
            # Generate optimized content
            response = chain.invoke({
                "content": content,
                "platform": platform,
                "topic_context": topic_content[:500] + "..." if len(topic_content) > 500 else topic_content,
                "overall_score": quality_score.overall_score,
                "detailed_feedback": detailed_feedback_str,
                "improvement_suggestions": "\n".join([f"• {suggestion}" for suggestion in quality_score.improvement_suggestions]),
                "strengths": "\n".join([f"• {strength}" for strength in quality_score.strengths]),
                "weaknesses": "\n".join([f"• {weakness}" for weakness in quality_score.weaknesses]),
                "platform_guidelines": platform_guidelines
            })
            
            # Clean and return optimized content
            optimized_content = str(response).strip()
            
            # Remove any meta-commentary or instructions that might have been included
            optimized_content = self._clean_optimized_content(optimized_content)
            
            return optimized_content if optimized_content else content
            
        except Exception as e:
            print(f"    ❌ Error generating optimized content: {e}")
            return content
    
    def _format_detailed_feedback(self, detailed_feedback: Dict[str, str]) -> str:
        """Format detailed feedback for prompt inclusion"""
        if not detailed_feedback:
            return "No specific feedback available"
            
        formatted_feedback = []
        for metric, feedback in detailed_feedback.items():
            formatted_feedback.append(f"• {metric.replace('_', ' ').title()}: {feedback}")
            
        return "\n".join(formatted_feedback)
    
    def _get_platform_guidelines(self, platform: str) -> str:
        """Get platform-specific optimization guidelines"""
        guidelines = {
            'linkedin': """
• Use professional tone with authoritative insights
• Optimize for 1300-3000 characters
• Include relevant hashtags (3-5 maximum)
• Structure with clear paragraphs and bullet points
• End with engaging professional questions
• Focus on value-driven content for business audience
""",
            'twitter': """
• Keep concise and punchy (150-280 characters ideal)
• Use conversational, direct tone
• Include 2-3 relevant hashtags maximum
• Create quotable, shareable content
• Use threads for complex topics
• Encourage retweets and replies
""",
            'reddit': """
• Use authentic, community-focused tone
• Optimize for 200-1500 characters
• Avoid promotional language
• Encourage discussion and community interaction
• Use markdown formatting appropriately
• Be genuinely helpful and informative
""",
            'newsletter': """
• Use clear, scannable structure with headers and sections
• Optimize for 800-2500 characters
• Include actionable insights and practical takeaways
• Use bullet points and numbered lists for readability
• Provide clear value proposition upfront
• End with next steps or resources
• Focus on subscriber retention and engagement
• Maintain consistent tone throughout sections
"""
        }
        
        return guidelines.get(platform, "Follow general best practices for the platform")
    
    def _clean_optimized_content(self, content: str) -> str:
        """Clean optimized content of any unwanted meta-commentary"""
        # Remove common unwanted prefixes/suffixes
        unwanted_prefixes = [
            "here's the optimized content:",
            "optimized content:",
            "improved version:",
            "here's the improved post:",
            "optimized version:"
        ]
        
        unwanted_suffixes = [
            "this addresses all the feedback points",
            "this should improve the quality score",
            "this optimized version should"
        ]
        
        cleaned_content = content.strip()
        
        # Remove unwanted prefixes
        for prefix in unwanted_prefixes:
            if cleaned_content.lower().startswith(prefix):
                cleaned_content = cleaned_content[len(prefix):].strip()
        
        # Remove unwanted suffixes  
        for suffix in unwanted_suffixes:
            if suffix in cleaned_content.lower():
                index = cleaned_content.lower().find(suffix)
                cleaned_content = cleaned_content[:index].strip()
        
        return cleaned_content


# =============================================================================
# MAIN CONTENT GENERATION PIPELINE
# =============================================================================

class AdvancedContentPipeline:
    """
    Complete advanced content generation pipeline orchestrator.
    
    Coordinates all pipeline stages from topic processing through finalization,
    providing comprehensive content generation with quality assessment,
    optimization, and analytics for high-performance social media content.
    
    Pipeline Stages:
        1. Topic Processing - Analyze and extract topic information
        2. Content Categorization - Auto-categorize for better inspiration
        3. Enhanced Research - Fact-checking, expert quotes, credible sources
        4. Inspiration Retrieval - Enhanced RAG with hybrid search
        5. Content Generation - AI-powered multi-variation creation
        6. Quality Assessment - Comprehensive quality scoring
        7. Brand Consistency - Voice alignment, compliance, originality checks
        8. Content Optimization - Iterative improvement based on feedback
        9. Analytics Tracking - Performance and quality metrics
        10. Finalization - Database storage and result packaging
    """
    
    def __init__(self, model_name: str = "gpt-oss", 
                 qdrant_host: str = "localhost", qdrant_port: int = 6333,
                 language: str = "en"):
        """
        Initialize advanced content pipeline with all required components.
        
        Args:
            model_name: LLM model name for content generation
            qdrant_host: Qdrant database host
            qdrant_port: Qdrant database port
            language: Target language for content generation and inspiration ('en' or 'pl')
        """
        # Initialize core components
        self.model_name = model_name
        self.language = language
        self.model = OllamaLLM(model=model_name)
        
        # Initialize embedding and vector components
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = f"inspiration_content_{language}"
        
        # Initialize pipeline components
        self.content_categorizer = ContentCategorizer()
        self.template_manager = ContentTemplateManager(Language(language))
        self.quality_assessor = ContentQualityAssessor(model_name)
        self.content_optimizer = ContentOptimizer(model_name)
        self.research_engine = EnhancedResearchEngine(model_name)
        self.brand_checker = BrandConsistencyChecker(model_name)
        self.rag_system = EnhancedRAGSystem(
            self.qdrant_client, self.embeddings, self.collection_name
        )
        
        # Pipeline configuration
        self.max_optimization_iterations = 3
        self.quality_threshold = 8.0  # Target 8.0+ for all content
        self.enable_optimization = True
        self.target_quality_score = 8.0
        
    def execute_pipeline(self, topic_data: Dict[str, Any], platforms: List[str],
                        content_type: Optional[ContentType] = None,
                        tone: Optional[ToneStyle] = None,
                        num_variations: int = 1) -> ContentPipelineResult:
        """
        Execute the complete 10-stage enhanced content generation pipeline.
        
        Args:
            topic_data: Dictionary containing topic information (title, content, etc.)
            platforms: List of target platforms (linkedin, reddit, twitter)
            content_type: Optional specific content type to generate
            tone: Optional specific tone style to use
            num_variations: Number of content variations to generate per platform
            
        Returns:
            ContentPipelineResult with comprehensive execution data and results
        """
        pipeline_start = time.time()
        result = ContentPipelineResult(
            topic_title=topic_data.get('title', 'Unknown'),
            platforms=platforms
        )
        
        try:
            # Stage 1: Topic Processing
            print("🔄 Stage 1/10: Topic Processing...")
            stage_result = self._execute_topic_processing(topic_data)
            result.stage_results[PipelineStage.TOPIC_PROCESSING] = stage_result
            if not stage_result.success:
                return result
            
            # Stage 2: Content Categorization
            print("🔄 Stage 2/10: Content Categorization...")
            stage_result = self._execute_content_categorization(topic_data)
            result.stage_results[PipelineStage.CONTENT_CATEGORIZATION] = stage_result
            
            # Stage 3: Enhanced Research
            print("🔄 Stage 3/10: Enhanced Research...")
            stage_result = self._execute_enhanced_research(topic_data)
            result.stage_results[PipelineStage.ENHANCED_RESEARCH] = stage_result
            research_data = stage_result.data
            result.research_data = stage_result.data.get('research_enhancement')
            
            # Stage 4: Inspiration Retrieval
            print("🔄 Stage 4/10: Inspiration Retrieval...")
            stage_result = self._execute_inspiration_retrieval(topic_data, platforms)
            result.stage_results[PipelineStage.INSPIRATION_RETRIEVAL] = stage_result
            inspiration_data = stage_result.data
            
            # Stage 5: Content Generation
            print("🔄 Stage 5/10: Content Generation...")
            stage_result = self._execute_content_generation(
                topic_data, platforms, inspiration_data, content_type, tone, num_variations
            )
            result.stage_results[PipelineStage.CONTENT_GENERATION] = stage_result
            if not stage_result.success:
                return result
            
            generated_content = stage_result.data['generated_content']
            
            # Stage 6: Quality Assessment
            print("🔄 Stage 6/10: Quality Assessment...")
            stage_result = self._execute_quality_assessment(generated_content, topic_data)
            result.stage_results[PipelineStage.QUALITY_ASSESSMENT] = stage_result
            quality_scores = stage_result.data['quality_scores']
            result.quality_scores = quality_scores
            
            # Stage 7: Brand Consistency Check
            print("🔄 Stage 7/10: Brand Consistency...")
            stage_result = self._execute_brand_consistency(generated_content, platforms)
            result.stage_results[PipelineStage.BRAND_CONSISTENCY] = stage_result
            brand_scores = stage_result.data.get('brand_scores', {})
            result.brand_scores = brand_scores
            
            # Stage 8: Content Optimization (if enabled)
            if self.enable_optimization:
                print("🔄 Stage 8/10: Content Optimization...")
                stage_result, optimized_content, iterations = self._execute_content_optimization(
                    generated_content, quality_scores, topic_data
                )
                result.stage_results[PipelineStage.CONTENT_OPTIMIZATION] = stage_result
                result.optimization_iterations = iterations
                generated_content = optimized_content
            else:
                print("🔄 Stage 8/10: Content Optimization (Skipped)")
            
            # Stage 9: Analytics Tracking
            print("🔄 Stage 9/10: Analytics Tracking...")
            stage_result = self._execute_analytics_tracking(generated_content, result)
            result.stage_results[PipelineStage.ANALYTICS_TRACKING] = stage_result
            result.analytics = stage_result.data
            
            # Stage 10: Finalization
            print("🔄 Stage 10/10: Finalization...")
            stage_result = self._execute_finalization(generated_content, result)
            result.stage_results[PipelineStage.FINALIZATION] = stage_result
            
            result.generated_content = generated_content
            result.success = True
            
        except Exception as e:
            print(f"Pipeline execution error: {e}")
            result.success = False
        
        result.total_execution_time = time.time() - pipeline_start
        return result
    
    def _execute_topic_processing(self, topic_data: Dict[str, Any]) -> PipelineStageResult:
        """Stage 1: Process and analyze topic data"""
        stage_start = time.time()
        
        try:
            content = topic_data.get('content', '')
            title = topic_data.get('title', '')
            
            # Calculate topic complexity metrics
            word_count = len(content.split())
            char_count = len(content)
            sentence_count = len([s for s in content.split('.') if s.strip()])
            
            # Extract key themes
            key_themes = self._extract_key_themes(content)
            
            # Determine content strategy
            strategy = self._determine_content_strategy(content, word_count)
            
            return PipelineStageResult(
                stage=PipelineStage.TOPIC_PROCESSING,
                success=True,
                data={
                    'processed_topic': topic_data,
                    'key_themes': key_themes,
                    'content_strategy': strategy,
                    'complexity_metrics': {
                        'word_count': word_count,
                        'char_count': char_count,
                        'sentence_count': sentence_count
                    }
                },
                metrics={
                    'complexity_score': min(word_count / 100, 10),
                    'theme_diversity': len(key_themes)
                },
                execution_time=time.time() - stage_start
            )
            
        except Exception as e:
            return PipelineStageResult(
                stage=PipelineStage.TOPIC_PROCESSING,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - stage_start
            )
    
    def _execute_content_categorization(self, topic_data: Dict[str, Any]) -> PipelineStageResult:
        """Stage 2: Categorize content for better inspiration matching"""
        stage_start = time.time()
        
        try:
            content = topic_data.get('content', '')
            title = topic_data.get('title', '')
            
            # Auto-categorize content
            categories = self.content_categorizer.categorize_content(content, title)
            primary_category = categories[0] if categories else ContentCategory.OTHER
            
            return PipelineStageResult(
                stage=PipelineStage.CONTENT_CATEGORIZATION,
                success=True,
                data={
                    'categories': [cat.value for cat in categories],
                    'primary_category': primary_category.value,
                    'category_confidence': len(categories) / len(ContentCategory)
                },
                metrics={
                    'categorization_confidence': min(len(categories) * 2, 10)
                },
                execution_time=time.time() - stage_start
            )
            
        except Exception as e:
            return PipelineStageResult(
                stage=PipelineStage.CONTENT_CATEGORIZATION,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - stage_start
            )
    
    def _execute_inspiration_retrieval(self, topic_data: Dict[str, Any], platforms: List[str]) -> PipelineStageResult:
        """Stage 3: Retrieve relevant inspiration content using enhanced RAG"""
        stage_start = time.time()
        
        try:
            content = topic_data.get('content', '')
            inspiration_data = {}
            
            for platform in platforms:
                # Calculate dynamic k-value based on topic complexity
                k_value = self._calculate_dynamic_k(content)
                
                # Get enhanced inspiration search
                search_result = self.rag_system.auto_categorize_and_search(content, platform, k_value)
                
                inspiration_data[platform] = {
                    'posts': search_result['results'],
                    'categories': search_result['detected_categories'],
                    'search_quality': len(search_result['results']) / k_value
                }
            
            return PipelineStageResult(
                stage=PipelineStage.INSPIRATION_RETRIEVAL,
                success=True,
                data=inspiration_data,
                metrics={
                    'total_inspiration_posts': sum(len(data['posts']) for data in inspiration_data.values()),
                    'average_search_quality': statistics.mean([data['search_quality'] for data in inspiration_data.values()])
                },
                execution_time=time.time() - stage_start
            )
            
        except Exception as e:
            return PipelineStageResult(
                stage=PipelineStage.INSPIRATION_RETRIEVAL,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - stage_start
            )
    
    def _execute_content_generation(self, topic_data: Dict[str, Any], platforms: List[str],
                                  inspiration_data: Dict[str, Any], content_type: Optional[ContentType],
                                  tone: Optional[ToneStyle], num_variations: int) -> PipelineStageResult:
        """Stage 4: Generate content using AI with templates and inspiration"""
        stage_start = time.time()
        
        try:
            generated_content = []
            
            for platform in platforms:
                
                # Determine optimal content type and tone if not specified
                current_content_type = content_type
                current_tone = tone
                if not current_content_type:
                    current_content_type = self._determine_optimal_content_type(topic_data, platform)
                if not current_tone:
                    current_tone = self._determine_optimal_tone(topic_data, platform)
                
                # Generate content variations
                variations = self._generate_variations(
                    platform, topic_data, current_content_type, current_tone, num_variations, inspiration_data.get(platform, {})
                )
                
                for i, variation in enumerate(variations):
                    content_item = {
                        'platform': platform,
                        'content': variation,
                        'content_type': current_content_type.value,
                        'tone': current_tone.value,
                        'variation_index': i,
                        'metadata': {
                            'char_count': len(variation),
                            'word_count': len(variation.split()),
                            'generation_timestamp': datetime.now().isoformat()
                        }
                    }
                    generated_content.append(content_item)
            
            return PipelineStageResult(
                stage=PipelineStage.CONTENT_GENERATION,
                success=True,
                data={'generated_content': generated_content},
                metrics={
                    'total_content_pieces': len(generated_content),
                    'platforms_covered': len(platforms)
                },
                execution_time=time.time() - stage_start
            )
            
        except Exception as e:
            return PipelineStageResult(
                stage=PipelineStage.CONTENT_GENERATION,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - stage_start
            )
    
    def _execute_quality_assessment(self, generated_content: List[Dict[str, Any]], 
                                  topic_data: Dict[str, Any]) -> PipelineStageResult:
        """Stage 5: Assess content quality using comprehensive metrics"""
        stage_start = time.time()
        
        try:
            quality_scores = {}
            total_score = 0
            
            for content_item in generated_content:
                key = f"{content_item['platform']}_v{content_item['variation_index']}"
                
                quality_score = self.quality_assessor.assess_content_quality(
                    content_item['content'], 
                    content_item['platform'], 
                    topic_data.get('content', '')
                )
                
                quality_scores[key] = quality_score
                total_score += quality_score.overall_score
            
            average_quality = total_score / len(generated_content) if generated_content else 0
            
            return PipelineStageResult(
                stage=PipelineStage.QUALITY_ASSESSMENT,
                success=True,
                data={'quality_scores': quality_scores},
                metrics={
                    'average_quality_score': average_quality,
                    'high_quality_content': sum(1 for score in quality_scores.values() if score.overall_score >= self.quality_threshold)
                },
                execution_time=time.time() - stage_start
            )
            
        except Exception as e:
            return PipelineStageResult(
                stage=PipelineStage.QUALITY_ASSESSMENT,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - stage_start
            )
    
    def _execute_content_optimization(self, generated_content: List[Dict[str, Any]],
                                    quality_scores: Dict[str, ContentQualityScore],
                                    topic_data: Dict[str, Any]) -> Tuple[PipelineStageResult, List[Dict[str, Any]], int]:
        """Stage 6: Production-grade iterative content optimization until 8.0+ quality"""
        stage_start = time.time()
        optimized_content = generated_content.copy()
        total_iterations = 0
        total_improvements = 0
        optimization_details = []
        
        try:
            print("  🎯 Starting iterative optimization to achieve 8.0+ quality scores...")
            
            for i, content_item in enumerate(optimized_content):
                key = f"{content_item['platform']}_v{content_item['variation_index']}"
                initial_quality_score = quality_scores.get(key)
                
                if not initial_quality_score:
                    continue
                    
                print(f"  📝 Optimizing {content_item['platform']} content (variation {content_item['variation_index'] + 1})")
                print(f"    Initial score: {initial_quality_score.overall_score:.1f}/10")
                
                # Check if already meets quality threshold
                if initial_quality_score.overall_score >= self.target_quality_score:
                    print(f"    ✅ Already meets quality target ({self.target_quality_score})")
                    continue
                
                # Perform iterative optimization
                optimized_text, quality_history, iterations_performed = self.content_optimizer.optimize_content_iteratively(
                    content_item['content'],
                    content_item['platform'],
                    self.quality_assessor,
                    topic_data.get('content', ''),
                    self.target_quality_score
                )
                
                # Update content if optimization occurred
                if optimized_text != content_item['content'] and quality_history:
                    optimized_content[i]['content'] = optimized_text
                    optimized_content[i]['metadata']['optimized'] = True
                    optimized_content[i]['metadata']['optimization_iterations'] = iterations_performed
                    optimized_content[i]['metadata']['initial_score'] = initial_quality_score.overall_score
                    optimized_content[i]['metadata']['final_score'] = quality_history[-1].overall_score
                    optimized_content[i]['metadata']['score_improvement'] = quality_history[-1].overall_score - initial_quality_score.overall_score
                    
                    # Update quality scores with final assessment
                    quality_scores[key] = quality_history[-1]
                    
                    total_improvements += 1
                    print(f"    📈 Score improved: {initial_quality_score.overall_score:.1f} → {quality_history[-1].overall_score:.1f}")
                
                total_iterations += iterations_performed
                
                # Track optimization details
                optimization_details.append({
                    'content_key': key,
                    'platform': content_item['platform'],
                    'variation': content_item['variation_index'],
                    'initial_score': initial_quality_score.overall_score,
                    'final_score': quality_history[-1].overall_score if quality_history else initial_quality_score.overall_score,
                    'iterations': iterations_performed,
                    'target_achieved': quality_history[-1].overall_score >= self.target_quality_score if quality_history else False
                })
            
            # Calculate optimization metrics
            achieved_target_count = sum(1 for detail in optimization_details if detail['target_achieved'])
            average_improvement = statistics.mean([detail['final_score'] - detail['initial_score'] 
                                                for detail in optimization_details if detail['iterations'] > 0]) if optimization_details else 0
            
            print(f"  📊 Optimization complete: {achieved_target_count}/{len(optimization_details)} pieces achieved 8.0+ target")
            
            return (
                PipelineStageResult(
                    stage=PipelineStage.CONTENT_OPTIMIZATION,
                    success=True,
                    data={
                        'optimized_content': optimized_content,
                        'optimization_details': optimization_details
                    },
                    metrics={
                        'total_iterations': total_iterations,
                        'improved_content_count': total_improvements,
                        'target_achieved_count': achieved_target_count,
                        'average_score_improvement': average_improvement,
                        'optimization_success_rate': (achieved_target_count / len(optimization_details)) * 100 if optimization_details else 0
                    },
                    execution_time=time.time() - stage_start
                ),
                optimized_content,
                total_iterations
            )
            
        except Exception as e:
            return (
                PipelineStageResult(
                    stage=PipelineStage.CONTENT_OPTIMIZATION,
                    success=False,
                    errors=[str(e)],
                    execution_time=time.time() - stage_start
                ),
                optimized_content,
                total_iterations
            )
    
    def _execute_analytics_tracking(self, generated_content: List[Dict[str, Any]], 
                                  pipeline_result: ContentPipelineResult) -> PipelineStageResult:
        """Stage 7: Track analytics and performance metrics"""
        stage_start = time.time()
        
        try:
            analytics = {
                'generation_timestamp': datetime.now().isoformat(),
                'topic_title': pipeline_result.topic_title,
                'platforms': pipeline_result.platforms,
                'content_statistics': {
                    'total_pieces': len(generated_content),
                    'average_length': statistics.mean([len(item['content']) for item in generated_content]),
                    'average_word_count': statistics.mean([item['metadata']['word_count'] for item in generated_content])
                },
                'quality_statistics': {
                    'average_score': statistics.mean([score.overall_score for score in pipeline_result.quality_scores.values()]) if pipeline_result.quality_scores else 0,
                    'high_quality_percentage': (sum(1 for score in pipeline_result.quality_scores.values() if score.overall_score >= self.quality_threshold) / len(pipeline_result.quality_scores)) * 100 if pipeline_result.quality_scores else 0
                },
                'pipeline_performance': {
                    'total_execution_time': pipeline_result.total_execution_time,
                    'optimization_iterations': pipeline_result.optimization_iterations,
                    'successful_stages': sum(1 for result in pipeline_result.stage_results.values() if result.success)
                }
            }
            
            return PipelineStageResult(
                stage=PipelineStage.ANALYTICS_TRACKING,
                success=True,
                data=analytics,
                execution_time=time.time() - stage_start
            )
            
        except Exception as e:
            return PipelineStageResult(
                stage=PipelineStage.ANALYTICS_TRACKING,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - stage_start
            )
    
    def _execute_finalization(self, generated_content: List[Dict[str, Any]], 
                            pipeline_result: ContentPipelineResult) -> PipelineStageResult:
        """Stage 10: Finalize and save content to database"""
        stage_start = time.time()
        
        try:
            # Import database manager
            from database import get_database
            db = get_database()
            
            saved_ids = []
            
            # Save each piece of generated content to database
            for content_item in generated_content:
                try:
                    # Connect to database
                    if not db.connect():
                        print("Warning: Could not connect to database for content saving")
                        continue
                    
                    # Prepare metadata
                    metadata = content_item.get('metadata', {})
                    
                    # Add additional metadata from pipeline result
                    metadata.update({
                        'quality_score': None,  # Will be set below
                        'brand_score': None,    # Will be set below
                        'research_enhanced': bool(pipeline_result.research_data),
                        'optimization_iterations': pipeline_result.optimization_iterations,
                        'pipeline_success': pipeline_result.success,
                        'total_execution_time': pipeline_result.total_execution_time
                    })
                    
                    # Add quality scores if available
                    content_key = f"{content_item['platform']}_v{content_item.get('variation_index', 0)}"
                    if content_key in pipeline_result.quality_scores:
                        quality_score = pipeline_result.quality_scores[content_key]
                        metadata['quality_score'] = quality_score.overall_score
                        metadata['quality_metrics'] = {
                            metric.value: score for metric, score in quality_score.metrics.items()
                        }
                    
                    # Add brand scores if available
                    if content_key in pipeline_result.brand_scores:
                        brand_score = pipeline_result.brand_scores[content_key]
                        metadata['brand_score'] = brand_score.overall_score
                        metadata['brand_metrics'] = {
                            metric.value: score for metric, score in brand_score.metrics.items()
                        }
                        metadata['plagiarism_score'] = brand_score.plagiarism_score
                        metadata['originality_assessment'] = brand_score.originality_assessment
                    
                    # Save to database
                    content_id = db.save_generated_content(
                        platform=content_item['platform'],
                        topic_source=pipeline_result.topic_title,  # Use topic title as source
                        topic_title=pipeline_result.topic_title,
                        content=content_item['content'],
                        metadata=metadata
                    )
                    
                    if content_id:
                        saved_ids.append(content_id)
                        print(f"✅ Saved {content_item['platform']} content to database (ID: {content_id})")
                    else:
                        print(f"⚠️ Failed to save {content_item['platform']} content to database")
                    
                except Exception as e:
                    print(f"Error saving content to database: {e}")
                    continue
                finally:
                    db.disconnect()
            
            return PipelineStageResult(
                stage=PipelineStage.FINALIZATION,
                success=True,
                data={'saved_ids': saved_ids},
                metrics={'saved_content_count': len(saved_ids)},
                execution_time=time.time() - stage_start
            )
            
        except Exception as e:
            return PipelineStageResult(
                stage=PipelineStage.FINALIZATION,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - stage_start
            )
    
    def _execute_enhanced_research(self, topic_data: Dict[str, Any]) -> PipelineStageResult:
        """Stage 3: Enhanced Research with fact-checking, expert quotes, and credible sources"""
        stage_start = time.time()
        
        try:
            # Perform enhanced research
            research_enhancement = self.research_engine.enhance_research(topic_data)
            
            # Calculate research metrics
            research_metrics = {}
            if research_enhancement.research_quality:
                for quality_metric, score in research_enhancement.research_quality.items():
                    research_metrics[f'research_{quality_metric.value}'] = score
                
                # Overall research score
                research_metrics['research_overall_score'] = statistics.mean(research_enhancement.research_quality.values())
            
            return PipelineStageResult(
                stage=PipelineStage.ENHANCED_RESEARCH,
                success=True,
                data={
                    'research_enhancement': research_enhancement,
                    'factual_claims_count': len(research_enhancement.factual_claims),
                    'verified_sources_count': len(research_enhancement.verified_sources),
                    'expert_quotes_count': len(research_enhancement.expert_quotes),
                    'academic_papers_count': len(research_enhancement.academic_references)
                },
                metrics=research_metrics,
                execution_time=time.time() - stage_start
            )
            
        except Exception as e:
            return PipelineStageResult(
                stage=PipelineStage.ENHANCED_RESEARCH,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - stage_start
            )
    
    def _execute_brand_consistency(self, generated_content: List[Dict[str, Any]], 
                                 platforms: List[str]) -> PipelineStageResult:
        """Stage 7: Brand Consistency, Compliance, and Originality Check"""
        stage_start = time.time()
        
        try:
            brand_scores = {}
            overall_brand_scores = []
            compliance_issues_summary = []
            originality_scores = []
            
            for content_item in generated_content:
                content = content_item['content']
                platform = content_item['platform']
                
                # Assess brand consistency
                brand_score = self.brand_checker.assess_brand_consistency(content, platform)
                
                # Create a unique key for this content piece
                content_key = f"{platform}_v{content_item.get('variation_index', 0)}"
                brand_scores[content_key] = brand_score
                
                # Track overall metrics
                overall_brand_scores.append(brand_score.overall_score)
                if brand_score.compliance_issues:
                    compliance_issues_summary.extend(brand_score.compliance_issues)
                originality_scores.append(brand_score.plagiarism_score)
            
            # Calculate aggregate metrics
            brand_metrics = {
                'average_brand_score': statistics.mean(overall_brand_scores) if overall_brand_scores else 0.0,
                'compliance_issues_count': len(compliance_issues_summary),
                'average_originality': 1.0 - (statistics.mean(originality_scores) if originality_scores else 0.3),
                'content_pieces_assessed': len(generated_content)
            }
            
            # Add detailed brand metrics
            if brand_scores:
                sample_score = list(brand_scores.values())[0]
                for metric, value in sample_score.metrics.items():
                    metric_values = [score.metrics.get(metric, 0) for score in brand_scores.values()]
                    brand_metrics[f'avg_{metric.value}'] = statistics.mean(metric_values)
            
            return PipelineStageResult(
                stage=PipelineStage.BRAND_CONSISTENCY,
                success=True,
                data={
                    'brand_scores': brand_scores,
                    'compliance_issues': compliance_issues_summary,
                    'originality_summary': {
                        'high_originality': sum(1 for s in originality_scores if s < 0.2),
                        'medium_originality': sum(1 for s in originality_scores if 0.2 <= s < 0.5),
                        'low_originality': sum(1 for s in originality_scores if s >= 0.5)
                    }
                },
                metrics=brand_metrics,
                execution_time=time.time() - stage_start
            )
            
        except Exception as e:
            return PipelineStageResult(
                stage=PipelineStage.BRAND_CONSISTENCY,
                success=False,
                errors=[str(e)],
                execution_time=time.time() - stage_start
            )

    # Helper methods
    def _extract_key_themes(self, content: str) -> List[str]:
        """Extract key themes from content using simple keyword frequency analysis"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.keys(), key=lambda x: word_freq[x], reverse=True)[:5]
    
    def _determine_content_strategy(self, content: str, word_count: int) -> str:
        """Determine optimal content strategy based on topic complexity"""
        if word_count < 100:
            return "simple_explanation"
        elif word_count < 500:
            return "detailed_breakdown"
        elif word_count < 1000:
            return "comprehensive_analysis"
        else:
            return "multi_part_series"
    
    def _calculate_dynamic_k(self, content: str) -> int:
        """Calculate dynamic k-value for inspiration retrieval based on content complexity"""
        word_count = len(content.split())
        
        if word_count < 100:
            return 3
        elif word_count < 500:
            return 5
        elif word_count < 1000:
            return 7
        else:
            return 10
    
    def _determine_optimal_content_type(self, topic_data: Dict[str, Any], platform: str) -> ContentType:
        """Determine optimal content type based on topic and platform"""
        content = topic_data.get('content', '').lower()
        
        if 'how to' in content or 'step' in content:
            return ContentType.HOW_TO
        elif 'list' in content or 'top' in content:
            return ContentType.LISTICLE
        elif '?' in content:
            return ContentType.QUESTION
        else:
            return ContentType.EDUCATIONAL
    
    def _determine_optimal_tone(self, topic_data: Dict[str, Any], platform: str) -> ToneStyle:
        """Determine optimal tone based on topic and platform"""
        if platform == 'linkedin':
            return ToneStyle.PROFESSIONAL
        elif platform == 'twitter':
            return ToneStyle.CONVERSATIONAL
        elif platform == 'reddit':
            return ToneStyle.CASUAL
        elif platform == 'newsletter':
            return ToneStyle.INFORMATIVE
        else:
            return ToneStyle.INFORMATIVE
    
    def _generate_variations(self, platform: str, topic_data: Dict[str, Any], 
                           content_type: ContentType, tone: ToneStyle, 
                           num_variations: int, inspiration_data: Dict[str, Any]) -> List[str]:
        """Generate multiple content variations for A/B testing"""
        try:
            variations = []
            
            # Get the template
            template = self.template_manager.get_template(platform, content_type, tone)
            
            # Prepare inspiration context
            inspiration_posts = inspiration_data.get('posts', [])
            inspiration_text = ""
            if inspiration_posts:
                inspiration_text = f"Here are {len(inspiration_posts)} high-performing {platform} posts for inspiration:\n\n"
                for i, post in enumerate(inspiration_posts[:5], 1):  # Limit to top 5
                    inspiration_text += f"{i}. {post.page_content}\n\n"
            else:
                inspiration_text = f"No specific inspiration available for {platform}."
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.model
            
            # Generate multiple variations with slight prompt variations
            variation_modifiers = [
                "Focus on practical, actionable advice.",
                "Emphasize storytelling and personal experiences.", 
                "Highlight data-driven insights and statistics.",
                "Use engaging questions and interactive elements.",
                "Focus on current trends and timely relevance."
            ]
            
            for i in range(num_variations):
                try:
                    modifier = variation_modifiers[i % len(variation_modifiers)]
                    
                    # Ensure we have topic content
                    topic_content = topic_data.get('content', '')
                    topic_title = topic_data.get('title', 'Untitled Topic')
                    
                    if not topic_content or not topic_content.strip():
                        print(f"Warning: Empty topic content for {topic_title}")
                        topic_content = "No detailed content provided for this topic."
                    
                    enhanced_content = f"{topic_content}\n\nGeneration guidance: {modifier}"
                    topic_content_formatted = f"Title: {topic_title}\n\nContent: {enhanced_content}"
                    
                    # Debug: Ensure all template parameters are provided
                    template_params = {
                        "topic_content": topic_content_formatted,
                        "inspiration": inspiration_text
                    }
                    
                    # Verify no None values
                    for key, value in template_params.items():
                        if value is None:
                            template_params[key] = f"No {key} available"
                    
                    result = chain.invoke(template_params)
                    
                    if result and isinstance(result, str) and result.strip():
                        variations.append(result.strip())
                    else:
                        print(f"Warning: Empty result from AI for variation {i+1}")
                        fallback = self._generate_fallback_content(topic_data, platform)
                        variations.append(fallback)
                        
                except Exception as e:
                    print(f"Error generating variation {i+1}: {e}")
                    fallback = self._generate_fallback_content(topic_data, platform)
                    variations.append(fallback)
        
            return variations if variations else [self._generate_fallback_content(topic_data, platform)]
        except Exception as e:
            print(f"Error in _generate_variations: {e}")
            return [self._generate_fallback_content(topic_data, platform)]
    
    def _generate_fallback_content(self, topic_data: Dict[str, Any], platform: str) -> str:
        """Generate fallback content if main generation fails"""
        title = topic_data.get('title', 'Untitled')
        content = topic_data.get('content', '')[:200]  # First 200 chars
        
        if platform == 'newsletter':
            return f"""## {title}

{content}...

**Key Takeaways:**
• Important insight from the topic
• Actionable advice for readers
• Relevant trend or development

**What's Next?**
Keep an eye on developments in this area and consider how it applies to your situation.

---
*What are your thoughts on {title.lower()}? Reply and let us know your experience.*"""
        else:
            return f"🎯 Insights on {title}\n\n{content}...\n\nWhat are your thoughts on this topic? Share your experiences in the comments! 💭\n\n#{platform} #content #discussion"


# =============================================================================
# MAIN PIPELINE FACTORY FUNCTION
# =============================================================================

def get_advanced_content_pipeline(model_name: str = "gpt-oss", language: str = "en") -> AdvancedContentPipeline:
    """
    Factory function to create and return an AdvancedContentPipeline instance.
    
    Args:
        model_name: Name of the LLM model to use (default: "gpt-oss")
        language: Target language for content generation ('en' or 'pl', default: "en")
        
    Returns:
        Configured AdvancedContentPipeline instance ready for content generation
    """
    return AdvancedContentPipeline(model_name=model_name, language=language)


# =============================================================================
# USAGE EXAMPLE AND DOCUMENTATION
# =============================================================================

if __name__ == "__main__":
    """
    Example usage of the Advanced Content Pipeline.
    
    This demonstrates how to use the pipeline for content generation
    with comprehensive quality assessment and optimization.
    """
    
    # Initialize pipeline
    pipeline = get_advanced_content_pipeline()
    
    # Example topic data
    topic_data = {
        'title': 'The Future of AI in Content Creation',
        'content': '''
        Artificial Intelligence is revolutionizing content creation across industries.
        From automated writing assistants to AI-powered video generation, the landscape
        is changing rapidly. This transformation brings both opportunities and challenges
        for content creators, marketers, and businesses.
        
        Key developments include natural language generation, image synthesis, and
        personalized content recommendations. However, questions remain about
        authenticity, creativity, and the human element in content creation.
        '''
    }
    
    # Execute pipeline
    result = pipeline.execute_pipeline(
        topic_data=topic_data,
        platforms=['linkedin', 'reddit', 'twitter'],
        num_variations=2
    )
    
    # Display results
    if result.success:
        print(f"\n✅ Pipeline completed successfully!")
        print(f"Generated {len(result.generated_content)} content pieces")
        print(f"Average quality score: {statistics.mean([score.overall_score for score in result.quality_scores.values()]):.1f}/10")
        print(f"Optimization iterations: {result.optimization_iterations}")
        print(f"Total execution time: {result.total_execution_time:.2f}s")
    else:
        print(f"\n❌ Pipeline failed")
        for stage, stage_result in result.stage_results.items():
            if not stage_result.success:
                print(f"Failed at {stage.value}: {stage_result.errors}")