from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import get_inspiration_retriever, get_enhanced_inspiration_search
from database import get_database
from content_templates import get_template_manager, ContentType, ToneStyle
from enhanced_rag import ContentCategory
from typing import Dict, List, Any, Optional
import json
import random
from enum import Enum


class ModelType(Enum):
    """Available model types"""
    GPT_OSS = "gpt-oss"
    # Future models can be added here while keeping gpt-oss as default
    # CLAUDE = "claude"
    # GPT4 = "gpt-4"
    # GEMINI = "gemini"


class ContentGenerator:
    """Base class for content generation with multi-model and tone support"""
    
    def __init__(self, model_name: str = "gpt-oss"):
        self.model = OllamaLLM(model=model_name)
        self.db = get_database()
        self.template_manager = get_template_manager()
    
    def get_inspiration_context(self, platform: str, topic: str, k: int = 5, use_enhanced: bool = True) -> str:
        """Get relevant inspiration content for the topic and platform with enhanced search"""
        try:
            if use_enhanced:
                # Use enhanced search with auto-categorization
                search_result = get_enhanced_inspiration_search(topic, platform, k, auto_categorize=True)
                relevant_posts = search_result['results']
                categories = search_result['detected_categories']
                
                context = f"Here are {len(relevant_posts)} high-performing {platform} posts for inspiration"
                if categories:
                    context += f" (detected categories: {', '.join(categories)})"
                context += ":\n\n"
                
                for i, post in enumerate(relevant_posts, 1):
                    context += f"{i}. {post.page_content}\n\n"
                
                return context
            else:
                # Use traditional search
                retriever = get_inspiration_retriever(platform=platform, k=k, use_enhanced=False)
                relevant_posts = retriever.invoke(topic)
                
                context = f"Here are {len(relevant_posts)} high-performing {platform} posts for inspiration:\n\n"
                for i, post in enumerate(relevant_posts, 1):
                    context += f"{i}. {post.page_content}\n\n"
                
                return context
        except Exception as e:
            print(f"Error getting inspiration context: {e}")
            return f"No specific inspiration available for {platform}."
    
    def calculate_dynamic_k(self, topic_content: str) -> int:
        """Calculate dynamic k-value based on topic complexity"""
        word_count = len(topic_content.split())
        
        if word_count < 100:
            return 3  # Simple topics need fewer examples
        elif word_count < 500:
            return 5  # Medium topics
        elif word_count < 1000:
            return 7  # Complex topics
        else:
            return 10  # Very complex topics need more inspiration
    
    def generate_variations(self, platform: str, topic_data: Dict[str, Any], 
                          content_type: ContentType, tone: ToneStyle, 
                          num_variations: int = 3) -> List[str]:
        """Generate multiple variations of content for A/B testing"""
        variations = []
        
        # Get the template
        template = self.template_manager.get_template(platform, content_type, tone)
        
        # Calculate dynamic k-value for inspiration
        k_value = self.calculate_dynamic_k(topic_data['content'])
        inspiration = self.get_inspiration_context(platform, topic_data['content'], k_value)
        
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
            modifier = variation_modifiers[i % len(variation_modifiers)]
            enhanced_template = template + f"\n\nAdditional focus for this variation: {modifier}"
            
            enhanced_prompt = ChatPromptTemplate.from_template(enhanced_template)
            enhanced_chain = enhanced_prompt | self.model
            
            try:
                variation = enhanced_chain.invoke({
                    "topic_content": f"Title: {topic_data['title']}\n\nContent: {topic_data['content']}",
                    "inspiration": inspiration
                })
                variations.append(variation)
            except Exception as e:
                print(f"Error generating variation {i+1}: {e}")
                continue
        
        return variations


class LinkedInGenerator(ContentGenerator):
    """Generate LinkedIn articles and posts with templates and variations"""
    
    def __init__(self):
        super().__init__()
        self.platform = "linkedin"
    
    def generate_post(self, topic_data: Dict[str, Any], content_type: ContentType = ContentType.EDUCATIONAL, 
                     tone: ToneStyle = ToneStyle.PROFESSIONAL, num_variations: int = 1) -> str:
        """Generate a LinkedIn post with specified type and tone"""
        if num_variations > 1:
            variations = self.generate_variations(self.platform, topic_data, content_type, tone, num_variations)
            return variations[0] if variations else self._fallback_generation(topic_data)
        
        return self._generate_single_post(topic_data, content_type, tone)
    
    def _generate_single_post(self, topic_data: Dict[str, Any], content_type: ContentType, tone: ToneStyle) -> str:
        """Generate a single LinkedIn post"""
        template = self.template_manager.get_template(self.platform, content_type, tone)
        k_value = self.calculate_dynamic_k(topic_data['content'])
        inspiration = self.get_inspiration_context(self.platform, topic_data['content'], k_value)
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model
        
        result = chain.invoke({
            "topic_content": f"Title: {topic_data['title']}\n\nContent: {topic_data['content']}",
            "inspiration": inspiration
        })
        
        return result
    
    def _fallback_generation(self, topic_data: Dict[str, Any]) -> str:
        """Fallback generation method if template fails"""
        fallback_template = """
You are an expert LinkedIn content creator who writes high-impact, professional content that drives engagement.

Topic Information:
{topic_content}

Create a LinkedIn post about the topic that provides valuable insights and encourages professional discussion.
"""
        
        prompt = ChatPromptTemplate.from_template(fallback_template)
        chain = prompt | self.model
        
        result = chain.invoke({
            "topic_content": f"Title: {topic_data['title']}\n\nContent: {topic_data['content']}"
        })
        
        return result


class RedditGenerator(ContentGenerator):
    """Generate Reddit posts and articles with templates and variations"""
    
    def __init__(self):
        super().__init__()
        self.platform = "reddit"
    
    def generate_post(self, topic_data: Dict[str, Any], content_type: ContentType = ContentType.EDUCATIONAL,
                     tone: ToneStyle = ToneStyle.CONVERSATIONAL, num_variations: int = 1) -> str:
        """Generate a Reddit post with specified type and tone"""
        if num_variations > 1:
            variations = self.generate_variations(self.platform, topic_data, content_type, tone, num_variations)
            return variations[0] if variations else self._fallback_generation(topic_data)
        
        return self._generate_single_post(topic_data, content_type, tone)
    
    def _generate_single_post(self, topic_data: Dict[str, Any], content_type: ContentType, tone: ToneStyle) -> str:
        """Generate a single Reddit post"""
        template = self.template_manager.get_template(self.platform, content_type, tone)
        k_value = self.calculate_dynamic_k(topic_data['content'])
        inspiration = self.get_inspiration_context(self.platform, topic_data['content'], k_value)
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model
        
        result = chain.invoke({
            "topic_content": f"Title: {topic_data['title']}\n\nContent: {topic_data['content']}",
            "inspiration": inspiration
        })
        
        return result
    
    def _fallback_generation(self, topic_data: Dict[str, Any]) -> str:
        """Fallback generation method if template fails"""
        fallback_template = """
You are an expert Reddit content creator who understands how to write posts that resonate with Reddit communities.

Topic Information:
{topic_content}

Create a Reddit post about the topic that provides genuine value and encourages community discussion.
"""
        
        prompt = ChatPromptTemplate.from_template(fallback_template)
        chain = prompt | self.model
        
        result = chain.invoke({
            "topic_content": f"Title: {topic_data['title']}\n\nContent: {topic_data['content']}"
        })
        
        return result


class TwitterGenerator(ContentGenerator):
    """Generate Twitter posts and threads with templates and variations"""
    
    def __init__(self):
        super().__init__()
        self.platform = "twitter"
    
    def generate_post(self, topic_data: Dict[str, Any], content_type: ContentType = ContentType.EDUCATIONAL,
                     tone: ToneStyle = ToneStyle.CONVERSATIONAL, num_variations: int = 1) -> str:
        """Generate a Twitter post with specified type and tone"""
        if num_variations > 1:
            variations = self.generate_variations(self.platform, topic_data, content_type, tone, num_variations)
            return variations[0] if variations else self._fallback_generation(topic_data)
        
        return self._generate_single_post(topic_data, content_type, tone)
    
    def _generate_single_post(self, topic_data: Dict[str, Any], content_type: ContentType, tone: ToneStyle) -> str:
        """Generate a single Twitter post"""
        template = self.template_manager.get_template(self.platform, content_type, tone)
        k_value = self.calculate_dynamic_k(topic_data['content'])
        inspiration = self.get_inspiration_context(self.platform, topic_data['content'], k_value)
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.model
        
        result = chain.invoke({
            "topic_content": f"Title: {topic_data['title']}\n\nContent: {topic_data['content']}",
            "inspiration": inspiration
        })
        
        return result
    
    def _fallback_generation(self, topic_data: Dict[str, Any]) -> str:
        """Fallback generation method if template fails"""
        fallback_template = """
You are an expert Twitter content creator who writes viral, engaging tweets and threads.

Topic Information:
{topic_content}

Create a Twitter post/thread about the topic that is engaging and optimized for Twitter.
"""
        
        prompt = ChatPromptTemplate.from_template(fallback_template)
        chain = prompt | self.model
        
        result = chain.invoke({
            "topic_content": f"Title: {topic_data['title']}\n\nContent: {topic_data['content']}"
        })
        
        return result


class ContentGenerationAgent:
    """Main agent that coordinates content generation across platforms"""
    
    def __init__(self):
        self.generators = {
            'linkedin': LinkedInGenerator(),
            'reddit': RedditGenerator(),
            'twitter': TwitterGenerator()
        }
        self.db = get_database()
    
    def generate_content_for_platform(self, platform: str, topic_data: Dict[str, Any], 
                                     content_type: ContentType = None, tone: ToneStyle = None, 
                                     num_variations: int = 1) -> Dict[str, Any]:
        """Generate content for a specific platform with customization options"""
        if platform not in self.generators:
            raise ValueError(f"Platform {platform} not supported. Choose from: {list(self.generators.keys())}")
        
        # Set defaults based on platform
        if content_type is None:
            content_type = ContentType.EDUCATIONAL
        
        if tone is None:
            platform_default_tones = {
                'linkedin': ToneStyle.PROFESSIONAL,
                'reddit': ToneStyle.CONVERSATIONAL, 
                'twitter': ToneStyle.CONVERSATIONAL
            }
            tone = platform_default_tones.get(platform, ToneStyle.PROFESSIONAL)
        
        generator = self.generators[platform]
        
        if num_variations > 1:
            # Generate multiple variations
            variations = generator.generate_variations(platform, topic_data, content_type, tone, num_variations)
            content = variations[0] if variations else generator.generate_post(topic_data, content_type, tone)
            
            # Store all variations in metadata
            metadata = {
                'word_count': len(content.split()),
                'char_count': len(content),
                'topic_file_type': topic_data.get('file_type', 'unknown'),
                'topic_word_count': topic_data.get('word_count', 0),
                'inspiration_used': True,
                'content_type': content_type.value,
                'tone': tone.value,
                'num_variations': len(variations),
                'variations': variations if len(variations) > 1 else None
            }
        else:
            # Generate single content
            content = generator.generate_post(topic_data, content_type, tone)
            
            metadata = {
                'word_count': len(content.split()),
                'char_count': len(content),
                'topic_file_type': topic_data.get('file_type', 'unknown'),
                'topic_word_count': topic_data.get('word_count', 0),
                'inspiration_used': True,
                'content_type': content_type.value,
                'tone': tone.value,
                'num_variations': 1
            }
        
        return {
            'platform': platform,
            'topic_source': topic_data['file_path'],
            'topic_title': topic_data['title'],
            'content': content,
            'metadata': metadata
        }
    
    def generate_content_for_all_platforms(self, topic_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate content for all platforms"""
        results = []
        
        for platform in self.generators.keys():
            try:
                result = self.generate_content_for_platform(platform, topic_data)
                results.append(result)
                print(f"Generated {platform} content: {len(result['content'])} characters")
            except Exception as e:
                print(f"Error generating content for {platform}: {e}")
        
        return results
    
    def save_generated_content(self, content_data: Dict[str, Any]) -> Optional[int]:
        """Save generated content to database"""
        try:
            if not self.db.connect():
                raise Exception("Could not connect to database")
            
            content_id = self.db.save_generated_content(
                platform=content_data['platform'],
                topic_source=content_data['topic_source'],
                topic_title=content_data['topic_title'],
                content=content_data['content'],
                metadata=content_data['metadata']
            )
            
            print(f"Saved {content_data['platform']} content with ID: {content_id}")
            return content_id
        except Exception as e:
            print(f"Error saving content: {e}")
            return None
        finally:
            self.db.disconnect()
    
    def generate_and_save_content(self, topic_data: Dict[str, Any], platforms: List[str] = None,
                                content_type: ContentType = None, tone: ToneStyle = None,
                                num_variations: int = 1) -> Dict[str, Any]:
        """Generate and save content for specified platforms with customization options"""
        if platforms is None:
            platforms = list(self.generators.keys())
        
        results = {
            'topic_title': topic_data['title'],
            'generated_content': [],
            'saved_ids': [],
            'errors': [],
            'generation_settings': {
                'content_type': content_type.value if content_type else 'auto',
                'tone': tone.value if tone else 'auto', 
                'num_variations': num_variations
            }
        }
        
        for platform in platforms:
            if platform not in self.generators:
                results['errors'].append(f"Platform {platform} not supported")
                continue
            
            try:
                # Generate content with customization
                content_data = self.generate_content_for_platform(
                    platform, topic_data, content_type, tone, num_variations
                )
                results['generated_content'].append(content_data)
                
                # Save to database (save primary content, variations stored in metadata)
                content_id = self.save_generated_content(content_data)
                if content_id:
                    results['saved_ids'].append({'platform': platform, 'id': content_id})
                
            except Exception as e:
                results['errors'].append(f"Error with {platform}: {str(e)}")
        
        return results
    
    def get_available_content_types(self, platform: str = None) -> List[ContentType]:
        """Get available content types for a platform or all platforms"""
        template_manager = get_template_manager()
        if platform:
            return template_manager.get_available_content_types(platform)
        else:
            # Return all content types
            return list(ContentType)
    
    def get_available_tones(self) -> List[ToneStyle]:
        """Get all available tone styles"""
        return list(ToneStyle)
    
    def preview_template(self, platform: str, content_type: ContentType, tone: ToneStyle) -> str:
        """Preview what a template looks like"""
        template_manager = get_template_manager()
        return template_manager.get_template_preview(platform, content_type, tone)