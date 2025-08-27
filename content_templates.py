"""
Content Templates System for Social Media Agent

Provides reusable templates for different content types and platforms
with customizable tone and style options.
"""

from typing import Dict, List, Optional, Any
from enum import Enum


class ContentType(Enum):
    """Content type categories"""
    HOW_TO = "how_to"
    OPINION = "opinion" 
    NEWS = "news"
    LISTICLE = "listicle"
    STORY = "story"
    QUESTION = "question"
    ANNOUNCEMENT = "announcement"
    EDUCATIONAL = "educational"


class ToneStyle(Enum):
    """Available tone styles"""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    AUTHORITATIVE = "authoritative"
    CONVERSATIONAL = "conversational"
    ENTHUSIASTIC = "enthusiastic"
    INFORMATIVE = "informative"
    INSPIRATIONAL = "inspirational"
    HUMOROUS = "humorous"


class ContentTemplateManager:
    """Manages content templates for different platforms and styles"""
    
    def __init__(self):
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
                ContentType.LISTICLE.value: """
You are creating a LinkedIn listicle post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a LinkedIn listicle that:
1. Has a compelling headline with a specific number (e.g., "5 Key Strategies...")
2. Provides valuable, actionable items in a numbered or bulleted list
3. Includes brief explanations or examples for each point
4. Uses engaging subheadings or emojis for visual appeal
5. Ends with a summary or call-to-action
6. Maintains a {tone} tone throughout
7. Is optimized for LinkedIn sharing and comments

{tone_guidance}
""",
                ContentType.EDUCATIONAL.value: """
You are creating a LinkedIn educational post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a LinkedIn educational post that:
1. Starts with a knowledge gap or common misconception
2. Provides clear, accurate information with credible sources
3. Uses analogies or simple examples to explain complex concepts
4. Includes actionable takeaways readers can implement
5. Encourages further learning or discussion
6. Maintains a {tone} tone while being authoritative
7. Is structured for easy comprehension and engagement

{tone_guidance}
"""
            },
            "reddit": {
                ContentType.HOW_TO.value: """
You are creating a Reddit how-to post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Reddit how-to post that:
1. Has a clear, descriptive title that fits Reddit culture
2. Provides detailed, step-by-step instructions
3. Uses Reddit-appropriate formatting (headers, bullet points, numbered lists)
4. Includes potential pitfalls or "pro tips" sections
5. Encourages community discussion and additional tips
6. Maintains a {tone} tone that fits the subreddit culture
7. Avoids promotional language - focus on genuine value

{tone_guidance}
""",
                ContentType.STORY.value: """
You are creating a Reddit story post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Reddit story post that:
1. Has an engaging title that doesn't give everything away
2. Tells a compelling narrative with clear beginning, middle, and end
3. Uses authentic Reddit voice and terminology
4. Includes relevant details that make the story believable
5. Connects the story to broader lessons or insights
6. Maintains a {tone} tone while being genuine and relatable
7. Encourages community sharing of similar experiences

{tone_guidance}
""",
                ContentType.QUESTION.value: """
You are creating a Reddit discussion question with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Reddit discussion post that:
1. Poses a thought-provoking question in the title
2. Provides context and background information
3. Shares your own perspective or experience with the topic
4. Uses open-ended questions to encourage diverse responses
5. Shows genuine curiosity and openness to different viewpoints
6. Maintains a {tone} tone that invites participation
7. Follows subreddit-specific discussion guidelines

{tone_guidance}
""",
                ContentType.EDUCATIONAL.value: """
You are creating a Reddit educational post with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Reddit educational post that:
1. Has an informative title that clearly states what will be learned
2. Breaks down complex information into digestible sections
3. Uses Reddit formatting for easy reading (headers, lists, emphasis)
4. Includes sources, links, or references when appropriate
5. Anticipates and answers common questions
6. Maintains a {tone} tone while being informative
7. Invites community discussion and additional insights

{tone_guidance}
"""
            },
            "twitter": {
                ContentType.HOW_TO.value: """
You are creating a Twitter how-to thread with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Twitter thread that:
1. Starts with a compelling hook in the first tweet
2. Breaks down the process into tweetable steps (each under 280 chars)
3. Numbers the thread clearly (1/n, 2/n, etc.)
4. Uses appropriate emojis and formatting for Twitter
5. Includes actionable tips in each tweet
6. Maintains a {tone} tone throughout the thread
7. Ends with a call-to-action or summary tweet

{tone_guidance}
""",
                ContentType.OPINION.value: """
You are creating a Twitter opinion thread with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Twitter opinion thread that:
1. Opens with a bold, attention-grabbing statement
2. Presents your viewpoint across multiple connected tweets
3. Uses Twitter-appropriate formatting and hashtags
4. Backs up opinions with examples or evidence
5. Maintains engagement across the entire thread
6. Uses a {tone} tone while being respectful
7. Invites discussion and retweets

{tone_guidance}
""",
                ContentType.LISTICLE.value: """
You are creating a Twitter listicle thread with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Twitter listicle thread that:
1. Starts with a numbered promise (e.g., "5 ways to...")
2. Dedicates one tweet to each main point
3. Uses visual elements like emojis or bullet points
4. Keeps each point concise but valuable
5. Builds momentum throughout the thread
6. Maintains a {tone} tone that matches Twitter culture
7. Ends with a strong conclusion or call-to-action

{tone_guidance}
""",
                ContentType.NEWS.value: """
You are creating a Twitter news thread with a {tone} tone.

Topic Information:
{topic_content}

Inspiration Content:
{inspiration}

Create a Twitter news thread that:
1. Leads with the most important information
2. Breaks down complex news into digestible tweets
3. Provides context and background information
4. Uses clear, journalistic language
5. Includes relevant hashtags and mentions
6. Maintains a {tone} tone while being informative
7. Encourages informed discussion

{tone_guidance}
"""
            }
        }
    
    def _initialize_tone_modifiers(self) -> Dict[str, str]:
        """Initialize tone-specific guidance for content generation"""
        return {
            ToneStyle.PROFESSIONAL.value: """
Tone Guidance: Maintain a professional, polished tone throughout. Use industry terminology appropriately, avoid slang, and focus on credibility and expertise. Structure content logically and use formal language while remaining accessible.
""",
            ToneStyle.CASUAL.value: """
Tone Guidance: Use a relaxed, friendly tone as if talking to a colleague. Include conversational language, contractions, and relatable examples. Keep it approachable while maintaining credibility.
""",
            ToneStyle.AUTHORITATIVE.value: """
Tone Guidance: Demonstrate expertise and confidence in your subject matter. Use definitive statements backed by evidence. Position yourself as a thought leader while remaining humble and fact-based.
""",
            ToneStyle.CONVERSATIONAL.value: """
Tone Guidance: Write as if you're having a one-on-one conversation with the reader. Ask questions, use "you" and "I" pronouns, and create a sense of dialogue. Be warm and engaging.
""",
            ToneStyle.ENTHUSIASTIC.value: """
Tone Guidance: Show excitement and passion for the topic. Use energetic language, exclamation points (sparingly), and convey genuine enthusiasm. Let your interest in the subject shine through.
""",
            ToneStyle.INFORMATIVE.value: """
Tone Guidance: Focus on providing clear, accurate information. Use neutral language, present facts objectively, and ensure content is educational and helpful. Prioritize clarity and comprehension.
""",
            ToneStyle.INSPIRATIONAL.value: """
Tone Guidance: Motivate and inspire the reader to take action or think differently. Use uplifting language, share success stories, and focus on possibilities and positive outcomes.
""",
            ToneStyle.HUMOROUS.value: """
Tone Guidance: Incorporate appropriate humor while maintaining professionalism. Use wit, clever observations, or light humor that enhances rather than detracts from the message. Keep it tasteful and relevant.
"""
        }
    
    def get_template(self, platform: str, content_type: ContentType, tone: ToneStyle = ToneStyle.PROFESSIONAL) -> str:
        """Get a specific template with tone modifications"""
        if platform not in self.templates:
            raise ValueError(f"Platform {platform} not supported")
        
        if content_type.value not in self.templates[platform]:
            # Fall back to educational template if specific type not found
            content_type = ContentType.EDUCATIONAL
        
        template = self.templates[platform][content_type.value]
        tone_guidance = self.tone_modifiers.get(tone.value, self.tone_modifiers[ToneStyle.PROFESSIONAL.value])
        
        return template.format(
            tone=tone.value,
            tone_guidance=tone_guidance,
            topic_content="{topic_content}",
            inspiration="{inspiration}"
        )
    
    def get_available_content_types(self, platform: str) -> List[ContentType]:
        """Get available content types for a platform"""
        if platform not in self.templates:
            return []
        
        available_types = []
        for content_type in ContentType:
            if content_type.value in self.templates[platform]:
                available_types.append(content_type)
        
        return available_types
    
    def get_all_tone_styles(self) -> List[ToneStyle]:
        """Get all available tone styles"""
        return list(ToneStyle)
    
    def add_custom_template(self, platform: str, content_type: str, template: str):
        """Add a custom template for a platform and content type"""
        if platform not in self.templates:
            self.templates[platform] = {}
        
        self.templates[platform][content_type] = template
    
    def get_template_preview(self, platform: str, content_type: ContentType, tone: ToneStyle) -> str:
        """Get a preview of what the template looks like"""
        template = self.get_template(platform, content_type, tone)
        
        # Create a sample preview
        sample_preview = template.format(
            topic_content="[Your topic content will appear here]",
            inspiration="[Relevant inspiration content will be inserted here]"
        )
        
        return sample_preview


# Global template manager instance
template_manager = ContentTemplateManager()

def get_template_manager():
    """Get the global template manager instance"""
    return template_manager