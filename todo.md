here are things i need you to implement:

  1. Content Quality & Personalization

  - Multi-model support: Add support for different LLMs (Claude, GPT-4, Gemini) for varied content styles BUT KEEP EVERYTHING FOR gpt-oss, i don't have api keys for anything else so the model should remain gpt-oss
  - Content variations: Generate multiple versions per platform with A/B testing capabilities
  - Tone customization: Allow users to specify brand voice, personality, or tone preferences
  - Content templates: Create reusable templates for different content types (how-to, opinion, news, etc.)

  2. Enhanced RAG System

  - Semantic search improvements: Implement hybrid search (keyword + semantic) for better inspiration matching
  - Content categorization: Auto-tag inspiration content by theme, industry, engagement type
  - Platform-specific filtering: Fix the disabled platform filtering in vector.py:128
  - Dynamic k-value: Adjust number of inspiration posts based on topic complexity