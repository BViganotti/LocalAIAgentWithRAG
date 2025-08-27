import json
import os
import hashlib
from typing import List, Dict, Any
from langchain_core.documents import Document


class InspirationLoader:
    """Loads and processes inspiration content from data/ directory"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.platforms = ["linkedin", "reddit", "twitter"]
    
    def load_json_inspiration(self, platform: str) -> List[Document]:
        """Load inspiration posts from JSON files"""
        json_file = os.path.join(self.data_dir, f"{platform}.json")
        
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found")
            return []
        
        documents = []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for post in data.get('posts', []):
            content = post.get('content', '')
            if content.strip():
                # Create metadata including platform-specific metrics
                metadata = {
                    'platform': platform,
                    'post_id': post.get('id', ''),
                    'word_count': post.get('word_count', 0),
                    'char_count': post.get('char_count', 0),
                    'source': 'json',
                    'content_hash': hashlib.md5(content.encode()).hexdigest()
                }
                
                document = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(document)
        
        print(f"Loaded {len(documents)} inspiration posts from {platform}.json")
        return documents
    
    def load_txt_inspiration(self, platform: str) -> List[Document]:
        """Load inspiration content from TXT files"""
        txt_file = os.path.join(self.data_dir, f"{platform}.txt")
        
        if not os.path.exists(txt_file):
            print(f"Warning: {txt_file} not found")
            return []
        
        documents = []
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into chunks (assuming posts are separated by double newlines)
        posts = content.split('\n\n')
        
        for i, post in enumerate(posts):
            if post.strip():
                metadata = {
                    'platform': platform,
                    'post_id': f"txt_{i+1}",
                    'word_count': len(post.split()),
                    'char_count': len(post),
                    'source': 'txt',
                    'content_hash': hashlib.md5(post.encode()).hexdigest()
                }
                
                document = Document(
                    page_content=post.strip(),
                    metadata=metadata
                )
                documents.append(document)
        
        print(f"Loaded {len(documents)} inspiration posts from {platform}.txt")
        return documents
    
    def load_all_inspiration(self) -> List[Document]:
        """Load all inspiration content from both JSON and TXT files"""
        all_documents = []
        
        for platform in self.platforms:
            # Load from JSON files
            json_docs = self.load_json_inspiration(platform)
            all_documents.extend(json_docs)
            
            # Load from TXT files
            txt_docs = self.load_txt_inspiration(platform)
            all_documents.extend(txt_docs)
        
        print(f"Total inspiration documents loaded: {len(all_documents)}")
        return all_documents
    
    def get_platform_specific_inspiration(self, platform: str) -> List[Document]:
        """Get inspiration content for a specific platform"""
        if platform not in self.platforms:
            raise ValueError(f"Platform {platform} not supported. Choose from: {self.platforms}")
        
        json_docs = self.load_json_inspiration(platform)
        txt_docs = self.load_txt_inspiration(platform)
        
        return json_docs + txt_docs