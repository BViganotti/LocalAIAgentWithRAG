#!/usr/bin/env python3
"""
Data Ingestion and Language Separation System

This module handles the ingestion of inspiration data, automatic language detection,
and organization of data into language-specific directories and Qdrant collections.
"""

import os
import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import shutil

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from language_utils import Language, LanguageDetector, LanguageManager


@dataclass
class DataIngestionResult:
    """Result of data ingestion process"""
    success: bool
    processed_files: int
    english_items: int
    polish_items: int
    skipped_items: int
    errors: List[str]
    execution_time: float


class LanguageSeparatedDataIngestion:
    """
    Handles ingestion and separation of inspiration data by language.
    
    This class processes JSON files containing social media posts, detects
    their language, and organizes them into language-specific directories
    and Qdrant collections.
    """
    
    def __init__(self, 
                 data_dir: str = "data",
                 qdrant_host: str = "localhost", 
                 qdrant_port: int = 6333):
        """
        Initialize data ingestion system.
        
        Args:
            data_dir: Base data directory path
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
        """
        self.data_dir = data_dir
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Create language-specific data directories
        self.lang_dirs = {
            Language.ENGLISH: os.path.join(data_dir, "en"),
            Language.POLISH: os.path.join(data_dir, "pl")
        }
        
        for lang_dir in self.lang_dirs.values():
            os.makedirs(lang_dir, exist_ok=True)
    
    def ingest_and_separate_data(self, force_reprocess: bool = False) -> DataIngestionResult:
        """
        Main ingestion process: detect languages and separate data.
        
        Args:
            force_reprocess: If True, reprocess all data regardless of changes
            
        Returns:
            DataIngestionResult with processing statistics
        """
        start_time = datetime.now()
        result = DataIngestionResult(
            success=False,
            processed_files=0,
            english_items=0,
            polish_items=0,
            skipped_items=0,
            errors=[],
            execution_time=0.0
        )
        
        try:
            # Check if reprocessing is needed
            if not force_reprocess and not self._should_reprocess():
                print("📋 No changes detected in data directory. Skipping ingestion.")
                result.success = True
                return result
            
            print("🔄 Starting language-separated data ingestion...")
            
            # Process each platform's data
            platforms = ["linkedin", "reddit", "twitter", "newsletter"]
            
            for platform in platforms:
                json_file = os.path.join(self.data_dir, f"{platform}.json")
                
                if not os.path.exists(json_file):
                    print(f"⚠️  File not found: {json_file}")
                    continue
                
                print(f"📂 Processing {platform}.json...")
                platform_result = self._process_platform_file(json_file, platform)
                
                result.processed_files += 1
                result.english_items += platform_result['english_count']
                result.polish_items += platform_result['polish_count']
                result.skipped_items += platform_result['skipped_count']
                result.errors.extend(platform_result['errors'])
            
            # Create/update Qdrant collections
            print("🔍 Setting up Qdrant collections...")
            self._setup_qdrant_collections()
            
            # Index data into Qdrant
            print("📊 Indexing data into Qdrant...")
            self._index_language_data()
            
            # Update hash for change detection
            self._update_data_hash()
            
            result.success = True
            
        except Exception as e:
            result.errors.append(f"Ingestion failed: {str(e)}")
            print(f"❌ Data ingestion failed: {e}")
        
        finally:
            result.execution_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _process_platform_file(self, json_file: str, platform: str) -> Dict[str, Any]:
        """
        Process a single platform JSON file and separate by language.
        
        Args:
            json_file: Path to JSON file
            platform: Platform name
            
        Returns:
            Processing statistics
        """
        result = {
            'english_count': 0,
            'polish_count': 0,
            'skipped_count': 0,
            'errors': []
        }
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Separate posts by language
            english_posts = []
            polish_posts = []
            
            posts = data.get('posts', [])
            
            for post in posts:
                content = post.get('content', '')
                
                if not content or not content.strip():
                    result['skipped_count'] += 1
                    continue
                
                # Detect language
                detected_lang = LanguageDetector.detect_language(content)
                
                # Add language metadata to post
                post_copy = post.copy()
                post_copy['language'] = detected_lang.value
                post_copy['language_detected'] = True
                
                if detected_lang == Language.ENGLISH:
                    english_posts.append(post_copy)
                    result['english_count'] += 1
                else:
                    polish_posts.append(post_copy)
                    result['polish_count'] += 1
            
            # Save separated data
            self._save_language_separated_data(platform, english_posts, polish_posts, data.get('metadata', {}))
            
        except Exception as e:
            result['errors'].append(f"Error processing {json_file}: {str(e)}")
        
        return result
    
    def _save_language_separated_data(self, platform: str, 
                                     english_posts: List[Dict], 
                                     polish_posts: List[Dict],
                                     original_metadata: Dict):
        """
        Save language-separated data to appropriate directories.
        
        Args:
            platform: Platform name
            english_posts: English posts list
            polish_posts: Polish posts list
            original_metadata: Original file metadata
        """
        # Save English data
        if english_posts:
            en_data = {
                'metadata': {
                    **original_metadata,
                    'language': 'en',
                    'total_posts': len(english_posts),
                    'separated_date': datetime.now().isoformat(),
                    'original_total_posts': original_metadata.get('total_posts', 0)
                },
                'posts': english_posts
            }
            
            en_file = os.path.join(self.lang_dirs[Language.ENGLISH], f"{platform}.json")
            with open(en_file, 'w', encoding='utf-8') as f:
                json.dump(en_data, f, indent=2, ensure_ascii=False)
            
            print(f"  📝 Saved {len(english_posts)} English posts to {en_file}")
        
        # Save Polish data
        if polish_posts:
            pl_data = {
                'metadata': {
                    **original_metadata,
                    'language': 'pl',
                    'total_posts': len(polish_posts),
                    'separated_date': datetime.now().isoformat(),
                    'original_total_posts': original_metadata.get('total_posts', 0)
                },
                'posts': polish_posts
            }
            
            pl_file = os.path.join(self.lang_dirs[Language.POLISH], f"{platform}.json")
            with open(pl_file, 'w', encoding='utf-8') as f:
                json.dump(pl_data, f, indent=2, ensure_ascii=False)
            
            print(f"  📝 Saved {len(polish_posts)} Polish posts to {pl_file}")
    
    def _setup_qdrant_collections(self):
        """Create language-specific Qdrant collections."""
        for language in [Language.ENGLISH, Language.POLISH]:
            collection_name = LanguageManager.get_collection_name(language)
            
            try:
                # Check if collection exists
                collections = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]
                
                if collection_name not in collection_names:
                    # Create collection
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                    )
                    print(f"  ✅ Created Qdrant collection: {collection_name}")
                else:
                    print(f"  ♻️  Collection already exists: {collection_name}")
                    
            except Exception as e:
                print(f"  ❌ Error setting up collection {collection_name}: {e}")
    
    def _index_language_data(self):
        """Index language-separated data into Qdrant collections."""
        for language in [Language.ENGLISH, Language.POLISH]:
            collection_name = LanguageManager.get_collection_name(language)
            lang_dir = self.lang_dirs[language]
            
            documents = []
            
            # Load all platform files for this language
            for platform in ["linkedin", "reddit", "twitter", "newsletter"]:
                platform_file = os.path.join(lang_dir, f"{platform}.json")
                
                if not os.path.exists(platform_file):
                    continue
                
                try:
                    with open(platform_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    posts = data.get('posts', [])
                    
                    for post in posts:
                        content = post.get('content', '')
                        
                        if not content.strip():
                            continue
                        
                        # Create document with comprehensive metadata
                        metadata = {
                            'platform': platform,
                            'language': language.value,
                            'post_id': post.get('id', f"{platform}_{post.get('post_number', 'unknown')}"),
                            'word_count': post.get('word_count', len(content.split())),
                            'char_count': post.get('char_count', len(content)),
                            'content_hash': hashlib.md5(content.encode()).hexdigest()
                        }
                        
                        doc = Document(
                            page_content=content,
                            metadata=metadata
                        )
                        
                        documents.append(doc)
                
                except Exception as e:
                    print(f"  ❌ Error processing {platform_file}: {e}")
            
            # Index documents into Qdrant
            if documents:
                try:
                    # Clear existing collection data
                    self.qdrant_client.delete_collection(collection_name)
                    self.qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                    )
                    
                    # Create vector store and add documents
                    vector_store = QdrantVectorStore(
                        client=self.qdrant_client,
                        collection_name=collection_name,
                        embedding=self.embeddings
                    )
                    
                    # Add documents in batches
                    batch_size = 50
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        vector_store.add_documents(batch)
                    
                    print(f"  ✅ Indexed {len(documents)} documents into {collection_name}")
                    
                except Exception as e:
                    print(f"  ❌ Error indexing {collection_name}: {e}")
    
    def _should_reprocess(self) -> bool:
        """Check if data directory has changed since last processing."""
        current_hash = self._calculate_data_hash()
        hash_file = os.path.join(self.data_dir, ".language_separation_hash.json")
        
        if not os.path.exists(hash_file):
            return True
        
        try:
            with open(hash_file, 'r') as f:
                stored_data = json.load(f)
            
            return stored_data.get('data_hash') != current_hash
        except:
            return True
    
    def _calculate_data_hash(self) -> str:
        """Calculate hash of all JSON files in data directory."""
        hasher = hashlib.md5()
        
        for platform in ["linkedin", "reddit", "twitter", "newsletter"]:
            json_file = os.path.join(self.data_dir, f"{platform}.json")
            
            if os.path.exists(json_file):
                with open(json_file, 'rb') as f:
                    hasher.update(f.read())
        
        return hasher.hexdigest()
    
    def _update_data_hash(self):
        """Update stored data hash."""
        current_hash = self._calculate_data_hash()
        hash_data = {
            'data_hash': current_hash,
            'last_processed': datetime.now().isoformat(),
            'ingestion_version': '1.0'
        }
        
        hash_file = os.path.join(self.data_dir, ".language_separation_hash.json")
        with open(hash_file, 'w') as f:
            json.dump(hash_data, f, indent=2)
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """
        Get current ingestion status and statistics.
        
        Returns:
            Dictionary with ingestion status information
        """
        status = {
            'english_collections': {},
            'polish_collections': {},
            'last_processed': None,
            'data_hash': self._calculate_data_hash()
        }
        
        # Check hash file
        hash_file = os.path.join(self.data_dir, ".language_separation_hash.json")
        if os.path.exists(hash_file):
            try:
                with open(hash_file, 'r') as f:
                    hash_data = json.load(f)
                status['last_processed'] = hash_data.get('last_processed')
            except:
                pass
        
        # Check language directories
        for language in [Language.ENGLISH, Language.POLISH]:
            lang_dir = self.lang_dirs[language]
            lang_key = f"{language.value}_collections"
            
            if os.path.exists(lang_dir):
                status[lang_key] = {}
                
                for platform in ["linkedin", "reddit", "twitter", "newsletter"]:
                    platform_file = os.path.join(lang_dir, f"{platform}.json")
                    
                    if os.path.exists(platform_file):
                        try:
                            with open(platform_file, 'r') as f:
                                data = json.load(f)
                            
                            status[lang_key][platform] = {
                                'posts_count': len(data.get('posts', [])),
                                'last_modified': os.path.getmtime(platform_file)
                            }
                        except:
                            status[lang_key][platform] = {'error': 'Failed to read file'}
        
        return status


def main():
    """Main function for running data ingestion."""
    print("🔄 Language-Separated Data Ingestion System")
    print("=" * 50)
    
    ingestion = LanguageSeparatedDataIngestion()
    
    # Show current status
    print("\n📊 Current Status:")
    status = ingestion.get_ingestion_status()
    
    if status['last_processed']:
        print(f"Last processed: {status['last_processed']}")
    else:
        print("Never processed before")
    
    # Run ingestion
    print("\n🚀 Starting ingestion process...")
    result = ingestion.ingest_and_separate_data(force_reprocess=False)
    
    # Display results
    print(f"\n📋 Ingestion Results:")
    print(f"Success: {'✅ Yes' if result.success else '❌ No'}")
    print(f"Files processed: {result.processed_files}")
    print(f"English items: {result.english_items}")
    print(f"Polish items: {result.polish_items}")
    print(f"Skipped items: {result.skipped_items}")
    print(f"Execution time: {result.execution_time:.2f}s")
    
    if result.errors:
        print(f"\n⚠️ Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  • {error}")


if __name__ == "__main__":
    main()