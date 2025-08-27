from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from inspiration_loader import InspirationLoader
from enhanced_rag import get_enhanced_rag_system, ContentCategorizer, ContentCategory
import os
import hashlib
import json
from pathlib import Path
from typing import Optional

# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Connect to Qdrant (assuming it's running via Docker Compose)
qdrant_client = QdrantClient(host="localhost", port=6333)

# Collection name for inspiration content
COLLECTION_NAME = "inspiration_content"

# Path to store the data directory hash
DATA_HASH_FILE = ".data_hash.json"
DATA_DIR = "data"

def get_data_directory_hash() -> str:
    """Calculate hash of all files in the data directory"""
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        return ""
    
    hash_md5 = hashlib.md5()
    for file_path in sorted(data_path.rglob("*")):
        if file_path.is_file():
            # Include file path and modification time
            hash_md5.update(str(file_path).encode())
            hash_md5.update(str(file_path.stat().st_mtime).encode())
            # Include file content
            with open(file_path, 'rb') as f:
                hash_md5.update(f.read())
    
    return hash_md5.hexdigest()

def should_reindex_data() -> bool:
    """Check if data directory has changed since last indexing"""
    current_hash = get_data_directory_hash()
    
    # Check if hash file exists
    hash_file_path = Path(DATA_HASH_FILE)
    if not hash_file_path.exists():
        return True
    
    try:
        with open(hash_file_path, 'r') as f:
            stored_data = json.load(f)
            stored_hash = stored_data.get('hash', '')
    except (json.JSONDecodeError, FileNotFoundError):
        return True
    
    return current_hash != stored_hash

def save_data_hash():
    """Save the current data directory hash"""
    current_hash = get_data_directory_hash()
    with open(DATA_HASH_FILE, 'w') as f:
        json.dump({'hash': current_hash, 'timestamp': os.path.getmtime(DATA_DIR) if os.path.exists(DATA_DIR) else 0}, f)

def initialize_qdrant():
    """Initialize Qdrant collection with inspiration data"""
    try:
        # Check if collection exists
        collections = qdrant_client.get_collections().collections
        collection_exists = any(col.name == COLLECTION_NAME for col in collections)
        
        # Check if we need to reindex data
        needs_reindexing = should_reindex_data()
        
        if not collection_exists or needs_reindexing:
            if not collection_exists:
                print("Creating Qdrant collection...")
                # Create collection
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)  # nomic-embed-text uses 768 dimensions
                )
            elif needs_reindexing:
                print("Data directory changed, recreating Qdrant collection...")
                # Delete existing collection and recreate
                qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
            
            # Load and index inspiration data
            loader = InspirationLoader()
            documents = loader.load_all_inspiration()
            
            if documents:
                vector_store = QdrantVectorStore.from_documents(
                    documents,
                    embeddings,
                    collection_name=COLLECTION_NAME,
                    url="http://localhost:6333"
                )
                print(f"Indexed {len(documents)} inspiration documents in Qdrant")
                # Save the current data hash
                save_data_hash()
            else:
                print("No inspiration documents found to index")
        else:
            print("Qdrant collection already exists and data unchanged")
            
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")
        print("Make sure Qdrant is running via Docker Compose")
        raise

def get_inspiration_retriever(platform: str = None, k: int = 5, use_enhanced: bool = True):
    """Get retriever for inspiration content, optionally filtered by platform"""
    try:
        if use_enhanced:
            # Use enhanced RAG system
            enhanced_rag = get_enhanced_rag_system(qdrant_client, embeddings, COLLECTION_NAME)
            
            class EnhancedRetriever:
                def __init__(self, rag_system, platform, k):
                    self.rag_system = rag_system
                    self.platform = platform
                    self.k = k
                
                def invoke(self, query: str):
                    return self.rag_system.search_inspiration(query, self.platform, k=self.k)
            
            return EnhancedRetriever(enhanced_rag, platform, k)
        else:
            # Use traditional retriever with fixed platform filtering
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                embedding=embeddings
            )
            
            search_kwargs = {"k": k}
            
            # Fixed platform filtering
            if platform:
                platform_filter = Filter(
                    must=[
                        FieldCondition(
                            key="platform",
                            match=MatchValue(value=platform)
                        )
                    ]
                )
                search_kwargs["filter"] = platform_filter
                
            return vector_store.as_retriever(search_kwargs=search_kwargs)
    
    except Exception as e:
        print(f"Error creating retriever: {e}")
        raise

def get_enhanced_inspiration_search(query: str, platform: str = None, k: int = 5, auto_categorize: bool = True):
    """Get inspiration using enhanced search with auto-categorization"""
    try:
        enhanced_rag = get_enhanced_rag_system(qdrant_client, embeddings, COLLECTION_NAME)
        
        if auto_categorize:
            return enhanced_rag.auto_categorize_and_search(query, platform, k)
        else:
            results = enhanced_rag.search_inspiration(query, platform, k=k)
            return {
                'results': results,
                'detected_categories': [],
                'primary_category': 'none',
                'search_metadata': {
                    'platform_filter': platform,
                    'results_count': len(results)
                }
            }
    except Exception as e:
        print(f"Error in enhanced inspiration search: {e}")
        return {
            'results': [],
            'detected_categories': [],
            'primary_category': 'error',
            'search_metadata': {'error': str(e)}
        }

# Initialize on import
try:
    initialize_qdrant()
    retriever = get_inspiration_retriever()
    print("Qdrant vector store initialized successfully")
    print("Enhanced RAG system ready")
except Exception as e:
    print(f"Failed to initialize Qdrant: {e}")
    retriever = None