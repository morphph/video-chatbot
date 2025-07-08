import os
import hashlib
import json
import pickle
from typing import List, Dict, Optional
import google.generativeai as genai
import logging
from chromadb.utils import embedding_functions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGeminiEmbeddings(embedding_functions.EmbeddingFunction):
    """
    Enhanced embedding service using Google's text-embedding-004 with caching and batch processing.
    Maintains consistent 768-dimensional embeddings for optimal retrieval performance.
    """
    
    def __init__(self, cache_dir: str = "embeddings_cache"):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.cache_dir = cache_dir
        self.embedding_cache: Dict[str, List[float]] = {}
        self.model_name = "models/text-embedding-004"
        self.dimension = 768  # Fixed dimension for consistency
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize Gemini
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for embedding service")
        
        genai.configure(api_key=self.gemini_api_key)
        
        # Validate API access
        try:
            # Test with a simple embedding
            test_result = genai.embed_content(
                model=self.model_name,
                content="test",
                task_type="retrieval_document"
            )
            if len(test_result['embedding']) != self.dimension:
                raise ValueError(f"Expected {self.dimension}d embeddings, got {len(test_result['embedding'])}d")
            logger.info(f"✅ Gemini {self.model_name} initialized successfully ({self.dimension}d embeddings)")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini embedding model: {e}")
        
        # Load existing cache
        self._load_cache()
    
    def _load_cache(self):
        """Load embedding cache from disk"""
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk"""
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate a hash for text to use as cache key"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with caching and batch processing"""
        if not texts:
            return []
        
        # Check cache first
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                embeddings.append(self.embedding_cache[text_hash])
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            logger.info(f"Computing embeddings for {len(uncached_texts)} new texts (cache hits: {len(texts) - len(uncached_texts)})")
            new_embeddings = self._embed_batch(uncached_texts)
            
            # Store in cache and update results
            for idx, (text, embedding) in enumerate(zip(uncached_texts, new_embeddings)):
                text_hash = self._get_text_hash(text)
                self.embedding_cache[text_hash] = embedding
                embeddings[uncached_indices[idx]] = embedding
            
            # Save cache
            self._save_cache()
        else:
            logger.info(f"All {len(texts)} embeddings retrieved from cache")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with caching"""
        result = self.embed_documents([text])
        return result[0] if result else [0.0] * self.dimension
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Required method for Chroma EmbeddingFunction interface"""
        return self.embed_documents(input)
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts using Gemini API with proper error handling"""
        embeddings = []
        
        for text in texts:
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                
                if 'embedding' not in result:
                    raise ValueError("No embedding in API response")
                
                embedding = result['embedding']
                if len(embedding) != self.dimension:
                    raise ValueError(f"Expected {self.dimension}d embedding, got {len(embedding)}d")
                
                embeddings.append(embedding)
                
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
                # Instead of zero embeddings, re-raise the error to maintain quality
                raise RuntimeError(f"Embedding generation failed: {e}")
        
        return embeddings
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self.embedding_cache = {}
        cache_file = os.path.join(self.cache_dir, "embedding_cache.pkl")
        if os.path.exists(cache_file):
            os.remove(cache_file)
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_embeddings": len(self.embedding_cache),
            "dimension": self.dimension,
            "model": self.model_name
        }

# Backward compatibility alias
GeminiEmbeddings = EnhancedGeminiEmbeddings