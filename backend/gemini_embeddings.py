import os
from typing import List
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEmbeddings:
    """Embedding service using Google's embedding models and sentence-transformers as fallback"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            try:
                # Try to use Gemini's embedding model
                self.use_gemini = True
                logger.info("Using Gemini embedding model")
            except Exception as e:
                logger.warning(f"Gemini embedding not available, falling back to sentence-transformers: {e}")
                self.use_gemini = False
        else:
            logger.warning("No Gemini API key found, using sentence-transformers")
            self.use_gemini = False
        
        # Fallback to sentence-transformers (free, local model)
        if not self.use_gemini:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Using sentence-transformers model: all-MiniLM-L6-v2")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if self.use_gemini:
            return self._embed_with_gemini(texts)
        else:
            return self._embed_with_sentence_transformers(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        if self.use_gemini:
            result = self._embed_with_gemini([text])
            return result[0] if result else []
        else:
            return self._embed_with_sentence_transformers([text])[0]
    
    def _embed_with_gemini(self, texts: List[str]) -> List[List[float]]:
        """Use Gemini's latest embedding model with fallback strategy"""
        # Try experimental model first (best performance)
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="models/text-embedding-004",  # Latest experimental model
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            logger.info("Using latest experimental embedding model")
            return embeddings
        except Exception as e:
            logger.warning(f"Experimental embedding model failed: {e}")
            
        # Fallback to stable model
        try:
            embeddings = []
            for text in texts:
                result = genai.embed_content(
                    model="text-embedding-004",  # Stable model
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            logger.info("Using stable text-embedding-004 model")
            return embeddings
        except Exception as e2:
            logger.error(f"All Gemini embedding attempts failed: {e2}")
            # Fallback to sentence transformers
            logger.info("Falling back to sentence-transformers")
            self.use_gemini = False
            return self._embed_with_sentence_transformers(texts)
    
    def _embed_with_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Use sentence-transformers as fallback"""
        try:
            embeddings = self.model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Sentence transformers embedding failed: {e}")
            # Return zero embeddings as last resort
            return [[0.0] * 384 for _ in texts]  # 384 is the dimension for all-MiniLM-L6-v2
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension"""
        if self.use_gemini:
            return 768  # text-embedding-004 dimension (stable model)
        else:
            return 384  # all-MiniLM-L6-v2 dimension