import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import hashlib
import json
import logging
import pickle
import numpy as np
from gemini_embeddings import GeminiEmbeddings

# BM25 and preprocessing imports
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    # Fallback for development
    class BM25Okapi:
        def __init__(self, corpus): pass
        def get_scores(self, query): return [0.5] * len(query)
        def get_top_n(self, query, documents, n=5): return documents[:n]

# Cross-encoder reranking imports
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    # Fallback for development
    class CrossEncoder:
        def __init__(self, model_name): pass
        def predict(self, pairs): return [0.5] * len(pairs)

import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embedding_model = GeminiEmbeddings()
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Collection for video transcripts
        self.collection_name = "video_transcripts"
        
        # BM25 indices storage (video_id -> BM25Okapi)
        self.bm25_indices: Dict[str, BM25Okapi] = {}
        self.bm25_documents: Dict[str, List[str]] = {}  # Store original documents
        self.bm25_cache_dir = os.path.join(persist_directory, "bm25_cache")
        os.makedirs(self.bm25_cache_dir, exist_ok=True)
        
        # Cross-encoder for reranking
        self.cross_encoder = None
        self._initialize_cross_encoder()
        
        logger.info("🔍 Enhanced VectorStore initialized with hybrid search + reranking capabilities")
        
    def _generate_chunk_id(self, video_id: str, chunk_index: int) -> str:
        """Generate unique ID for a chunk"""
        return f"{video_id}_chunk_{chunk_index}"
    
    def _generate_video_collection_name(self, video_id: str) -> str:
        """Generate collection name for a specific video"""
        # Chroma collection names must be 3-63 characters, start/end with alphanumeric
        safe_id = video_id.replace("-", "_")[:50]
        return f"video_{safe_id}"
    
    def create_video_index(self, video_id: str, chunks: List[str], metadata: Dict[str, Any]) -> bool:
        """Create both vector and BM25 indices for a specific video"""
        try:
            collection_name = self._generate_video_collection_name(video_id)
            
            # Create vector index (existing functionality)
            # Get or create collection
            try:
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_model
                )
            except Exception:
                # Collection already exists, delete and recreate
                self.chroma_client.delete_collection(name=collection_name)
                collection = self.chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_model
                )
            
            # Create embeddings and add to collection
            texts = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "video_id": video_id,
                    "chunk_index": i,
                    "video_title": metadata.get("title", ""),
                    "video_author": metadata.get("author", ""),
                    "total_chunks": len(chunks),
                    "timestamp_range": metadata.get("timestamp_ranges", {}).get(str(i), ""),
                    "start_time": metadata.get("start_times", {}).get(str(i), 0),
                    "end_time": metadata.get("end_times", {}).get(str(i), 60)
                }
                
                texts.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(self._generate_chunk_id(video_id, i))
            
            # Add documents to collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Create BM25 index
            self._create_bm25_index(video_id, chunks)
            
            logger.info(f"✅ Created hybrid index for video {video_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating hybrid index: {e}")
            return False
    
    def search_video_content(self, video_id: str, query: str, k: int = 5) -> List[Document]:
        """Search for relevant content in a specific video"""
        try:
            collection_name = self._generate_video_collection_name(video_id)
            
            # Get the collection
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_model
            )
            
            # Perform similarity search
            results = collection.query(
                query_texts=[query],
                n_results=k
            )
            
            # Convert to Document format
            documents = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i, doc_text in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    documents.append(Document(
                        page_content=doc_text,
                        metadata=metadata
                    ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching video content: {e}")
            return []
    
    def search_all_videos(self, query: str, video_ids: List[str], k: int = 5) -> List[Document]:
        """Search across multiple videos using semantic search"""
        all_results = []
        
        for video_id in video_ids:
            results = self.search_video_content(video_id, query, k=k)
            all_results.extend(results)
        
        # Sort by relevance (this is simplified, could use scores)
        return all_results[:k]
    
    def _create_bm25_index(self, video_id: str, chunks: List[str]) -> bool:
        """Create BM25 index for a video"""
        try:
            # Preprocess documents for BM25
            tokenized_corpus = [self._preprocess_text(chunk) for chunk in chunks]
            
            # Create BM25 index
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Store in memory
            self.bm25_indices[video_id] = bm25
            self.bm25_documents[video_id] = chunks
            
            # Persist to disk
            self._save_bm25_index(video_id, bm25, chunks)
            
            logger.info(f"🔍 Created BM25 index for video {video_id} with {len(chunks)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error creating BM25 index for {video_id}: {e}")
            return False
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 tokenization"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        # Split into tokens and remove empty strings
        tokens = [token for token in text.split() if token and len(token) > 2]
        return tokens
    
    def _save_bm25_index(self, video_id: str, bm25: BM25Okapi, chunks: List[str]):
        """Save BM25 index to disk"""
        try:
            cache_file = os.path.join(self.bm25_cache_dir, f"{video_id}_bm25.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'bm25': bm25,
                    'documents': chunks,
                    'video_id': video_id
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save BM25 index for {video_id}: {e}")
    
    def _load_bm25_index(self, video_id: str) -> bool:
        """Load BM25 index from disk"""
        try:
            cache_file = os.path.join(self.bm25_cache_dir, f"{video_id}_bm25.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                self.bm25_indices[video_id] = data['bm25']
                self.bm25_documents[video_id] = data['documents']
                return True
        except Exception as e:
            logger.warning(f"Failed to load BM25 index for {video_id}: {e}")
        return False
    
    def search_video_bm25(self, video_id: str, query: str, k: int = 5) -> List[Document]:
        """Search video content using BM25 keyword search"""
        try:
            # Load BM25 index if not in memory
            if video_id not in self.bm25_indices:
                if not self._load_bm25_index(video_id):
                    logger.warning(f"No BM25 index found for video {video_id}")
                    return []
            
            bm25 = self.bm25_indices[video_id]
            documents = self.bm25_documents[video_id]
            
            # Preprocess query
            tokenized_query = self._preprocess_text(query)
            
            # Get BM25 scores
            scores = bm25.get_scores(tokenized_query)
            
            # Get top-k results
            top_indices = np.argsort(scores)[::-1][:k]
            
            # Convert to Document format
            results = []
            for idx in top_indices:
                if scores[idx] > 0:  # Only include relevant results
                    doc = Document(
                        page_content=documents[idx],
                        metadata={
                            "video_id": video_id,
                            "chunk_index": idx,
                            "bm25_score": float(scores[idx]),
                            "search_type": "bm25"
                        }
                    )
                    results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in BM25 search for video {video_id}: {e}")
            return []
    
    def search_video_hybrid(
        self, 
        video_id: str, 
        query: str, 
        k: int = 5, 
        semantic_weight: float = 0.7, 
        bm25_weight: float = 0.3
    ) -> List[Document]:
        """Search video content using hybrid semantic + BM25 approach"""
        try:
            # Get semantic results
            semantic_results = self.search_video_content(video_id, query, k=k*2)
            
            # Get BM25 results
            bm25_results = self.search_video_bm25(video_id, query, k=k*2)
            
            # Create a combined scoring system
            chunk_scores = {}
            
            # Process semantic results
            for i, doc in enumerate(semantic_results):
                chunk_idx = doc.metadata.get("chunk_index", i)
                # Higher rank = lower score, so invert
                semantic_score = 1.0 / (i + 1)
                chunk_scores[chunk_idx] = {
                    "doc": doc,
                    "semantic_score": semantic_score,
                    "bm25_score": 0.0
                }
            
            # Process BM25 results
            for i, doc in enumerate(bm25_results):
                chunk_idx = doc.metadata.get("chunk_index", i)
                bm25_score = doc.metadata.get("bm25_score", 1.0 / (i + 1))
                
                if chunk_idx in chunk_scores:
                    chunk_scores[chunk_idx]["bm25_score"] = bm25_score
                else:
                    chunk_scores[chunk_idx] = {
                        "doc": doc,
                        "semantic_score": 0.0,
                        "bm25_score": bm25_score
                    }
            
            # Calculate hybrid scores
            for chunk_data in chunk_scores.values():
                semantic_norm = chunk_data["semantic_score"]
                bm25_norm = min(1.0, chunk_data["bm25_score"] / 10.0)  # Normalize BM25 scores
                hybrid_score = (semantic_weight * semantic_norm) + (bm25_weight * bm25_norm)
                chunk_data["hybrid_score"] = hybrid_score
                
                # Update document metadata
                chunk_data["doc"].metadata.update({
                    "semantic_score": semantic_norm,
                    "bm25_score": chunk_data["bm25_score"],
                    "hybrid_score": hybrid_score,
                    "search_type": "hybrid"
                })
            
            # Sort by hybrid score and return top-k
            sorted_chunks = sorted(
                chunk_scores.values(), 
                key=lambda x: x["hybrid_score"], 
                reverse=True
            )
            
            return [chunk["doc"] for chunk in sorted_chunks[:k]]
            
        except Exception as e:
            logger.error(f"Error in hybrid search for video {video_id}: {e}")
            # Fallback to semantic search
            return self.search_video_content(video_id, query, k)
    
    def search_all_videos_hybrid(
        self, 
        query: str, 
        video_ids: List[str], 
        k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Document]:
        """Search across multiple videos using hybrid approach"""
        all_results = []
        
        for video_id in video_ids:
            results = self.search_video_hybrid(
                video_id, query, k=k, 
                semantic_weight=semantic_weight, 
                bm25_weight=bm25_weight
            )
            all_results.extend(results)
        
        # Sort by hybrid score across all videos
        all_results.sort(
            key=lambda x: x.metadata.get("hybrid_score", 0), 
            reverse=True
        )
        
        return all_results[:k]
    
    def _initialize_cross_encoder(self):
        """Initialize cross-encoder model for reranking"""
        try:
            # Use a lightweight cross-encoder model for better relevance scoring
            model_name = "cross-encoder/ms-marco-MiniLM-L-2-v2"
            self.cross_encoder = CrossEncoder(model_name)
            logger.info(f"✅ Cross-encoder {model_name} initialized for reranking")
        except Exception as e:
            logger.warning(f"Cross-encoder initialization failed: {e}. Reranking will use fallback scoring.")
            self.cross_encoder = None
    
    def _rerank_with_cross_encoder(self, query: str, documents: List[Document], top_k: int = None) -> List[Document]:
        """Rerank documents using cross-encoder for better relevance scoring"""
        if not documents:
            return documents
        
        if top_k is None:
            top_k = len(documents)
        
        try:
            if self.cross_encoder is None:
                logger.warning("Cross-encoder not available, using original ranking")
                return documents[:top_k]
            
            # Prepare query-document pairs for cross-encoder
            pairs = []
            for doc in documents:
                # Truncate content to avoid token limits
                content = doc.page_content[:512]  # Limit to ~512 chars
                pairs.append([query, content])
            
            # Get cross-encoder scores
            ce_scores = self.cross_encoder.predict(pairs)
            
            # Create scored documents
            scored_docs = []
            for i, doc in enumerate(documents):
                doc_copy = Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata.copy()
                )
                doc_copy.metadata['cross_encoder_score'] = float(ce_scores[i])
                doc_copy.metadata['original_rank'] = i
                scored_docs.append(doc_copy)
            
            # Sort by cross-encoder score
            reranked = sorted(scored_docs, key=lambda x: x.metadata['cross_encoder_score'], reverse=True)
            
            logger.info(f"Cross-encoder reranked {len(documents)} documents")
            return reranked[:top_k]
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            return documents[:top_k]
    
    def search_video_hybrid_reranked(
        self, 
        video_id: str, 
        query: str, 
        k: int = 5, 
        semantic_weight: float = 0.7, 
        bm25_weight: float = 0.3,
        rerank_top_n: int = 20
    ) -> List[Document]:
        """Search with hybrid approach and cross-encoder reranking"""
        try:
            # Get initial hybrid results with more candidates for reranking
            initial_results = self.search_video_hybrid(
                video_id, query, k=rerank_top_n, 
                semantic_weight=semantic_weight, 
                bm25_weight=bm25_weight
            )
            
            if not initial_results:
                return []
            
            # Apply cross-encoder reranking
            reranked_results = self._rerank_with_cross_encoder(query, initial_results, top_k=k)
            
            # Update metadata to indicate reranking was applied
            for i, doc in enumerate(reranked_results):
                doc.metadata['search_type'] = 'hybrid_reranked'
                doc.metadata['final_rank'] = i
                doc.metadata['reranking_applied'] = True
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Hybrid reranked search failed: {e}")
            # Fallback to regular hybrid search
            return self.search_video_hybrid(video_id, query, k, semantic_weight, bm25_weight)
    
    def search_all_videos_hybrid_reranked(
        self, 
        query: str, 
        video_ids: List[str], 
        k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        rerank_top_n: int = 20
    ) -> List[Document]:
        """Search across videos with hybrid approach and cross-encoder reranking"""
        try:
            # Get initial hybrid results across all videos
            all_results = []
            
            for video_id in video_ids:
                results = self.search_video_hybrid(
                    video_id, query, k=rerank_top_n//len(video_ids) + 2, 
                    semantic_weight=semantic_weight, 
                    bm25_weight=bm25_weight
                )
                all_results.extend(results)
            
            if not all_results:
                return []
            
            # Sort by hybrid score and take top candidates for reranking
            all_results.sort(
                key=lambda x: x.metadata.get("hybrid_score", 0), 
                reverse=True
            )
            top_candidates = all_results[:rerank_top_n]
            
            # Apply cross-encoder reranking
            reranked_results = self._rerank_with_cross_encoder(query, top_candidates, top_k=k)
            
            # Update metadata
            for i, doc in enumerate(reranked_results):
                doc.metadata['search_type'] = 'multi_video_hybrid_reranked'
                doc.metadata['final_rank'] = i
                doc.metadata['reranking_applied'] = True
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Multi-video hybrid reranked search failed: {e}")
            # Fallback to regular hybrid search
            return self.search_all_videos_hybrid(query, video_ids, k, semantic_weight, bm25_weight)
    
    def delete_video_index(self, video_id: str) -> bool:
        """Delete both vector and BM25 indices for a specific video"""
        try:
            # Delete vector index
            collection_name = self._generate_video_collection_name(video_id)
            self.chroma_client.delete_collection(name=collection_name)
            
            # Delete BM25 index from memory
            if video_id in self.bm25_indices:
                del self.bm25_indices[video_id]
            if video_id in self.bm25_documents:
                del self.bm25_documents[video_id]
            
            # Delete BM25 index from disk
            cache_file = os.path.join(self.bm25_cache_dir, f"{video_id}_bm25.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
            
            logger.info(f"Deleted hybrid index for video {video_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting hybrid index: {e}")
            return False
    
    def get_video_stats(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics about a video's indices"""
        try:
            collection_name = self._generate_video_collection_name(video_id)
            collection = self.chroma_client.get_collection(name=collection_name)
            
            stats = {
                "video_id": video_id,
                "chunk_count": collection.count(),
                "collection_name": collection_name,
                "has_bm25_index": video_id in self.bm25_indices,
                "cross_encoder_available": self.cross_encoder is not None,
                "search_capabilities": [
                    "semantic",
                    "bm25" if video_id in self.bm25_indices else None,
                    "hybrid" if video_id in self.bm25_indices else None,
                    "cross_encoder_reranked" if self.cross_encoder is not None else None
                ]
            }
            
            # Filter out None capabilities
            stats["search_capabilities"] = [cap for cap in stats["search_capabilities"] if cap is not None]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting video stats: {e}")
            return None