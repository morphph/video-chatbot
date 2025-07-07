import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
import hashlib
import json
import logging
from gemini_embeddings import GeminiEmbeddings

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
        
    def _generate_chunk_id(self, video_id: str, chunk_index: int) -> str:
        """Generate unique ID for a chunk"""
        return f"{video_id}_chunk_{chunk_index}"
    
    def _generate_video_collection_name(self, video_id: str) -> str:
        """Generate collection name for a specific video"""
        # Chroma collection names must be 3-63 characters, start/end with alphanumeric
        safe_id = video_id.replace("-", "_")[:50]
        return f"video_{safe_id}"
    
    def create_video_index(self, video_id: str, chunks: List[str], metadata: Dict[str, Any]) -> bool:
        """Create a vector index for a specific video"""
        try:
            collection_name = self._generate_video_collection_name(video_id)
            
            # Create documents with metadata
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    "video_id": video_id,
                    "chunk_index": i,
                    "video_title": metadata.get("title", ""),
                    "video_author": metadata.get("author", ""),
                    "total_chunks": len(chunks)
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                ))
                metadatas.append(chunk_metadata)
                ids.append(self._generate_chunk_id(video_id, i))
            
            # Create vector store for this video
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                collection_name=collection_name,
                persist_directory=self.persist_directory,
                ids=ids
            )
            
            vector_store.persist()
            logger.info(f"Created vector index for video {video_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            return False
    
    def search_video_content(self, video_id: str, query: str, k: int = 5) -> List[Document]:
        """Search for relevant content in a specific video"""
        try:
            collection_name = self._generate_video_collection_name(video_id)
            
            # Load the vector store for this video
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory
            )
            
            # Perform similarity search
            results = vector_store.similarity_search(query, k=k)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching video content: {e}")
            return []
    
    def search_all_videos(self, query: str, video_ids: List[str], k: int = 5) -> List[Document]:
        """Search across multiple videos"""
        all_results = []
        
        for video_id in video_ids:
            results = self.search_video_content(video_id, query, k=k)
            all_results.extend(results)
        
        # Sort by relevance (this is simplified, could use scores)
        return all_results[:k]
    
    def delete_video_index(self, video_id: str) -> bool:
        """Delete the vector index for a specific video"""
        try:
            collection_name = self._generate_video_collection_name(video_id)
            self.chroma_client.delete_collection(name=collection_name)
            logger.info(f"Deleted vector index for video {video_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting vector index: {e}")
            return False
    
    def get_video_stats(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics about a video's vector index"""
        try:
            collection_name = self._generate_video_collection_name(video_id)
            collection = self.chroma_client.get_collection(name=collection_name)
            
            return {
                "video_id": video_id,
                "chunk_count": collection.count(),
                "collection_name": collection_name
            }
        except Exception as e:
            logger.error(f"Error getting video stats: {e}")
            return None