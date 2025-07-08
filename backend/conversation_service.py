import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
import logging
import json

from vector_store import VectorStore
from conversation_storage import PersistentConversationStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedConversationService:
    def __init__(self, vector_store: VectorStore, db_path: str = "conversations.db"):
        self.vector_store = vector_store
        self.storage = PersistentConversationStorage(db_path)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        logger.info("✅ Enhanced conversation service initialized with persistent storage")
        
    def _create_system_prompt(self, video_metadata: Dict[str, Any], transcript_method: str = "unknown") -> str:
        """Create a system prompt with video context"""
        # Determine content quality based on transcript method
        content_quality_note = ""
        if transcript_method == "gemini_direct":
            content_quality_note = "✅ High-quality content: This video was processed using advanced AI video understanding, providing accurate transcripts and comprehensive analysis."
        elif transcript_method == "youtube_api":
            content_quality_note = "✅ High-quality content: This video includes official captions/subtitles for accurate reference."
        elif transcript_method == "langchain":
            content_quality_note = "⚠️  Standard quality: Transcript extracted using general-purpose tools."
        else:
            content_quality_note = "⚠️  Limited quality: Content generated from video metadata only due to processing limitations."
        
        return f"""You are an intelligent assistant helping users understand and discuss YouTube video content.
        
Current Video Information:
- Title: {video_metadata.get('title', 'Unknown')}
- Author: {video_metadata.get('author', 'Unknown')}
- Duration: {video_metadata.get('duration', 'Unknown')} seconds

{content_quality_note}

Your responses should:
1. **Be accurate and helpful** - Use the provided transcript content to answer questions
2. **Include timestamps** - When referencing specific parts, include approximate timestamps when available
3. **Maintain conversation context** - Remember and reference previous parts of our conversation
4. **Be specific and detailed** - Use exact quotes and specific details from the video content
5. **Provide source attribution** - Reference which parts of the video your information comes from

When answering questions:
- Quote specific parts of the transcript when relevant
- Explain concepts thoroughly using the video's explanations
- Connect ideas across different parts of the video
- Reference timestamps or chunk numbers when possible
- Maintain a helpful, conversational tone"""
    
    def _format_sources(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """Format source documents for response"""
        sources = []
        for doc in documents:
            source = {
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "content_preview": doc.page_content[:200] + "...",
                "video_id": doc.metadata.get("video_id", "")
            }
            sources.append(source)
        return sources
    
    def create_conversation(self, user_id: str, video_id: str, video_metadata: Dict[str, Any], title: str = None) -> str:
        """Initialize a new conversation with persistent storage"""
        conversation_id = self.storage.create_conversation(
            user_id=user_id,
            video_id=video_id,
            video_metadata=video_metadata,
            title=title
        )
        
        logger.info(f"Created conversation {conversation_id} for user {user_id} with video {video_id}")
        return conversation_id
    
    def add_video_to_conversation(self, conversation_id: str, video_id: str, user_id: str) -> bool:
        """Add another video to an existing conversation"""
        return self.storage.add_video_to_conversation(conversation_id, video_id, user_id)
    
    def process_message(
        self, 
        conversation_id: str, 
        message: str, 
        user_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a user message and generate response with persistent storage"""
        
        # Get conversation from storage
        conversation = self.storage.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ValueError("Conversation not found")
        
        # Search for relevant content across all videos in conversation
        relevant_docs = self.vector_store.search_all_videos(
            query=message,
            video_ids=conversation["video_ids"],
            k=5
        )
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"[Chunk {doc.metadata.get('chunk_index', 'Unknown')}]: {doc.page_content}"
            for doc in relevant_docs
        ])
        
        # Get recent conversation history for context
        recent_messages = self.storage.get_conversation_memory_summary(
            conversation_id, user_id, max_messages=10
        )
        
        # Convert to LangChain message format
        chat_history = []
        for msg in recent_messages:
            if msg['role'] == 'user':
                chat_history.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                chat_history.append(AIMessage(content=msg['content']))
        
        # Get transcript method from video metadata if available
        transcript_method = conversation["metadata"].get("transcript_method", "unknown")
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._create_system_prompt(conversation["metadata"], transcript_method)),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """Based on the following context from the video transcript, please answer the question.
            
Context:
{context}

Question: {question}

Remember to cite specific parts of the video and include approximate timestamps when possible.""")
        ])
        
        # Format the prompt
        formatted_prompt = prompt_template.format_messages(
            chat_history=chat_history,
            context=context,
            question=message
        )
        
        # Generate response
        response = self.llm.invoke(formatted_prompt)
        
        # Format sources
        sources = self._format_sources(relevant_docs)
        
        # Save messages to persistent storage
        self.storage.add_message(
            conversation_id=conversation_id,
            role="user",
            content=message,
            user_id=user_id,
            sources=None,
            token_count=len(message.split())  # Rough token estimate
        )
        
        self.storage.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response.content,
            user_id=user_id,
            sources=sources,
            token_count=len(response.content.split())  # Rough token estimate
        )
        
        logger.info(f"Processed message in conversation {conversation_id} for user {user_id}")
        return response.content, sources
    
    def get_conversation_history(self, conversation_id: str, user_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Retrieve conversation history with pagination"""
        conversation = self.storage.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ValueError("Conversation not found")
        
        messages = self.storage.get_conversation_messages(conversation_id, user_id, limit, offset)
        
        return {
            "conversation_id": conversation_id,
            "title": conversation["title"],
            "video_ids": conversation["video_ids"],
            "messages": messages,
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"],
            "metadata": conversation["metadata"]
        }
    
    def get_user_conversations(self, user_id: str, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all conversations for a user"""
        return self.storage.get_user_conversations(user_id, limit, offset)
    
    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation"""
        return self.storage.delete_conversation(conversation_id, user_id)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service and storage statistics"""
        storage_stats = self.storage.get_storage_stats()
        
        # Try to get embedding cache stats if available
        embedding_cache_stats = {}
        try:
            if hasattr(self.vector_store, 'embedding_function'):
                embedding_func = self.vector_store.embedding_function
                if hasattr(embedding_func, 'get_cache_stats'):
                    embedding_cache_stats = embedding_func.get_cache_stats()
        except Exception:
            pass  # Ignore if not available
        
        return {
            "storage": storage_stats,
            "embedding_cache": embedding_cache_stats,
            "service_version": "enhanced_v1.0"
        }

# Backward compatibility alias
ConversationService = EnhancedConversationService