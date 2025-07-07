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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConversationService:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        self.conversations = {}  # In-memory storage (should be Redis in production)
        
    def _create_system_prompt(self, video_metadata: Dict[str, Any]) -> str:
        """Create a system prompt with video context"""
        return f"""You are an intelligent assistant helping users understand and discuss YouTube video content.
        
Current Video Information:
- Title: {video_metadata.get('title', 'Unknown')}
- Author: {video_metadata.get('author', 'Unknown')}
- Duration: {video_metadata.get('duration', 'Unknown')} seconds

Your responses should:
1. Be grounded in the actual video content
2. Provide timestamps when referencing specific parts
3. Be conversational and helpful
4. Acknowledge when information is not available in the video
5. Maintain context from previous messages in the conversation

When answering questions:
- Always cite the relevant parts of the video
- If the information isn't in the video, say so clearly
- Provide context and explain concepts when needed"""
    
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
    
    def create_conversation(self, user_id: str, video_id: str, video_metadata: Dict[str, Any]) -> str:
        """Initialize a new conversation"""
        conversation_id = str(uuid.uuid4())
        
        self.conversations[conversation_id] = {
            "user_id": user_id,
            "video_ids": [video_id],
            "metadata": video_metadata,
            "memory": ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            ),
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        return conversation_id
    
    def add_video_to_conversation(self, conversation_id: str, video_id: str, video_metadata: Dict[str, Any]) -> bool:
        """Add another video to an existing conversation"""
        if conversation_id not in self.conversations:
            return False
        
        conversation = self.conversations[conversation_id]
        if video_id not in conversation["video_ids"]:
            conversation["video_ids"].append(video_id)
            conversation["updated_at"] = datetime.now().isoformat()
        
        return True
    
    def process_message(
        self, 
        conversation_id: str, 
        message: str, 
        user_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a user message and generate response"""
        
        if conversation_id not in self.conversations:
            raise ValueError("Conversation not found")
        
        conversation = self.conversations[conversation_id]
        if conversation["user_id"] != user_id:
            raise ValueError("Unauthorized access to conversation")
        
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
        
        # Create prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._create_system_prompt(conversation["metadata"])),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """Based on the following context from the video transcript, please answer the question.
            
Context:
{context}

Question: {question}

Remember to cite specific parts of the video when possible.""")
        ])
        
        # Format the prompt
        formatted_prompt = prompt_template.format_messages(
            chat_history=conversation["memory"].chat_memory.messages,
            context=context,
            question=message
        )
        
        # Generate response
        response = self.llm.invoke(formatted_prompt)
        
        # Update conversation memory
        conversation["memory"].chat_memory.add_user_message(message)
        conversation["memory"].chat_memory.add_ai_message(response.content)
        
        # Store message history
        conversation["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        conversation["messages"].append({
            "role": "assistant",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        conversation["updated_at"] = datetime.now().isoformat()
        
        # Format sources
        sources = self._format_sources(relevant_docs)
        
        return response.content, sources
    
    def get_conversation_history(self, conversation_id: str, user_id: str) -> Dict[str, Any]:
        """Retrieve conversation history"""
        if conversation_id not in self.conversations:
            raise ValueError("Conversation not found")
        
        conversation = self.conversations[conversation_id]
        if conversation["user_id"] != user_id:
            raise ValueError("Unauthorized access to conversation")
        
        return {
            "conversation_id": conversation_id,
            "video_ids": conversation["video_ids"],
            "messages": conversation["messages"],
            "created_at": conversation["created_at"],
            "updated_at": conversation["updated_at"]
        }
    
    def clear_conversation_memory(self, conversation_id: str, user_id: str) -> bool:
        """Clear conversation memory while preserving the conversation"""
        if conversation_id not in self.conversations:
            return False
        
        conversation = self.conversations[conversation_id]
        if conversation["user_id"] != user_id:
            return False
        
        conversation["memory"].clear()
        conversation["messages"] = []
        conversation["updated_at"] = datetime.now().isoformat()
        
        return True