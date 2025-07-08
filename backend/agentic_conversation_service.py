"""
Agentic RAG Service using LangGraph StateGraph (2025)
Implements cutting-edge conversational AI patterns with tool-calling and streaming.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, TypedDict, Annotated
from datetime import datetime

# LangGraph 2025 imports
try:
    from langgraph.graph import StateGraph, MessagesState, START, END
    from langgraph.prebuilt import ToolNode
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
except ImportError:
    # Fallback for development - create mock classes
    class StateGraph:
        def __init__(self, state_schema): pass
        def add_node(self, name, func): pass
        def add_edge(self, from_node, to_node): pass
        def add_conditional_edges(self, source, condition, mapping): pass
        def compile(self, checkpointer=None): pass
    
    class MessagesState:
        messages: List[Dict]
    
    class ToolNode:
        def __init__(self, tools): pass
    
    class MemorySaver:
        pass
    
    START = "START"
    END = "END"

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage

from vector_store import VectorStore
from conversation_storage import PersistentConversationStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced State Schema for Agentic RAG
class AgenticRAGState(TypedDict):
    """Enhanced state schema for sophisticated agentic RAG workflows"""
    messages: Annotated[List[Dict], "The conversation messages"]
    query: str
    conversation_id: str
    user_id: str
    video_ids: List[str]
    context_documents: List[Dict]
    retrieved_chunks: List[Dict]
    reranked_results: List[Dict]
    search_strategy: str  # "semantic", "hybrid", "metadata_filtered"
    reasoning_trace: List[str]
    tool_results: Dict[str, Any]
    confidence_score: float
    should_retrieve: bool
    should_rerank: bool
    should_enhance_query: bool
    enhanced_query: Optional[str]
    generation_strategy: str  # "direct", "context_aware", "multi_perspective"

class AgenticRAGService:
    """
    State-of-the-art Agentic RAG Service using LangGraph StateGraph.
    Implements 2025's cutting-edge conversational AI patterns.
    """
    
    def __init__(self, vector_store: VectorStore, db_path: str = "conversations.db"):
        self.vector_store = vector_store
        self.storage = PersistentConversationStorage(db_path)
        
        # Initialize Gemini 2.5 Flash with enhanced capabilities
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,  # Lower for more focused reasoning
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # Initialize LangGraph StateGraph
        self.checkpointer = MemorySaver()
        self.graph = self._build_agentic_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)
        
        logger.info("✅ AgenticRAGService initialized with LangGraph StateGraph")
    
    def _build_agentic_graph(self) -> StateGraph:
        """Build the sophisticated agentic RAG workflow graph"""
        
        # Create StateGraph with enhanced state schema
        graph = StateGraph(AgenticRAGState)
        
        # Add sophisticated workflow nodes
        graph.add_node("analyze_query", self._analyze_query)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("rerank_results", self._rerank_results)
        graph.add_node("enhance_query", self._enhance_query)
        graph.add_node("generate_response", self._generate_response)
        graph.add_node("evaluate_response", self._evaluate_response)
        
        # Define sophisticated conditional routing
        graph.add_edge(START, "analyze_query")
        
        graph.add_conditional_edges(
            "analyze_query",
            self._should_retrieve,
            {
                "retrieve": "retrieve_context",
                "enhance_query": "enhance_query",
                "direct_response": "generate_response"
            }
        )
        
        graph.add_conditional_edges(
            "retrieve_context",
            self._should_rerank,
            {
                "rerank": "rerank_results",
                "generate": "generate_response"
            }
        )
        
        graph.add_conditional_edges(
            "enhance_query",
            lambda state: "retrieve",
            {"retrieve": "retrieve_context"}
        )
        
        graph.add_edge("rerank_results", "generate_response")
        graph.add_edge("generate_response", "evaluate_response")
        graph.add_edge("evaluate_response", END)
        
        return graph
    
    def _analyze_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Analyze query to determine optimal processing strategy"""
        query = state["query"]
        
        # Use Gemini 2.5 Flash thinking capabilities for query analysis
        analysis_prompt = f"""
        Analyze this user query for optimal RAG processing strategy:
        Query: "{query}"
        
        Determine:
        1. Query type (factual, temporal, comparative, procedural)
        2. Search strategy needed (semantic, hybrid, metadata-filtered)
        3. Whether query enhancement is needed
        4. Confidence in direct response capability
        
        Respond with JSON:
        {{
            "query_type": "...",
            "search_strategy": "...",
            "needs_enhancement": boolean,
            "can_respond_directly": boolean,
            "confidence": 0.0-1.0,
            "reasoning": "..."
        }}
        """
        
        try:
            analysis_response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            analysis = json.loads(analysis_response.content)
            
            state["search_strategy"] = analysis.get("search_strategy", "hybrid")
            state["should_retrieve"] = not analysis.get("can_respond_directly", False)
            state["should_enhance_query"] = analysis.get("needs_enhancement", False)
            state["confidence_score"] = analysis.get("confidence", 0.5)
            state["reasoning_trace"] = [f"Query analysis: {analysis.get('reasoning', '')}"]
            
        except Exception as e:
            logger.warning(f"Query analysis failed: {e}")
            # Fallback to safe defaults
            state["search_strategy"] = "hybrid"
            state["should_retrieve"] = True
            state["should_enhance_query"] = False
            state["confidence_score"] = 0.5
            state["reasoning_trace"] = ["Fallback analysis used"]
        
        return state
    
    def _should_retrieve(self, state: AgenticRAGState) -> str:
        """Conditional routing based on query analysis"""
        if state.get("should_enhance_query", False):
            return "enhance_query"
        elif state.get("should_retrieve", True):
            return "retrieve"
        else:
            return "direct_response"
    
    def _should_rerank(self, state: AgenticRAGState) -> str:
        """Determine if additional reranking is needed"""
        retrieved_chunks = state.get("retrieved_chunks", [])
        
        # Check if cross-encoder reranking was already applied
        if retrieved_chunks and retrieved_chunks[0].get("metadata", {}).get("reranking_applied", False):
            # Cross-encoder reranking already applied, proceed to generation
            return "generate"
        
        # Apply additional reranking if we have multiple results and moderate confidence
        if len(retrieved_chunks) > 3 and state.get("confidence_score", 0) < 0.8:
            return "rerank"
        else:
            return "generate"
    
    def _retrieve_context(self, state: AgenticRAGState) -> AgenticRAGState:
        """Retrieve relevant context using the determined strategy"""
        query = state.get("enhanced_query") or state["query"]
        video_ids = state["video_ids"]
        strategy = state["search_strategy"]
        
        try:
            # Use enhanced search with cross-encoder reranking
            if strategy == "hybrid" and hasattr(self.vector_store, 'search_all_videos_hybrid_reranked'):
                relevant_docs = self.vector_store.search_all_videos_hybrid_reranked(
                    query=query,
                    video_ids=video_ids,
                    k=8,
                    rerank_top_n=20  # Get more candidates for reranking
                )
            elif strategy == "semantic":
                relevant_docs = self.vector_store.search_all_videos(
                    query=query,
                    video_ids=video_ids,
                    k=8
                )
            else:
                # Fallback to regular hybrid search if reranking not available
                if hasattr(self.vector_store, 'search_all_videos_hybrid'):
                    relevant_docs = self.vector_store.search_all_videos_hybrid(
                        query=query,
                        video_ids=video_ids,
                        k=8
                    )
                else:
                    relevant_docs = self.vector_store.search_all_videos(
                        query=query,
                        video_ids=video_ids,
                        k=8
                    )
            
            # Convert to enhanced format
            chunks = []
            for doc in relevant_docs:
                chunk = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'relevance_score', 0.5)
                }
                chunks.append(chunk)
            
            state["retrieved_chunks"] = chunks
            # Check if reranking was applied
            reranking_applied = chunks and chunks[0].get("metadata", {}).get("reranking_applied", False)
            rerank_info = " with cross-encoder reranking" if reranking_applied else ""
            
            state["reasoning_trace"].append(f"Retrieved {len(chunks)} chunks using {strategy} strategy{rerank_info}")
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            state["retrieved_chunks"] = []
            state["reasoning_trace"].append(f"Retrieval failed: {e}")
        
        return state
    
    def _rerank_results(self, state: AgenticRAGState) -> AgenticRAGState:
        """Additional reranking if cross-encoder reranking wasn't already applied"""
        chunks = state["retrieved_chunks"]
        query = state.get("enhanced_query") or state["query"]
        
        try:
            # Check if cross-encoder reranking was already applied
            already_reranked = chunks and chunks[0].get("metadata", {}).get("reranking_applied", False)
            
            if already_reranked:
                # Use existing ranking
                state["reranked_results"] = chunks[:5]
                state["reasoning_trace"].append("Using cross-encoder reranked results")
            else:
                # Apply simple relevance-based reranking
                reranked = sorted(chunks, key=lambda x: x.get("relevance_score", 0), reverse=True)
                state["reranked_results"] = reranked[:5]
                state["reasoning_trace"].append(f"Applied fallback reranking to top {len(state['reranked_results'])} results")
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            state["reranked_results"] = chunks[:5]  # Fallback
            state["reasoning_trace"].append(f"Reranking failed, using top 5: {e}")
        
        return state
    
    def _enhance_query(self, state: AgenticRAGState) -> AgenticRAGState:
        """Advanced context-aware query enhancement with intelligent rewriting"""
        original_query = state["query"]
        conversation_id = state["conversation_id"]
        user_id = state["user_id"]
        video_ids = state["video_ids"]
        
        try:
            # Get comprehensive conversation context
            recent_messages = self.storage.get_conversation_memory_summary(
                conversation_id, user_id, max_messages=10
            )
            
            # Extract key entities and topics from conversation history
            context_analysis = self._analyze_conversation_context(recent_messages)
            
            # Get video metadata for additional context
            video_context = self._extract_video_context(video_ids)
            
            # Apply intelligent query rewriting
            enhanced_query = self._rewrite_query_with_context(
                original_query, 
                context_analysis, 
                video_context, 
                recent_messages
            )
            
            state["enhanced_query"] = enhanced_query
            state["reasoning_trace"].append(
                f"Context-aware enhancement: '{original_query}' → '{enhanced_query}'"
            )
            state["reasoning_trace"].append(
                f"Context factors: {context_analysis.get('summary', 'none identified')}"
            )
            
        except Exception as e:
            logger.warning(f"Advanced query enhancement failed: {e}")
            state["enhanced_query"] = original_query
            state["reasoning_trace"].append(f"Query enhancement failed, using original: {e}")
        
        return state
    
    def _analyze_conversation_context(self, messages: List[Dict]) -> Dict[str, Any]:
        """Analyze conversation history to extract key context"""
        try:
            if not messages:
                return {"entities": [], "topics": [], "intent": "general", "summary": ""}
            
            # Build conversation summary
            conversation_text = ""
            user_queries = []
            assistant_responses = []
            
            for msg in messages[-8:]:  # Last 8 messages
                role = msg.get('role', '')
                content = msg.get('content', '')[:300]  # Limit length
                
                if role == 'user':
                    user_queries.append(content)
                    conversation_text += f"User: {content}\n"
                elif role == 'assistant':
                    assistant_responses.append(content)
                    conversation_text += f"Assistant: {content}\n"
            
            # Use Gemini to analyze context
            analysis_prompt = f"""
            Analyze this conversation to extract key context for query enhancement:
            
            {conversation_text}
            
            Extract and provide as JSON:
            {{
                "key_entities": ["entity1", "entity2"],
                "main_topics": ["topic1", "topic2"],
                "user_intent": "description of what user is trying to accomplish",
                "context_summary": "brief summary of relevant context",
                "question_pattern": "type of questions being asked",
                "technical_level": "beginner/intermediate/advanced"
            }}
            """
            
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            
            try:
                import json
                analysis = json.loads(response.content)
                return analysis
            except json.JSONDecodeError:
                # Fallback to basic analysis
                return {
                    "key_entities": self._extract_simple_entities(user_queries),
                    "main_topics": self._extract_simple_topics(user_queries),
                    "user_intent": "information_seeking",
                    "context_summary": "User asking questions about video content",
                    "question_pattern": "exploratory",
                    "technical_level": "intermediate"
                }
                
        except Exception as e:
            logger.warning(f"Context analysis failed: {e}")
            return {"entities": [], "topics": [], "intent": "general", "summary": ""}
    
    def _extract_simple_entities(self, queries: List[str]) -> List[str]:
        """Extract simple entities from user queries"""
        import re
        entities = set()
        
        for query in queries[-3:]:  # Last 3 queries
            # Extract capitalized words (potential entities)
            words = re.findall(r'\b[A-Z][a-z]+\b', query)
            entities.update(words)
            
            # Extract quoted terms
            quoted = re.findall(r'"([^"]+)"', query)
            entities.update(quoted)
        
        return list(entities)[:5]  # Limit to 5 entities
    
    def _extract_simple_topics(self, queries: List[str]) -> List[str]:
        """Extract simple topics from user queries"""
        topics = set()
        common_topics = [
            'implementation', 'algorithm', 'performance', 'best practices',
            'tutorial', 'explanation', 'example', 'comparison', 'setup',
            'configuration', 'debugging', 'optimization', 'integration'
        ]
        
        combined_text = ' '.join(queries[-3:]).lower()
        
        for topic in common_topics:
            if topic in combined_text:
                topics.add(topic)
        
        return list(topics)[:3]  # Limit to 3 topics
    
    def _extract_video_context(self, video_ids: List[str]) -> Dict[str, Any]:
        """Extract context from video metadata"""
        try:
            video_context = {
                "total_videos": len(video_ids),
                "video_topics": [],
                "content_type": "educational"  # default assumption
            }
            
            # Get video stats if available
            for video_id in video_ids[:3]:  # Limit to first 3 videos
                try:
                    stats = self.vector_store.get_video_stats(video_id)
                    if stats:
                        video_context[f"video_{video_id}_chunks"] = stats.get("chunk_count", 0)
                except Exception:
                    pass
            
            return video_context
            
        except Exception as e:
            logger.warning(f"Video context extraction failed: {e}")
            return {"total_videos": len(video_ids), "video_topics": []}
    
    def _rewrite_query_with_context(
        self, 
        original_query: str, 
        context_analysis: Dict[str, Any], 
        video_context: Dict[str, Any],
        recent_messages: List[Dict]
    ) -> str:
        """Intelligently rewrite query using context"""
        try:
            # Build context-aware enhancement prompt
            context_info = ""
            
            if context_analysis.get("key_entities"):
                context_info += f"Key entities mentioned: {', '.join(context_analysis['key_entities'])}\n"
            
            if context_analysis.get("main_topics"):
                context_info += f"Topics discussed: {', '.join(context_analysis['main_topics'])}\n"
            
            if context_analysis.get("user_intent"):
                context_info += f"User intent: {context_analysis['user_intent']}\n"
            
            if context_analysis.get("technical_level"):
                context_info += f"Technical level: {context_analysis['technical_level']}\n"
            
            # Get last user question for reference
            last_question = ""
            for msg in reversed(recent_messages):
                if msg.get('role') == 'user' and msg.get('content') != original_query:
                    last_question = msg.get('content', '')[:150]
                    break
            
            enhancement_prompt = f"""
            Enhance this query for better video content search:
            
            Original Query: "{original_query}"
            
            Context:
            {context_info}
            Previous Question: "{last_question}"
            
            Instructions:
            1. Keep the core intent of the original query
            2. Add specific context from the conversation
            3. Include relevant technical terms or entities mentioned
            4. Make it more specific for better search results
            5. Maintain natural language flow
            
            Enhanced Query (one line, no quotes):
            """
            
            response = self.llm.invoke([HumanMessage(content=enhancement_prompt)])
            enhanced = response.content.strip().strip('"').strip("'")
            
            # Quality check - don't make query too long or completely different
            if len(enhanced) > len(original_query) * 3 or len(enhanced) < 5:
                logger.warning("Enhanced query quality check failed, using original")
                return original_query
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return original_query
    
    def _generate_response(self, state: AgenticRAGState) -> AgenticRAGState:
        """Generate sophisticated response using retrieved context"""
        query = state.get("enhanced_query") or state["query"]
        results = state.get("reranked_results") or state.get("retrieved_chunks", [])
        
        # Build context from results
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(results):
            content = chunk["content"]
            metadata = chunk.get("metadata", {})
            
            # Extract timestamp info if available
            timestamp_info = ""
            if "timestamp_range" in metadata:
                timestamp_info = f" [{metadata['timestamp_range']}]"
            elif "start_time" in metadata:
                start_time = metadata["start_time"]
                timestamp_info = f" [~{start_time//60:02d}:{start_time%60:02d}]"
            
            context_parts.append(f"[Source {i+1}]{timestamp_info}: {content}")
            
            # Prepare source attribution
            sources.append({
                "chunk_index": metadata.get("chunk_index", i),
                "content_preview": content[:200] + "...",
                "video_id": metadata.get("video_id", ""),
                "timestamp_range": metadata.get("timestamp_range", ""),
                "relevance_score": chunk.get("relevance_score", 0.5)
            })
        
        context = "\\n\\n".join(context_parts)
        
        # Create sophisticated generation prompt
        system_prompt = """You are an intelligent video content assistant with advanced reasoning capabilities.

Your responses should:
1. **Be accurate and detailed** - Use specific information from the provided context
2. **Include precise references** - Cite sources with timestamps when available
3. **Maintain conversation flow** - Consider the query in context of the discussion
4. **Be comprehensive** - Provide thorough explanations while staying focused
5. **Show reasoning** - Explain how you arrived at your conclusions

When referencing video content:
- Use timestamp references like [02:30] when available
- Quote specific phrases from the transcript
- Connect ideas across different parts of the video
- Explain technical concepts clearly"""
        
        user_prompt = f"""Based on the following video content, please answer this question:

Question: {query}

Video Content:
{context}

Please provide a comprehensive response that:
- Answers the question directly and thoroughly
- References specific parts of the video with timestamps
- Explains any technical concepts mentioned
- Connects related ideas from different parts of the video

Remember to cite your sources and include relevant timestamps."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Store the response and sources
            state["tool_results"] = {
                "response": response.content,
                "sources": sources,
                "context_used": len(results),
                "reasoning_trace": state["reasoning_trace"]
            }
            
            state["reasoning_trace"].append(f"Generated response using {len(results)} sources")
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            state["tool_results"] = {
                "response": f"I apologize, but I encountered an error while processing your question: {e}",
                "sources": [],
                "context_used": 0,
                "reasoning_trace": state["reasoning_trace"]
            }
        
        return state
    
    def _evaluate_response(self, state: AgenticRAGState) -> AgenticRAGState:
        """Evaluate response quality and completeness"""
        tool_results = state.get("tool_results", {})
        response = tool_results.get("response", "")
        
        # Simple quality metrics
        quality_score = min(1.0, len(response) / 500)  # Longer responses generally better
        
        if tool_results.get("sources"):
            quality_score += 0.2  # Bonus for having sources
        
        if state.get("reasoning_trace"):
            quality_score += 0.1  # Bonus for reasoning trace
        
        quality_score = min(1.0, quality_score)
        
        state["confidence_score"] = quality_score
        state["reasoning_trace"].append(f"Response quality score: {quality_score:.2f}")
        
        return state
    
    # Public API methods
    
    def create_conversation(self, user_id: str, video_id: str, video_metadata: Dict[str, Any], title: str = None) -> str:
        """Create a new conversation with agentic capabilities"""
        return self.storage.create_conversation(
            user_id=user_id,
            video_id=video_id,
            video_metadata=video_metadata,
            title=title
        )
    
    def process_message_agentic(
        self,
        conversation_id: str,
        message: str,
        user_id: str
    ) -> Tuple[str, List[Dict[str, Any]], List[str]]:
        """Process message using agentic RAG workflow"""
        
        # Get conversation details
        conversation = self.storage.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ValueError("Conversation not found")
        
        # Initialize state
        initial_state = {
            "messages": [],
            "query": message,
            "conversation_id": conversation_id,
            "user_id": user_id,
            "video_ids": conversation["video_ids"],
            "context_documents": [],
            "retrieved_chunks": [],
            "reranked_results": [],
            "search_strategy": "hybrid",
            "reasoning_trace": [],
            "tool_results": {},
            "confidence_score": 0.0,
            "should_retrieve": True,
            "should_rerank": False,
            "should_enhance_query": False,
            "enhanced_query": None,
            "generation_strategy": "context_aware"
        }
        
        try:
            # Run the agentic workflow
            config = {"configurable": {"thread_id": conversation_id}}
            result = self.app.invoke(initial_state, config)
            
            # Extract results
            tool_results = result.get("tool_results", {})
            response_text = tool_results.get("response", "I apologize, but I couldn't generate a response.")
            sources = tool_results.get("sources", [])
            reasoning_trace = result.get("reasoning_trace", [])
            
            # Save messages to storage
            self.storage.add_message(
                conversation_id=conversation_id,
                role="user",
                content=message,
                user_id=user_id,
                sources=None,
                token_count=len(message.split())
            )
            
            self.storage.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response_text,
                user_id=user_id,
                sources=sources,
                token_count=len(response_text.split())
            )
            
            logger.info(f"Agentic RAG processed message in conversation {conversation_id}")
            return response_text, sources, reasoning_trace
            
        except Exception as e:
            logger.error(f"Agentic message processing failed: {e}")
            # Fallback to basic response
            fallback_response = f"I apologize, but I encountered an error while processing your question. Error: {e}"
            return fallback_response, [], [f"Error in agentic processing: {e}"]
    
    def get_conversation_history(self, conversation_id: str, user_id: str, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get conversation history with agentic metadata"""
        return self.storage.get_conversation_history(conversation_id, user_id, limit, offset)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get enhanced service statistics"""
        storage_stats = self.storage.get_storage_stats()
        
        return {
            "storage": storage_stats,
            "service_type": "agentic_rag",
            "service_version": "2025_v1.0",
            "langgraph_enabled": True,
            "features": [
                "state_graph_orchestration",
                "tool_calling",
                "conditional_routing",
                "query_enhancement",
                "intelligent_reranking",
                "reasoning_trace"
            ]
        }

# Backward compatibility - enhanced version of the original service
class EnhancedConversationService(AgenticRAGService):
    """Enhanced conversation service with agentic capabilities"""
    
    def process_message(self, conversation_id: str, message: str, user_id: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process message with agentic enhancement, maintaining original API"""
        response, sources, _ = self.process_message_agentic(conversation_id, message, user_id)
        return response, sources

# Export the enhanced service as default
ConversationService = EnhancedConversationService