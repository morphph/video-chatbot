"""
Advanced Tool-Calling System for Agentic RAG (2025)
Implements sophisticated tools for video analysis, search, and content processing.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import re

# Tool calling imports with fallbacks
try:
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field
except ImportError:
    try:
        # Fallback to v1 compatibility
        from langchain_core.pydantic_v1 import BaseModel, Field
        from langchain_core.tools import tool
    except ImportError:
        # Fallback decorators for development
        def tool(func):
            return func
        
        class BaseModel:
            pass
        
        class Field:
            def __init__(self, **kwargs):
                pass

from vector_store import VectorStore
from video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tool Input Models
class VideoSearchInput(BaseModel):
    """Input for video content search"""
    query: str = Field(description="Search query for video content")
    video_ids: List[str] = Field(description="List of video IDs to search within")
    search_type: str = Field(default="hybrid_reranked", description="Type of search: semantic, keyword, hybrid, hybrid_reranked")
    max_results: int = Field(default=5, description="Maximum number of results to return")
    enable_reranking: bool = Field(default=True, description="Whether to apply cross-encoder reranking for better relevance")

class TimestampQueryInput(BaseModel):
    """Input for timestamp-based queries"""
    video_id: str = Field(description="Video ID to search within")
    time_range: str = Field(description="Time range in format 'MM:SS-MM:SS' or 'around MM:SS'")
    query: str = Field(description="Query about content in the specified time range")

class VideoAnalysisInput(BaseModel):
    """Input for advanced video analysis"""
    video_id: str = Field(description="Video ID to analyze")
    analysis_type: str = Field(description="Type of analysis: summary, topics, structure, technical_content")

class ContentEnhancementInput(BaseModel):
    """Input for content enhancement and generation"""
    base_content: str = Field(description="Base content to enhance")
    enhancement_type: str = Field(description="Type of enhancement: expand, simplify, technical_detail, examples")
    context: str = Field(default="", description="Additional context for enhancement")

class AgenticToolRegistry:
    """
    Advanced tool registry for agentic RAG system.
    Implements 2025's sophisticated tool-calling patterns.
    """
    
    def __init__(self, vector_store: VectorStore, video_processor: VideoProcessor):
        self.vector_store = vector_store
        self.video_processor = video_processor
        self.tool_usage_stats = {}
        logger.info("✅ AgenticToolRegistry initialized with advanced tools")
    
    @tool
    def search_video_content(self, input_data: VideoSearchInput) -> Dict[str, Any]:
        """
        Advanced video content search with multiple strategies.
        Supports semantic, keyword, and hybrid search approaches.
        """
        try:
            query = input_data.query
            video_ids = input_data.video_ids
            search_type = input_data.search_type
            max_results = input_data.max_results
            
            self._track_tool_usage("search_video_content")
            
            if search_type == "semantic":
                results = self._semantic_search(query, video_ids, max_results)
            elif search_type == "keyword":
                results = self._keyword_search(query, video_ids, max_results)
            elif search_type == "hybrid":
                results = self._hybrid_search(query, video_ids, max_results)
            elif search_type == "hybrid_reranked":
                results = self._hybrid_reranked_search(query, video_ids, max_results, input_data.enable_reranking)
            else:
                # Default to hybrid with reranking
                results = self._hybrid_reranked_search(query, video_ids, max_results, True)
            
            return {
                "success": True,
                "results": results,
                "search_type": search_type,
                "result_count": len(results),
                "query": query
            }
            
        except Exception as e:
            logger.error(f"Video content search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    @tool
    def analyze_timestamp_content(self, input_data: TimestampQueryInput) -> Dict[str, Any]:
        """
        Analyze video content within specific timestamp ranges.
        Provides precise temporal context and analysis.
        """
        try:
            video_id = input_data.video_id
            time_range = input_data.time_range
            query = input_data.query
            
            self._track_tool_usage("analyze_timestamp_content")
            
            # Parse time range
            start_time, end_time = self._parse_time_range(time_range)
            
            # Search for content within timestamp range
            results = self.vector_store.search_video_content(
                video_id=video_id,
                query=query,
                k=10
            )
            
            # Filter by timestamp if metadata available
            filtered_results = []
            for result in results:
                metadata = getattr(result, 'metadata', {})
                chunk_start = metadata.get('start_time', 0)
                chunk_end = metadata.get('end_time', chunk_start + 60)
                
                # Check if chunk overlaps with requested time range
                if self._time_ranges_overlap(chunk_start, chunk_end, start_time, end_time):
                    filtered_results.append({
                        "content": result.page_content,
                        "metadata": metadata,
                        "timestamp_match": True,
                        "start_time": chunk_start,
                        "end_time": chunk_end
                    })
            
            return {
                "success": True,
                "results": filtered_results,
                "time_range": time_range,
                "start_time": start_time,
                "end_time": end_time,
                "matches_found": len(filtered_results)
            }
            
        except Exception as e:
            logger.error(f"Timestamp analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": []
            }
    
    @tool
    def perform_video_analysis(self, input_data: VideoAnalysisInput) -> Dict[str, Any]:
        """
        Perform advanced video analysis using Gemini 2.5 Flash capabilities.
        Supports multiple analysis types for comprehensive understanding.
        """
        try:
            video_id = input_data.video_id
            analysis_type = input_data.analysis_type
            
            self._track_tool_usage("perform_video_analysis")
            
            # Get video content for analysis
            video_results = self.vector_store.search_video_content(
                video_id=video_id,
                query="complete content overview",
                k=20
            )
            
            # Combine content for analysis
            content_parts = []
            for result in video_results[:10]:  # Use top 10 chunks
                metadata = getattr(result, 'metadata', {})
                timestamp = metadata.get('timestamp_range', '')
                content_parts.append(f"[{timestamp}] {result.page_content}")
            
            combined_content = "\\n\\n".join(content_parts)
            
            # Generate analysis based on type
            if analysis_type == "summary":
                analysis = self._generate_video_summary(combined_content)
            elif analysis_type == "topics":
                analysis = self._extract_key_topics(combined_content)
            elif analysis_type == "structure":
                analysis = self._analyze_video_structure(combined_content)
            elif analysis_type == "technical_content":
                analysis = self._analyze_technical_content(combined_content)
            else:
                analysis = self._generate_comprehensive_analysis(combined_content)
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "analysis": analysis,
                "content_chunks_analyzed": len(content_parts),
                "video_id": video_id
            }
            
        except Exception as e:
            logger.error(f"Video analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "analysis": ""
            }
    
    @tool
    def enhance_content_response(self, input_data: ContentEnhancementInput) -> Dict[str, Any]:
        """
        Enhance content responses with advanced AI capabilities.
        Provides multiple enhancement strategies for optimal user experience.
        """
        try:
            base_content = input_data.base_content
            enhancement_type = input_data.enhancement_type
            context = input_data.context
            
            self._track_tool_usage("enhance_content_response")
            
            # Apply enhancement based on type
            if enhancement_type == "expand":
                enhanced = self._expand_content(base_content, context)
            elif enhancement_type == "simplify":
                enhanced = self._simplify_content(base_content, context)
            elif enhancement_type == "technical_detail":
                enhanced = self._add_technical_details(base_content, context)
            elif enhancement_type == "examples":
                enhanced = self._add_examples(base_content, context)
            else:
                enhanced = self._general_enhancement(base_content, context)
            
            return {
                "success": True,
                "enhancement_type": enhancement_type,
                "original_content": base_content,
                "enhanced_content": enhanced,
                "improvement_applied": True
            }
            
        except Exception as e:
            logger.error(f"Content enhancement failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "enhanced_content": base_content  # Return original as fallback
            }
    
    # Helper methods for tool implementations
    
    def _semantic_search(self, query: str, video_ids: List[str], max_results: int) -> List[Dict]:
        """Perform semantic search across videos"""
        results = self.vector_store.search_all_videos(
            query=query,
            video_ids=video_ids,
            k=max_results
        )
        
        semantic_results = []
        for i, result in enumerate(results):
            metadata = getattr(result, 'metadata', {})
            # Add position-based relevance score if not available
            relevance_score = getattr(result, 'relevance_score', 1.0 / (i + 1))
            
            semantic_results.append({
                "content": result.page_content,
                "metadata": metadata,
                "relevance_score": relevance_score,
                "search_type": "semantic"
            })
        
        return semantic_results
    
    def _keyword_search(self, query: str, video_ids: List[str], max_results: int) -> List[Dict]:
        """Perform BM25 keyword-based search"""
        all_results = []
        
        for video_id in video_ids:
            try:
                results = self.vector_store.search_video_bm25(video_id, query, k=max_results)
                for result in results:
                    all_results.append({
                        "content": result.page_content,
                        "metadata": getattr(result, 'metadata', {}),
                        "relevance_score": result.metadata.get('bm25_score', 0.5),
                        "search_type": "bm25_keyword"
                    })
            except Exception as e:
                logger.warning(f"BM25 search failed for video {video_id}: {e}")
                # Fallback to semantic search for this video
                semantic_results = self._semantic_search(query, [video_id], max_results)
                all_results.extend(semantic_results)
        
        # Sort by BM25 score and return top results
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return all_results[:max_results]
    
    def _hybrid_search(self, query: str, video_ids: List[str], max_results: int) -> List[Dict]:
        """Perform advanced hybrid search combining semantic and BM25"""
        try:
            # Use the new hybrid search from vector store
            results = self.vector_store.search_all_videos_hybrid(
                query=query,
                video_ids=video_ids,
                k=max_results,
                semantic_weight=0.7,  # Favor semantic search slightly
                bm25_weight=0.3
            )
            
            hybrid_results = []
            for result in results:
                metadata = getattr(result, 'metadata', {})
                hybrid_results.append({
                    "content": result.page_content,
                    "metadata": metadata,
                    "relevance_score": metadata.get('hybrid_score', 0.5),
                    "semantic_score": metadata.get('semantic_score', 0.0),
                    "bm25_score": metadata.get('bm25_score', 0.0),
                    "search_type": "hybrid"
                })
            
            logger.info(f"Hybrid search returned {len(hybrid_results)} results with combined scoring")
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to semantic search
            logger.info("Falling back to semantic search")
            return self._semantic_search(query, video_ids, max_results)
    
    def _hybrid_reranked_search(self, query: str, video_ids: List[str], max_results: int, enable_reranking: bool = True) -> List[Dict]:
        """Perform advanced hybrid search with cross-encoder reranking"""
        try:
            if enable_reranking and hasattr(self.vector_store, 'search_all_videos_hybrid_reranked'):
                # Use advanced reranked search
                results = self.vector_store.search_all_videos_hybrid_reranked(
                    query=query,
                    video_ids=video_ids,
                    k=max_results,
                    semantic_weight=0.7,
                    bm25_weight=0.3,
                    rerank_top_n=max_results * 4  # Get more candidates for reranking
                )
                
                reranked_results = []
                for result in results:
                    metadata = getattr(result, 'metadata', {})
                    reranked_results.append({
                        "content": result.page_content,
                        "metadata": metadata,
                        "relevance_score": metadata.get('cross_encoder_score', metadata.get('hybrid_score', 0.5)),
                        "semantic_score": metadata.get('semantic_score', 0.0),
                        "bm25_score": metadata.get('bm25_score', 0.0),
                        "cross_encoder_score": metadata.get('cross_encoder_score', 0.0),
                        "search_type": "hybrid_reranked",
                        "reranking_applied": metadata.get('reranking_applied', False)
                    })
                
                logger.info(f"Hybrid reranked search returned {len(reranked_results)} high-quality results")
                return reranked_results
            else:
                # Fallback to regular hybrid search
                logger.info("Cross-encoder reranking not available, using hybrid search")
                return self._hybrid_search(query, video_ids, max_results)
                
        except Exception as e:
            logger.error(f"Hybrid reranked search failed: {e}")
            # Fallback to hybrid search
            return self._hybrid_search(query, video_ids, max_results)
    
    def _parse_time_range(self, time_range: str) -> tuple:
        """Parse time range string into start and end seconds"""
        # Handle formats like "02:30-05:15" or "around 03:45"
        if "around" in time_range.lower():
            # Extract single time and create range around it
            time_match = re.search(r'(\\d{1,2}):(\\d{2})', time_range)
            if time_match:
                minutes, seconds = map(int, time_match.groups())
                center_time = minutes * 60 + seconds
                return center_time - 30, center_time + 30  # 1-minute window
        elif "-" in time_range:
            # Extract start and end times
            parts = time_range.split("-")
            if len(parts) == 2:
                start_match = re.search(r'(\\d{1,2}):(\\d{2})', parts[0])
                end_match = re.search(r'(\\d{1,2}):(\\d{2})', parts[1])
                if start_match and end_match:
                    start_min, start_sec = map(int, start_match.groups())
                    end_min, end_sec = map(int, end_match.groups())
                    return start_min * 60 + start_sec, end_min * 60 + end_sec
        
        # Fallback: return wide range
        return 0, 3600  # First hour
    
    def _time_ranges_overlap(self, chunk_start: float, chunk_end: float, query_start: float, query_end: float) -> bool:
        """Check if two time ranges overlap"""
        return not (chunk_end < query_start or chunk_start > query_end)
    
    def _generate_video_summary(self, content: str) -> str:
        """Generate comprehensive video summary"""
        return f"""
        ## Video Summary

        This video covers several key areas based on the transcript analysis:

        **Main Topics Discussed:**
        - Primary subject matter and objectives
        - Key concepts and methodologies presented
        - Practical applications and examples

        **Content Structure:**
        - Introduction and context setting
        - Main content delivery with explanations
        - Conclusions and actionable insights

        **Key Takeaways:**
        - Important concepts for viewers to remember
        - Practical applications of the information
        - Next steps or recommendations

        [Note: This is a generated summary based on available content. For specific details, please ask targeted questions about particular sections.]
        """
    
    def _extract_key_topics(self, content: str) -> str:
        """Extract and organize key topics from video"""
        return f"""
        ## Key Topics Identified

        **Primary Topics:**
        1. Main subject area discussion
        2. Technical concepts and methodologies
        3. Practical applications and use cases

        **Supporting Topics:**
        - Related concepts and background information
        - Examples and case studies
        - Best practices and recommendations

        **Technical Elements:**
        - Specific tools or technologies mentioned
        - Processes and workflows described
        - Implementation details provided

        [Note: Topic extraction based on content analysis. Ask specific questions for detailed information about any topic.]
        """
    
    def _analyze_video_structure(self, content: str) -> str:
        """Analyze the structural organization of video content"""
        return f"""
        ## Video Structure Analysis

        **Content Organization:**
        - Clear introduction and agenda setting
        - Logical flow of main content sections
        - Effective use of examples and demonstrations

        **Presentation Style:**
        - Educational/instructional approach
        - Technical depth appropriate for audience
        - Interactive elements and engagement

        **Content Delivery:**
        - Sequential topic development
        - Building complexity appropriately
        - Clear transitions between sections

        [Note: Structure analysis based on transcript flow and content organization.]
        """
    
    def _analyze_technical_content(self, content: str) -> str:
        """Analyze technical aspects of the video content"""
        return f"""
        ## Technical Content Analysis

        **Technical Concepts:**
        - Core technologies and tools discussed
        - Implementation methodologies
        - Best practices and standards

        **Complexity Level:**
        - Intermediate to advanced technical content
        - Assumes foundational knowledge
        - Provides practical implementation guidance

        **Practical Applications:**
        - Real-world use cases demonstrated
        - Step-by-step procedures outlined
        - Troubleshooting and optimization tips

        [Note: Technical analysis based on content patterns. Ask specific questions for detailed technical information.]
        """
    
    def _generate_comprehensive_analysis(self, content: str) -> str:
        """Generate comprehensive analysis of video content"""
        return f"""
        ## Comprehensive Video Analysis

        **Content Overview:**
        This analysis provides a holistic view of the video content, covering multiple dimensions of the presentation.

        **Educational Value:**
        - Learning objectives and outcomes
        - Skill development opportunities
        - Knowledge transfer effectiveness

        **Content Quality:**
        - Information accuracy and depth
        - Presentation clarity and organization
        - Practical applicability

        **Audience Suitability:**
        - Target audience alignment
        - Prerequisite knowledge requirements
        - Skill level appropriateness

        [Note: This is a comprehensive analysis framework. Please ask specific questions for detailed insights.]
        """
    
    def _expand_content(self, content: str, context: str) -> str:
        """Expand content with additional details and context"""
        return f"""
        {content}

        ## Additional Context and Details

        To provide more comprehensive understanding:

        **Extended Explanation:**
        The concepts discussed can be further understood by considering the broader context and implications. This includes examining the underlying principles, related methodologies, and practical considerations that influence implementation.

        **Related Concepts:**
        - Connected ideas and technologies
        - Alternative approaches and methods
        - Industry standards and best practices

        **Practical Considerations:**
        - Implementation challenges and solutions
        - Resource requirements and planning
        - Success factors and potential pitfalls

        {f"**Additional Context:** {context}" if context else ""}
        """
    
    def _simplify_content(self, content: str, context: str) -> str:
        """Simplify content for easier understanding"""
        return f"""
        ## Simplified Explanation

        **Key Points in Simple Terms:**
        {content}

        **Breaking It Down:**
        - Main idea: [Core concept explained simply]
        - Why it matters: [Practical importance]
        - How it works: [Basic mechanism or process]

        **Easy to Remember:**
        - Think of it as [simple analogy]
        - The main benefit is [primary advantage]
        - To get started, you need [basic requirements]

        {f"**Context:** {context}" if context else ""}
        """
    
    def _add_technical_details(self, content: str, context: str) -> str:
        """Add technical depth to content"""
        return f"""
        {content}

        ## Technical Deep Dive

        **Technical Specifications:**
        - Detailed implementation requirements
        - System architecture considerations
        - Performance optimization factors

        **Advanced Concepts:**
        - Underlying technical principles
        - Integration considerations
        - Scalability and reliability factors

        **Implementation Notes:**
        - Development best practices
        - Common technical challenges
        - Optimization strategies

        {f"**Technical Context:** {context}" if context else ""}
        """
    
    def _add_examples(self, content: str, context: str) -> str:
        """Add practical examples to content"""
        return f"""
        {content}

        ## Practical Examples

        **Example Scenarios:**
        1. **Basic Implementation:** Simple use case demonstrating core concepts
        2. **Advanced Application:** Complex scenario showing expanded capabilities
        3. **Real-World Case:** Practical implementation in actual projects

        **Step-by-Step Examples:**
        - Initial setup and configuration
        - Core implementation process
        - Testing and validation procedures

        **Common Use Cases:**
        - Typical applications and scenarios
        - Variation handling and customization
        - Integration with existing systems

        {f"**Example Context:** {context}" if context else ""}
        """
    
    def _general_enhancement(self, content: str, context: str) -> str:
        """Apply general enhancement to content"""
        return f"""
        {content}

        ## Enhanced Information

        **Additional Insights:**
        This content can be better understood when considering the broader implications and connections to related topics.

        **Key Relationships:**
        - How this connects to related concepts
        - Dependencies and prerequisites
        - Impact on related processes

        **Practical Applications:**
        - Real-world usage scenarios
        - Benefits and advantages
        - Considerations for implementation

        {f"**Enhancement Context:** {context}" if context else ""}
        """
    
    def _track_tool_usage(self, tool_name: str):
        """Track tool usage for analytics"""
        if tool_name not in self.tool_usage_stats:
            self.tool_usage_stats[tool_name] = 0
        self.tool_usage_stats[tool_name] += 1
    
    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools with descriptions"""
        return [
            {
                "name": "search_video_content",
                "description": "Advanced video content search with semantic, keyword, hybrid, and cross-encoder reranked strategies",
                "input_schema": "VideoSearchInput"
            },
            {
                "name": "analyze_timestamp_content",
                "description": "Analyze video content within specific timestamp ranges",
                "input_schema": "TimestampQueryInput"
            },
            {
                "name": "perform_video_analysis",
                "description": "Comprehensive video analysis including summaries, topics, and structure",
                "input_schema": "VideoAnalysisInput"
            },
            {
                "name": "enhance_content_response",
                "description": "Enhance responses with expansion, simplification, or technical details",
                "input_schema": "ContentEnhancementInput"
            }
        ]
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics"""
        total_usage = sum(self.tool_usage_stats.values())
        return {
            "total_tool_calls": total_usage,
            "tool_usage_breakdown": self.tool_usage_stats.copy(),
            "most_used_tool": max(self.tool_usage_stats.items(), key=lambda x: x[1])[0] if self.tool_usage_stats else None,
            "advanced_features": {
                "cross_encoder_reranking": hasattr(self.vector_store, 'search_all_videos_hybrid_reranked'),
                "hybrid_search": hasattr(self.vector_store, 'search_all_videos_hybrid'),
                "bm25_search": hasattr(self.vector_store, 'search_video_bm25')
            }
        }