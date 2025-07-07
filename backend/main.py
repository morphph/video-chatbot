from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import uvicorn
from dotenv import load_dotenv
import os
import asyncio

from video_processor import VideoProcessor
from vector_store import VectorStore
from conversation_service import ConversationService
from auth import get_current_user_optional
from validation import validate_youtube_url, validate_message_content, sanitize_input

load_dotenv()

app = FastAPI(title="YouTube Video Chatbot API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class VideoProcessRequest(BaseModel):
    url: HttpUrl

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None

class ChatRequest(BaseModel):
    video_id: str
    message: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[dict]
    conversation_id: str

# Initialize services
video_processor = VideoProcessor()
vector_store = VectorStore()
conversation_service = ConversationService(vector_store)

# In-memory storage for video processing status (should be Redis in production)
video_processing_status = {}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "video-chatbot-api"}

async def process_video_background(url: str, user_id: str):
    """Background task for video processing"""
    try:
        # Process the video
        result = video_processor.process_video(url)
        video_id = result["video_id"]
        
        # Update status
        video_processing_status[video_id] = {
            "status": "indexing",
            "metadata": result["metadata"]
        }
        
        # Create vector index
        success = vector_store.create_video_index(
            video_id=video_id,
            chunks=result["chunks"],
            metadata=result["metadata"]
        )
        
        # Update final status
        video_processing_status[video_id] = {
            "status": "completed" if success else "failed",
            "metadata": result["metadata"],
            "chunk_count": result["chunk_count"]
        }
        
    except Exception as e:
        if "video_id" in locals():
            video_processing_status[video_id] = {
                "status": "failed",
                "error": str(e)
            }

# Video processing endpoint
@app.post("/api/videos/process")
async def process_video(
    request: VideoProcessRequest, 
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user_optional)
):
    """Process a YouTube video and extract transcript"""
    try:
        # Validate YouTube URL
        if not validate_youtube_url(str(request.url)):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL format")
        
        # Extract video ID first to check if already processing
        video_id = video_processor.extract_video_id(str(request.url))
        if not video_id:
            raise HTTPException(status_code=400, detail="Could not extract video ID from URL")
        
        # Check if already processing
        if video_id in video_processing_status:
            return {
                "video_id": video_id,
                "status": video_processing_status[video_id]["status"],
                "message": "Video already processed or processing"
            }
        
        # Initialize status
        user_id = current_user.get("sub", "anonymous")
        video_processing_status[video_id] = {
            "status": "processing",
            "user_id": user_id
        }
        
        # Start background processing
        background_tasks.add_task(process_video_background, str(request.url), user_id)
        
        return {
            "video_id": video_id,
            "status": "processing",
            "message": "Video processing started"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user_optional)
):
    """Handle chat messages for a specific video"""
    try:
        # Validate message content
        if not validate_message_content(request.message):
            raise HTTPException(status_code=400, detail="Invalid message content")
        
        # Sanitize message input
        sanitized_message = sanitize_input(request.message)
        
        # Check if video is processed
        if request.video_id not in video_processing_status:
            raise HTTPException(status_code=404, detail="Video not found")
        
        if video_processing_status[request.video_id]["status"] != "completed":
            raise HTTPException(status_code=400, detail="Video processing not completed")
        
        # Create or continue conversation
        user_id = current_user.get("sub", "anonymous")
        if not request.conversation_id:
            # Create new conversation
            conversation_id = conversation_service.create_conversation(
                user_id=user_id,
                video_id=request.video_id,
                video_metadata=video_processing_status[request.video_id]["metadata"]
            )
        else:
            conversation_id = request.conversation_id
        
        # Process message
        response, sources = conversation_service.process_message(
            conversation_id=conversation_id,
            message=sanitized_message,
            user_id=user_id
        )
        
        return ChatResponse(
            response=response,
            sources=sources,
            conversation_id=conversation_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get video status
@app.get("/api/videos/{video_id}/status")
async def get_video_status(video_id: str):
    """Check the processing status of a video"""
    try:
        if video_id not in video_processing_status:
            raise HTTPException(status_code=404, detail="Video not found")
        
        status_info = video_processing_status[video_id]
        return {
            "video_id": video_id,
            "status": status_info["status"],
            "metadata": status_info.get("metadata", {}),
            "chunk_count": status_info.get("chunk_count", 0),
            "error": status_info.get("error")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Get conversation history
@app.get("/api/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str, 
    current_user: dict = Depends(get_current_user_optional)
):
    """Retrieve conversation history"""
    try:
        user_id = current_user.get("sub", "anonymous")
        history = conversation_service.get_conversation_history(conversation_id, user_id)
        return history
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)