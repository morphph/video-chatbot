#!/usr/bin/env python3
"""
Enhanced Phase 2 Server Startup Script
Starts the YouTube Video Chatbot with all advanced features
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import and run the server
if __name__ == "__main__":
    print("🚀 Starting Phase 2 Enhanced YouTube Video Chatbot Server...")
    print("✅ Features enabled:")
    print("   • Gemini 2.5 Flash multimodal video processing")
    print("   • BM25 + Semantic + Cross-Encoder hybrid search")
    print("   • LangGraph StateGraph agentic orchestration")
    print("   • Context-aware query enhancement")
    print("   • Intelligent conversation memory compression")
    print("   • Advanced tool-calling with video analysis")
    print()
    
    # Run the FastAPI server
    import uvicorn
    from main import app
    
    print("🌐 Server starting at http://localhost:8000")
    print("📚 API documentation at http://localhost:8000/docs")
    print("🎥 Frontend will be available at http://localhost:3000")
    print()
    print("Press Ctrl+C to stop the server")
    print()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)