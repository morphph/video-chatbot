#!/usr/bin/env python3

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

from vector_store import VectorStore

def debug_video_content(video_id):
    """Debug what content is stored for a video"""
    vector_store = VectorStore()
    
    # Try to search for general content
    results = vector_store.search_video_content(video_id, "what is this video about", k=10)
    
    print(f"=== Content for video {video_id} ===")
    print(f"Found {len(results)} chunks:")
    print()
    
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(f"Content: {doc.page_content[:500]}...")
        print(f"Metadata: {doc.metadata}")
        print()

if __name__ == "__main__":
    import sys
    video_id = sys.argv[1] if len(sys.argv) > 1 else "f8RnRuaxee8"
    debug_video_content(video_id)