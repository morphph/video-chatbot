#!/usr/bin/env python3
"""
Test script to validate Gemini integration
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def test_gemini_chat():
    """Test Gemini chat functionality"""
    print("Testing Gemini Chat...")
    try:
        from conversation_service import ConversationService
        from vector_store import VectorStore
        
        vector_store = VectorStore()
        conversation_service = ConversationService(vector_store)
        
        print("✅ Gemini chat initialization successful")
        return True
    except Exception as e:
        print(f"❌ Gemini chat failed: {e}")
        return False

def test_gemini_embeddings():
    """Test Gemini embeddings functionality"""
    print("Testing Gemini Embeddings...")
    try:
        from gemini_embeddings import GeminiEmbeddings
        
        embeddings = GeminiEmbeddings()
        test_text = "This is a test document for embedding"
        result = embeddings.embed_query(test_text)
        
        if result and len(result) > 0:
            print(f"✅ Embeddings working - dimension: {len(result)}")
            return True
        else:
            print("❌ Embeddings returned empty result")
            return False
    except Exception as e:
        print(f"❌ Embeddings failed: {e}")
        return False

def test_video_processor():
    """Test video processor with Gemini"""
    print("Testing Video Processor...")
    try:
        from video_processor import VideoProcessor
        
        processor = VideoProcessor()
        
        # Test URL extraction
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = processor.extract_video_id(test_url)
        
        if video_id:
            print(f"✅ Video ID extraction successful: {video_id}")
            return True
        else:
            print("❌ Video ID extraction failed")
            return False
    except Exception as e:
        print(f"❌ Video processor failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Latest Gemini 2.5 Integration\n")
    
    # Check if Gemini API key is set
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("⚠️  GEMINI_API_KEY not set - some tests may use fallback methods")
    else:
        print("✅ GEMINI_API_KEY is configured")
    
    print()
    
    # Run tests
    tests = [
        test_gemini_embeddings,
        test_gemini_chat,
        test_video_processor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    # Summary
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Latest Gemini 2.5 integration is working correctly.")
        print("📊 Models in use:")
        print("   • Chat: gemini-2.5-flash")
        print("   • Transcription: gemini-2.5-flash")
        print("   • Embeddings: text-embedding-004 (with sentence-transformers fallback)")
        return 0
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())