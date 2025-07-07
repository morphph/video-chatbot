#!/usr/bin/env python3
"""
Simple test to verify Gemini API key works
"""
import os
import sys

def test_gemini_key():
    print("🧪 Testing Gemini API Key...\n")
    
    # Check if API key is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("\n📝 To set your API key:")
        print("export GEMINI_API_KEY=your_api_key_here")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        # Try to import google.generativeai
        import google.generativeai as genai
        print("✅ google-generativeai package available")
        
        # Configure and test
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Simple test
        response = model.generate_content("Say hello in exactly 3 words")
        print(f"✅ Gemini 2.5 Flash response: {response.text}")
        
        return True
        
    except ImportError:
        print("❌ google-generativeai package not installed")
        print("Install with: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_key()
    if success:
        print("\n🎉 Gemini API is working! You can now run the full application.")
    else:
        print("\n⚠️  Please fix the issues above before running the application.")
    
    sys.exit(0 if success else 1)