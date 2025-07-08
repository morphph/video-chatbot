#!/usr/bin/env python3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if Gemini API key is properly configured"""
    key = os.getenv('GEMINI_API_KEY')
    
    if not key or key == 'PASTE_YOUR_REAL_API_KEY_HERE':
        print("❌ API key not set properly")
        print("Please edit /Users/bytedance/Desktop/video_chatbot/backend/.env")
        print("Replace 'PASTE_YOUR_REAL_API_KEY_HERE' with your actual Gemini API key")
        print("Get your key from: https://aistudio.google.com/app/apikey")
        return False
    
    print("✅ API key is set")
    
    # Test API connectivity
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content('Say hello in one word')
        print("✅ Gemini API connectivity successful")
        print(f"Response: {response.text.strip()}")
        return True
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

if __name__ == "__main__":
    test_api_key()