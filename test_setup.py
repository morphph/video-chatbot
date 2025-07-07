#!/usr/bin/env python3
"""
Test the complete setup
"""
import os
import subprocess
import sys
import time
import threading

def load_env():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

def test_api_key():
    """Test if Gemini API key works"""
    load_env()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content("Say 'Hello from Gemini!' in exactly those words")
        print(f"✅ Gemini test: {response.text.strip()}")
        return True
    except ImportError:
        print("⚠️  google-generativeai not installed - will use demo mode")
        return True  # Still okay for demo
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

def check_frontend():
    """Check if frontend is accessible"""
    try:
        import urllib.request
        response = urllib.request.urlopen('http://localhost:3000', timeout=5)
        if response.getcode() == 200:
            print("✅ Frontend running at http://localhost:3000")
            return True
    except:
        pass
    
    print("⚠️  Frontend not running - start with: npm run dev")
    return False

def start_backend():
    """Start the minimal backend"""
    print("🚀 Starting minimal backend server...")
    try:
        # Run the minimal backend
        backend_process = subprocess.Popen([
            sys.executable, 'minimal_backend.py'
        ], cwd=os.path.dirname(__file__))
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if it's running
        if backend_process.poll() is None:
            print("✅ Backend started successfully")
            return backend_process
        else:
            print("❌ Backend failed to start")
            return None
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def main():
    print("🧪 Testing Complete Setup\n")
    
    # Test API key
    api_ok = test_api_key()
    print()
    
    # Check frontend
    frontend_ok = check_frontend()
    print()
    
    # Start backend
    backend_process = start_backend()
    backend_ok = backend_process is not None
    print()
    
    # Summary
    print("📊 Setup Summary:")
    print(f"   API Key: {'✅' if api_ok else '❌'}")
    print(f"   Frontend: {'✅' if frontend_ok else '⚠️'}")
    print(f"   Backend: {'✅' if backend_ok else '❌'}")
    
    if api_ok and backend_ok:
        print("\n🎉 Setup is working!")
        print("🌐 Access your app at:")
        print("   • Frontend: http://localhost:3000")
        print("   • Backend API: http://localhost:8000/health")
        print("   • Simple HTML: open simple-frontend.html in browser")
        
        if backend_process:
            try:
                input("\nPress Enter to stop the backend server...")
                backend_process.terminate()
                print("👋 Backend stopped")
            except KeyboardInterrupt:
                backend_process.terminate()
                print("\n👋 Backend stopped")
    else:
        print("\n⚠️  Some components need attention")
        if backend_process:
            backend_process.terminate()

if __name__ == "__main__":
    main()