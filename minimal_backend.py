#!/usr/bin/env python3
"""
Minimal backend server for testing - works without external dependencies
"""
import json
import os
import sys
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time

# Load environment variables from .env file
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

load_env()

class VideoProcessor:
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:watch\?v=)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def get_mock_metadata(self, video_id):
        """Return mock metadata for testing"""
        return {
            "title": f"Demo Video ({video_id})",
            "author": "Demo Channel",
            "duration": 300,
            "description": "This is a demo video for testing the chatbot functionality."
        }

class ChatbotHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.video_processor = VideoProcessor()
        self.video_status = {}
        super().__init__(*args, **kwargs)

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def send_cors_headers(self):
        """Send CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_GET(self):
        """Handle GET requests"""        
        if self.path == '/health':
            self.send_response(200)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "service": "minimal-video-chatbot-api"}
            self.wfile.write(json.dumps(response).encode())
            
        elif self.path.startswith('/api/videos/') and self.path.endswith('/status'):
            # Extract video_id from path
            video_id = self.path.split('/')[-2]
            
            self.send_response(200)
            self.send_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            if video_id in self.video_status:
                self.wfile.write(json.dumps(self.video_status[video_id]).encode())
            else:
                response = {"error": "Video not found"}
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests"""
        
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            if self.path == '/api/videos/process':
                url = data.get('url')
                video_id = self.video_processor.extract_video_id(url)
                
                if not video_id:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {"detail": "Invalid YouTube URL"}
                    self.wfile.write(json.dumps(response).encode())
                    return
                
                # Mock processing
                metadata = self.video_processor.get_mock_metadata(video_id)
                self.video_status[video_id] = {
                    "video_id": video_id,
                    "status": "completed",
                    "metadata": metadata,
                    "chunk_count": 5
                }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "video_id": video_id,
                    "status": "completed",
                    "message": "Video processed successfully (demo mode)"
                }
                self.wfile.write(json.dumps(response).encode())
                
            elif self.path == '/api/chat':
                video_id = data.get('video_id')
                message = data.get('message')
                conversation_id = data.get('conversation_id', 'demo_conversation')
                
                # Mock chat response
                api_key = os.getenv('GEMINI_API_KEY')
                if api_key:
                    try:
                        # Try to use real Gemini API if available
                        import google.generativeai as genai
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-2.5-flash')
                        
                        # Create a context-aware prompt
                        if video_id in self.video_status:
                            video_info = self.video_status[video_id]['metadata']
                            prompt = f"""You are helping a user understand a YouTube video titled "{video_info['title']}" by {video_info['author']}. 
                            
User question: {message}

Please provide a helpful response about this video. Since this is a demo, you can provide general information about the topic or suggest what might typically be discussed in such a video."""
                        else:
                            prompt = f"User asked: {message}\n\nPlease provide a helpful response about YouTube videos in general."
                        
                        response_text = model.generate_content(prompt).text
                        
                    except Exception as e:
                        response_text = f"I'm a demo chatbot! You asked: '{message}'. In a real setup with video processing, I would analyze the actual video content and provide specific answers based on the transcript."
                else:
                    response_text = f"Demo response: I understand you're asking about '{message}'. With proper video processing, I would analyze the actual video content to give you specific answers!"
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    "response": response_text,
                    "sources": [{"chunk_index": 0, "content_preview": "Demo source from video content..."}],
                    "conversation_id": conversation_id
                }
                self.wfile.write(json.dumps(response).encode())
                
            else:
                self.send_response(404)
                self.end_headers()
                
        except Exception as e:
            print(f"Error: {e}")
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"detail": str(e)}
            self.wfile.write(json.dumps(response).encode())

def run_server(port=8000):
    """Run the minimal backend server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, ChatbotHandler)
    print(f"🚀 Minimal backend server running at http://localhost:{port}")
    print("📊 Features available:")
    print("   • Health check: GET /health")
    print("   • Video processing: POST /api/videos/process") 
    print("   • Chat: POST /api/chat")
    print("   • Video status: GET /api/videos/{video_id}/status")
    
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        print(f"🔑 Gemini API key: {api_key[:10]}... (configured)")
    else:
        print("⚠️  No Gemini API key found - running in basic demo mode")
    
    print("\n🌐 Test with: http://localhost:3000 or open simple-frontend.html")
    print("Press Ctrl+C to stop\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        httpd.shutdown()

if __name__ == "__main__":
    run_server()