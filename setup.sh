#!/bin/bash

# YouTube Video Chatbot Quick Setup Script
echo "🚀 Setting up YouTube Video Chatbot..."

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required. Download from https://python.org"
    exit 1
fi
echo "✅ Python 3 found"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required. Download from https://nodejs.org"
    exit 1
fi
echo "✅ Node.js found"

# Check if in git repo, if not initialize
if [ ! -d ".git" ]; then
    echo "📁 Initializing git repository..."
    git init
fi

# Create .env from example if it doesn't exist
if [ ! -f ".env" ]; then
    echo "🔧 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your Gemini API key!"
    echo "   Get your key from: https://aistudio.google.com/app/apikey"
else
    echo "✅ .env file exists"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install fastapi uvicorn google-generativeai python-dotenv youtube-transcript-api pytube langchain langchain-google-genai chromadb sentence-transformers || {
    echo "⚠️  Some Python packages failed to install. You can still use minimal_backend.py"
}

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
cd frontend
npm install || {
    echo "⚠️  Frontend dependencies failed to install"
    cd ..
    exit 1
}
cd ..

# Test the setup
echo "🧪 Testing setup..."
python3 simple_test.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit .env and add your Gemini API key"
echo "2. Start backend: python3 minimal_backend.py"
echo "3. Start frontend: cd frontend && npm run dev"
echo "4. Open http://localhost:3000"
echo ""
echo "📚 See DEVELOPMENT_SETUP.md for detailed instructions"