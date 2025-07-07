# Development Setup Guide

This guide will help you set up the YouTube Video Chatbot project on any computer for continued development.

## 🚀 Quick Start Checklist

### Prerequisites (Install These First)
- [ ] **Python 3.9+** - Download from [python.org](https://python.org)
- [ ] **Node.js 18+** - Download from [nodejs.org](https://nodejs.org) 
- [ ] **Git** - Download from [git-scm.com](https://git-scm.com)
- [ ] **Gemini API Key** - Get free key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Setup Steps

#### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd video_chatbot
```

#### 2. Set Up Environment Variables
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your actual API key
# Replace 'your_api_key_here' with your real Gemini API key
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

#### 3. Install Dependencies

**Backend:**
```bash
pip3 install fastapi uvicorn google-generativeai python-dotenv youtube-transcript-api pytube langchain langchain-google-genai chromadb sentence-transformers
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

#### 4. Test the Setup
```bash
# Test API key and basic functionality
python3 simple_test.py

# Should show: ✅ API key found and ✅ Gemini test response
```

#### 5. Start Development Servers

**Terminal 1 - Backend:**
```bash
# Option A: Full backend (requires all dependencies)
cd backend
python3 main.py

# Option B: Minimal backend (for quick testing)
python3 minimal_backend.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

#### 6. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Simple HTML**: Open `simple-frontend.html` in browser

## 🔧 Development Environment Details

### Project Structure
```
video_chatbot/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API server
│   ├── video_processor.py   # YouTube processing
│   ├── conversation_service.py # Chat logic
│   ├── gemini_embeddings.py # Embedding service
│   └── requirements.txt    # Python dependencies
├── frontend/               # Next.js frontend
│   ├── app/               # Next.js 15 app directory
│   ├── components/        # React components
│   └── package.json       # Node.js dependencies
├── minimal_backend.py     # Lightweight server for testing
├── simple-frontend.html   # Backup frontend
├── .env.example          # Environment template
└── .env                  # Your API keys (not in git)
```

### Environment Variables Required
```bash
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional (for production features)
CLERK_SECRET_KEY=your_clerk_secret_key_here
CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key_here
```

### Tech Stack Overview
- **Backend**: FastAPI + Python
- **AI Models**: Google Gemini 2.5 Flash + Text Embedding 004
- **Frontend**: Next.js 15 + React 18 + TypeScript + Tailwind CSS
- **Database**: Chroma (vector database) + local storage
- **Authentication**: Clerk (optional)

## 🧪 Testing & Development

### Quick Test Commands
```bash
# Test Gemini API connection
python3 simple_test.py

# Test complete setup
python3 test_setup.py

# Test backend endpoints
curl http://localhost:8000/health
```

### Development Modes

**Full Development Mode:**
- Uses all production dependencies
- Full feature set
- Requires internet for package installation

**Minimal Testing Mode:**
- Uses `minimal_backend.py`
- Works with minimal dependencies
- Good for quick testing and demos

## 🚨 Troubleshooting

### Common Issues

**"GEMINI_API_KEY not set"**
```bash
# Make sure .env file exists and has your key
cat .env
# Should show: GEMINI_API_KEY=AIza...
```

**"Backend not running"**
```bash
# Check if backend is running
curl http://localhost:8000/health
# or
python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

**"Module not found"**
```bash
# Install missing Python packages
pip3 install <missing-package-name>

# Or install all backend requirements
pip3 install -r backend/requirements.txt
```

**"npm install fails"**
```bash
# Clear npm cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

### Package Installation Issues
If you have network/firewall issues installing packages:
1. Use the minimal backend (`python3 minimal_backend.py`)
2. Use the simple HTML frontend (`simple-frontend.html`)
3. This provides core functionality for testing

## 🔄 Git Workflow

### Making Changes
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes...

# Commit changes
git add .
git commit -m "Description of changes"

# Push to remote
git push origin feature/your-feature-name
```

### Syncing Between Computers
```bash
# Pull latest changes
git pull origin main

# Install any new dependencies
pip3 install -r backend/requirements.txt
cd frontend && npm install && cd ..

# Update your .env file if needed
cp .env.example .env
# Edit .env with your API keys
```

## 📋 Development Checklist

When setting up on a new computer:
- [ ] Install Python 3.9+, Node.js 18+, Git
- [ ] Clone repository
- [ ] Create .env file with your API keys
- [ ] Install Python dependencies
- [ ] Install Node.js dependencies  
- [ ] Test with `python3 simple_test.py`
- [ ] Start backend and frontend servers
- [ ] Test with a YouTube URL

## 🎯 What's Working

✅ **Core Features:**
- YouTube video processing
- Gemini 2.5 Flash chat responses
- Vector embeddings with fallbacks
- React frontend with real-time chat
- CORS-enabled API

✅ **Development Tools:**
- Multiple backend options (full/minimal)
- Test scripts for validation
- Environment variable protection
- Git repository with proper .gitignore

## 🔮 Next Development Steps

Consider implementing:
- [ ] Database persistence (PostgreSQL)
- [ ] Redis caching
- [ ] User authentication (Clerk)
- [ ] Enhanced error handling
- [ ] Video metadata caching
- [ ] Multi-video conversations
- [ ] Production deployment scripts

This setup ensures you can continue development seamlessly on any computer!