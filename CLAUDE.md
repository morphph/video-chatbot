# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Backend Development
```bash
# Start full backend server (requires all dependencies)
cd backend && python main.py

# Start lightweight backend for testing
python minimal_backend.py

# Run tests
python simple_test.py  # Test Gemini API connection
python test_setup.py   # Test complete setup
cd backend && python test_api.py  # Test API endpoints

# Install/update dependencies
pip install -r backend/requirements.txt
```

### Frontend Development
```bash
# Start frontend development server
cd frontend && npm run dev

# Build for production
cd frontend && npm run build

# Lint frontend code
cd frontend && npm run lint

# Install/update dependencies
cd frontend && npm install
```

### Quick Setup
```bash
# Use automated setup script
./setup.sh

# Manual setup
cp .env.example .env
# Add GEMINI_API_KEY to .env file
```

## Architecture Overview

### Backend (FastAPI + LangChain + Gemini)
- **main.py**: FastAPI application with CORS, health checks, and API endpoints
- **video_processor.py**: YouTube transcript extraction with Gemini 2.5 Flash fallback
- **conversation_service.py**: RAG-based chat using LangChain and Gemini 2.5 Flash
- **vector_store.py**: Chroma vector database interface for semantic search
- **gemini_embeddings.py**: Google text-embedding-004 integration with sentence-transformers fallback
- **auth.py**: Optional Clerk authentication integration
- **validation.py**: Input validation and security checks

The backend uses a Retrieval-Augmented Generation (RAG) architecture:
1. Video transcripts are chunked and embedded using text-embedding-004
2. Embeddings are stored in Chroma vector database
3. User queries trigger semantic search to find relevant chunks
4. Gemini 2.5 Flash generates responses with retrieved context

### Frontend (Next.js 15 + TypeScript)
- **app/**: Next.js App Router structure
- **components/**: React components (chat-interface, video-input, auth-header)
- **lib/api.ts**: Centralized API client for backend communication
- **middleware.ts**: Next.js middleware for auth and routing

The frontend provides a real-time chat interface with:
- YouTube URL input and processing status
- Streaming chat responses with source attribution
- Optional Clerk authentication
- Responsive Tailwind CSS design

### Key Dependencies
- **Gemini 2.5 Flash**: Primary conversation model (thinking-enabled)
- **text-embedding-004**: State-of-the-art embeddings (multilingual)
- **Chroma**: Local vector database (can upgrade to hosted)
- **youtube-transcript-api**: Primary transcript extraction
- **sentence-transformers**: Free local embedding fallback

## Environment Variables

Required:
- `GEMINI_API_KEY`: Google AI Studio API key (get free at https://aistudio.google.com/app/apikey)

Optional:
- `CLERK_SECRET_KEY`: Backend Clerk authentication
- `CLERK_PUBLISHABLE_KEY`: Frontend Clerk authentication
- `DATABASE_URL`: PostgreSQL connection (for production)
- `REDIS_URL`: Redis connection (for caching)

Frontend-specific:
- `NEXT_PUBLIC_API_URL`: Backend API URL (default: http://localhost:8000)
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`: Clerk public key

## Development Workflow

### Adding New Features
1. Backend endpoints: Add to `main.py`, implement logic in separate modules
2. Frontend components: Create in `components/`, integrate via `lib/api.ts`
3. Vector operations: Extend `vector_store.py` for new embedding features
4. Always validate inputs using `validation.py` patterns

### Testing Strategy
- Use `minimal_backend.py` for rapid iteration without full dependencies
- Test with `simple-frontend.html` when Next.js setup is problematic
- Run `test_api.py` after backend changes
- Always test with actual YouTube URLs

### Common Tasks

#### Processing a New Video
1. Validate URL format and security
2. Extract transcript (YouTube API → Gemini fallback)
3. Chunk text into ~1000 char segments with overlap
4. Generate embeddings (text-embedding-004 → sentence-transformers fallback)
5. Store in Chroma with metadata

#### Handling Chat Requests
1. Validate message and video_id
2. Perform semantic search in Chroma
3. Assemble context from top-k chunks
4. Generate response with Gemini 2.5 Flash
5. Include source attributions with timestamps

## Important Notes

- **Model Selection**: Uses Gemini 2.5 Flash (1.5 models deprecated April 2025)
- **Cost Optimization**: Gemini provides 70%+ cost reduction vs OpenAI
- **Fallback Strategy**: Always implement fallbacks (embeddings, transcripts)
- **Rate Limits**: Be mindful of YouTube API and Gemini quotas
- **Security**: Never expose API keys, validate all inputs
- **Performance**: Chunk size affects retrieval quality (default ~1000 chars)