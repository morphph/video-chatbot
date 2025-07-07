# YouTube Video Chatbot

An intelligent chatbot platform that allows users to interact conversationally with YouTube video content using natural language queries. Built with FastAPI, React, LangChain, and powered by Google's latest Gemini 2.5 models.

## Features

- **YouTube Video Processing**: Automatic transcript extraction with Gemini 2.5 Flash fallback
- **Advanced AI**: Powered by Google's latest Gemini 2.5 Flash with built-in thinking capabilities
- **State-of-the-Art Embeddings**: Uses text-embedding-004 for superior retrieval performance
- **RAG Architecture**: Retrieval-Augmented Generation using Chroma vector database
- **Multilingual Support**: Enhanced support for 100+ languages via latest embedding models
- **User Authentication**: Secure authentication via Clerk
- **Real-time Processing**: Background video processing with status updates
- **Source Attribution**: Responses include references to specific video timestamps
- **Cost-Effective**: 70%+ cost reduction compared to OpenAI with superior performance
- **Robust Fallbacks**: Multiple fallback strategies including free local models

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **LangChain**: AI/ML pipeline orchestration
- **Google Gemini 2.5 Flash**: Latest thinking model for fast, intelligent conversation
- **Text Embedding 004**: Google's state-of-the-art embedding model
- **Chroma**: Vector database for semantic search
- **Clerk**: User authentication and management
- **YouTube Transcript API**: Primary transcript extraction
- **Sentence Transformers**: Free local embedding fallback

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Clerk**: Frontend authentication components

## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- **Google AI Studio API key** (get free at https://aistudio.google.com/app/apikey)
- Clerk account (optional for development)

### Why Gemini 2.5?
- **Latest Models**: Uses Gemini 2.5 Flash (1.5 models deprecated April 29, 2025)
- **Superior Performance**: Built-in thinking capabilities and faster responses
- **Cost Effective**: Significantly cheaper than OpenAI while maintaining quality
- **Future Proof**: Latest stable models with ongoing Google support

### Backend Setup

1. **Clone and navigate to backend**
   ```bash
   cd backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment configuration**
   ```bash
   cp ../.env.example .env
   ```
   
   Fill in your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   CLERK_SECRET_KEY=your_clerk_secret_key_here  # Optional
   ```

4. **Start the server**
   ```bash
   python main.py
   ```
   
   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Environment configuration**
   ```bash
   echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
   echo "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key" >> .env.local
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```
   
   The frontend will be available at `http://localhost:3000`

## Usage

1. **Access the application** at `http://localhost:3000`
2. **Sign in** (optional) using Clerk authentication
3. **Enter a YouTube URL** in the input field
4. **Wait for processing** - the system will extract and index the video transcript
5. **Start chatting** - ask questions about the video content
6. **Get responses** with source attribution and timestamps

## API Documentation

### Core Endpoints

#### Process Video
```http
POST /api/videos/process
Content-Type: application/json

{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID"
}
```

#### Send Chat Message
```http
POST /api/chat
Content-Type: application/json

{
  "video_id": "VIDEO_ID",
  "message": "What is this video about?",
  "conversation_id": "optional_conversation_id"
}
```

#### Check Video Status
```http
GET /api/videos/{video_id}/status
```

#### Get Conversation History
```http
GET /api/conversations/{conversation_id}
```

### Authentication

All endpoints support optional Clerk authentication. Include the JWT token in the Authorization header:

```http
Authorization: Bearer YOUR_JWT_TOKEN
```

## Architecture

### Video Processing Pipeline
1. **URL Validation**: Validates YouTube URL format and security
2. **Transcript Extraction**: Primary via YouTube API, fallback to Gemini 2.5 Flash
3. **Text Chunking**: Splits transcript into semantically meaningful chunks
4. **Embedding Generation**: Creates vector embeddings using text-embedding-004
5. **Vector Storage**: Stores embeddings in Chroma database with metadata

### Conversation Flow
1. **Query Processing**: Validates and sanitizes user input
2. **Semantic Search**: Finds relevant transcript chunks using advanced embeddings
3. **Context Assembly**: Combines relevant chunks with conversation memory
4. **Response Generation**: Uses Gemini 2.5 Flash with thinking capabilities
5. **Source Attribution**: Links responses to specific video timestamps

### Model Selection Strategy
- **Gemini 2.5 Flash**: Optimal balance of speed, cost, and intelligence
- **Text Embedding 004**: Best-in-class retrieval performance with multilingual support
- **Fallback Models**: Sentence-transformers ensures reliability without API dependency

## Development

### Project Structure
```
video_chatbot/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── video_processor.py   # YouTube processing logic
│   ├── vector_store.py      # Chroma database interface
│   ├── conversation_service.py # Chat logic
│   ├── auth.py             # Clerk authentication
│   ├── validation.py       # Input validation
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── app/                # Next.js app directory
│   ├── components/         # React components
│   ├── lib/               # API client and utilities
│   └── package.json       # Node.js dependencies
└── .env.example           # Environment template
```

### Adding New Features

1. **Backend**: Add new endpoints in `main.py` and supporting logic in separate modules
2. **Frontend**: Create new components and integrate with the API client in `lib/api.ts`
3. **Database**: Extend vector store operations in `vector_store.py`

## Deployment

### Backend (Recommended: Railway, Heroku, or DigitalOcean)
1. Set environment variables in your hosting platform
2. Install dependencies and start with `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Frontend (Vercel - Recommended)
1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Environment Variables for Production
```env
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional but recommended
CLERK_SECRET_KEY=your_clerk_secret_key
CLERK_PUBLISHABLE_KEY=your_clerk_publishable_key

# Database (for production scaling)
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port/db
```

## Capabilities & Limitations

### ✅ **Capabilities**
- **Multilingual Support**: 100+ languages via text-embedding-004
- **Advanced Reasoning**: Built-in thinking capabilities with Gemini 2.5 Flash
- **Robust Fallbacks**: Multiple model fallback strategies for reliability
- **Cost Optimization**: Significant cost savings compared to OpenAI solutions
- **Future Proof**: Uses latest stable Google AI models

### ⚠️ **Current Limitations**
- **Video Length**: Optimized for videos under 2 hours
- **Rate Limits**: Subject to YouTube API and Gemini API quotas
- **Storage**: Uses local Chroma database (consider hosted solutions for production)
- **Model Availability**: Requires Gemini API access (free tier available)

## Troubleshooting

### Common Issues

**"Invalid YouTube URL"**
- Ensure the URL is a valid YouTube video link
- Supported formats: youtube.com/watch?v=ID, youtu.be/ID

**"Video processing failed"**
- Check if the video has captions/subtitles available
- Verify your API keys are correctly set
- Some videos may be age-restricted or private

**"Authentication failed"**
- Verify Clerk keys are correctly configured
- Check that the frontend and backend are using matching Clerk settings

**"Model not found" or "API Error"**
- Ensure you're using a valid Gemini API key from AI Studio
- Check if you have access to Gemini 2.5 models
- Verify your API quota hasn't been exceeded

### Debug Mode
Set `LOG_LEVEL=DEBUG` in your environment to enable detailed logging.

### Model Testing
Run the test script to validate your Gemini integration:
```bash
cd backend
python test_gemini.py
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Support

For issues and questions:
- Check the [GitHub Issues](https://github.com/your-repo/issues)
- Review the troubleshooting section above
- Ensure all API keys are properly configured