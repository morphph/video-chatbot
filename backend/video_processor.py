import os
from typing import Optional, List, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain.schema import Document
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self):
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_video_id(self, url: str) -> Optional[str]:
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
    
    def get_video_metadata(self, url: str) -> Dict[str, Any]:
        """Extract video metadata"""
        try:
            yt = YouTube(url)
            return {
                "title": yt.title,
                "description": yt.description,
                "duration": yt.length,
                "author": yt.author,
                "thumbnail_url": yt.thumbnail_url,
                "publish_date": yt.publish_date.isoformat() if yt.publish_date else None
            }
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}
    
    def get_transcript_youtube_api(self, video_id: str) -> Optional[str]:
        """Try to get transcript using youtube-transcript-api"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = " ".join([entry['text'] for entry in transcript_list])
            return transcript
        except Exception as e:
            logger.warning(f"YouTube API transcript extraction failed: {e}")
            return None
    
    def get_transcript_langchain(self, url: str) -> Optional[str]:
        """Try to get transcript using LangChain's YoutubeLoader"""
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            documents = loader.load()
            if documents:
                return " ".join([doc.page_content for doc in documents])
            return None
        except Exception as e:
            logger.warning(f"LangChain transcript extraction failed: {e}")
            return None
    
    def generate_transcript_gemini(self, url: str, video_metadata: Dict[str, Any]) -> Optional[str]:
        """Generate transcript using Gemini as fallback based on video metadata"""
        if not self.gemini_api_key:
            logger.error("Gemini API key not configured")
            return None
        
        try:
            # Use video metadata to create a helpful context
            title = video_metadata.get('title', 'Unknown video')
            description = video_metadata.get('description', '')[:500]  # Limit description length
            
            prompt = f"""Based on the YouTube video metadata below, please provide a helpful summary that can serve as a transcript substitute:

Title: {title}
Description: {description}

Please create a detailed summary that covers the main topics, key points, and important information that would likely be discussed in this video. Format it as if it were a transcript with natural speech patterns."""
            
            response = self.gemini_model.generate_content(prompt)
            logger.info("Generated synthetic transcript using Gemini based on video metadata")
            return response.text
        except Exception as e:
            logger.error(f"Gemini transcript generation failed: {e}")
            return None
    
    def chunk_transcript(self, transcript: str) -> List[Document]:
        """Split transcript into chunks for vector storage"""
        documents = [Document(page_content=transcript)]
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_video(self, url: str) -> Dict[str, Any]:
        """Main processing function for YouTube videos"""
        video_id = self.extract_video_id(url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        # Get metadata
        metadata = self.get_video_metadata(url)
        
        # Try to get transcript (with fallbacks)
        transcript = (
            self.get_transcript_youtube_api(video_id) or
            self.get_transcript_langchain(url) or
            self.generate_transcript_gemini(url, metadata)
        )
        
        if not transcript:
            raise ValueError("Could not extract transcript from video")
        
        # Chunk the transcript
        chunks = self.chunk_transcript(transcript)
        
        return {
            "video_id": video_id,
            "url": url,
            "metadata": metadata,
            "transcript": transcript,
            "chunks": [chunk.page_content for chunk in chunks],
            "chunk_count": len(chunks)
        }