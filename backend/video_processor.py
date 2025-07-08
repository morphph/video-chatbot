import os
from typing import Optional, List, Dict, Any, Tuple
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
        """Extract video metadata using multiple methods"""
        metadata = {}
        
        # Try YouTube oEmbed API first (no authentication required)
        try:
            import requests
            oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
            response = requests.get(oembed_url, timeout=10)
            if response.status_code == 200:
                oembed_data = response.json()
                metadata = {
                    "title": oembed_data.get("title", ""),
                    "author": oembed_data.get("author_name", ""),
                    "thumbnail_url": oembed_data.get("thumbnail_url", ""),
                    "description": "",  # oEmbed doesn't provide description
                    "duration": None,   # oEmbed doesn't provide duration
                    "publish_date": None
                }
                logger.info(f"oEmbed metadata extracted: {metadata.get('title', 'No title')}")
                return metadata
        except Exception as e:
            logger.warning(f"oEmbed extraction failed: {e}")
        
        # Fallback to PyTube
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
    
    def get_transcript_youtube_api(self, video_id: str) -> Optional[Tuple[str, List[Dict]]]:
        """Try to get transcript using youtube-transcript-api, returning both text and timestamp data"""
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Extract text and preserve timestamp information
            transcript_text = " ".join([entry['text'] for entry in transcript_list])
            
            # Format timestamp data for enhanced chunking
            timestamp_data = []
            for entry in transcript_list:
                timestamp_data.append({
                    'text': entry['text'],
                    'start': entry.get('start', 0),
                    'duration': entry.get('duration', 0),
                    'end': entry.get('start', 0) + entry.get('duration', 0)
                })
            
            return transcript_text, timestamp_data
        except Exception as e:
            logger.warning(f"YouTube API transcript extraction failed: {e}")
            return None
    
    def get_transcript_langchain(self, url: str) -> Optional[Tuple[str, List[Dict]]]:
        """Try to get transcript using LangChain's YoutubeLoader"""
        try:
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
            documents = loader.load()
            if documents:
                transcript_text = " ".join([doc.page_content for doc in documents])
                # No timestamp data available from LangChain
                return transcript_text, []
            return None
        except Exception as e:
            logger.warning(f"LangChain transcript extraction failed: {e}")
            return None
    
    def get_transcript_gemini_direct(self, url: str) -> Optional[Tuple[str, List[Dict]]]:
        """Extract transcript using Gemini's direct YouTube URL processing"""
        if not self.gemini_api_key:
            logger.error("Gemini API key not configured")
            return None
        
        try:
            # Use Gemini's direct YouTube URL processing capability
            logger.info(f"Attempting Gemini direct YouTube processing for: {url}")
            
            # Create the content with YouTube URL and transcription request
            content = [
                {
                    "file_data": {
                        "file_uri": url
                    }
                },
                """Please provide a complete, accurate transcript of this video with approximate timestamps. 

Format the response as follows:
- Include all spoken content with proper punctuation and paragraph breaks
- Add approximate timestamps in [MM:SS] format throughout the transcript
- Group related content into logical segments
- Do not include explanatory text or summaries - just the transcript with timestamps

Example format:
[00:00] Welcome to this video about...
[00:30] Today we'll be discussing...
[01:15] The first point I want to make is..."""
            ]
            
            response = self.gemini_model.generate_content(content)
            
            if response and response.text:
                logger.info(f"Gemini direct transcript successful, length: {len(response.text)}")
                
                # Parse timestamps from Gemini response if available
                timestamp_data = self._parse_timestamps_from_text(response.text)
                
                return response.text, timestamp_data
            else:
                logger.warning("Gemini direct processing returned empty response")
                return None
                
        except Exception as e:
            logger.warning(f"Gemini direct YouTube processing failed: {e}")
            # Check if it's a public video requirement issue
            if "public" in str(e).lower() or "private" in str(e).lower():
                logger.warning("Video may not be public - Gemini requires public videos")
            return None
    
    def _parse_timestamps_from_text(self, text: str) -> List[Dict]:
        """Parse timestamp data from Gemini-generated transcript text"""
        import re
        
        timestamp_data = []
        # Pattern to match [MM:SS] or [HH:MM:SS] timestamps
        timestamp_pattern = r'\[(\d{1,2}:\d{2}(?::\d{2})?)\]'
        
        lines = text.split('\n')
        current_position = 0
        
        for line in lines:
            if line.strip():
                # Look for timestamp in the line
                timestamp_match = re.search(timestamp_pattern, line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    # Convert timestamp to seconds
                    time_parts = timestamp_str.split(':')
                    if len(time_parts) == 2:  # MM:SS
                        start_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
                    elif len(time_parts) == 3:  # HH:MM:SS
                        start_seconds = int(time_parts[0]) * 3600 + int(time_parts[1]) * 60 + int(time_parts[2])
                    else:
                        start_seconds = current_position
                    
                    # Remove timestamp from text
                    clean_text = re.sub(timestamp_pattern, '', line).strip()
                    
                    if clean_text:
                        timestamp_data.append({
                            'text': clean_text,
                            'start': start_seconds,
                            'duration': 30,  # Estimate 30 seconds per segment
                            'end': start_seconds + 30
                        })
                        current_position = start_seconds + 30
        
        return timestamp_data
    
    def generate_transcript_gemini(self, url: str, video_metadata: Dict[str, Any]) -> Optional[Tuple[str, List[Dict]]]:
        """Generate transcript using Gemini as fallback based on video metadata"""
        if not self.gemini_api_key:
            logger.error("Gemini API key not configured")
            return None
        
        try:
            # Extract video ID for context
            video_id = self.extract_video_id(url)
            title = video_metadata.get('title', 'Unknown video')
            description = video_metadata.get('description', '')[:500]  # Limit description length
            
            # Create an honest response when video content is not accessible
            if title == 'Unknown video' or not title:
                prompt = f"""I need to create an honest and helpful response about a YouTube video that I cannot directly access.

Video ID: {video_id}
URL: {url}

The video's metadata and transcript could not be extracted due to API limitations or restrictions. Instead of creating fictional content, please generate a transparent response that:

1. **Clearly explains the limitation**: Acknowledge that the actual video content cannot be accessed
2. **Provides helpful alternatives**: Suggest practical ways users can get information about this video
3. **Offers general guidance**: Give useful tips for understanding YouTube videos when direct access isn't available
4. **Remains helpful**: Be supportive and constructive rather than just saying "I can't help"

The response should be honest, helpful, and never present fictional content as if it were real video information. Make it clear that you cannot watch or analyze the actual video content.

Keep the response concise (300-500 words) and focused on being genuinely helpful while maintaining transparency about limitations."""
            else:
                prompt = f"""Based on the YouTube video metadata below, create a helpful summary about what this video likely contains.

Title: {title}
Description: {description}
Author: {video_metadata.get('author', 'Unknown')}
URL: {url}

IMPORTANT: Since the actual video transcript is not available, you are creating content based only on the metadata above. Please:

1. **Be transparent**: Start by clearly stating this is based on metadata, not actual video content
2. **Be helpful**: Provide useful information about what viewers can expect
3. **Be accurate**: Base your summary on the real title and description provided
4. **Add disclaimer**: Include a note that this is generated content, not a real transcript

Create a helpful summary (300-500 words) that accurately reflects what this video is likely about based on its title and metadata. Make it clear this is synthetic content based on available information."""
            
            response = self.gemini_model.generate_content(prompt)
            logger.info("Generated synthetic transcript using Gemini based on video metadata")
            # No timestamp data for metadata-based generation
            return response.text, []
        except Exception as e:
            logger.error(f"Gemini transcript generation failed: {e}")
            return None
    
    def chunk_transcript_enhanced(
        self, 
        transcript: str, 
        timestamp_data: List[Dict], 
        video_metadata: Dict[str, Any],
        transcript_method: str
    ) -> List[Document]:
        """Enhanced chunking with timestamp awareness and semantic boundaries"""
        
        if timestamp_data:
            # Use timestamp-aware chunking
            return self._chunk_with_timestamps(transcript, timestamp_data, video_metadata, transcript_method)
        else:
            # Fall back to semantic chunking
            return self._chunk_semantic(transcript, video_metadata, transcript_method)
    
    def _chunk_with_timestamps(
        self, 
        transcript: str, 
        timestamp_data: List[Dict], 
        video_metadata: Dict[str, Any],
        transcript_method: str
    ) -> List[Document]:
        """Create chunks based on timestamp boundaries"""
        chunks = []
        
        # Group timestamp entries into logical segments
        segments = self._group_timestamp_segments(timestamp_data)
        
        for i, segment in enumerate(segments):
            # Combine text from segment entries
            segment_text = " ".join([entry['text'] for entry in segment])
            
            # Get time boundaries
            start_time = segment[0]['start']
            end_time = segment[-1]['end']
            
            # Format timestamp for display
            start_formatted = self._seconds_to_timestamp(start_time)
            end_formatted = self._seconds_to_timestamp(end_time)
            
            # Create enhanced metadata
            metadata = {
                'video_id': video_metadata.get('video_id', ''),
                'chunk_index': i,
                'chunk_type': 'timestamp_segment',
                'start_time': start_time,
                'end_time': end_time,
                'timestamp_range': f"{start_formatted}-{end_formatted}",
                'duration': end_time - start_time,
                'title': video_metadata.get('title', ''),
                'author': video_metadata.get('author', ''),
                'transcript_method': transcript_method,
                'segment_length': len(segment_text)
            }
            
            # Add timestamp prefix to content for better context
            content_with_timestamp = f"[{start_formatted}-{end_formatted}] {segment_text}"
            
            chunks.append(Document(
                page_content=content_with_timestamp,
                metadata=metadata
            ))
        
        logger.info(f"Created {len(chunks)} timestamp-aware chunks")
        return chunks
    
    def _chunk_semantic(
        self, 
        transcript: str, 
        video_metadata: Dict[str, Any],
        transcript_method: str
    ) -> List[Document]:
        """Create semantic chunks with sentence boundaries"""
        # Use enhanced text splitter that respects sentence boundaries
        enhanced_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Slightly smaller for better semantic coherence
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            add_start_index=True
        )
        
        # Create base document
        base_doc = Document(page_content=transcript)
        chunks = enhanced_splitter.split_documents([base_doc])
        
        # Enhance metadata for each chunk
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Estimate timestamp based on position in text
            estimated_start = (i * 60)  # Rough estimate: 1 minute per chunk
            estimated_end = estimated_start + 60
            
            metadata = {
                'video_id': video_metadata.get('video_id', ''),
                'chunk_index': i,
                'chunk_type': 'semantic_segment',
                'estimated_start_time': estimated_start,
                'estimated_end_time': estimated_end,
                'timestamp_range': f"~{self._seconds_to_timestamp(estimated_start)}",
                'title': video_metadata.get('title', ''),
                'author': video_metadata.get('author', ''),
                'transcript_method': transcript_method,
                'segment_length': len(chunk.page_content)
            }
            
            # Update chunk metadata
            chunk.metadata.update(metadata)
            enhanced_chunks.append(chunk)
        
        logger.info(f"Created {len(enhanced_chunks)} semantic chunks")
        return enhanced_chunks
    
    def _group_timestamp_segments(self, timestamp_data: List[Dict], max_segment_duration: int = 120) -> List[List[Dict]]:
        """Group timestamp entries into logical segments"""
        if not timestamp_data:
            return []
        
        segments = []
        current_segment = [timestamp_data[0]]
        segment_start = timestamp_data[0]['start']
        
        for entry in timestamp_data[1:]:
            # Check if we should start a new segment
            segment_duration = entry['end'] - segment_start
            
            if (segment_duration > max_segment_duration or 
                entry['start'] - current_segment[-1]['end'] > 10):  # 10 second gap
                
                # Finalize current segment
                segments.append(current_segment)
                current_segment = [entry]
                segment_start = entry['start']
            else:
                current_segment.append(entry)
        
        # Add final segment
        if current_segment:
            segments.append(current_segment)
        
        return segments
    
    def _seconds_to_timestamp(self, seconds: float) -> str:
        """Convert seconds to MM:SS timestamp format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    # Backward compatibility method
    def chunk_transcript(self, transcript: str) -> List[Document]:
        """Basic chunking for backward compatibility"""
        documents = [Document(page_content=transcript)]
        chunks = self.text_splitter.split_documents(documents)
        return chunks
    
    def process_video(self, url: str) -> Dict[str, Any]:
        """Main processing function for YouTube videos"""
        logger.info(f"Starting video processing for URL: {url}")
        
        video_id = self.extract_video_id(url)
        if not video_id:
            logger.error(f"Could not extract video ID from URL: {url}")
            raise ValueError("Invalid YouTube URL")
        
        logger.info(f"Extracted video ID: {video_id}")
        
        # Get metadata
        logger.info(f"Extracting metadata for video {video_id}")
        metadata = self.get_video_metadata(url)
        logger.info(f"Metadata extracted: {metadata.get('title', 'Unknown title')}")
        
        # Try to get transcript (with enhanced fallback strategy)
        logger.info(f"Starting transcript extraction for video {video_id}")
        
        transcript = None
        timestamp_data = []
        transcript_method = "unknown"
        
        # Try YouTube API first (free and fastest)
        logger.info("Attempting YouTube API transcript extraction")
        result_tuple = self.get_transcript_youtube_api(video_id)
        if result_tuple:
            transcript, timestamp_data = result_tuple
            logger.info(f"YouTube API transcript successful, length: {len(transcript)}, timestamps: {len(timestamp_data)}")
            transcript_method = "youtube_api"
        else:
            logger.warning("YouTube API transcript extraction failed")
            
            # Try Gemini Direct YouTube Processing (most accurate)
            logger.info("Attempting Gemini direct YouTube processing")
            result_tuple = self.get_transcript_gemini_direct(url)
            if result_tuple:
                transcript, timestamp_data = result_tuple
                logger.info(f"Gemini direct transcript successful, length: {len(transcript)}, timestamps: {len(timestamp_data)}")
                transcript_method = "gemini_direct"
            else:
                logger.warning("Gemini direct YouTube processing failed")
                
                # Try LangChain
                logger.info("Attempting LangChain transcript extraction")
                result_tuple = self.get_transcript_langchain(url)
                if result_tuple:
                    transcript, timestamp_data = result_tuple
                    logger.info(f"LangChain transcript successful, length: {len(transcript)}")
                    transcript_method = "langchain"
                else:
                    logger.warning("LangChain transcript extraction failed")
                    
                    # Try Gemini metadata-based fallback (last resort)
                    logger.info("Attempting Gemini metadata-based transcript generation")
                    result_tuple = self.generate_transcript_gemini(url, metadata)
                    if result_tuple:
                        transcript, timestamp_data = result_tuple
                        logger.info(f"Gemini metadata transcript successful, length: {len(transcript)}")
                        transcript_method = "gemini_metadata"
                    else:
                        logger.error("All transcript extraction methods failed")
        
        if not transcript:
            logger.error(f"Could not extract transcript from video {video_id}")
            raise ValueError("Could not extract transcript from video")
        
        # Add transcript method to metadata for enhanced system prompts
        metadata["transcript_method"] = transcript_method
        metadata["video_id"] = video_id
        
        # Enhanced chunking with timestamp awareness
        logger.info(f"Enhanced chunking for video {video_id}")
        chunks = self.chunk_transcript_enhanced(transcript, timestamp_data, metadata, transcript_method)
        logger.info(f"Created {len(chunks)} enhanced chunks for video {video_id}")
        
        result = {
            "video_id": video_id,
            "url": url,
            "metadata": metadata,
            "transcript": transcript,
            "transcript_method": transcript_method,
            "timestamp_data": timestamp_data,
            "chunks": [chunk.page_content for chunk in chunks],
            "chunk_metadata": [chunk.metadata for chunk in chunks],
            "chunk_count": len(chunks)
        }
        
        logger.info(f"Video processing completed successfully for {video_id}")
        return result