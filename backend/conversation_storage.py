import sqlite3
import json
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import os
import re
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PersistentConversationStorage:
    """
    SQLite-based persistent storage for conversations with enhanced performance and reliability.
    Provides conversation management, message history, and user session tracking.
    """
    
    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self._init_database()
        logger.info(f"✅ Conversation storage initialized: {db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            # Conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    video_ids TEXT NOT NULL,  -- JSON array
                    metadata TEXT NOT NULL,  -- JSON object
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            """)
            
            # Messages table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,  -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    sources TEXT,  -- JSON array of sources
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    token_count INTEGER DEFAULT 0,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
                )
            """)
            
            # User sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    user_id TEXT PRIMARY KEY,
                    last_conversation_id TEXT,
                    session_data TEXT,  -- JSON object for additional data
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_last_active ON user_sessions(last_active)")
            
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign key constraints
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_conversation(self, user_id: str, video_id: str, video_metadata: Dict[str, Any], title: str = None) -> str:
        """Create a new conversation"""
        conversation_id = str(uuid.uuid4())
        
        # Generate title from video metadata if not provided
        if not title:
            title = video_metadata.get('title', 'Untitled Video')[:100]  # Truncate long titles
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO conversations (id, user_id, title, video_ids, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                conversation_id,
                user_id,
                title,
                json.dumps([video_id]),
                json.dumps(video_metadata)
            ))
            
            # Update user session
            conn.execute("""
                INSERT OR REPLACE INTO user_sessions (user_id, last_conversation_id, last_active)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """, (user_id, conversation_id))
            
            conn.commit()
        
        logger.info(f"Created conversation {conversation_id} for user {user_id}")
        return conversation_id
    
    def add_video_to_conversation(self, conversation_id: str, video_id: str, user_id: str) -> bool:
        """Add a video to an existing conversation"""
        with self._get_connection() as conn:
            # Check if conversation exists and belongs to user
            row = conn.execute("""
                SELECT video_ids FROM conversations 
                WHERE id = ? AND user_id = ? AND is_active = TRUE
            """, (conversation_id, user_id)).fetchone()
            
            if not row:
                return False
            
            # Update video_ids
            video_ids = json.loads(row['video_ids'])
            if video_id not in video_ids:
                video_ids.append(video_id)
                
                conn.execute("""
                    UPDATE conversations 
                    SET video_ids = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (json.dumps(video_ids), conversation_id))
                
                conn.commit()
                logger.info(f"Added video {video_id} to conversation {conversation_id}")
            
            return True
    
    def add_message(
        self, 
        conversation_id: str, 
        role: str, 
        content: str, 
        user_id: str,
        sources: List[Dict] = None,
        token_count: int = 0
    ) -> bool:
        """Add a message to conversation"""
        with self._get_connection() as conn:
            # Verify conversation belongs to user
            row = conn.execute("""
                SELECT id FROM conversations 
                WHERE id = ? AND user_id = ? AND is_active = TRUE
            """, (conversation_id, user_id)).fetchone()
            
            if not row:
                return False
            
            # Add message
            conn.execute("""
                INSERT INTO messages (conversation_id, role, content, sources, token_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                conversation_id,
                role,
                content,
                json.dumps(sources) if sources else None,
                token_count
            ))
            
            # Update conversation timestamp
            conn.execute("""
                UPDATE conversations 
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (conversation_id,))
            
            conn.commit()
            return True
    
    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation with its metadata"""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM conversations 
                WHERE id = ? AND user_id = ? AND is_active = TRUE
            """, (conversation_id, user_id)).fetchone()
            
            if not row:
                return None
            
            return {
                'conversation_id': row['id'],
                'user_id': row['user_id'],
                'title': row['title'],
                'video_ids': json.loads(row['video_ids']),
                'metadata': json.loads(row['metadata']),
                'created_at': row['created_at'],
                'updated_at': row['updated_at']
            }
    
    def get_conversation_messages(
        self, 
        conversation_id: str, 
        user_id: str, 
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation with pagination"""
        with self._get_connection() as conn:
            # Verify conversation belongs to user
            conv_row = conn.execute("""
                SELECT id FROM conversations 
                WHERE id = ? AND user_id = ? AND is_active = TRUE
            """, (conversation_id, user_id)).fetchone()
            
            if not conv_row:
                return []
            
            # Get messages
            rows = conn.execute("""
                SELECT * FROM messages 
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
                LIMIT ? OFFSET ?
            """, (conversation_id, limit, offset)).fetchall()
            
            messages = []
            for row in rows:
                message = {
                    'id': row['id'],
                    'role': row['role'],
                    'content': row['content'],
                    'timestamp': row['timestamp'],
                    'token_count': row['token_count']
                }
                if row['sources']:
                    message['sources'] = json.loads(row['sources'])
                messages.append(message)
            
            return messages
    
    def get_user_conversations(
        self, 
        user_id: str, 
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get all conversations for a user"""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT id, title, video_ids, created_at, updated_at,
                       (SELECT COUNT(*) FROM messages WHERE conversation_id = conversations.id) as message_count
                FROM conversations 
                WHERE user_id = ? AND is_active = TRUE
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            """, (user_id, limit, offset)).fetchall()
            
            conversations = []
            for row in rows:
                conversations.append({
                    'conversation_id': row['id'],
                    'title': row['title'],
                    'video_ids': json.loads(row['video_ids']),
                    'message_count': row['message_count'],
                    'created_at': row['created_at'],
                    'updated_at': row['updated_at']
                })
            
            return conversations
    
    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Soft delete a conversation (mark as inactive)"""
        with self._get_connection() as conn:
            result = conn.execute("""
                UPDATE conversations 
                SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
            """, (conversation_id, user_id))
            
            conn.commit()
            
            if result.rowcount > 0:
                logger.info(f"Deleted conversation {conversation_id} for user {user_id}")
                return True
            return False
    
    def get_conversation_memory_summary(
        self, 
        conversation_id: str, 
        user_id: str, 
        max_messages: int = 10,
        enable_compression: bool = True
    ) -> List[Dict[str, Any]]:
        """Get conversation memory with intelligent compression"""
        with self._get_connection() as conn:
            # Verify conversation belongs to user
            conv_row = conn.execute("""
                SELECT id FROM conversations 
                WHERE id = ? AND user_id = ? AND is_active = TRUE
            """, (conversation_id, user_id)).fetchone()
            
            if not conv_row:
                return []
            
            # Get recent messages with sources
            rows = conn.execute("""
                SELECT role, content, timestamp, sources FROM messages 
                WHERE conversation_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (conversation_id, max_messages * 2)).fetchall()  # Get more for compression
            
            # Convert to list of dicts
            recent_messages = []
            for row in reversed(rows):
                sources = json.loads(row['sources']) if row['sources'] else []
                recent_messages.append({
                    'role': row['role'],
                    'content': row['content'],
                    'created_at': row['timestamp'],
                    'sources': sources
                })
            
            # Apply intelligent compression if enabled and we have more than threshold
            if enable_compression and len(recent_messages) > max_messages:
                compressed_memory = self._compress_conversation_memory(
                    recent_messages, 
                    conversation_id, 
                    user_id,
                    target_length=max_messages
                )
                return compressed_memory
            else:
                return recent_messages[:max_messages]  # Simple truncation
    
    def cleanup_old_conversations(self, days_old: int = 30) -> int:
        """Clean up old inactive conversations"""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        with self._get_connection() as conn:
            # Delete messages first (foreign key constraint)
            conn.execute("""
                DELETE FROM messages 
                WHERE conversation_id IN (
                    SELECT id FROM conversations 
                    WHERE is_active = FALSE AND updated_at < ?
                )
            """, (cutoff_date,))
            
            # Delete conversations
            result = conn.execute("""
                DELETE FROM conversations 
                WHERE is_active = FALSE AND updated_at < ?
            """, (cutoff_date,))
            
            conn.commit()
            
            deleted_count = result.rowcount
            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old conversations")
            
            return deleted_count
    
    def _compress_conversation_memory(
        self, 
        messages: List[Dict], 
        conversation_id: str, 
        user_id: str,
        target_length: int = 10
    ) -> List[Dict]:
        """Intelligently compress conversation memory while preserving important context"""
        try:
            if len(messages) <= target_length:
                return messages
            
            # Always keep the most recent messages
            recent_keep = target_length // 3
            recent_messages = messages[-recent_keep:]
            
            # Compress older messages
            older_messages = messages[:-recent_keep]
            
            # Identify important messages to preserve
            important_messages = self._identify_important_messages(older_messages)
            
            # Create conversation summary for the compressed part
            compression_summary = self._create_conversation_summary(
                older_messages, important_messages
            )
            
            # Combine: summary + important messages + recent messages
            compressed = []
            
            # Add summary as a special message
            if compression_summary:
                compressed.append({
                    'role': 'system',
                    'content': f"[Previous conversation summary: {compression_summary}]",
                    'created_at': older_messages[-1]['created_at'] if older_messages else datetime.now().isoformat(),
                    'sources': [],
                    'is_summary': True
                })
            
            # Add selected important messages
            compressed.extend(important_messages[-3:])  # Keep last 3 important ones
            
            # Add recent messages
            compressed.extend(recent_messages)
            
            # Ensure chronological order
            compressed.sort(key=lambda x: x['created_at'])
            
            logger.info(f"Compressed {len(messages)} messages to {len(compressed)} with intelligent preservation")
            return compressed
            
        except Exception as e:
            logger.error(f"Memory compression failed: {e}")
            # Fallback to simple truncation
            return messages[-target_length:]
    
    def _identify_important_messages(self, messages: List[Dict]) -> List[Dict]:
        """Identify important messages to preserve during compression"""
        important = []
        
        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', '')
            sources = msg.get('sources', [])
            
            # Criteria for importance
            is_important = False
            
            # 1. Messages with sources (contain factual information)
            if sources:
                is_important = True
                msg['importance_reason'] = 'has_sources'
            
            # 2. Long, detailed responses (likely contain valuable info)
            elif len(content) > 300 and role == 'assistant':
                is_important = True
                msg['importance_reason'] = 'detailed_response'
            
            # 3. Questions about specific topics or entities
            elif role == 'user' and self._contains_specific_entities(content):
                is_important = True
                msg['importance_reason'] = 'specific_query'
            
            # 4. Messages with technical terms or code
            elif self._contains_technical_content(content):
                is_important = True
                msg['importance_reason'] = 'technical_content'
            
            # 5. Error messages or clarifications
            elif any(word in content.lower() for word in ['error', 'clarify', 'explain', 'confused']):
                is_important = True
                msg['importance_reason'] = 'clarification'
            
            if is_important:
                important.append(msg)
        
        return important
    
    def _contains_specific_entities(self, text: str) -> bool:
        """Check if text contains specific entities (proper nouns, quoted terms, etc.)"""
        # Look for capitalized words (potential entities)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        
        # Look for quoted terms
        quoted_terms = re.findall(r'"[^"]+"', text)
        
        # Look for specific patterns
        has_numbers = bool(re.search(r'\d+', text))
        has_technical_refs = bool(re.search(r'\b(API|URL|HTTP|JSON|XML|SQL)\b', text, re.IGNORECASE))
        
        return len(capitalized_words) >= 2 or len(quoted_terms) > 0 or has_technical_refs
    
    def _contains_technical_content(self, text: str) -> bool:
        """Check if text contains technical content"""
        technical_indicators = [
            'function', 'class', 'method', 'algorithm', 'implementation',
            'code', 'syntax', 'parameter', 'variable', 'return',
            'import', 'library', 'framework', 'database', 'server',
            'configuration', 'setup', 'install', 'deploy', 'debug'
        ]
        
        text_lower = text.lower()
        technical_count = sum(1 for term in technical_indicators if term in text_lower)
        
        return technical_count >= 2
    
    def _create_conversation_summary(self, messages: List[Dict], important_messages: List[Dict]) -> str:
        """Create a concise summary of conversation history"""
        try:
            if not messages:
                return ""
            
            # Extract key topics and entities
            user_queries = []
            assistant_topics = []
            
            for msg in messages:
                content = msg.get('content', '')
                role = msg.get('role', '')
                
                if role == 'user':
                    # Extract main topic from user questions
                    query_keywords = self._extract_keywords(content)
                    user_queries.extend(query_keywords)
                elif role == 'assistant' and len(content) > 100:
                    # Extract topics from substantial assistant responses
                    response_keywords = self._extract_keywords(content)
                    assistant_topics.extend(response_keywords)
            
            # Count and prioritize topics
            all_topics = Counter(user_queries + assistant_topics)
            top_topics = [topic for topic, count in all_topics.most_common(5)]
            
            # Build summary
            summary_parts = []
            
            if top_topics:
                summary_parts.append(f"Discussed: {', '.join(top_topics)}")
            
            if important_messages:
                reasons = [msg.get('importance_reason', '') for msg in important_messages]
                reason_counts = Counter(reasons)
                if reason_counts:
                    top_reason = reason_counts.most_common(1)[0][0]
                    summary_parts.append(f"Focus: {top_reason.replace('_', ' ')}")
            
            summary_parts.append(f"History: {len(messages)} exchanges compressed")
            
            return "; ".join(summary_parts)
            
        except Exception as e:
            logger.warning(f"Summary creation failed: {e}")
            return f"Previous conversation with {len(messages)} messages"
    
    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract meaningful keywords from text"""
        try:
            # Simple keyword extraction
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            
            # Filter out common stop words
            stop_words = {
                'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                'by', 'this', 'that', 'these', 'those', 'can', 'could', 'would', 'should',
                'will', 'what', 'how', 'when', 'where', 'why', 'who', 'which', 'have',
                'has', 'had', 'are', 'was', 'were', 'been', 'being', 'you', 'your',
                'please', 'thank', 'thanks', 'help', 'get', 'make', 'use', 'need'
            }
            
            keywords = [word for word in words if word not in stop_words and len(word) > 3]
            
            # Count frequency and return most common
            keyword_counts = Counter(keywords)
            return [word for word, count in keyword_counts.most_common(max_keywords)]
            
        except Exception:
            return []
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics including memory management info"""
        with self._get_connection() as conn:
            # Get conversation count
            conv_count = conn.execute("""
                SELECT COUNT(*) as count FROM conversations WHERE is_active = TRUE
            """).fetchone()['count']
            
            # Get message count
            msg_count = conn.execute("""
                SELECT COUNT(*) as count FROM messages
            """).fetchone()['count']
            
            # Get user count
            user_count = conn.execute("""
                SELECT COUNT(DISTINCT user_id) as count FROM conversations WHERE is_active = TRUE
            """).fetchone()['count']
            
            # Get average messages per conversation
            avg_messages = conn.execute("""
                SELECT AVG(message_count) as avg FROM (
                    SELECT COUNT(*) as message_count 
                    FROM messages 
                    GROUP BY conversation_id
                )
            """).fetchone()['avg'] or 0
            
            # Get memory compression statistics
            long_conversations = conn.execute("""
                SELECT conversation_id, COUNT(*) as msg_count
                FROM messages 
                GROUP BY conversation_id 
                HAVING COUNT(*) > 15
            """).fetchall()
            
            compression_candidates = len(long_conversations)
            
            # Get recent activity (last 7 days)
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            recent_message_count = conn.execute("""
                SELECT COUNT(*) as count FROM messages 
                WHERE timestamp > ?
            """, (week_ago,)).fetchone()['count']
            
            return {
                'active_conversations': conv_count,
                'total_messages': msg_count,
                'unique_users': user_count,
                'avg_messages_per_conversation': round(avg_messages, 2),
                'database_size_mb': round(os.path.getsize(self.db_path) / (1024 * 1024), 2) if os.path.exists(self.db_path) else 0,
                'memory_management': {
                    'compression_candidates': compression_candidates,
                    'recent_activity_7d': recent_message_count,
                    'compression_enabled': True,
                    'intelligent_preservation': True
                },
                'features': [
                    'intelligent_memory_compression',
                    'important_message_preservation',
                    'context_aware_summaries',
                    'technical_content_recognition',
                    'entity_extraction'
                ]
            }