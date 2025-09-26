"""
Conversation Persistence Service
Database-backed conversation storage for Ultimate CVA Agent and Orchestrator
Week 2 Day 7 Implementation
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from core.database.connection import DatabaseManager

logger = logging.getLogger(__name__)

@dataclass
class ConversationRecord:
    """Conversation record structure"""
    conversation_id: str
    user_id: str
    agent_type: str
    agent_id: str
    message_count: int
    created_at: datetime
    last_updated: datetime
    metadata: Dict[str, Any]

@dataclass
class MessageRecord:
    """Message record structure"""
    message_id: str
    conversation_id: str
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Dict[str, Any]

class ConversationPersistenceService:
    """
    Production-ready conversation persistence service

    Features:
    - Database-backed conversation storage
    - Message history tracking
    - Context summarization
    - Performance analytics
    - Multi-agent conversation support
    """

    def __init__(self):
        self.db_manager = DatabaseManager()
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure conversation tables exist"""
        try:
            # In a production system, this would use proper migrations
            # For now, we'll create tables if they don't exist
            with self.db_manager.get_session() as session:
                # Create conversations table
                session.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id VARCHAR(36) PRIMARY KEY,
                    user_id VARCHAR(100) NOT NULL,
                    agent_type VARCHAR(50) NOT NULL,
                    agent_id VARCHAR(100) NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
                """)

                # Create messages table
                session.execute("""
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id VARCHAR(36) PRIMARY KEY,
                    conversation_id VARCHAR(36) REFERENCES conversations(conversation_id),
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
                """)

                # Create indexes for performance
                session.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user_id
                ON conversations(user_id)
                """)

                session.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_agent_type
                ON conversations(agent_type)
                """)

                session.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON conversation_messages(conversation_id)
                """)

                session.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_timestamp
                ON conversation_messages(timestamp)
                """)

                session.commit()
                logger.info("Conversation persistence tables initialized")

        except Exception as e:
            logger.error(f"Failed to initialize conversation tables: {e}")

    async def create_conversation(
        self,
        user_id: str,
        agent_type: str,
        agent_id: str,
        initial_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new conversation"""
        try:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"
            now = datetime.utcnow()

            conversation_record = {
                'conversation_id': conversation_id,
                'user_id': user_id,
                'agent_type': agent_type,
                'agent_id': agent_id,
                'message_count': 0,
                'created_at': now,
                'last_updated': now,
                'metadata': json.dumps(metadata or {})
            }

            with self.db_manager.get_session() as session:
                # Insert conversation record
                session.execute("""
                INSERT INTO conversations
                (conversation_id, user_id, agent_type, agent_id, message_count, created_at, last_updated, metadata)
                VALUES (%(conversation_id)s, %(user_id)s, %(agent_type)s, %(agent_id)s,
                        %(message_count)s, %(created_at)s, %(last_updated)s, %(metadata)s)
                """, conversation_record)

                # Add initial message if provided
                if initial_message:
                    await self._add_message_to_conversation(
                        session, conversation_id, "user", initial_message, {}
                    )

                session.commit()

            logger.info(f"Created conversation {conversation_id} for user {user_id}")
            return conversation_id

        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            return None

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a message to an existing conversation"""
        try:
            with self.db_manager.get_session() as session:
                message_id = await self._add_message_to_conversation(
                    session, conversation_id, role, content, metadata or {}
                )

                # Update conversation last_updated and message count
                session.execute("""
                UPDATE conversations
                SET last_updated = %(timestamp)s,
                    message_count = message_count + 1
                WHERE conversation_id = %(conversation_id)s
                """, {
                    'timestamp': datetime.utcnow(),
                    'conversation_id': conversation_id
                })

                session.commit()

            logger.debug(f"Added message {message_id} to conversation {conversation_id}")
            return message_id

        except Exception as e:
            logger.error(f"Failed to add message to conversation {conversation_id}: {e}")
            return None

    async def _add_message_to_conversation(
        self,
        session,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Internal method to add message within a session"""
        message_id = f"msg_{uuid.uuid4().hex[:8]}"

        message_record = {
            'message_id': message_id,
            'conversation_id': conversation_id,
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow(),
            'metadata': json.dumps(metadata)
        }

        session.execute("""
        INSERT INTO conversation_messages
        (message_id, conversation_id, role, content, timestamp, metadata)
        VALUES (%(message_id)s, %(conversation_id)s, %(role)s, %(content)s,
                %(timestamp)s, %(metadata)s)
        """, message_record)

        return message_id

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = 50,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """Get conversation message history"""
        try:
            with self.db_manager.get_session() as session:
                query = """
                SELECT message_id, role, content, timestamp, metadata
                FROM conversation_messages
                WHERE conversation_id = %(conversation_id)s
                ORDER BY timestamp ASC
                """

                if limit:
                    query += f" LIMIT {limit}"

                result = session.execute(query, {'conversation_id': conversation_id})
                messages = []

                for row in result.fetchall():
                    message = {
                        'message_id': row[0],
                        'role': row[1],
                        'content': row[2],
                        'timestamp': row[3].isoformat() if row[3] else None
                    }

                    if include_metadata and row[4]:
                        try:
                            message['metadata'] = json.loads(row[4])
                        except (json.JSONDecodeError, TypeError):
                            message['metadata'] = {}

                    messages.append(message)

                logger.debug(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
                return messages

        except Exception as e:
            logger.error(f"Failed to get conversation history for {conversation_id}: {e}")
            return []

    async def get_user_conversations(
        self,
        user_id: str,
        limit: Optional[int] = 20,
        agent_type: Optional[str] = None
    ) -> List[ConversationRecord]:
        """Get conversation list for a user"""
        try:
            with self.db_manager.get_session() as session:
                query = """
                SELECT conversation_id, user_id, agent_type, agent_id,
                       message_count, created_at, last_updated, metadata
                FROM conversations
                WHERE user_id = %(user_id)s
                """
                params = {'user_id': user_id}

                if agent_type:
                    query += " AND agent_type = %(agent_type)s"
                    params['agent_type'] = agent_type

                query += " ORDER BY last_updated DESC"

                if limit:
                    query += f" LIMIT {limit}"

                result = session.execute(query, params)
                conversations = []

                for row in result.fetchall():
                    try:
                        metadata = json.loads(row[7]) if row[7] else {}
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}

                    conversation = ConversationRecord(
                        conversation_id=row[0],
                        user_id=row[1],
                        agent_type=row[2],
                        agent_id=row[3],
                        message_count=row[4],
                        created_at=row[5],
                        last_updated=row[6],
                        metadata=metadata
                    )
                    conversations.append(conversation)

                logger.debug(f"Retrieved {len(conversations)} conversations for user {user_id}")
                return conversations

        except Exception as e:
            logger.error(f"Failed to get conversations for user {user_id}: {e}")
            return []

    async def get_conversation_context(
        self,
        conversation_id: str,
        max_tokens: int = 4000
    ) -> Dict[str, Any]:
        """Get conversation context optimized for LLM input"""
        try:
            # Get recent messages
            messages = await self.get_conversation_history(conversation_id, limit=20)

            if not messages:
                return {
                    'conversation_id': conversation_id,
                    'messages': [],
                    'context_length': 0,
                    'summary': 'No conversation history found'
                }

            # Estimate token count (rough approximation: 1 token = 4 chars)
            total_chars = sum(len(msg['content']) for msg in messages)
            estimated_tokens = total_chars // 4

            context = {
                'conversation_id': conversation_id,
                'messages': messages,
                'context_length': estimated_tokens,
                'message_count': len(messages)
            }

            # If context is too long, summarize older messages
            if estimated_tokens > max_tokens:
                context = await self._summarize_conversation_context(conversation_id, messages, max_tokens)

            return context

        except Exception as e:
            logger.error(f"Failed to get conversation context for {conversation_id}: {e}")
            return {
                'conversation_id': conversation_id,
                'messages': [],
                'context_length': 0,
                'error': str(e)
            }

    async def _summarize_conversation_context(
        self,
        conversation_id: str,
        messages: List[Dict],
        max_tokens: int
    ) -> Dict[str, Any]:
        """Summarize conversation context to fit within token limit"""
        try:
            # Keep the most recent messages and summarize older ones
            recent_messages = messages[-10:]  # Keep last 10 messages
            older_messages = messages[:-10]

            # Create summary of older messages
            if older_messages:
                summary_points = []
                for msg in older_messages[::2]:  # Sample every other message
                    if len(msg['content']) > 100:
                        summary_points.append(f"{msg['role']}: {msg['content'][:100]}...")
                    else:
                        summary_points.append(f"{msg['role']}: {msg['content']}")

                summary = f"Previous conversation summary ({len(older_messages)} messages): " + " | ".join(summary_points)
            else:
                summary = "No previous conversation history"

            return {
                'conversation_id': conversation_id,
                'messages': recent_messages,
                'conversation_summary': summary,
                'context_length': (len(summary) + sum(len(msg['content']) for msg in recent_messages)) // 4,
                'message_count': len(messages),
                'summarized': True
            }

        except Exception as e:
            logger.error(f"Failed to summarize conversation context: {e}")
            return {
                'conversation_id': conversation_id,
                'messages': messages[-5:],  # Fallback to last 5 messages
                'context_length': 0,
                'error': str(e)
            }

    async def save_agent_interaction(
        self,
        conversation_id: str,
        user_message: str,
        agent_response: str,
        agent_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Save a complete user-agent interaction"""
        try:
            # Add user message
            user_msg_id = await self.add_message(
                conversation_id, "user", user_message,
                {'interaction_type': 'user_input'}
            )

            # Add agent response
            agent_msg_id = await self.add_message(
                conversation_id, "assistant", agent_response,
                {
                    'interaction_type': 'agent_response',
                    'agent_metadata': agent_metadata or {},
                    'user_message_id': user_msg_id
                }
            )

            return bool(user_msg_id and agent_msg_id)

        except Exception as e:
            logger.error(f"Failed to save agent interaction: {e}")
            return False

    async def get_conversation_analytics(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get conversation analytics"""
        try:
            with self.db_manager.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)

                # Base query
                base_query = """
                FROM conversations c
                LEFT JOIN conversation_messages m ON c.conversation_id = m.conversation_id
                WHERE c.created_at >= %(cutoff_date)s
                """
                params = {'cutoff_date': cutoff_date}

                if conversation_id:
                    base_query += " AND c.conversation_id = %(conversation_id)s"
                    params['conversation_id'] = conversation_id

                if user_id:
                    base_query += " AND c.user_id = %(user_id)s"
                    params['user_id'] = user_id

                # Get conversation count
                conv_result = session.execute(f"""
                SELECT COUNT(DISTINCT c.conversation_id) as conversation_count,
                       COUNT(m.message_id) as total_messages,
                       AVG(c.message_count) as avg_messages_per_conversation
                {base_query}
                """, params)

                conv_row = conv_result.fetchone()

                # Get agent type distribution
                agent_result = session.execute(f"""
                SELECT c.agent_type, COUNT(DISTINCT c.conversation_id) as count
                {base_query}
                GROUP BY c.agent_type
                """, params)

                agent_distribution = {row[0]: row[1] for row in agent_result.fetchall()}

                analytics = {
                    'period_days': days,
                    'conversation_count': conv_row[0] or 0,
                    'total_messages': conv_row[1] or 0,
                    'avg_messages_per_conversation': float(conv_row[2] or 0),
                    'agent_type_distribution': agent_distribution,
                    'analytics_timestamp': datetime.utcnow().isoformat()
                }

                return analytics

        except Exception as e:
            logger.error(f"Failed to get conversation analytics: {e}")
            return {
                'error': str(e),
                'analytics_timestamp': datetime.utcnow().isoformat()
            }

    async def cleanup_old_conversations(self, days_to_keep: int = 30) -> int:
        """Clean up old conversations beyond retention period"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)

            with self.db_manager.get_session() as session:
                # Get conversations to delete
                result = session.execute("""
                SELECT conversation_id FROM conversations
                WHERE created_at < %(cutoff_date)s
                """, {'cutoff_date': cutoff_date})

                conversation_ids = [row[0] for row in result.fetchall()]

                if conversation_ids:
                    # Delete messages first (foreign key constraint)
                    session.execute("""
                    DELETE FROM conversation_messages
                    WHERE conversation_id = ANY(%(conversation_ids)s)
                    """, {'conversation_ids': conversation_ids})

                    # Delete conversations
                    session.execute("""
                    DELETE FROM conversations
                    WHERE conversation_id = ANY(%(conversation_ids)s)
                    """, {'conversation_ids': conversation_ids})

                    session.commit()

                logger.info(f"Cleaned up {len(conversation_ids)} old conversations")
                return len(conversation_ids)

        except Exception as e:
            logger.error(f"Failed to cleanup old conversations: {e}")
            return 0

# Global instance
_conversation_service: Optional[ConversationPersistenceService] = None

def get_conversation_service() -> ConversationPersistenceService:
    """Get global conversation persistence service"""
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationPersistenceService()
    return _conversation_service