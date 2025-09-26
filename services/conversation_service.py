"""
Conversation Persistence Service
Handles persistent storage and retrieval of CVA conversations for learning
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from core.database.models import CVAConversation, UserSession
from core.database.connection import db_manager
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

class ConversationService:
    """Service for managing persistent CVA conversations"""
    
    @staticmethod
    def save_conversation(
        session_id: str,
        user_id: str,
        user_message: str,
        cva_response: str,
        conversation_turn: int,
        response_metadata: Dict[str, Any] = None,
        performance_data: Dict[str, Any] = None,
        conversation_history: List[Dict] = None
    ) -> str:
        """
        Save a conversation turn to database
        Returns the conversation ID
        """
        try:
            # Ensure database is initialized before using it
            if not db_manager._is_initialized:
                logger.warning("Database manager not initialized, initializing now...")
                if not db_manager.initialize():
                    raise Exception("Failed to initialize database manager")
            
            with db_manager.get_session() as db_session:
                # Create context summary from conversation history
                context_summary = None
                if conversation_history:
                    context_summary = ConversationService._create_context_summary(conversation_history)
                
                conversation = CVAConversation(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=user_message,
                    cva_response=cva_response,
                    conversation_turn=conversation_turn,
                    context_summary=context_summary,
                    response_metadata=response_metadata or {},
                    performance_data=performance_data or {}
                )
                
                db_session.add(conversation)
                db_session.commit()
                db_session.refresh(conversation)
                
                logger.info(f"Saved conversation turn {conversation_turn} for session {session_id}")
                return conversation.id
                
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise
    
    @staticmethod
    def get_conversation_history(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session
        Returns list of conversation turns in chronological order
        """
        try:
            # Ensure database is initialized before using it
            if not db_manager._is_initialized:
                logger.warning("Database manager not initialized for get_conversation_history")
                if not db_manager.initialize():
                    logger.error("Failed to initialize database manager")
                    return []
            
            with db_manager.get_session() as db_session:
                conversations = db_session.query(CVAConversation).filter(
                    CVAConversation.session_id == session_id
                ).order_by(CVAConversation.conversation_turn).limit(limit).all()
                
                history = []
                for conv in conversations:
                    history.extend([
                        {"role": "user", "content": conv.user_message},
                        {"role": "assistant", "content": conv.cva_response}
                    ])
                
                logger.debug(f"Retrieved {len(history)} messages for session {session_id}")
                return history
                
        except Exception as e:
            logger.error(f"Failed to get conversation history: {e}")
            return []
    
    @staticmethod
    def get_learning_data(limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get conversation data for learning and analytics
        Returns conversations with performance and feedback data
        """
        try:
            # Ensure database is initialized before using it
            if not db_manager._is_initialized:
                logger.warning("Database manager not initialized for get_learning_data")
                if not db_manager.initialize():
                    logger.error("Failed to initialize database manager")
                    return []
            
            with db_manager.get_session() as db_session:
                conversations = db_session.query(CVAConversation).filter(
                    CVAConversation.user_feedback.isnot(None)
                ).order_by(desc(CVAConversation.created_at)).limit(limit).all()
                
                learning_data = []
                for conv in conversations:
                    learning_data.append({
                        "conversation_id": conv.id,
                        "user_message": conv.user_message,
                        "cva_response": conv.cva_response,
                        "user_feedback": conv.user_feedback,
                        "feedback_details": conv.feedback_details,
                        "performance_data": conv.performance_data,
                        "conversation_topic": conv.conversation_topic,
                        "keywords": conv.keywords,
                        "created_at": conv.created_at.isoformat()
                    })
                
                logger.info(f"Retrieved {len(learning_data)} learning samples")
                return learning_data
                
        except Exception as e:
            logger.error(f"Failed to get learning data: {e}")
            return []
    
    @staticmethod
    def add_feedback(conversation_id: str, feedback: str, details: str = None) -> bool:
        """
        Add user feedback to a conversation for learning
        """
        try:
            # Ensure database is initialized before using it
            if not db_manager._is_initialized:
                logger.warning("Database manager not initialized for add_feedback")
                if not db_manager.initialize():
                    logger.error("Failed to initialize database manager")
                    return False
            
            with db_manager.get_session() as db_session:
                conversation = db_session.query(CVAConversation).filter(
                    CVAConversation.id == conversation_id
                ).first()
                
                if conversation:
                    conversation.user_feedback = feedback
                    conversation.feedback_details = details
                    db_session.commit()
                    
                    logger.info(f"Added feedback '{feedback}' to conversation {conversation_id}")
                    return True
                else:
                    logger.warning(f"Conversation {conversation_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to add feedback: {e}")
            return False
    
    @staticmethod
    def get_conversation_analytics(days: int = 30) -> Dict[str, Any]:
        """
        Get analytics data for conversations over specified days
        """
        try:
            from sqlalchemy import func, and_
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            # Ensure database is initialized before using it
            if not db_manager._is_initialized:
                logger.warning("Database manager not initialized for get_conversation_analytics")
                if not db_manager.initialize():
                    logger.error("Failed to initialize database manager")
                    return {}
            
            with db_manager.get_session() as db_session:
                # Total conversations
                total_conversations = db_session.query(CVAConversation).filter(
                    CVAConversation.created_at >= cutoff_date
                ).count()
                
                # Feedback distribution
                feedback_stats = db_session.query(
                    CVAConversation.user_feedback,
                    func.count(CVAConversation.id).label('count')
                ).filter(
                    and_(
                        CVAConversation.created_at >= cutoff_date,
                        CVAConversation.user_feedback.isnot(None)
                    )
                ).group_by(CVAConversation.user_feedback).all()
                
                # Most common topics
                topic_stats = db_session.query(
                    CVAConversation.conversation_topic,
                    func.count(CVAConversation.id).label('count')
                ).filter(
                    and_(
                        CVAConversation.created_at >= cutoff_date,
                        CVAConversation.conversation_topic.isnot(None)
                    )
                ).group_by(CVAConversation.conversation_topic).order_by(
                    func.count(CVAConversation.id).desc()
                ).limit(10).all()
                
                analytics = {
                    "period_days": days,
                    "total_conversations": total_conversations,
                    "feedback_distribution": {stat.user_feedback: stat.count for stat in feedback_stats},
                    "top_topics": [{"topic": stat.conversation_topic, "count": stat.count} for stat in topic_stats],
                    "generated_at": datetime.utcnow().isoformat()
                }
                
                logger.info(f"Generated analytics for {days} days: {total_conversations} conversations")
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {}
    
    @staticmethod
    def _create_context_summary(conversation_history: List[Dict]) -> str:
        """
        Create a brief summary of conversation context for storage
        """
        if not conversation_history:
            return ""
        
        # Simple context summary - could be enhanced with AI summarization
        user_messages = [msg["content"] for msg in conversation_history if msg["role"] == "user"]
        topics = []
        
        # Extract key topics/keywords (simple approach)
        for message in user_messages[-3:]:  # Last 3 user messages
            words = message.lower().split()
            key_words = [word for word in words if len(word) > 4 and word not in ["have", "would", "could", "should"]]
            topics.extend(key_words[:2])  # Top 2 words per message
        
        context = f"Topics discussed: {', '.join(set(topics[:5]))}" if topics else "General conversation"
        return context