"""
Database Models - Basic user and role models voor FASE 0.4
Extended with KnowledgeEntry model voor FASE 3
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Float, Index
from sqlalchemy.dialects.postgresql import JSONB, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

Base = declarative_base()

class Role(Base):
    __tablename__ = "roles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50), unique=True, nullable=False)

    # Relationship
    users = relationship("User", back_populates="role")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role_id = Column(Integer, ForeignKey("roles.id"), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    role = relationship("Role", back_populates="users")

class KnowledgeEntry(Base):
    """
    KnowledgeEntry model voor FASE 3 kennisbeheer systeem
    Zoals gespecificeerd in STAP 3.1: velden voor id, content (Text), type (String), metadata (JSONB), en confidence_score (Float)
    Enhanced with TAAK 3.2 performance indexes
    """
    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    type = Column(String(100), nullable=False, index=True)
    entry_metadata = Column(JSONB, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    confidence_score = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)  # TAAK 3.2: Added index
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # TAAK 3.2: Performance-critical composite indexes
    __table_args__ = (
        # Index for filtering by type and ordering by confidence score
        Index('idx_knowledge_type_confidence', 'type', 'confidence_score'),
        # Index for recent knowledge queries filtered by type
        Index('idx_knowledge_created_type', 'created_at', 'type'),
        # Composite index for complex search queries
        Index('idx_knowledge_search', 'type', 'created_at', 'confidence_score'),
        # Index for high-confidence knowledge retrieval
        Index('idx_knowledge_confidence_created', 'confidence_score', 'created_at'),
    )


class RegisteredAgent(Base):
    """
    RegisteredAgent model voor FASE 6 agent registry persistentie
    Zoals gespecificeerd in STAP 6.1.1: velden voor agent_id, address, capabilities (JSONB), en last_heartbeat
    Enhanced with TAAK 3.2 performance indexes for agent registry queries
    """
    __tablename__ = "registered_agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(String(255), nullable=False, unique=True, index=True)
    address = Column(String(500), nullable=False)
    capabilities = Column(JSONB, nullable=True)
    last_heartbeat = Column(DateTime(timezone=True), nullable=True, index=True)  # TAAK 3.2: Added index for heartbeat queries
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # TAAK 3.2: Performance indexes for agent registry queries
    __table_args__ = (
        # Index for active agent queries (heartbeat-based)
        Index('idx_agent_heartbeat', 'last_heartbeat'),
        # Index for agent health monitoring
        Index('idx_agent_status', 'last_heartbeat', 'created_at'),
    )


class LearningExperience(Base):
    """
    LearningExperience model voor FASE 6 adaptive learning engine persistentie
    Zoals gespecificeerd in STAP 6.1.3: persistentie voor learning events
    Enhanced with TAAK 3.2 performance indexes for adaptive learning queries
    """
    __tablename__ = "learning_experiences"

    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(String(255), nullable=False, unique=True, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    agent_id = Column(String(255), nullable=True, index=True)
    task_type = Column(String(100), nullable=True, index=True)
    success = Column(Boolean, nullable=True, index=True)  # TAAK 3.2: Added index for success filtering
    duration = Column(Float, nullable=True)
    context = Column(JSONB, nullable=True)
    metrics = Column(JSONB, nullable=True)
    user_id = Column(String(255), nullable=True, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    confidence = Column(Float, nullable=False, default=1.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)  # TAAK 3.2: Added index
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    # TAAK 3.2: Performance indexes for adaptive learning queries
    __table_args__ = (
        # Index for agent performance analysis (most common query)
        Index('idx_learning_agent_task', 'agent_id', 'task_type'),
        # Index for success rate analysis over time
        Index('idx_learning_success_time', 'success', 'created_at'),
        # Composite index for comprehensive performance queries
        Index('idx_learning_performance', 'agent_id', 'task_type', 'success', 'created_at'),
        # Index for recent learning experiences
        Index('idx_learning_recent', 'created_at', 'agent_id'),
        # Index for event type analysis
        Index('idx_learning_event_time', 'event_type', 'created_at'),
    )


class CVAConversation(Base):
    """CVA conversation history for persistent chat sessions"""
    __tablename__ = "cva_conversations"

    id = Column(String(50), primary_key=True, default=lambda: f"conv_{uuid.uuid4().hex[:8]}")
    session_id = Column(String(50), nullable=False)  # Links to UserSession
    user_id = Column(String(100), nullable=False, default="user")

    # Conversation content
    user_message = Column(Text, nullable=False)
    cva_response = Column(Text, nullable=False)

    # Response metadata
    response_metadata = Column(JSON, default=dict)  # provider, model, tokens, etc.
    performance_data = Column(JSON, default=dict)   # execution_time, cost, etc.

    # Context tracking
    conversation_turn = Column(Integer, nullable=False)  # 1, 2, 3, etc.
    context_summary = Column(Text, nullable=True)  # Summary of conversation so far

    # Learning data
    user_feedback = Column(String(20), nullable=True)  # positive, negative, neutral
    feedback_details = Column(Text, nullable=True)

    # File attachments
    attached_files = Column(JSON, default=list)  # List of file paths/metadata

    # Classification
    conversation_topic = Column(String(200), nullable=True)
    conversation_category = Column(String(100), nullable=True)
    keywords = Column(JSON, default=list)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Indexes
    __table_args__ = (
        Index('ix_cva_session_id', 'session_id'),
        Index('ix_cva_user_id', 'user_id'),
        Index('ix_cva_created_at', 'created_at'),
        Index('ix_cva_topic', 'conversation_topic'),
        Index('ix_cva_turn', 'conversation_turn'),
    )


class UserSession(Base):
    """User sessions for the Mission Control interface"""
    __tablename__ = "user_sessions"

    id = Column(String(50), primary_key=True, default=lambda: f"session_{uuid.uuid4().hex[:8]}")
    user_id = Column(String(100), nullable=False)
    session_token = Column(String(200), nullable=False, unique=True)

    # Session data
    session_data = Column(JSON, default=dict)
    user_preferences = Column(JSON, default=dict)

    # Activity tracking
    last_activity = Column(DateTime(timezone=True), nullable=True)
    page_views = Column(Integer, default=0)
    actions_count = Column(Integer, default=0)

    # Status
    is_active = Column(Boolean, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)

    # Client info
    client_ip = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Indexes
    __table_args__ = (
        Index('ix_session_token', 'session_token'),
        Index('ix_session_user', 'user_id'),
        Index('ix_session_activity', 'last_activity'),
        Index('ix_session_expires', 'expires_at'),
    )