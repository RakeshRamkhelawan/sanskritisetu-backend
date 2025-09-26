"""
FastAPI Dependencies for Sanskriti Setu API
Database connections, authentication, and shared dependencies
"""

from typing import Generator, Optional, Dict, Any, AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from core.config.unified_config import config
from core.database.database import get_db as get_async_db_session
from core.auth.jwt_handler import jwt_handler

# Security
security = HTTPBearer(auto_error=False)

# Database setup
engine = None
SessionLocal = None

def init_database():
    """Initialize database connection"""
    global engine, SessionLocal
    
    if engine is None:
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        db_config = config.get_database_config()
        engine = create_engine(
            db_config["url"],
            pool_size=db_config["pool_size"],
            pool_pre_ping=True,
            echo=db_config["echo"]
        )
        
        SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=engine
        )

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    UNIVERSAL database dependency - ONLY way to access database
    Uses async session from centralized database.py
    """
    async for session in get_async_db_session():
        yield session

def get_database() -> Generator[Session, None, None]:
    """DEPRECATED: Use get_db() instead for async sessions"""
    if SessionLocal is None:
        init_database()

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token using production JWT handler"""
    return jwt_handler.create_access_token(data, expires_delta)

def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token using production JWT handler"""
    return jwt_handler.verify_token(token)

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user with production JWT validation"""
    if not credentials:
        # No authentication bypass allowed - return None for unauthenticated requests
        return None

    token = credentials.credentials

    # Development mode: Allow special development token
    if config.is_development() and token == "dev-token-safe-development":
        return {
            "user_id": "dev_user",
            "role": "user",
            "security_level": "authenticated",
            "development_mode": True
        }

    # Production JWT validation
    payload = verify_token(token)

    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Add security level for middleware compatibility
    if "security_level" not in payload:
        payload["security_level"] = "authenticated"

    return payload

async def require_authenticated_user(
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Require authenticated user with production validation"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Production mode: Strict validation
    if config.is_production() and current_user.get("user_id") in ["anonymous", "dev_user"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Valid authentication required in production",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return current_user

async def require_admin_user(
    current_user: Dict[str, Any] = Depends(require_authenticated_user)
) -> Dict[str, Any]:
    """Require admin user"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user