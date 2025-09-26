"""
Unified Database Connection Manager - Centralized session management
Eliminates duplicate engines and standardizes session lifecycle
"""

import os
import logging
import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError, OperationalError, TimeoutError as SQLTimeoutError
from sqlalchemy import text

from core.config.unified_config import config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Centralized database manager - single source of truth for all database operations
    Eliminates duplicate engines and provides consistent session management
    """

    def __init__(self):
        self.engine = None
        self.async_session_maker = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the centralized database engine"""
        if self._initialized:
            return True

        try:
            db_config = config.get_database_config()

            self.engine = create_async_engine(
                db_config["url"],
                echo=db_config["echo"],
                pool_size=db_config["pool_size"],
                max_overflow=db_config["max_overflow"],
                pool_pre_ping=True,
                pool_recycle=3600  # Recycle connections every hour
            )

            self.async_session_maker = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            self._initialized = True
            logger.info("Unified database manager initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize unified database manager: {e}")
            return False

    @asynccontextmanager
    async def get_session(self, max_retries: int = 3) -> AsyncGenerator[AsyncSession, None]:
        """
        Centralized session context manager with comprehensive error handling and retry logic

        Args:
            max_retries: Maximum number of retry attempts for connection issues
        """
        if not self._initialized:
            if not await self.initialize():
                raise RuntimeError("Database manager not initialized")

        retry_count = 0
        last_exception = None

        while retry_count <= max_retries:
            session = None
            try:
                session = self.async_session_maker()

                # Test session with a simple query
                await session.execute(text("SELECT 1"))

                yield session
                await session.commit()
                return  # Success - exit retry loop

            except (DisconnectionError, OperationalError, SQLTimeoutError) as e:
                # Connection-related errors - attempt retry
                retry_count += 1
                last_exception = e

                if session:
                    try:
                        await session.rollback()
                        await session.close()
                    except Exception:
                        pass  # Ignore cleanup errors

                if retry_count <= max_retries:
                    wait_time = min(2 ** retry_count, 10)  # Exponential backoff, max 10s
                    logger.warning(f"Database connection error (attempt {retry_count}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)

                    # Try to reinitialize connection
                    try:
                        await self.engine.dispose()
                        self._initialized = False
                        await self.initialize()
                    except Exception as init_error:
                        logger.error(f"Failed to reinitialize database connection: {init_error}")
                else:
                    logger.error(f"Database connection failed after {max_retries} retries: {e}")
                    raise

            except SQLAlchemyError as e:
                # Non-connection SQLAlchemy errors - don't retry
                if session:
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                logger.error(f"Database SQLAlchemy error: {e}")
                raise

            except Exception as e:
                # Other unexpected errors - don't retry
                if session:
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                logger.error(f"Unexpected database session error: {e}")
                raise

            finally:
                if session:
                    try:
                        await session.close()
                    except Exception as close_error:
                        logger.warning(f"Error closing session: {close_error}")

        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError("Database session creation failed")

    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            self._initialized = False
            logger.info("Database connections closed")

# Global database manager instance
db_manager = DatabaseManager()

# Legacy compatibility functions
engine = None
async_session_maker = None

async def initialize_database() -> bool:
    """
    Legacy compatibility function - delegates to unified database manager
    """
    global engine, async_session_maker

    result = await db_manager.initialize()
    if result:
        # Set legacy global variables for backward compatibility
        engine = db_manager.engine
        async_session_maker = db_manager.async_session_maker

    return result

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Standardized session generator using unified database manager
    """
    async with db_manager.get_session() as session:
        yield session

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions
    Uses unified database manager for consistent session handling
    """
    async for session in get_async_session():
        yield session

async def close_database():
    """
    Legacy compatibility function - delegates to unified database manager
    """
    await db_manager.close()
