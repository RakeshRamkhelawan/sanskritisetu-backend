"""
Unified FastAPI Server - FASE 9 Implementation
Consolidated single-server architecture with agent sub-modules
Compliant with Single Server Policy from Instructions.md
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Unified configuration system
from core.config.unified_config import config

# Security middleware
from core.security.security_middleware import SecurityMiddleware

# Unified database initialization
from core.database.database import initialize_database, close_database
from core.database.connection import db_manager
from core.autonomy.autonomous_agent import create_autonomous_agent

# Import all core routers
from core.api.auth_endpoints import router as auth_router
from core.api.admin_endpoints import router as admin_router
from core.api.cache_endpoints import router as cache_router
from core.api.agent_endpoints import router as agent_router
from core.api.goal_endpoints import router as goal_router

# Import agent routers - NOW INTERNAL MODULES
from agents.knowledge_agent.routes import router as knowledge_router
from agents.sentiment_agent.routes import router as sentiment_router
from agents.diagnostics_agent.routes import router as diagnostics_router
from agents.translation_agent.routes import router as translation_router

# Import for internal agent registration
from core.collaboration.agent_mesh import get_agent_registry, RegisteredAgent


async def register_internal_agents():
    """
    Register internal agents in AgentRegistry for AutonomousAgent system
    Replaces self-registration mechanism for unified architecture
    """
    agent_registry = get_agent_registry()

    # Define internal agents with their capabilities
    internal_agents = [
        {
            "agent_id": "knowledge_agent",
            "agent_name": "Knowledge Agent",
            "agent_type": "knowledge_processor",
            "address": "localhost",
            "port": 8000,  # Now unified server port
            "endpoints": {
                "execute_task": "/api/v1/knowledge/execute-task",
                "health": "/api/v1/knowledge/health",
                "status": "/api/v1/knowledge/status"
            },
            "capabilities": ["knowledge_management", "crud_operations"],
            "supported_task_types": ["create_knowledge", "get_knowledge", "update_knowledge", "delete_knowledge"],
            "max_concurrent_tasks": 5
        },
        {
            "agent_id": "sentiment_agent",
            "agent_name": "Sentiment Agent",
            "agent_type": "sentiment_processor",
            "address": "localhost",
            "port": 8000,  # Now unified server port
            "endpoints": {
                "execute_task": "/api/v1/sentiment/execute_task",
                "health": "/api/v1/sentiment/health",
                "status": "/api/v1/sentiment/status"
            },
            "capabilities": ["sentiment_analysis", "text_processing"],
            "supported_task_types": ["sentiment_analysis"],
            "max_concurrent_tasks": 10
        },
        {
            "agent_id": "diagnostics_agent",
            "agent_name": "Diagnostics Agent",
            "agent_type": "diagnostics_processor",
            "address": "localhost",
            "port": 8000,  # Now unified server port
            "endpoints": {
                "execute_task": "/api/v1/diagnostics/execute_task",
                "health": "/api/v1/diagnostics/health",
                "status": "/api/v1/diagnostics/status"
            },
            "capabilities": ["system_diagnostics"],
            "supported_task_types": ["diagnose_system"],
            "max_concurrent_tasks": 3
        },
        {
            "agent_id": "translation_agent",
            "agent_name": "Translation Agent",
            "agent_type": "translation_processor",
            "address": "localhost",
            "port": 8000,  # Now unified server port
            "endpoints": {
                "execute_task": "/api/v1/translation/execute_task",
                "health": "/api/v1/translation/health",
                "status": "/api/v1/translation/status"
            },
            "capabilities": ["text_translation", "language_processing"],
            "supported_task_types": ["translate_text", "detect_language"],
            "max_concurrent_tasks": 8
        }
    ]

    # Register each internal agent
    for agent_data in internal_agents:
        registered_agent = RegisteredAgent(**agent_data)
        await agent_registry.register_agent(registered_agent)
        print(f"INFO: Registered internal agent: {agent_data['agent_id']}")

    print(f"INFO: Successfully registered {len(internal_agents)} internal agents")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Unified lifespan manager for single server architecture
    Handles database initialization and autonomous agent setup
    """
    print("INFO: Unified Server starting up...")

    # Validate configuration before starting services
    print("INFO: Validating configuration...")
    try:
        config.validate_startup_configuration()
        print("INFO: Configuration validation successful")
    except Exception as e:
        print(f"CRITICAL ERROR: Configuration validation failed: {e}")
        raise RuntimeError(f"Server startup aborted due to configuration issues: {e}")

    # Initialize database
    if await initialize_database():
        # Store database access in app state for dependency injection
        app.state.db_initialized = True
        print("INFO: Async database system initialized.")

        print("INFO: Starting internal agent registration...")
        # Register internal agents in AgentRegistry
        await register_internal_agents()
        print("INFO: Internal agent registration completed.")

        app.state.autonomous_agent = create_autonomous_agent()
        print("INFO: Database and autonomous agent successfully initialized.")
    else:
        raise RuntimeError("CRITICAL ERROR: Database initialization failed. Server stopping.")

    print("INFO: All agent modules loaded as internal routers")
    print("INFO: Unified Server startup complete")

    yield

    # Shutdown
    print("INFO: Unified Server shutting down...")
    await close_database()
    print("INFO: Database connection closed.")


# Create unified FastAPI application
app = FastAPI(
    title="Sanskriti Setu Unified Backend",
    description="Production-grade unified server architecture - Single Server Policy compliant",
    version="9.0.0",
    lifespan=lifespan
)

# Add security middleware (production security)
if config.security_middleware_enabled and not config.is_development():
    app.add_middleware(SecurityMiddleware, enable_security=True)
    print("INFO: Production security middleware enabled")
elif config.is_development():
    print("INFO: Development mode - security middleware disabled")

# Add CORS middleware with environment-specific configuration
cors_config = config.get_cors_config()
app.add_middleware(
    CORSMiddleware,
    **cors_config
)
print(f"INFO: CORS configured for {config.environment.value} environment")

# --- Core API Router Registration ---
API_PREFIX = "/api/v1"
app.include_router(auth_router, prefix=f"{API_PREFIX}/auth", tags=["Authentication"])
app.include_router(admin_router, prefix=f"{API_PREFIX}/admin", tags=["Admin"])
app.include_router(cache_router, prefix=f"{API_PREFIX}/cache", tags=["Cache"])
app.include_router(agent_router, prefix=f"{API_PREFIX}/agents", tags=["Agents"])
app.include_router(goal_router, prefix=f"{API_PREFIX}/goals", tags=["Goals"])

# --- Agent Module Router Registration ---
# These were previously separate FastAPI servers on ports 8003, 8004, 8005
# Now consolidated as internal sub-routers under single server (port 8000)
app.include_router(
    knowledge_router,
    prefix=f"{API_PREFIX}/knowledge",
    tags=["Knowledge Agent"]
)
app.include_router(
    sentiment_router,
    prefix=f"{API_PREFIX}/sentiment",
    tags=["Sentiment Agent"]
)
app.include_router(
    diagnostics_router,
    prefix=f"{API_PREFIX}/diagnostics",
    tags=["Diagnostics Agent"]
)
app.include_router(
    translation_router,
    prefix=f"{API_PREFIX}/translation",
    tags=["Translation Agent"]
)

# --- Root Endpoints ---
@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint to verify unified API accessibility"""
    return {
        "message": "Sanskriti Setu Unified Backend API",
        "status": "operational",
        "architecture": "unified_single_server",
        "version": "9.0.0",
        "agents": {
            "knowledge": "/api/v1/knowledge",
            "sentiment": "/api/v1/sentiment",
            "diagnostics": "/api/v1/diagnostics",
            "translation": "/api/v1/translation"
        }
    }


@app.get("/health", tags=["Health Check"])
async def health_check():
    """Health check endpoint for monitoring and service discovery"""
    return {
        "status": "healthy",
        "service": "sanskriti-setu-unified-backend",
        "architecture": "single_server",
        "compliance": "instructions_md_single_server_policy"
    }


# Server startup for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)