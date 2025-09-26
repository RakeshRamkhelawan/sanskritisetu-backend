"""
Agent Endpoints for FastAPI
Handles agent registration and retrieval via database-backed AgentRegistry
FASE 7: Functional implementation with complete agent registration logic
"""
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from core.agents.agent_registry import get_agent_registry, AgentRegistry
from core.shared.interfaces import Agent, AgentCapability, AgentType
from core.database.database import get_async_session
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents"])


async def get_agent_registry_dependency(
    session: AsyncSession = Depends(get_async_session)
) -> AgentRegistry:
    """FastAPI dependency to get AgentRegistry with database session"""
    return get_agent_registry(session)


class AgentRegistrationRequest(BaseModel):
    agent_id: str
    address: str
    capabilities: List[str]
    agent_type: str = "specialist"


class AgentRegistrationResponse(BaseModel):
    success: bool
    message: str
    agent_id: str


class AgentListResponse(BaseModel):
    agents: List[Dict[str, Any]]
    total_count: int


@router.post("/register", response_model=AgentRegistrationResponse)
async def register_agent(
    request: AgentRegistrationRequest,
    registry: AgentRegistry = Depends(get_agent_registry_dependency)
):
    """
    Register a new agent in the system via AgentRegistry database persistence.

    This endpoint receives agent registration data and creates a proper Agent
    object to register through the AgentRegistry system.
    """
    try:
        logger.info(f"Registering agent: {request.agent_id} at {request.address}")

        # Convert agent_type string to AgentType enum
        try:
            agent_type = AgentType(request.agent_type.lower())
        except ValueError:
            logger.warning(f"Unknown agent type '{request.agent_type}', defaulting to SPECIALIST")
            agent_type = AgentType.SPECIALIST

        # Create AgentCapability objects from strings
        capabilities = [
            AgentCapability(name=cap, description=f"Capability: {cap}")
            for cap in request.capabilities
        ]

        # Create a minimal Agent implementation for registration
        class RegistrationAgent:
            def __init__(self, agent_id: str, address: str, agent_type: AgentType, capabilities: List[AgentCapability]):
                self.agent_id = agent_id
                self.address = address
                self.agent_type = agent_type
                self._capabilities = capabilities

            async def get_capabilities(self) -> List[AgentCapability]:
                return self._capabilities

            async def process_task(self, task_data):
                # This is a minimal implementation for registration only
                raise NotImplementedError("Agent registered but not fully implemented")

        # Create agent instance for registration
        agent = RegistrationAgent(
            agent_id=request.agent_id,
            address=request.address,
            agent_type=agent_type,
            capabilities=capabilities
        )

        # Register through AgentRegistry
        success = await registry.register_agent(agent)

        if success:
            logger.info(f"Successfully registered agent {request.agent_id}")
            return AgentRegistrationResponse(
                success=True,
                message=f"Agent {request.agent_id} registered successfully",
                agent_id=request.agent_id
            )
        else:
            logger.error(f"Failed to register agent {request.agent_id}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to register agent {request.agent_id}"
            )

    except Exception as e:
        logger.error(f"Error during agent registration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Agent registration failed: {str(e)}"
        )


@router.get("/", response_model=AgentListResponse)
async def get_all_agents(registry: AgentRegistry = Depends(get_agent_registry_dependency)):
    """
    Get all registered agents from the database-backed AgentRegistry.

    Returns comprehensive agent information including capabilities,
    last heartbeat, and registration timestamps.
    """
    try:
        logger.info("Retrieving all registered agents")

        # Get agents from registry database
        agents = await registry.get_all_agents()

        logger.info(f"Retrieved {len(agents)} registered agents")

        return AgentListResponse(
            agents=agents,
            total_count=len(agents)
        )

    except Exception as e:
        logger.error(f"Error retrieving agents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve agents: {str(e)}"
        )


@router.get("/health")
async def agent_system_health(registry: AgentRegistry = Depends(get_agent_registry_dependency)):
    """
    Get system health status for the agent infrastructure.

    Returns comprehensive system metrics including agent counts,
    load distribution, and success rates.
    """
    try:
        logger.info("Getting agent system health status")

        status = await registry.get_system_status()

        return {
            "status": "healthy",
            "system_metrics": status,
            "timestamp": status.get("uptime")
        }

    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system health: {str(e)}"
        )