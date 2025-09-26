"""
Subagent API Endpoints - RESTful Interface for Subagent Management
Provides HTTP endpoints for discovering, loading, and managing specialized subagents
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from core.agents.subagent_loader import get_loader, AgentStatus

logger = logging.getLogger(__name__)

# Pydantic models for API
class AgentInfo(BaseModel):
    """Agent information model"""
    name: str
    type: str
    status: str
    description: str
    capabilities: List[str]
    priority: int
    error: Optional[str] = None

class AgentListResponse(BaseModel):
    """Response model for agent listing"""
    total_agents: int
    agents: List[AgentInfo]
    status_summary: Dict[str, int]

class AgentLoadRequest(BaseModel):
    """Request model for loading agents"""
    agent_names: List[str] = Field(..., description="List of agent names to load")
    force_reload: bool = Field(False, description="Force reload if already loaded")

class AgentInstantiateRequest(BaseModel):
    """Request model for instantiating agents"""
    agent_name: str = Field(..., description="Name of the agent to instantiate")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")

class AgentActionRequest(BaseModel):
    """Request model for agent actions"""
    agent_name: str = Field(..., description="Name of the target agent")
    action: str = Field(..., description="Action to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Action parameters")

class LoaderMetrics(BaseModel):
    """Loader metrics model"""
    total_agents: int
    loaded_agents: int
    active_agents: int
    status_distribution: Dict[str, int]
    agent_types: List[str]
    loaded_modules: int

# Create router
router = APIRouter(prefix="/api/v1/subagents", tags=["subagents"])

@router.get("/discover", response_model=AgentListResponse)
async def discover_agents():
    """
    Discover all available subagents

    Returns a list of all discoverable agents with their metadata
    """
    try:
        loader = get_loader()

        # Perform discovery
        discovered = await loader.discover_agents()

        # Get current agent list with status
        agent_list = loader.list_available_agents()

        # Create response
        agents = [
            AgentInfo(
                name=agent["name"],
                type=agent["type"],
                status=agent["status"],
                description=agent["description"],
                capabilities=agent["capabilities"],
                priority=agent["priority"],
                error=agent["error"]
            )
            for agent in agent_list
        ]

        # Status summary
        status_summary = {}
        for status in AgentStatus:
            status_summary[status.value] = sum(
                1 for agent in agent_list if agent["status"] == status.value
            )

        return AgentListResponse(
            total_agents=len(agents),
            agents=agents,
            status_summary=status_summary
        )

    except Exception as e:
        logger.error(f"Error discovering agents: {e}")
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")

@router.get("/list", response_model=AgentListResponse)
async def list_agents():
    """
    List all known agents with their current status

    Returns agents that have been discovered previously
    """
    try:
        loader = get_loader()
        agent_list = loader.list_available_agents()

        agents = [
            AgentInfo(
                name=agent["name"],
                type=agent["type"],
                status=agent["status"],
                description=agent["description"],
                capabilities=agent["capabilities"],
                priority=agent["priority"],
                error=agent["error"]
            )
            for agent in agent_list
        ]

        # Status summary
        status_summary = {}
        for status in AgentStatus:
            status_summary[status.value] = sum(
                1 for agent in agent_list if agent["status"] == status.value
            )

        return AgentListResponse(
            total_agents=len(agents),
            agents=agents,
            status_summary=status_summary
        )

    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=f"Listing failed: {str(e)}")

@router.post("/load")
async def load_agents(request: AgentLoadRequest, background_tasks: BackgroundTasks):
    """
    Load specific agents by name

    Loads the specified agents into memory, making them available for instantiation
    """
    try:
        loader = get_loader()
        results = {}

        for agent_name in request.agent_names:
            if request.force_reload:
                success = await loader.reload_agent(agent_name)
            else:
                success = await loader.load_agent(agent_name)

            results[agent_name] = {
                "success": success,
                "status": loader.get_agent_status(agent_name).value if loader.get_agent_status(agent_name) else "unknown"
            }

        return {
            "message": f"Processed {len(request.agent_names)} agents",
            "results": results,
            "total_loaded": sum(1 for r in results.values() if r["success"])
        }

    except Exception as e:
        logger.error(f"Error loading agents: {e}")
        raise HTTPException(status_code=500, detail=f"Loading failed: {str(e)}")

@router.post("/instantiate")
async def instantiate_agent(request: AgentInstantiateRequest):
    """
    Instantiate a loaded agent

    Creates an active instance of the specified agent with the given configuration
    """
    try:
        loader = get_loader()

        # Check if agent is loaded
        status = loader.get_agent_status(request.agent_name)
        if not status or status == AgentStatus.UNLOADED:
            raise HTTPException(
                status_code=400,
                detail=f"Agent {request.agent_name} is not loaded. Load it first."
            )

        # Instantiate the agent
        instance = await loader.instantiate_agent(request.agent_name, **request.config)

        if not instance:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to instantiate agent {request.agent_name}"
            )

        return {
            "message": f"Agent {request.agent_name} instantiated successfully",
            "agent_name": request.agent_name,
            "status": "active",
            "instance_type": type(instance).__name__
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error instantiating agent {request.agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Instantiation failed: {str(e)}")

@router.get("/status/{agent_name}")
async def get_agent_status(agent_name: str):
    """
    Get the status of a specific agent

    Returns detailed status information for the specified agent
    """
    try:
        loader = get_loader()

        # Check if agent exists
        if agent_name not in loader.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

        agent_def = loader.agents[agent_name]
        instance = loader.get_agent(agent_name)

        # Get instance info if available
        instance_info = None
        if instance:
            instance_info = {
                "type": type(instance).__name__,
                "has_execute": hasattr(instance, "execute"),
                "has_status": hasattr(instance, "get_status"),
                "methods": [method for method in dir(instance) if not method.startswith("_")]
            }

        return {
            "name": agent_def.name,
            "type": agent_def.agent_type,
            "status": agent_def.status.value,
            "description": agent_def.description,
            "capabilities": agent_def.capabilities,
            "dependencies": agent_def.dependencies,
            "priority": agent_def.priority,
            "module_path": agent_def.module_path,
            "class_name": agent_def.class_name,
            "load_error": agent_def.load_error,
            "instance_info": instance_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/action")
async def execute_agent_action(request: AgentActionRequest):
    """
    Execute an action on an active agent

    Performs the specified action on the target agent with given parameters
    """
    try:
        loader = get_loader()
        agent = loader.get_agent(request.agent_name)

        if not agent:
            raise HTTPException(
                status_code=404,
                detail=f"Active agent {request.agent_name} not found"
            )

        # Execute the action based on the action type
        if request.action == "execute":
            if hasattr(agent, "execute"):
                if asyncio.iscoroutinefunction(agent.execute):
                    result = await agent.execute(**request.parameters)
                else:
                    result = agent.execute(**request.parameters)
                return {"success": True, "result": result}
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Agent {request.agent_name} does not support execute action"
                )

        elif request.action == "status":
            if hasattr(agent, "get_status"):
                if asyncio.iscoroutinefunction(agent.get_status):
                    status = await agent.get_status()
                else:
                    status = agent.get_status()
                return {"success": True, "status": status}
            else:
                return {"success": True, "status": "active"}

        elif request.action == "capabilities":
            if hasattr(agent, "get_capabilities"):
                if asyncio.iscoroutinefunction(agent.get_capabilities):
                    capabilities = await agent.get_capabilities()
                else:
                    capabilities = agent.get_capabilities()
                return {"success": True, "capabilities": capabilities}
            else:
                agent_def = loader.agents[request.agent_name]
                return {"success": True, "capabilities": agent_def.capabilities}

        else:
            # Try to call a custom method
            if hasattr(agent, request.action):
                method = getattr(agent, request.action)
                if callable(method):
                    if asyncio.iscoroutinefunction(method):
                        result = await method(**request.parameters)
                    else:
                        result = method(**request.parameters)
                    return {"success": True, "result": result}

            raise HTTPException(
                status_code=400,
                detail=f"Action {request.action} not supported by agent {request.agent_name}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing action {request.action} on {request.agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Action execution failed: {str(e)}")

@router.delete("/unload/{agent_name}")
async def unload_agent(agent_name: str):
    """
    Unload an agent

    Stops and removes the agent from active memory
    """
    try:
        loader = get_loader()

        if agent_name not in loader.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

        # Stop active instance if it exists
        if agent_name in loader.active_agents:
            agent = loader.active_agents[agent_name]
            if hasattr(agent, "stop"):
                if asyncio.iscoroutinefunction(agent.stop):
                    await agent.stop()
                else:
                    agent.stop()

            del loader.active_agents[agent_name]

        # Reset agent status
        agent_def = loader.agents[agent_name]
        agent_def.status = AgentStatus.UNLOADED
        agent_def.instance = None
        agent_def.load_error = None

        return {
            "message": f"Agent {agent_name} unloaded successfully",
            "agent_name": agent_name,
            "status": "unloaded"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unloading agent {agent_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Unload failed: {str(e)}")

@router.get("/metrics", response_model=LoaderMetrics)
async def get_loader_metrics():
    """
    Get subagent loader metrics

    Returns comprehensive metrics about the loader and agent states
    """
    try:
        loader = get_loader()
        metrics = loader.get_metrics()

        return LoaderMetrics(
            total_agents=metrics["total_agents"],
            loaded_agents=metrics["loaded_agents"],
            active_agents=metrics["active_agents"],
            status_distribution=metrics["status_distribution"],
            agent_types=metrics["agent_types"],
            loaded_modules=metrics["loaded_modules"]
        )

    except Exception as e:
        logger.error(f"Error getting loader metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")

@router.post("/load-all")
async def load_all_agents():
    """
    Load all discovered agents

    Attempts to load all agents that have been discovered, respecting priority order
    """
    try:
        loader = get_loader()

        # Ensure agents are discovered first
        await loader.discover_agents()

        # Load all agents
        results = await loader.load_all_agents()

        successful_loads = sum(1 for success in results.values() if success)

        return {
            "message": f"Loaded {successful_loads} out of {len(results)} agents",
            "results": results,
            "total_agents": len(results),
            "successful_loads": successful_loads,
            "failed_loads": len(results) - successful_loads
        }

    except Exception as e:
        logger.error(f"Error loading all agents: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk loading failed: {str(e)}")

@router.get("/types/{agent_type}")
async def get_agents_by_type(agent_type: str):
    """
    Get all agents of a specific type

    Returns information about all agents matching the specified type
    """
    try:
        loader = get_loader()

        # Get agents of the specified type
        matching_agents = [
            agent for agent in loader.list_available_agents()
            if agent["type"] == agent_type
        ]

        # Get active instances
        active_instances = loader.get_agents_by_type(agent_type)

        return {
            "agent_type": agent_type,
            "total_agents": len(matching_agents),
            "active_instances": len(active_instances),
            "agents": matching_agents,
            "active_agent_names": [
                name for name, agent in loader.active_agents.items()
                if loader.agents[name].agent_type == agent_type
            ]
        }

    except Exception as e:
        logger.error(f"Error getting agents by type {agent_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Type query failed: {str(e)}")

# Health check endpoint
@router.get("/health")
async def subagent_health():
    """
    Health check for the subagent system

    Returns the current health status of the subagent management system
    """
    try:
        loader = get_loader()
        metrics = loader.get_metrics()

        # Determine health status
        health_status = "healthy"
        if metrics["active_agents"] == 0:
            health_status = "warning"

        failed_agents = metrics["status_distribution"].get("error", 0)
        if failed_agents > 0:
            health_status = "degraded" if failed_agents < metrics["total_agents"] / 2 else "unhealthy"

        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "message": f"Subagent system operational with {metrics['active_agents']} active agents"
        }

    except Exception as e:
        logger.error(f"Error in subagent health check: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "message": "Subagent health check failed"
        }

# Export router for main.py compatibility
subagent_router = router