"""
Ultimate Perfect Orchestrator API Endpoints
FastAPI routes for orchestrator integration with real API communication
Week 2 Day 7 Implementation
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from core.agents.ultimate_perfect_orchestrator import UltimatePerfectOrchestrator, create_ultimate_perfect_orchestrator

logger = logging.getLogger(__name__)

# Router for orchestrator endpoints
router = APIRouter(prefix="/api/v1/orchestrator", tags=["orchestrator"])

# Global orchestrator instance
_orchestrator: Optional[UltimatePerfectOrchestrator] = None

async def get_orchestrator() -> UltimatePerfectOrchestrator:
    """Get or create orchestrator instance"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = await create_ultimate_perfect_orchestrator()
    return _orchestrator


class OrchestrationRequest(BaseModel):
    """Request model for orchestration"""
    command: str
    parameters: Optional[Dict[str, Any]] = {}
    masterprompt: Optional[str] = ""
    priority: Optional[int] = 3
    ceo_initiated: Optional[bool] = False


class OrchestrationResponse(BaseModel):
    """Response model for orchestration"""
    success: bool
    orchestration_id: Optional[str] = None
    command: str
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    orchestrator_mode: str
    error: Optional[str] = None


class OrchestratorStatusResponse(BaseModel):
    """Response model for orchestrator status"""
    mode: str
    active_tasks: int
    completed_tasks: int
    agent_load: Dict[str, float]
    performance_metrics: Dict[str, Any]
    api_connections: Dict[str, bool]
    llm_integration: str
    orchestrator_health: str
    real_api_integration: bool


@router.post("/execute", response_model=OrchestrationResponse)
async def execute_orchestration(request: OrchestrationRequest):
    """Execute orchestration request"""
    try:
        # Get orchestrator
        orchestrator = await get_orchestrator()

        # Convert request to dict
        orchestration_request = {
            "command": request.command,
            "parameters": request.parameters,
            "masterprompt": request.masterprompt,
            "priority": request.priority,
            "ceo_initiated": request.ceo_initiated
        }

        # Process orchestration
        result = await orchestrator.process_orchestration_request(orchestration_request)

        return OrchestrationResponse(
            success=result.get("success", False),
            orchestration_id=result.get("orchestration_id"),
            command=result.get("command", request.command),
            result=result.get("result"),
            execution_time=result.get("execution_time"),
            orchestrator_mode=result.get("orchestrator_mode", "unknown"),
            error=result.get("error")
        )

    except Exception as e:
        logger.error(f"Orchestration execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")


@router.get("/status", response_model=OrchestratorStatusResponse)
async def get_orchestrator_status():
    """Get orchestrator status and metrics"""
    try:
        orchestrator = await get_orchestrator()
        status = orchestrator.get_orchestrator_status()

        return OrchestratorStatusResponse(
            mode=status["mode"],
            active_tasks=status["active_tasks"],
            completed_tasks=status["completed_tasks"],
            agent_load=status["agent_load"],
            performance_metrics=status["performance_metrics"],
            api_connections=status["api_connections"],
            llm_integration=status["llm_integration"],
            orchestrator_health=status["orchestrator_health"],
            real_api_integration=status["real_api_integration"]
        )

    except Exception as e:
        logger.error(f"Orchestrator status error: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")


@router.get("/dashboard")
async def get_orchestrator_dashboard():
    """Get orchestrator dashboard data"""
    try:
        orchestrator = await get_orchestrator()
        status = orchestrator.get_orchestrator_status()

        # Create dashboard summary
        dashboard = {
            "orchestrator_overview": {
                "mode": status["mode"],
                "health": status["orchestrator_health"],
                "api_integration": status["real_api_integration"],
                "llm_system": status["llm_integration"]
            },
            "task_metrics": {
                "active_tasks": status["active_tasks"],
                "completed_tasks": status["completed_tasks"],
                "success_rate": status["performance_metrics"].get("success_rate", 0),
                "average_completion_time": status["performance_metrics"].get("average_completion_time", 0)
            },
            "agent_utilization": status["agent_load"],
            "api_connections": status["api_connections"],
            "performance_summary": {
                "total_tasks": status["performance_metrics"].get("tasks_completed", 0) + status["performance_metrics"].get("tasks_failed", 0),
                "real_api_calls": status["performance_metrics"].get("real_api_calls", 0),
                "cva_directives": status["performance_metrics"].get("cva_directives_processed", 0)
            }
        }

        return {
            "dashboard": dashboard,
            "timestamp": "2025-01-17T12:00:00Z",  # Would use datetime.utcnow() in production
            "status": "operational"
        }

    except Exception as e:
        logger.error(f"Dashboard retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Dashboard failed: {str(e)}")


@router.get("/agents")
async def get_available_agents():
    """Get available agents and their status"""
    try:
        orchestrator = await get_orchestrator()
        status = orchestrator.get_orchestrator_status()

        # Get agent information from task routing rules
        agents_info = []
        for task_type, rules in orchestrator.task_routing_rules.items():
            for agent in rules["agents"]:
                current_load = status["agent_load"].get(agent, 0.0)
                utilization = status["performance_metrics"]["agent_utilization"].get(agent, {"tasks": 0, "successes": 0})

                agents_info.append({
                    "agent_id": agent,
                    "specialization": task_type,
                    "status": "active" if current_load >= 0 else "inactive",
                    "current_load": current_load,
                    "total_tasks": utilization.get("tasks", 0),
                    "success_count": utilization.get("successes", 0),
                    "success_rate": (utilization.get("successes", 0) / max(utilization.get("tasks", 1), 1)) * 100,
                    "api_endpoint": rules["api_endpoint"],
                    "capabilities": [task_type, "general"]
                })

        return {
            "agents": agents_info,
            "total_agents": len(agents_info),
            "active_agents": len([a for a in agents_info if a["status"] == "active"]),
            "specializations": list(orchestrator.task_routing_rules.keys())
        }

    except Exception as e:
        logger.error(f"Agents retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Agents retrieval failed: {str(e)}")


@router.post("/task")
async def delegate_task(
    task_description: str,
    task_type: Optional[str] = "general",
    priority: Optional[int] = 3,
    masterprompt: Optional[str] = ""
):
    """Delegate a task through the orchestrator"""
    try:
        orchestrator = await get_orchestrator()

        # Create orchestration request
        request = {
            "command": f"delegate_{task_type}",
            "parameters": {
                "description": task_description,
                "task_type": task_type
            },
            "masterprompt": masterprompt,
            "priority": priority,
            "ceo_initiated": False
        }

        # Process through orchestrator
        result = await orchestrator.process_orchestration_request(request)

        return {
            "task_delegation": result,
            "task_type": task_type,
            "priority": priority,
            "masterprompt_used": bool(masterprompt)
        }

    except Exception as e:
        logger.error(f"Task delegation error: {e}")
        raise HTTPException(status_code=500, detail=f"Task delegation failed: {str(e)}")


@router.get("/metrics")
async def get_orchestrator_metrics():
    """Get detailed orchestrator performance metrics"""
    try:
        orchestrator = await get_orchestrator()
        status = orchestrator.get_orchestrator_status()

        metrics = {
            "performance_metrics": status["performance_metrics"],
            "agent_utilization": status["agent_load"],
            "api_health": status["api_connections"],
            "orchestrator_mode": status["mode"],
            "system_health": status["orchestrator_health"]
        }

        return {
            "metrics": metrics,
            "collection_timestamp": "2025-01-17T12:00:00Z",
            "metrics_version": "1.0"
        }

    except Exception as e:
        logger.error(f"Metrics retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics retrieval failed: {str(e)}")


# Health check for orchestrator service
@router.get("/health")
async def orchestrator_health_check():
    """Orchestrator service health check"""
    try:
        orchestrator = await get_orchestrator()
        status = orchestrator.get_orchestrator_status()

        health_status = "healthy" if status["orchestrator_health"] == "operational" else "degraded"

        # Check API connections
        api_health = sum(1 for connected in status["api_connections"].values() if connected)
        total_apis = len(status["api_connections"])

        return {
            "service": "orchestrator",
            "status": health_status,
            "mode": status["mode"],
            "api_connections": f"{api_health}/{total_apis}",
            "active_tasks": status["active_tasks"],
            "real_api_integration": status["real_api_integration"],
            "llm_integration": status["llm_integration"]
        }

    except Exception as e:
        logger.error(f"Orchestrator health check error: {e}")
        return {
            "service": "orchestrator",
            "status": "error",
            "error": str(e)
        }


@router.post("/shutdown")
async def shutdown_orchestrator():
    """Gracefully shutdown orchestrator"""
    global _orchestrator
    try:
        if _orchestrator:
            _orchestrator.shutdown()
            _orchestrator = None

        return {
            "message": "Orchestrator shutdown completed",
            "timestamp": "2025-01-17T12:00:00Z"
        }

    except Exception as e:
        logger.error(f"Orchestrator shutdown error: {e}")
        raise HTTPException(status_code=500, detail=f"Shutdown failed: {str(e)}")


@router.post("/cva-directive")
async def process_cva_directive(
    directive: str,
    masterprompt: str,
    priority: Optional[int] = 1
):
    """Process a CVA CEO directive with masterprompt"""
    try:
        orchestrator = await get_orchestrator()

        # Create CVA directive request
        request = {
            "command": "cva_directive",
            "parameters": {
                "description": directive,
                "directive_type": "ceo_command"
            },
            "masterprompt": masterprompt,
            "priority": priority,
            "ceo_initiated": True
        }

        # Process directive
        result = await orchestrator.process_orchestration_request(request)

        return {
            "cva_directive_result": result,
            "directive": directive,
            "masterprompt_processed": True,
            "priority": priority
        }

    except Exception as e:
        logger.error(f"CVA directive processing error: {e}")
        raise HTTPException(status_code=500, detail=f"CVA directive failed: {str(e)}")


# Import datetime for timestamps
from datetime import datetime


@router.post("/knowledge-query")
async def process_knowledge_query(
    query: str,
    priority: Optional[int] = 1
):
    """Process knowledge query via CVA agent through orchestrator"""
    try:
        orchestrator = await get_orchestrator()

        # Create knowledge query request
        request = {
            "command": "knowledge_query",
            "parameters": {
                "query": query,
                "task_type": "knowledge_query"
            },
            "masterprompt": "Provide direct, factual answer to this knowledge question. Use ASCII-only characters.",
            "priority": priority,
            "ceo_initiated": False
        }

        # Process through orchestrator
        result = await orchestrator.process_orchestration_request(request)

        return {
            "knowledge_query_result": result,
            "query": query,
            "orchestrator_routing": "cva_agent",
            "priority": priority
        }

    except Exception as e:
        logger.error(f"Knowledge query processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge query failed: {str(e)}")