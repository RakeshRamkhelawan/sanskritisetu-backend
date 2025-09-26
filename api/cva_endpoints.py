"""
CVA Agent API Endpoints
FastAPI routes for Ultimate CVA Agent integration with dynamic LLM system
Week 2 Day 7 Implementation - Ultimate CVA Integration
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from core.agents.ultimate_cva_agent import UltimateCVAAgent, create_ultimate_cva_agent
from core.shared.interfaces import TaskData, AgentType

logger = logging.getLogger(__name__)

# Router for CVA endpoints
router = APIRouter(prefix="/api/v1/cva", tags=["cva"])

# Global CVA agent instance
_cva_agent: Optional[UltimateCVAAgent] = None

async def get_cva_agent() -> UltimateCVAAgent:
    """Get or create CVA agent instance"""
    global _cva_agent
    if _cva_agent is None:
        _cva_agent = await create_ultimate_cva_agent()
    return _cva_agent


class CVAChatRequest(BaseModel):
    """Request model for CVA chat"""
    message: str
    conversation_history: Optional[List[Dict[str, str]]] = []
    user_id: Optional[str] = "anonymous"
    context: Optional[Dict[str, Any]] = {}


class CVAChatResponse(BaseModel):
    """Response model for CVA chat"""
    response: str
    success: bool
    confidence: float
    execution_time: float
    suggestions: List[str]
    llm_provider_used: Optional[str] = None
    orchestration_used: bool = False
    performance_metrics: Optional[Dict[str, Any]] = None
    conversation_id: Optional[str] = None


class CVAStatusResponse(BaseModel):
    """Response model for CVA status"""
    agent_id: str
    mode: str
    active: bool
    capabilities: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    execution_history_size: int
    last_optimization: str


@router.post("/chat", response_model=CVAChatResponse)
async def chat_with_cva(request: CVAChatRequest):
    """Chat with Ultimate CVA Agent"""
    try:
        # Get CVA agent
        cva_agent = await get_cva_agent()

        # Process the command with CVA
        result = await cva_agent.process_command(
            user_input=request.message,
            context={
                "conversation_history": request.conversation_history,
                "user_id": request.user_id,
                **request.context
            }
        )

        # Generate conversation ID for persistence
        import uuid
        conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

        # Extract response content
        response_content = result.response.get('llm_analysis',
                           result.response.get('message',
                           str(result.response)))

        # Clean up response content for ASCII compliance
        if isinstance(response_content, str):
            response_content = response_content.encode('ascii', errors='replace').decode('ascii')

        return CVAChatResponse(
            response=response_content,
            success=result.success,
            confidence=result.confidence,
            execution_time=result.execution_time,
            suggestions=result.suggestions,
            llm_provider_used=result.llm_provider_used,
            orchestration_used=result.orchestration_used,
            performance_metrics=result.performance_metrics,
            conversation_id=conversation_id
        )

    except Exception as e:
        logger.error(f"CVA chat error: {e}")
        raise HTTPException(status_code=500, detail=f"CVA chat failed: {str(e)}")


@router.get("/status", response_model=CVAStatusResponse)
async def get_cva_status():
    """Get Ultimate CVA Agent status"""
    try:
        cva_agent = await get_cva_agent()

        return CVAStatusResponse(
            agent_id=cva_agent.agent_id,
            mode=cva_agent.cva_mode.value,
            active=cva_agent.autonomous_triggers_active,
            capabilities={
                "natural_language_understanding": cva_agent.capabilities.natural_language_understanding,
                "proactive_monitoring": cva_agent.capabilities.proactive_monitoring,
                "circuit_breaker_protection": cva_agent.capabilities.circuit_breaker_protection,
                "intelligent_task_routing": cva_agent.capabilities.intelligent_task_routing,
                "multi_command_processing": cva_agent.capabilities.multi_command_processing,
                "performance_optimization": cva_agent.capabilities.performance_optimization,
                "predictive_alerting": cva_agent.capabilities.predictive_alerting,
                "dynamic_llm_integration": cva_agent.capabilities.dynamic_llm_integration,
                "ceo_level_orchestration": cva_agent.capabilities.ceo_level_orchestration
            },
            performance_metrics=cva_agent.performance_metrics,
            execution_history_size=len(cva_agent.execution_history),
            last_optimization=cva_agent.last_optimization.isoformat()
        )

    except Exception as e:
        logger.error(f"CVA status error: {e}")
        raise HTTPException(status_code=500, detail=f"CVA status retrieval failed: {str(e)}")


@router.get("/conversations")
async def get_conversations(limit: int = 10):
    """Get conversation history"""
    try:
        cva_agent = await get_cva_agent()

        # Return recent conversation context from memory
        conversations = []
        for memory in cva_agent.context_memory[-limit:]:
            conversations.append({
                "timestamp": memory["timestamp"],
                "command": memory["command"],
                "intent": memory["intent"],
                "confidence": memory["confidence"],
                "success": memory["success"],
                "llm_provider": memory.get("llm_provider", "unknown")
            })

        return {
            "conversations": conversations,
            "total": len(cva_agent.context_memory),
            "limit": limit
        }

    except Exception as e:
        logger.error(f"Conversations retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Conversations retrieval failed: {str(e)}")


@router.post("/optimize")
async def optimize_cva():
    """Trigger CVA system optimization"""
    try:
        cva_agent = await get_cva_agent()

        # Trigger optimization
        cva_agent._optimize_system_configuration()

        return {
            "message": "CVA optimization completed",
            "timestamp": cva_agent.last_optimization.isoformat(),
            "performance_metrics": cva_agent._get_performance_metrics()
        }

    except Exception as e:
        logger.error(f"CVA optimization error: {e}")
        raise HTTPException(status_code=500, detail=f"CVA optimization failed: {str(e)}")


@router.get("/analytics")
async def get_cva_analytics():
    """Get CVA performance analytics"""
    try:
        cva_agent = await get_cva_agent()

        # Calculate analytics
        total_commands = len(cva_agent.execution_history)
        successful_commands = len([ex for ex in cva_agent.execution_history if ex.success])

        analytics = {
            "total_commands_processed": total_commands,
            "success_rate": (successful_commands / max(total_commands, 1)) * 100,
            "average_execution_time": sum(ex.execution_time for ex in cva_agent.execution_history) / max(total_commands, 1),
            "average_confidence": sum(ex.confidence for ex in cva_agent.execution_history) / max(total_commands, 1),
            "llm_providers_used": list(set(ex.llm_provider_used for ex in cva_agent.execution_history if ex.llm_provider_used)),
            "orchestration_usage_rate": (len([ex for ex in cva_agent.execution_history if ex.orchestration_used]) / max(total_commands, 1)) * 100,
            "learning_enabled": cva_agent.learning_enabled,
            "context_memory_size": len(cva_agent.context_memory),
            "performance_cache_size": len(cva_agent.performance_cache),
            "capabilities_active": sum(1 for capability in cva_agent.capabilities.__dict__.values() if capability)
        }

        return {
            "analytics": analytics,
            "timestamp": cva_agent.last_optimization.isoformat(),
            "agent_mode": cva_agent.cva_mode.value
        }

    except Exception as e:
        logger.error(f"CVA analytics error: {e}")
        raise HTTPException(status_code=500, detail=f"CVA analytics retrieval failed: {str(e)}")


# Health check for CVA service
@router.get("/health")
async def cva_health_check():
    """CVA service health check"""
    try:
        cva_agent = await get_cva_agent()

        health_status = "healthy" if cva_agent.autonomous_triggers_active else "degraded"

        return {
            "service": "cva",
            "status": health_status,
            "agent_id": cva_agent.agent_id,
            "mode": cva_agent.cva_mode.value,
            "capabilities_count": sum(1 for capability in cva_agent.capabilities.__dict__.values() if capability),
            "llm_integration": "active",
            "learning_enabled": cva_agent.learning_enabled,
            "execution_history_size": len(cva_agent.execution_history)
        }

    except Exception as e:
        logger.error(f"CVA health check error: {e}")
        return {
            "service": "cva",
            "status": "error",
            "error": str(e)
        }


@router.post("/shutdown")
async def shutdown_cva():
    """Gracefully shutdown CVA agent"""
    global _cva_agent
    try:
        if _cva_agent:
            _cva_agent.shutdown()
            _cva_agent = None

        return {
            "message": "CVA agent shutdown completed",
            "timestamp": str(datetime.now())
        }

    except Exception as e:
        logger.error(f"CVA shutdown error: {e}")
        raise HTTPException(status_code=500, detail=f"CVA shutdown failed: {str(e)}")


# Import datetime for shutdown endpoint
from datetime import datetime