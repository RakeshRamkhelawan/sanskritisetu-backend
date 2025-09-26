"""
CVA Autonomous Trigger API Endpoints
Complete control over autonomous CVA operation
"""

import logging
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import the autonomous trigger system
from core.agents.cva_autonomous_trigger import (
    start_cva_autonomous,
    stop_cva_autonomous, 
    get_cva_autonomous_status,
    trigger_manual_cva_breakthrough
)

logger = logging.getLogger(__name__)

# Create router for CVA autonomous endpoints
cva_autonomous_router = APIRouter(
    prefix="/api/v1/cva-autonomous",
    tags=["cva-autonomous", "breakthrough", "autonomous-ai"],
    responses={404: {"description": "Not found"}}
)

# Request/Response Models
class ManualTriggerRequest(BaseModel):
    """Request model for manual breakthrough trigger"""
    reason: str = Field("manual_request", description="Reason for manual trigger")
    priority: int = Field(1, description="Priority level (1=highest, 5=lowest)")

class AutonomousStatusResponse(BaseModel):
    """Response model for autonomous status"""
    system_name: str
    version: str
    is_running: bool
    pending_triggers: int
    last_breakthrough: str = None
    performance_metrics: Dict[str, Any]
    configuration: Dict[str, Any]

class TriggerResponse(BaseModel):
    """Response model for trigger actions"""
    status: str
    message: str
    timestamp: str

# API Endpoints

@cva_autonomous_router.get("/status", response_model=AutonomousStatusResponse)
async def get_autonomous_status():
    """
    Get CVA Autonomous Trigger System status
    
    Returns complete status of the autonomous system including:
    - Running status
    - Pending triggers
    - Performance metrics
    - Configuration settings
    """
    try:
        status = get_cva_autonomous_status()
        
        return AutonomousStatusResponse(
            system_name=status['system_name'],
            version=status['version'],
            is_running=status['is_running'],
            pending_triggers=status['pending_triggers'],
            last_breakthrough=status['last_breakthrough'],
            performance_metrics=status['performance_metrics'],
            configuration=status['configuration']
        )
        
    except Exception as e:
        logger.error(f"Error getting autonomous status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")

@cva_autonomous_router.post("/start", response_model=TriggerResponse)
async def start_autonomous_system():
    """
    Start CVA Autonomous Operation
    
    Initiates the complete autonomous trigger system:
    - Scheduled breakthrough sessions (every 6 hours)
    - Intelligent monitoring (every 5 minutes)
    - Event-driven triggers
    - Emergency response system
    """
    try:
        result = start_cva_autonomous()
        
        return TriggerResponse(
            status="started" if result and result.get('autonomous_mode') else "already_running",
            message="CVA Autonomous Trigger System activated - Full self-operation initiated",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error starting autonomous system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start autonomous system: {str(e)}")

@cva_autonomous_router.post("/stop", response_model=TriggerResponse)
async def stop_autonomous_system():
    """
    Stop CVA Autonomous Operation
    
    Gracefully stops all autonomous triggers:
    - Scheduled sessions
    - Intelligent monitoring
    - Event processing
    """
    try:
        stop_cva_autonomous()
        
        return TriggerResponse(
            status="stopped",
            message="CVA Autonomous Trigger System deactivated",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error stopping autonomous system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop autonomous system: {str(e)}")

@cva_autonomous_router.post("/trigger", response_model=TriggerResponse)
async def manual_breakthrough_trigger(request: ManualTriggerRequest):
    """
    Manually Trigger CVA Breakthrough Session
    
    Immediately initiates a CVA breakthrough session with:
    - High priority processing
    - Real LLM integration
    - Complete logging to sandboxlog.md
    """
    try:
        result = trigger_manual_cva_breakthrough(request.reason)
        
        return TriggerResponse(
            status=result.get('status', 'triggered'),
            message=f"Manual breakthrough triggered: {request.reason}",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error triggering manual breakthrough: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger breakthrough: {str(e)}")

@cva_autonomous_router.get("/metrics")
async def get_autonomous_metrics():
    """
    Get detailed autonomous system metrics
    
    Returns comprehensive metrics including:
    - Total breakthroughs generated
    - Success rates
    - System uptime
    - Performance statistics
    """
    try:
        status = get_cva_autonomous_status()
        
        metrics = status.get('performance_metrics', {})
        config = status.get('configuration', {})
        
        return {
            "autonomous_metrics": {
                "total_triggers": metrics.get('total_triggers', 0),
                "successful_breakthroughs": metrics.get('successful_breakthroughs', 0),
                "success_rate": round(
                    (metrics.get('successful_breakthroughs', 0) / max(metrics.get('total_triggers', 1), 1)) * 100, 2
                ),
                "system_uptime": metrics.get('system_uptime', 0),
                "last_innovation": metrics.get('last_innovation_time'),
                "configuration": {
                    "scheduled_interval_hours": config.get('scheduled_interval_hours'),
                    "intelligent_monitoring_interval": config.get('intelligent_monitoring_interval'),
                    "max_daily_breakthroughs": config.get('max_daily_breakthroughs')
                }
            },
            "system_status": {
                "is_running": status['is_running'],
                "pending_triggers": status['pending_triggers'],
                "last_breakthrough": status['last_breakthrough']
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting autonomous metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@cva_autonomous_router.get("/health")
async def autonomous_system_health():
    """
    Health check for autonomous system
    
    Quick health status of the CVA autonomous trigger system
    """
    try:
        status = get_cva_autonomous_status()
        
        health_status = "healthy" if status['is_running'] else "stopped"
        
        return {
            "health_status": health_status,
            "system_running": status['is_running'],
            "last_check": datetime.now().isoformat(),
            "autonomous_capability": "fully_operational" if status['is_running'] else "standby"
        }
        
    except Exception as e:
        logger.error(f"Error checking autonomous health: {e}")
        return {
            "health_status": "error",
            "system_running": False,
            "error": str(e),
            "last_check": datetime.now().isoformat()
        }