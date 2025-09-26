"""
Goal Endpoints - FASE 3+ STAP 3+.3 Implementation
REST API endpoints voor autonome goal execution.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from core.autonomy.autonomous_agent import AutonomousAgent
from core.autonomy.task_planner import Goal
from core.auth.rbac import require_permission, Permission
# --- CORRECTIE ---
# Importeer de dependency uit het centrale 'dependencies' bestand
# om circulaire imports te voorkomen.
from core.dependencies import get_autonomous_agent_dependency

logger = logging.getLogger(__name__)

# Maak de router aan
router = APIRouter()

class GoalCreateRequest(BaseModel):
    """Request model voor het creëren van goals."""
    description: str = Field(description="Goal description (e.g., 'archive: content')")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Aanvullende goal metadata")

class GoalExecutionResponse(BaseModel):
    """Response model voor de resultaten van een goal executie."""
    success: bool
    message: str
    goal_id: str
    execution_result: Optional[Dict[str, Any]] = None

@router.post("/", response_model=GoalExecutionResponse, status_code=status.HTTP_202_ACCEPTED)
async def execute_goal(
    request: GoalCreateRequest,
    autonomous_agent: AutonomousAgent = Depends(get_autonomous_agent_dependency),
    # _: None = Depends(require_permission(Permission.ADMIN)) # Tijdelijk uitgeschakeld voor testgemak
) -> GoalExecutionResponse:
    """
    Voert een autonoom doel uit.
    
    Dit endpoint accepteert een `Goal`, geeft deze door aan de `AutonomousAgent`
    en retourneert het resultaat van de executie.
    """
    try:
        logger.info(f"Nieuw doel ontvangen: {request.description}")
        goal = Goal(description=request.description, metadata=request.metadata or {})
        
        logger.info(f"Doel {goal.id} wordt uitgevoerd door AutonomousAgent...")
        execution_result = await autonomous_agent.execute_goal(goal)
        
        if execution_result.get("success"):
            logger.info(f"Doel {goal.id} succesvol uitgevoerd.")
            return GoalExecutionResponse(
                success=True,
                message="Doel succesvol uitgevoerd door de autonome agent.",
                goal_id=goal.id,
                execution_result=execution_result,
            )
        else:
            error_message = execution_result.get('error', 'Onbekende fout')
            logger.error(f"Fout bij uitvoeren van doel {goal.id}: {error_message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Uitvoering van doel mislukt: {error_message}"
            )

    except Exception as e:
        logger.exception(f"Onverwachte fout in execute_goal endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interne serverfout: {e}"
        )

@router.get("/status", response_model=Dict[str, Any])
async def get_autonomous_agent_status(
    autonomous_agent: AutonomousAgent = Depends(get_autonomous_agent_dependency),
    # _: None = Depends(require_permission(Permission.READ)) # Tijdelijk uitgeschakeld
) -> Dict[str, Any]:
    """Haalt de huidige status van de AutonomousAgent op."""
    try:
        return {
            "success": True,
            "timestamp": datetime.utcnow().isoformat(),
            "autonomous_agent_status": autonomous_agent.get_current_status()
        }
    except Exception as e:
        logger.exception(f"Fout bij ophalen van agent status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interne serverfout: {e}"
        )
