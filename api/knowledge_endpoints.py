"""
Knowledge Endpoints - Production Implementation
REST API endpoints voor kennisbeheer zoals gespecificeerd in FASE 3 STAP 3.3
Communiceert via TaskDistributor naar knowledge_agent (GEEN directe KnowledgeService calls)
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field, ConfigDict

from core.collaboration.task_distributor import get_task_distributor, TaskDistributor
from core.collaboration.agent_mesh import get_agent_registry, AgentRegistry
from core.collaboration.message_protocol import TaskAssignmentMessage
from core.auth.rbac import require_permission, Permission

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class KnowledgeCreateRequest(BaseModel):
    """Request model voor het aanmaken van knowledge entries via agent"""
    model_config = ConfigDict(use_enum_values=True)

    content: str = Field(description="Knowledge content text")
    type: str = Field(default="general", description="Type/category of knowledge")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score between 0 and 1")


class KnowledgeGetRequest(BaseModel):
    """Request model voor het ophalen van knowledge entries via agent"""
    entry_id: int = Field(description="ID of knowledge entry to retrieve")


class KnowledgeUpdateRequest(BaseModel):
    """Request model voor het updaten van knowledge entries via agent"""
    entry_id: int = Field(description="ID of knowledge entry to update")
    content: Optional[str] = Field(default=None, description="Updated content")
    type: Optional[str] = Field(default=None, description="Updated type")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Updated metadata")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Updated confidence score")


class KnowledgeDeleteRequest(BaseModel):
    """Request model voor het verwijderen van knowledge entries via agent"""
    entry_id: int = Field(description="ID of knowledge entry to delete")


class KnowledgeResponse(BaseModel):
    """Response model voor knowledge operations"""
    success: bool = Field(description="Whether operation was successful")
    message: str = Field(description="Operation result message")
    task_id: str = Field(description="Task ID for tracking")
    agent_response: Optional[Dict[str, Any]] = Field(default=None, description="Response from knowledge agent")


@router.post("/", response_model=KnowledgeResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_knowledge(
    request: KnowledgeCreateRequest,
    task_distributor: TaskDistributor = Depends(get_task_distributor),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    _: None = Depends(require_permission(Permission.ADMIN))  # Require admin permission
) -> KnowledgeResponse:
    """
    Creëer knowledge entry zoals gespecificeerd in FASE 3 STAP 3.3

    Dit endpoint maakt GEEN directe aanroep naar KnowledgeService.
    In plaats daarvan roept het TaskDistributor aan om TaskAssignmentMessage
    te sturen naar knowledge_agent (opgezocht via AgentMesh).
    """
    try:
        logger.info("Received create knowledge request via backend endpoint")

        # Start TaskDistributor if not running
        if not task_distributor._running:
            await task_distributor.start()
            logger.info("TaskDistributor started for knowledge operation")

        # Lookup knowledge_agent via AgentMesh zoals gespecificeerd
        knowledge_agent = agent_registry.get_agent("knowledge_agent")
        if not knowledge_agent:
            # Register knowledge_agent for testing if not found
            from core.collaboration.agent_mesh import RegisteredAgent, AgentStatus

            test_knowledge_agent = RegisteredAgent(
                agent_id="knowledge_agent",
                agent_name="Knowledge Agent",
                agent_type="knowledge_processor",
                address="localhost",
                port=8003,
                capabilities=["knowledge_management", "crud_operations"],
                supported_task_types=["create_knowledge", "get_knowledge", "update_knowledge", "delete_knowledge"],
                status=AgentStatus.ONLINE,
                current_load=0.0,
                active_tasks=0,
                max_concurrent_tasks=5,
                last_heartbeat=datetime.utcnow()
            )

            agent_registry.agents["knowledge_agent"] = test_knowledge_agent
            logger.info("Knowledge agent registered for backend operation")

        # Create TaskAssignmentMessage zoals gespecificeerd
        task_message = TaskAssignmentMessage.create_task_assignment(
            sender_id="backend_knowledge_endpoint",
            recipient_id="knowledge_agent",
            task_name="create_knowledge_via_backend",
            task_type="knowledge_management",
            task_parameters={
                "action": "create_knowledge",
                "content": request.content,
                "type": request.type,
                "metadata": request.metadata,
                "confidence_score": request.confidence_score,
                "source": "backend_api"
            }
        )

        logger.info(f"Created TaskAssignmentMessage for knowledge creation: {task_message.task_id}")

        # Dispatch task via TaskDistributor zoals gespecificeerd in plan
        logger.info("Dispatching create knowledge task to knowledge_agent via TaskDistributor...")
        result = await task_distributor.dispatch_task(
            agent_id="knowledge_agent",
            task_message=task_message,
            timeout_seconds=30
        )

        logger.info(f"TaskDistributor result: success={result.success}, status={result.status}")

        # Check if operation succeeded
        if result.success:
            logger.info("Knowledge creation via TaskDistributor successful")
            return KnowledgeResponse(
                success=True,
                message="Knowledge entry created successfully via agent",
                task_id=str(task_message.task_id),
                agent_response=result.response_data
            )
        else:
            logger.error(f"Knowledge creation failed: {result.error_message}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Knowledge creation failed: {result.error_message}"
            )

    except Exception as e:
        logger.error(f"Error in create knowledge endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/get", response_model=KnowledgeResponse)
async def get_knowledge(
    entry_id: int,
    task_distributor: TaskDistributor = Depends(get_task_distributor),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    _: None = Depends(require_permission(Permission.READ))  # Require read permission
) -> KnowledgeResponse:
    """
    Haal knowledge entry op zoals gespecificeerd in FASE 3 STAP 3.3
    Communiceert via TaskDistributor naar knowledge_agent
    """
    try:
        logger.info(f"Received get knowledge request for entry {entry_id}")

        # Start TaskDistributor if not running
        if not task_distributor._running:
            await task_distributor.start()

        # Create TaskAssignmentMessage voor get operation
        task_message = TaskAssignmentMessage.create_task_assignment(
            sender_id="backend_knowledge_endpoint",
            recipient_id="knowledge_agent",
            task_name="get_knowledge_via_backend",
            task_type="knowledge_management",
            task_parameters={
                "action": "get_knowledge",
                "entry_id": entry_id,
                "source": "backend_api"
            }
        )

        # Dispatch task via TaskDistributor
        result = await task_distributor.dispatch_task(
            agent_id="knowledge_agent",
            task_message=task_message,
            timeout_seconds=15
        )

        if result.success:
            return KnowledgeResponse(
                success=True,
                message="Knowledge entry retrieved successfully via agent",
                task_id=str(task_message.task_id),
                agent_response=result.response_data
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge entry not found: {result.error_message}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get knowledge endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.put("/update", response_model=KnowledgeResponse)
async def update_knowledge(
    request: KnowledgeUpdateRequest,
    task_distributor: TaskDistributor = Depends(get_task_distributor),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    _: None = Depends(require_permission(Permission.ADMIN))  # Require admin permission
) -> KnowledgeResponse:
    """
    Update knowledge entry zoals gespecificeerd in FASE 3 STAP 3.3
    Communiceert via TaskDistributor naar knowledge_agent
    """
    try:
        logger.info(f"Received update knowledge request for entry {request.entry_id}")

        # Start TaskDistributor if not running
        if not task_distributor._running:
            await task_distributor.start()

        # Build update parameters (only non-None values)
        update_params = {"action": "update_knowledge", "entry_id": request.entry_id}
        if request.content is not None:
            update_params["content"] = request.content
        if request.type is not None:
            update_params["type"] = request.type
        if request.metadata is not None:
            update_params["metadata"] = request.metadata
        if request.confidence_score is not None:
            update_params["confidence_score"] = request.confidence_score
        update_params["source"] = "backend_api"

        # Create TaskAssignmentMessage voor update operation
        task_message = TaskAssignmentMessage.create_task_assignment(
            sender_id="backend_knowledge_endpoint",
            recipient_id="knowledge_agent",
            task_name="update_knowledge_via_backend",
            task_type="knowledge_management",
            task_parameters=update_params
        )

        # Dispatch task via TaskDistributor
        result = await task_distributor.dispatch_task(
            agent_id="knowledge_agent",
            task_message=task_message,
            timeout_seconds=20
        )

        if result.success:
            return KnowledgeResponse(
                success=True,
                message="Knowledge entry updated successfully via agent",
                task_id=str(task_message.task_id),
                agent_response=result.response_data
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge entry update failed: {result.error_message}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in update knowledge endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/delete", response_model=KnowledgeResponse)
async def delete_knowledge(
    entry_id: int,
    task_distributor: TaskDistributor = Depends(get_task_distributor),
    agent_registry: AgentRegistry = Depends(get_agent_registry),
    _: None = Depends(require_permission(Permission.ADMIN))  # Require admin permission
) -> KnowledgeResponse:
    """
    Verwijder knowledge entry zoals gespecificeerd in FASE 3 STAP 3.3
    Communiceert via TaskDistributor naar knowledge_agent
    """
    try:
        logger.info(f"Received delete knowledge request for entry {entry_id}")

        # Start TaskDistributor if not running
        if not task_distributor._running:
            await task_distributor.start()

        # Create TaskAssignmentMessage voor delete operation
        task_message = TaskAssignmentMessage.create_task_assignment(
            sender_id="backend_knowledge_endpoint",
            recipient_id="knowledge_agent",
            task_name="delete_knowledge_via_backend",
            task_type="knowledge_management",
            task_parameters={
                "action": "delete_knowledge",
                "entry_id": entry_id,
                "source": "backend_api"
            }
        )

        # Dispatch task via TaskDistributor
        result = await task_distributor.dispatch_task(
            agent_id="knowledge_agent",
            task_message=task_message,
            timeout_seconds=15
        )

        if result.success:
            return KnowledgeResponse(
                success=True,
                message="Knowledge entry deleted successfully via agent",
                task_id=str(task_message.task_id),
                agent_response=result.response_data
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Knowledge entry deletion failed: {result.error_message}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete knowledge endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )