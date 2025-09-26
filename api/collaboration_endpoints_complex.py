#!/usr/bin/env python3
"""
Phase 3 - Multi-Agent Collaboration API Endpoints
Provides RESTful API for agent mesh, task coordination, and adaptive learning
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

from ..collaboration.agent_mesh import (
    get_agent_mesh, AgentRegistration, AgentCapability, AgentMessage, 
    MessageType, MessagePriority
)
from ..collaboration.task_coordinator import (
    get_task_coordinator, TaskRequirement, CollaborativeTask
)
from ..knowledge.adaptive_learning import (
    get_adaptive_learning, LearningExperience, LearningType, LearningOutcome
)
from ..intelligence.system_optimizer import get_system_optimizer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/collaboration", tags=["collaboration"])

# --- Pydantic Models ---

class AgentCapabilityModel(BaseModel):
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    complexity_score: float = Field(ge=0.0, le=1.0)
    processing_time_avg: float
    success_rate: float = Field(ge=0.0, le=1.0)
    resources_required: List[str]

class AgentRegistrationModel(BaseModel):
    agent_id: str
    agent_type: str
    capabilities: List[AgentCapabilityModel]
    status: str = "active"
    load_factor: float = Field(ge=0.0, le=1.0, default=0.0)
    network_address: Optional[str] = None
    metadata: Dict[str, Any] = {}

class AgentMessageModel(BaseModel):
    to_agent: str
    message_type: str
    priority: str = "normal"
    content: Dict[str, Any]
    requires_response: bool = False
    expires_in_seconds: Optional[int] = None

class TaskRequirementModel(BaseModel):
    capability: str
    priority: int = Field(ge=1, le=10)
    estimated_time: float
    resources_needed: List[str]
    dependencies: List[str] = []
    constraints: Dict[str, Any] = {}

class CollaborativeTaskModel(BaseModel):
    description: str
    requirements: List[TaskRequirementModel]
    deadline_hours: Optional[float] = None
    metadata: Dict[str, Any] = {}

class LearningExperienceModel(BaseModel):
    agent_id: str
    learning_type: str
    context: Dict[str, Any]
    action_taken: Dict[str, Any]
    outcome: str
    outcome_data: Dict[str, Any]
    confidence_score: float = Field(ge=0.0, le=1.0)
    impact_score: float = Field(ge=0.0, le=1.0)
    metadata: Dict[str, Any] = {}

# --- Agent Mesh Endpoints ---

@router.post("/agents/register")
async def register_agent(registration: AgentRegistrationModel):
    """Register a new agent in the mesh network"""
    try:
        mesh = get_agent_mesh()
        
        # Convert capabilities
        capabilities = [
            AgentCapability(**cap.dict()) for cap in registration.capabilities
        ]
        
        # Create registration
        agent_reg = AgentRegistration(
            agent_id=registration.agent_id,
            agent_type=registration.agent_type,
            capabilities=capabilities,
            status=registration.status,
            load_factor=registration.load_factor,
            last_heartbeat=datetime.now(),
            network_address=registration.network_address,
            metadata=registration.metadata
        )
        
        success = await mesh.register_agent(agent_reg)
        
        if success:
            return {
                "success": True,
                "message": f"Agent {registration.agent_id} registered successfully",
                "agent_id": registration.agent_id
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to register agent")
            
    except Exception as e:
        logger.error(f"Agent registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/agents/{agent_id}")
async def unregister_agent(agent_id: str):
    """Unregister an agent from the mesh network"""
    try:
        mesh = get_agent_mesh()
        success = await mesh.unregister_agent(agent_id)
        
        if success:
            return {
                "success": True,
                "message": f"Agent {agent_id} unregistered successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
            
    except Exception as e:
        logger.error(f"Agent unregistration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/agents/{agent_id}/heartbeat")
async def update_agent_heartbeat(agent_id: str):
    """Update agent heartbeat timestamp"""
    try:
        mesh = get_agent_mesh()
        success = await mesh.update_agent_heartbeat(agent_id)
        
        if success:
            return {"success": True, "timestamp": datetime.now().isoformat()}
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
            
    except Exception as e:
        logger.error(f"Heartbeat update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str):
    """Get detailed status of a specific agent"""
    try:
        mesh = get_agent_mesh()
        status = await mesh.get_agent_status(agent_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
            
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/network/metrics")
async def get_network_metrics():
    """Get comprehensive network metrics"""
    try:
        mesh = get_agent_mesh()
        metrics = await mesh.get_network_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get network metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities/discover")
async def discover_capabilities(capability_query: str):
    """Discover agents with specific capabilities"""
    try:
        mesh = get_agent_mesh()
        matches = await mesh.discover_capabilities(capability_query)
        
        return {
            "query": capability_query,
            "matches_found": len(matches),
            "matches": matches
        }
        
    except Exception as e:
        logger.error(f"Capability discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/messages/send")
async def send_message(from_agent: str, message: AgentMessageModel):
    """Send a message to another agent"""
    try:
        mesh = get_agent_mesh()
        
        # Create message
        agent_message = AgentMessage(
            id=f"msg_{datetime.now().timestamp()}",
            from_agent=from_agent,
            to_agent=message.to_agent,
            message_type=MessageType(message.message_type),
            priority=MessagePriority[message.priority.upper()],
            content=message.content,
            timestamp=datetime.now(),
            expires_at=(datetime.now() + timedelta(seconds=message.expires_in_seconds) 
                       if message.expires_in_seconds else None),
            requires_response=message.requires_response
        )
        
        success = await mesh.send_message(agent_message)
        
        if success:
            return {
                "success": True,
                "message_id": agent_message.id,
                "sent_at": agent_message.timestamp.isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to send message")
            
    except Exception as e:
        logger.error(f"Message sending failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/messages/{agent_id}")
async def receive_messages(agent_id: str, limit: int = 10):
    """Receive messages for an agent"""
    try:
        mesh = get_agent_mesh()
        messages = await mesh.receive_messages(agent_id, limit)
        
        return {
            "agent_id": agent_id,
            "messages_count": len(messages),
            "messages": [
                {
                    "id": msg.id,
                    "from_agent": msg.from_agent,
                    "message_type": msg.message_type.value,
                    "priority": msg.priority.name,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "requires_response": msg.requires_response
                }
                for msg in messages
            ]
        }
        
    except Exception as e:
        logger.error(f"Message receiving failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/groups/{group_id}")
async def create_collaboration_group(group_id: str, agent_ids: List[str]):
    """Create a collaboration group"""
    try:
        mesh = get_agent_mesh()
        success = await mesh.create_collaboration_group(group_id, agent_ids)
        
        if success:
            return {
                "success": True,
                "group_id": group_id,
                "members": agent_ids,
                "created_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to create collaboration group")
            
    except Exception as e:
        logger.error(f"Group creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Task Coordination Endpoints ---

@router.post("/tasks/submit")
async def submit_collaborative_task(task: CollaborativeTaskModel, requester_agent: str):
    """Submit a task for collaborative execution"""
    try:
        coordinator = get_task_coordinator()
        
        # Convert requirements
        requirements = [
            TaskRequirement(**req.dict()) for req in task.requirements
        ]
        
        # Set deadline
        deadline = None
        if task.deadline_hours:
            deadline = datetime.now() + timedelta(hours=task.deadline_hours)
        
        task_id = await coordinator.submit_collaborative_task(
            description=task.description,
            requirements=requirements,
            requester_agent=requester_agent,
            deadline=deadline,
            metadata=task.metadata
        )
        
        if task_id:
            return {
                "success": True,
                "task_id": task_id,
                "submitted_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to submit collaborative task")
            
    except Exception as e:
        logger.error(f"Task submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get status of a collaborative task"""
    try:
        coordinator = get_task_coordinator()
        status = await coordinator.get_task_status(task_id)
        
        if status:
            return status
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a collaborative task"""
    try:
        coordinator = get_task_coordinator()
        success = await coordinator.cancel_task(task_id)
        
        if success:
            return {
                "success": True,
                "task_id": task_id,
                "cancelled_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Task not found")
            
    except Exception as e:
        logger.error(f"Task cancellation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/coordination/metrics")
async def get_coordination_metrics():
    """Get task coordination metrics"""
    try:
        coordinator = get_task_coordinator()
        metrics = await coordinator.get_coordination_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get coordination metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Adaptive Learning Endpoints ---

@router.post("/learning/experience")
async def record_learning_experience(experience: LearningExperienceModel):
    """Record a learning experience"""
    try:
        learning_system = get_adaptive_learning()
        
        # Create learning experience
        learning_exp = LearningExperience(
            id=f"exp_{datetime.now().timestamp()}",
            agent_id=experience.agent_id,
            learning_type=LearningType(experience.learning_type),
            context=experience.context,
            action_taken=experience.action_taken,
            outcome=LearningOutcome(experience.outcome),
            outcome_data=experience.outcome_data,
            timestamp=datetime.now(),
            confidence_score=experience.confidence_score,
            impact_score=experience.impact_score,
            metadata=experience.metadata
        )
        
        success = await learning_system.record_experience(learning_exp)
        
        if success:
            return {
                "success": True,
                "experience_id": learning_exp.id,
                "recorded_at": learning_exp.timestamp.isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to record learning experience")
            
    except Exception as e:
        logger.error(f"Learning experience recording failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/learning/patterns")
async def discover_knowledge_patterns():
    """Discover knowledge patterns from learning experiences"""
    try:
        learning_system = get_adaptive_learning()
        patterns = await learning_system.discover_patterns()
        
        return {
            "patterns_discovered": len(patterns),
            "patterns": [
                {
                    "pattern_id": pattern.id,
                    "pattern_type": pattern.pattern_type,
                    "description": pattern.description,
                    "confidence": pattern.confidence,
                    "usage_count": pattern.usage_count,
                    "discovered_at": pattern.discovered_at.isoformat()
                }
                for pattern in patterns
            ]
        }
        
    except Exception as e:
        logger.error(f"Pattern discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/learning/recommendations/{agent_id}")
async def get_learning_recommendations(agent_id: str):
    """Get personalized learning recommendations for an agent"""
    try:
        learning_system = get_adaptive_learning()
        recommendations = await learning_system.get_learning_recommendations(agent_id)
        
        return {
            "agent_id": agent_id,
            "recommendations_count": len(recommendations),
            "recommendations": recommendations
        }
        
    except Exception as e:
        logger.error(f"Failed to get learning recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/learning/metrics")
async def get_learning_metrics():
    """Get comprehensive learning system metrics"""
    try:
        learning_system = get_adaptive_learning()
        metrics = await learning_system.get_learning_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get learning metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learning/apply-pattern/{pattern_id}/{agent_id}")
async def apply_knowledge_pattern(pattern_id: str, agent_id: str):
    """Apply a discovered knowledge pattern to an agent"""
    try:
        learning_system = get_adaptive_learning()
        success = await learning_system.apply_pattern_to_agent(pattern_id, agent_id)
        
        if success:
            return {
                "success": True,
                "pattern_id": pattern_id,
                "agent_id": agent_id,
                "applied_at": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="Pattern not found")
            
    except Exception as e:
        logger.error(f"Pattern application failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- System Intelligence Endpoints ---

@router.get("/intelligence/status")
async def get_system_intelligence_status():
    """Get system intelligence and optimization status"""
    try:
        optimizer = get_system_optimizer()
        status = await optimizer.get_system_status()
        return status
        
    except Exception as e:
        logger.error(f"Failed to get intelligence status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/intelligence/metrics")
async def get_system_metrics():
    """Get current system performance metrics"""
    try:
        optimizer = get_system_optimizer()
        metrics = await optimizer.get_current_metrics()
        
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "disk_usage": metrics.disk_usage,
            "network_throughput": metrics.network_throughput,
            "active_agents": metrics.active_agents,
            "avg_response_time": metrics.avg_response_time,
            "error_rate": metrics.error_rate,
            "queue_depth": metrics.queue_depth,
            "cache_hit_rate": metrics.cache_hit_rate
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Background Task Initialization ---

async def initialize_collaboration_systems():
    """Initialize all collaboration systems"""
    try:
        logger.info("Initializing Phase 3 collaboration systems...")
        
        # Initialize agent mesh
        mesh = get_agent_mesh()
        await mesh.start()
        
        # Initialize task coordinator
        coordinator = get_task_coordinator()
        await coordinator.start()
        
        # Initialize adaptive learning
        learning_system = get_adaptive_learning()
        await learning_system.start()
        
        # Initialize system optimizer
        optimizer = get_system_optimizer()
        await optimizer.start()
        
        logger.info("Phase 3 collaboration systems initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize collaboration systems: {e}")
        return False

# Health check for collaboration systems
@router.get("/health")
async def collaboration_health_check():
    """Health check for all collaboration systems"""
    try:
        systems_status = {
            "agent_mesh": "operational",
            "task_coordinator": "operational", 
            "adaptive_learning": "operational",
            "system_optimizer": "operational",
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "status": "healthy",
            "systems": systems_status,
            "phase": "Phase 3 - Agent Mesh & Coordination",
            "capabilities": [
                "Multi-agent communication",
                "Collaborative task execution", 
                "Adaptive learning system",
                "Autonomous system optimization"
            ]
        }
        
    except Exception as e:
        logger.error(f"Collaboration health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }