#!/usr/bin/env python3
"""
Phase 4 - Autonomous Systems API Endpoints
Provides RESTful API for agent factory, self-healing, code generation, and emergent intelligence
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio

from .error_handling import handle_endpoint_error, get_template_error

from ..autonomy.autonomous_agent_factory import (
    get_agent_factory, AgentArchetype, AgentTemplate, GenerationRequest
)
from ..autonomy.self_healing_system import (
    get_self_healing_system, FailureType, RepairStrategy, SystemFailure
)
from ..generation.dynamic_code_generator import (
    get_code_generator, CodeType, GenerationStrategy, CodeRequirement
)
from ..emergence.emergent_intelligence import (
    get_emergent_intelligence, EmergenceType, IntelligenceLevel
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/autonomous", tags=["autonomous"])

# --- Pydantic Models ---

class AgentCreationRequest(BaseModel):
    archetype: str
    name: str
    description: str
    capabilities: List[str]
    tools: List[str]
    parameters: Dict[str, Any] = {}

class AgentTemplateModel(BaseModel):
    archetype: str
    name: str
    description: str
    core_capabilities: List[str]
    required_tools: List[str]
    performance_metrics: Dict[str, float] = {}
    initialization_parameters: Dict[str, Any] = {}

class FailureReportModel(BaseModel):
    component: str
    failure_type: str
    severity: str
    description: str
    context: Dict[str, Any] = {}
    auto_repair: bool = True

class CodeGenerationRequest(BaseModel):
    code_type: str
    functionality: str
    performance_constraints: Dict[str, float] = {}
    integration_points: List[str] = []
    quality_requirements: Dict[str, float] = {}
    security_constraints: List[str] = []
    strategy: str = "template_based"

class EmergenceFacilitationRequest(BaseModel):
    target_behavior: str
    participant_agents: List[str]
    conditions: Dict[str, Any] = {}
    intelligence_level: int = 2

# --- Agent Factory Endpoints ---

@router.post("/agents/create")
async def create_autonomous_agent(request: AgentCreationRequest):
    """Create a new autonomous agent dynamically using full GenerationRequest workflow"""
    try:
        print("=== UNIQUE DEBUG MESSAGE: ROUTE HIT 1234567890 ===")
        logger.error(f"[UNIQUE-DEBUG] API endpoint was hit! Request: {request}")
        logger.info(f"[FLOW-1] API endpoint called with request: {request}")
        factory = get_agent_factory()
        logger.info(f"[FLOW-2] Got factory: {type(factory)}")
        
        # Convert string to archetype enum
        try:
            archetype = AgentArchetype(request.archetype.lower())
            logger.info(f"[FLOW-3] Archetype converted: {archetype}")
        except ValueError:
            logger.error(f"[FLOW-ERROR] Invalid archetype: {request.archetype}")
            raise HTTPException(status_code=400, detail=f"Invalid archetype: {request.archetype}")
        
        # Create proper GenerationRequest for full autonomous workflow
        import uuid
        logger.info(f"[FLOW-4] About to create GenerationRequest with parameters:")
        logger.info(f"[FLOW-4] - request_id: {str(uuid.uuid4())}")
        logger.info(f"[FLOW-4] - archetype: {archetype}")
        logger.info(f"[FLOW-4] - specialization: {request.name.lower()}")
        logger.info(f"[FLOW-4] - performance_requirements: {{'success_rate': 0.95, 'response_time': 1.0}}")
        logger.info(f"[FLOW-4] - context_data: {{'capabilities': {request.capabilities}, 'tools': {request.tools}, 'parameters': {request.parameters}, 'description': '{request.description}'}}")
        logger.info(f"[FLOW-4] - urgency_level: 4")
        logger.info(f"[FLOW-4] - requester_agent: 'api_endpoint'")
        
        generation_request = GenerationRequest(
            request_id=str(uuid.uuid4()),
            archetype=archetype,
            specialization=request.name.lower(),
            performance_requirements={"success_rate": 0.95, "response_time": 1.0},
            context_data={
                "capabilities": request.capabilities,
                "tools": request.tools,
                "parameters": request.parameters,
                "description": request.description
            },
            urgency_level=4,
            requester_agent="api_endpoint"
        )
        logger.info(f"[FLOW-5] GenerationRequest created successfully: {generation_request}")
        
        # Use full autonomous generation workflow
        logger.info(f"[FLOW-6] About to call factory.request_agent_generation with: {generation_request}")
        request_id = await factory.request_agent_generation(generation_request)
        logger.info(f"[FLOW-7] request_agent_generation returned: {request_id}")
        
        # For immediate response, also use direct create for now
        # This gives us both autonomous queuing AND immediate results
        template_id = f"{request.archetype.lower()}_template"
        config = {
            "specialization": request.name.lower(),
            "performance_requirements": {"success_rate": 0.95, "response_time": 1.0},
            "capabilities": request.capabilities,
            "tools": request.tools,
            "parameters": request.parameters
        }
        logger.info(f"[FLOW-8] About to call factory.create_agent with template_id: {template_id}, config: {config}")
        agent_result = await factory.create_agent(template_id, config)
        logger.info(f"[FLOW-9] create_agent returned: {agent_result}")
        
        if agent_result.get("success"):
            return {
                "success": True,
                "agent_id": agent_result.get("agent_id"),
                "archetype": request.archetype,
                "specialization": agent_result.get("specialization"),
                "class_name": agent_result.get("class_name"),
                "created_at": datetime.now().isoformat(),
                "message": f"Agent '{agent_result.get('agent_id')}' created successfully",
                # Autonomous workflow tracking
                "generation_request_id": request_id,
                "autonomous_queue": "active",
                "workflow_type": "hybrid_immediate_and_autonomous"
            }
        else:
            error_msg = agent_result.get("error", "Agent creation failed")
            logger.error(f"Agent creation failed for archetype '{request.archetype}': {error_msg}")
            
            # Provide specific error codes and troubleshooting hints
            if "template" in error_msg.lower():
                raise HTTPException(
                    status_code=404, 
                    detail={
                        "error": "Template not found",
                        "message": error_msg,
                        "troubleshooting": f"Available archetypes: analyzer, executor, monitor",
                        "requested_archetype": request.archetype
                    }
                )
            elif "invalid" in error_msg.lower():
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Invalid request parameters", 
                        "message": error_msg,
                        "troubleshooting": "Check archetype name and required parameters"
                    }
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "Agent creation failed",
                        "message": error_msg,
                        "troubleshooting": "Check system logs for detailed error information",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
    except HTTPException:
        # Re-raise HTTP exceptions (they're already properly formatted)
        raise
    except Exception as e:
        logger.error(f"[FLOW-EXCEPTION] Unexpected error in agent creation: {e}", exc_info=True)
        logger.error(f"[FLOW-EXCEPTION] Error type: {type(e).__name__}")
        logger.error(f"[FLOW-EXCEPTION] Error message: {str(e)}")
        
        # Check for the mysterious 'requested_capabilities' issue
        if "requested_capabilities" in str(e):
            logger.error(f"[FLOW-EXCEPTION] FOUND requested_capabilities error!")
            import traceback
            logger.error(f"[FLOW-EXCEPTION] Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(
            status_code=500, 
            detail={
                "error": "Internal server error",
                "message": f"Unexpected error occurred: {str(e)[:200]}",
                "error_type": type(e).__name__,
                "troubleshooting": "This appears to be a system error. Please check server logs.",
                "timestamp": datetime.now().isoformat()
            }
        )

@router.get("/agents/templates")
async def get_agent_templates():
    """Get available agent templates"""
    try:
        factory = get_agent_factory()
        templates = await factory.get_agent_templates()
        
        return {
            "templates": templates,
            "count": len(templates),
            "status": "success"
        }
        
    except Exception as e:
        raise handle_endpoint_error(e, "get_agent_templates")

@router.post("/agents/templates")
async def create_agent_template(template: AgentTemplateModel):
    """Create a new agent template"""
    try:
        factory = get_agent_factory()
        
        archetype = AgentArchetype(template.archetype.lower())
        
        agent_template = AgentTemplate(
            archetype=archetype,
            name=template.name,
            description=template.description,
            core_capabilities=template.core_capabilities,
            required_tools=template.required_tools,
            performance_metrics=template.performance_metrics,
            initialization_parameters=template.initialization_parameters
        )
        
        template_id = await factory.register_template(agent_template)
        
        return {
            "success": True,
            "template_id": template_id,
            "archetype": template.archetype,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Template creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/factory/metrics")
async def get_factory_metrics():
    """Get agent factory performance metrics"""
    try:
        factory = get_agent_factory()
        metrics = await factory.get_factory_metrics()
        
        return {
            "agents_created": metrics.get("total_agents_created", 0),
            "active_agents": metrics.get("active_agents", 0),
            "success_rate": metrics.get("creation_success_rate", 0.0),
            "avg_creation_time": metrics.get("avg_creation_time", 0.0),
            "templates_available": metrics.get("templates_count", 0),
            "archetypes_used": metrics.get("archetype_distribution", {}),
            "performance_metrics": metrics.get("performance_summary", {})
        }
        
    except Exception as e:
        logger.error(f"Failed to get factory metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Self-Healing System Endpoints ---

@router.post("/healing/report-failure")
async def report_system_failure(failure: FailureReportModel):
    """Report a system failure for autonomous healing"""
    try:
        healing_system = get_self_healing_system()
        
        failure_type = FailureType(failure.failure_type.lower())
        
        system_failure = SystemFailure(
            failure_id=f"failure_{datetime.now().timestamp()}",
            component=failure.component,
            failure_type=failure_type,
            severity=failure.severity,
            description=failure.description,
            timestamp=datetime.now(),
            context=failure.context
        )
        
        # Report failure and get repair recommendation
        repair_plan = await healing_system.detect_and_diagnose_failure(system_failure)
        
        # Auto-repair if requested
        if failure.auto_repair and repair_plan:
            repair_result = await healing_system.execute_repair_plan(repair_plan)
            
            return {
                "failure_id": system_failure.failure_id,
                "repair_plan": {
                    "strategy": repair_plan.strategy.value,
                    "steps": repair_plan.repair_steps,
                    "estimated_time": repair_plan.estimated_repair_time
                },
                "repair_result": {
                    "success": repair_result.success,
                    "time_taken": repair_result.execution_time,
                    "status": repair_result.status
                },
                "auto_repaired": True
            }
        else:
            return {
                "failure_id": system_failure.failure_id,
                "repair_plan": {
                    "strategy": repair_plan.strategy.value if repair_plan else "no_plan",
                    "steps": repair_plan.repair_steps if repair_plan else [],
                    "estimated_time": repair_plan.estimated_repair_time if repair_plan else 0
                },
                "auto_repaired": False,
                "recommendation": "Manual intervention required" if not repair_plan else "Ready for repair execution"
            }
            
    except Exception as e:
        logger.error(f"Failure reporting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/healing/status")
async def get_healing_system_status():
    """Get self-healing system status and metrics"""
    try:
        healing_system = get_self_healing_system()
        status = await healing_system.get_system_status()
        
        return {
            "overall_health": status.get("system_status", "unknown"),
            "active_failures": status.get("failure_history_count", 0),
            "repairs_in_progress": 0 if not status.get("monitoring_active", False) else 1,
            "repairs_completed": status.get("repair_history_count", 0),
            "success_rate": status.get("healing_metrics", {}).get("healing_success_rate", 0.0),
            "avg_repair_time": status.get("healing_metrics", {}).get("mean_time_to_repair", 0.0),
            "system_uptime": status.get("uptime", 0.0),
            "component_health": status.get("components", {}),
            "recent_failures": status.get("recent_failures", 0),
            "monitoring_active": status.get("monitoring_active", False),
            "registered_components": status.get("registered_components", 0),
            "emergency_protocols": status.get("emergency_protocols_available", 0)
        }
        
    except Exception as e:
        logger.error(f"Failed to get healing status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/healing/execute-repair/{failure_id}")
async def execute_repair(failure_id: str):
    """Execute repair plan for a specific failure"""
    try:
        healing_system = get_self_healing_system()
        
        # Get repair plan for failure
        repair_plan = await healing_system.get_repair_plan(failure_id)
        if not repair_plan:
            raise HTTPException(status_code=404, detail="Repair plan not found")
        
        # Execute repair
        repair_result = await healing_system.execute_repair_plan(repair_plan)
        
        return {
            "failure_id": failure_id,
            "repair_success": repair_result.success,
            "execution_time": repair_result.execution_time,
            "status": repair_result.status,
            "details": repair_result.details,
            "completed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Repair execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Dynamic Code Generation Endpoints ---

@router.post("/code/generate")
async def generate_code(request: CodeGenerationRequest):
    """Generate code dynamically based on requirements"""
    try:
        code_generator = get_code_generator()
        
        code_type = CodeType(request.code_type.lower())
        strategy = GenerationStrategy(request.strategy.lower())
        
        code_req = CodeRequirement(
            functionality=request.functionality,
            performance_constraints=request.performance_constraints,
            integration_points=request.integration_points,
            quality_requirements=request.quality_requirements,
            security_constraints=request.security_constraints
        )
        
        generation_result = await code_generator.generate_code(
            code_type=code_type,
            requirements=code_req,
            strategy=strategy
        )
        
        if generation_result.success:
            return {
                "success": True,
                "generation_id": generation_result.generation_id,
                "code": generation_result.generated_code,
                "metadata": {
                    "lines_of_code": generation_result.metrics.get("lines_of_code", 0),
                    "complexity_score": generation_result.metrics.get("complexity", 0.0),
                    "quality_score": generation_result.metrics.get("quality", 0.0),
                    "generation_time": generation_result.generation_time
                },
                "documentation": generation_result.documentation,
                "tests": generation_result.test_code
            }
        else:
            raise HTTPException(status_code=500, detail=generation_result.error_message)
            
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/code/templates")
async def get_code_templates():
    """Get available code generation templates"""
    try:
        code_generator = get_code_generator()
        templates = await code_generator.get_templates()
        
        return {
            "templates": templates,  # templates is already a list of dicts
            "count": len(templates)
        }
        
    except Exception as e:
        logger.error(f"Failed to get code templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/code/metrics")
async def get_generation_metrics():
    """Get code generation performance metrics"""
    try:
        code_generator = get_code_generator()
        metrics = await code_generator.get_generation_metrics()
        
        return {
            "total_generations": metrics.get("total_generations", 0),
            "successful_generations": metrics.get("successful_generations", 0),
            "success_rate": metrics.get("success_rate", 0.0),
            "avg_generation_time": metrics.get("avg_generation_time", 0.0),
            "code_types_generated": metrics.get("code_type_distribution", {}),
            "quality_metrics": metrics.get("quality_summary", {}),
            "recent_generations": metrics.get("recent_generations", [])
        }
        
    except Exception as e:
        logger.error(f"Failed to get generation metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Emergent Intelligence Endpoints ---

@router.post("/emergence/facilitate")
async def facilitate_emergence(request: EmergenceFacilitationRequest):
    """Facilitate emergent intelligence behaviors"""
    try:
        emergence_system = get_emergent_intelligence()
        
        emergence_type = EmergenceType(request.target_behavior.lower())
        intelligence_level = IntelligenceLevel(request.intelligence_level)
        
        facilitation_result = await emergence_system.facilitate_emergence(
            emergence_type=emergence_type,
            participant_agents=request.participant_agents,
            conditions=request.conditions,
            target_level=intelligence_level
        )
        
        return {
            "success": facilitation_result.success,
            "emergence_id": facilitation_result.emergence_id,
            "behavior_type": request.target_behavior,
            "participants": request.participant_agents,
            "intelligence_level": request.intelligence_level,
            "initial_conditions": facilitation_result.initial_state,
            "expected_outcomes": facilitation_result.expected_behaviors,
            "monitoring_started": facilitation_result.monitoring_active
        }
        
    except Exception as e:
        logger.error(f"Emergence facilitation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/emergence/behaviors")
async def get_emergent_behaviors():
    """Get detected emergent behaviors"""
    try:
        emergence_system = get_emergent_intelligence()
        behaviors = await emergence_system.get_detected_behaviors()
        
        return {
            "detected_behaviors": [
                {
                    "behavior_id": behavior.behavior_id,
                    "type": behavior.emergence_type.value,
                    "participants": behavior.participants,
                    "intelligence_level": behavior.intelligence_level.value,
                    "emergence_strength": behavior.emergence_strength,
                    "stability": behavior.stability_score,
                    "outcomes": behavior.observed_outcomes,
                    "discovered_at": behavior.emergence_timestamp.isoformat()
                }
                for behavior in behaviors
            ],
            "count": len(behaviors)
        }
        
    except Exception as e:
        logger.error(f"Failed to get emergent behaviors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/emergence/intelligence-levels")
async def get_intelligence_analysis():
    """Get system-wide intelligence level analysis"""
    try:
        emergence_system = get_emergent_intelligence()
        analysis = await emergence_system.analyze_collective_intelligence()
        
        return {
            "collective_intelligence_level": analysis.get("collective_level", 0),
            "individual_levels": analysis.get("individual_levels", {}),
            "emerging_capabilities": analysis.get("new_capabilities", []),
            "intelligence_trends": analysis.get("trends", []),
            "complexity_metrics": analysis.get("complexity", {}),
            "adaptation_rate": analysis.get("adaptation_rate", 0.0),
            "innovation_potential": analysis.get("innovation_score", 0.0)
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze intelligence levels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- System Integration Endpoints ---

@router.get("/status")
async def get_autonomous_systems_status():
    """Get comprehensive status of all autonomous systems"""
    try:
        # Get status from all Phase 4 systems
        factory = get_agent_factory()
        healing_system = get_self_healing_system()
        code_generator = get_code_generator()
        emergence_system = get_emergent_intelligence()
        
        factory_status = await factory.get_system_status()
        healing_status = await healing_system.get_system_status()
        generation_status = await code_generator.get_system_status()
        emergence_status = await emergence_system.get_system_status()
        
        return {
            "phase": "Phase 4 - Autonomous Systems",
            "overall_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "systems": {
                "agent_factory": {
                    "status": factory_status.get("status", "unknown"),
                    "agents_active": factory_status.get("active_agents", 0),
                    "templates_available": factory_status.get("templates_count", 0)
                },
                "self_healing": {
                    "status": healing_status.get("overall_health", "unknown"),
                    "active_repairs": healing_status.get("repairs_in_progress", 0),
                    "system_uptime": healing_status.get("system_uptime", 0.0)
                },
                "code_generation": {
                    "status": generation_status.get("status", "unknown"),
                    "generations_completed": generation_status.get("total_generations", 0),
                    "success_rate": generation_status.get("success_rate", 0.0)
                },
                "emergent_intelligence": {
                    "status": emergence_status.get("status", "unknown"),
                    "behaviors_detected": emergence_status.get("behaviors_count", 0),
                    "intelligence_level": emergence_status.get("collective_intelligence", 1)
                }
            },
            "capabilities": [
                "Dynamic agent creation and management",
                "Autonomous failure detection and repair",
                "Real-time code generation and optimization", 
                "Emergent intelligence facilitation",
                "Self-improving system architecture"
            ]
        }
        
    except Exception as e:
        logger.error(f"Failed to get autonomous systems status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check for autonomous systems
@router.get("/health")
async def autonomous_health_check():
    """Health check for all Phase 4 autonomous systems"""
    try:
        return {
            "status": "healthy",
            "phase": "Phase 4 - Autonomous Systems",
            "components": {
                "agent_factory": "operational",
                "self_healing": "operational", 
                "code_generation": "operational",
                "emergent_intelligence": "operational"
            },
            "timestamp": datetime.now().isoformat(),
            "autonomy_level": "full"
        }
        
    except Exception as e:
        logger.error(f"Autonomous health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Background task initialization
async def initialize_autonomous_systems():
    """Initialize all Phase 4 autonomous systems"""
    try:
        logger.info("Initializing Phase 4 autonomous systems...")
        
        # Initialize systems in dependency order
        factory = get_agent_factory()
        await factory.start()
        
        healing_system = get_self_healing_system()
        await healing_system.start()
        
        code_generator = get_code_generator()
        await code_generator.start()
        
        emergence_system = get_emergent_intelligence()
        await emergence_system.start()
        
        logger.info("Phase 4 autonomous systems initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize autonomous systems: {e}")
        return False