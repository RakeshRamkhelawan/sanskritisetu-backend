#!/usr/bin/env python3
"""
FastAPI endpoints for Workflow Engine
Phase 2 - Workflow Orchestration API Integration
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from core.agents.workflow_engine import (
    get_workflow_engine, WorkflowExecutionMode, TaskPriority, TaskStatus,
    WorkflowDefinition, WorkflowTask, WorkflowExecution
)

# API Models
class TaskCreateRequest(BaseModel):
    """Request model for creating a task"""
    name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    agent_type: str = Field(..., description="Required agent type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    priority: str = Field(default="MEDIUM", description="Task priority (LOW, MEDIUM, HIGH, CRITICAL)")

class WorkflowCreateRequest(BaseModel):
    """Request model for creating a workflow"""
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    execution_mode: str = Field(default="SEQUENTIAL", description="Execution mode (SEQUENTIAL, PARALLEL, MIXED)")
    timeout: int = Field(default=300, description="Timeout in seconds")
    tasks: List[TaskCreateRequest] = Field(default_factory=list, description="Workflow tasks")

class WorkflowExecuteRequest(BaseModel):
    """Request model for executing a workflow"""
    workflow_id: str = Field(..., description="Workflow ID to execute")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")

class TaskResponse(BaseModel):
    """Response model for task information"""
    task_id: str
    name: str
    description: str
    agent_type: str
    parameters: Dict[str, Any]
    dependencies: List[str]
    priority: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

class WorkflowResponse(BaseModel):
    """Response model for workflow information"""
    workflow_id: str
    name: str
    description: str
    execution_mode: str
    tasks: List[TaskResponse]
    metadata: Dict[str, Any]
    timeout: int

class ExecutionResponse(BaseModel):
    """Response model for execution information"""
    execution_id: str
    workflow_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    tasks_completed: int
    tasks_failed: int
    tasks_total: int
    results: Optional[Dict[str, Any]] = None

class WorkflowStatsResponse(BaseModel):
    """Response model for workflow statistics"""
    total_workflows: int
    total_executions: int
    active_executions: int
    successful_executions: int
    failed_executions: int
    subagents_available: int

# Create router
workflow_router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])

# Global workflow engine instance
engine = get_workflow_engine()

def _convert_task_to_response(task: WorkflowTask) -> TaskResponse:
    """Convert WorkflowTask to TaskResponse"""
    return TaskResponse(
        task_id=task.task_id,
        name=task.name,
        description=task.description,
        agent_type=task.agent_type,
        parameters=task.parameters,
        dependencies=task.dependencies,
        priority=task.priority.name,
        status=task.status.value,
        result=task.result,
        error=task.error,
        start_time=task.start_time,
        end_time=task.end_time,
        duration=task.duration
    )

def _convert_workflow_to_response(workflow: WorkflowDefinition) -> WorkflowResponse:
    """Convert WorkflowDefinition to WorkflowResponse"""
    return WorkflowResponse(
        workflow_id=workflow.workflow_id,
        name=workflow.name,
        description=workflow.description,
        execution_mode=workflow.execution_mode.value,
        tasks=[_convert_task_to_response(task) for task in workflow.tasks],
        metadata=workflow.metadata,
        timeout=workflow.timeout
    )

def _convert_execution_to_response(execution: WorkflowExecution) -> ExecutionResponse:
    """Convert WorkflowExecution to ExecutionResponse"""
    return ExecutionResponse(
        execution_id=execution.execution_id,
        workflow_id=execution.workflow_id,
        status=execution.status.value,
        start_time=execution.start_time,
        end_time=execution.end_time,
        duration=execution.duration,
        tasks_completed=execution.tasks_completed,
        tasks_failed=execution.tasks_failed,
        tasks_total=execution.tasks_total,
        results=execution.results
    )

@workflow_router.post("/", response_model=Dict[str, str])
async def create_workflow(request: WorkflowCreateRequest):
    """Create a new workflow"""
    try:
        # Validate execution mode
        try:
            execution_mode = WorkflowExecutionMode(request.execution_mode)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid execution mode: {request.execution_mode}")
        
        # Create workflow
        workflow_id = engine.create_workflow(
            name=request.name,
            description=request.description,
            execution_mode=execution_mode
        )
        
        # Add tasks
        for task_req in request.tasks:
            try:
                priority = TaskPriority[task_req.priority]
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid task priority: {task_req.priority}")
            
            engine.add_task(
                workflow_id=workflow_id,
                name=task_req.name,
                description=task_req.description,
                agent_type=task_req.agent_type,
                parameters=task_req.parameters,
                dependencies=task_req.dependencies,
                priority=priority
            )
        
        return {"workflow_id": workflow_id, "status": "created"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")

@workflow_router.get("/", response_model=List[WorkflowResponse])
async def list_workflows():
    """List all workflows"""
    workflows = engine.list_workflows()
    return [_convert_workflow_to_response(workflow) for workflow in workflows]

@workflow_router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str):
    """Get a specific workflow"""
    workflow = engine.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    return _convert_workflow_to_response(workflow)

@workflow_router.post("/{workflow_id}/tasks", response_model=Dict[str, str])
async def add_task_to_workflow(workflow_id: str, task: TaskCreateRequest):
    """Add a task to an existing workflow"""
    try:
        # Validate priority
        try:
            priority = TaskPriority[task.priority]
        except KeyError:
            raise HTTPException(status_code=400, detail=f"Invalid task priority: {task.priority}")
        
        task_id = engine.add_task(
            workflow_id=workflow_id,
            name=task.name,
            description=task.description,
            agent_type=task.agent_type,
            parameters=task.parameters,
            dependencies=task.dependencies,
            priority=priority
        )
        
        return {"task_id": task_id, "status": "added"}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add task: {str(e)}")

async def _execute_workflow_background(workflow_id: str, context: Dict[str, Any]):
    """Background task for workflow execution"""
    try:
        execution_id = await engine.execute_workflow(workflow_id, context)
        return execution_id
    except Exception as e:
        print(f"Background workflow execution failed: {str(e)}")

@workflow_router.post("/execute", response_model=Dict[str, str])
async def execute_workflow(request: WorkflowExecuteRequest, background_tasks: BackgroundTasks):
    """Execute a workflow"""
    try:
        workflow = engine.get_workflow(request.workflow_id)
        if not workflow:
            raise HTTPException(status_code=404, detail=f"Workflow {request.workflow_id} not found")
        
        # Start background execution
        background_tasks.add_task(_execute_workflow_background, request.workflow_id, request.context)
        
        return {"status": "execution_started", "workflow_id": request.workflow_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute workflow: {str(e)}")

@workflow_router.get("/executions/", response_model=List[ExecutionResponse])
async def list_executions():
    """List all workflow executions"""
    executions = engine.list_executions()
    return [_convert_execution_to_response(execution) for execution in executions]

@workflow_router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution(execution_id: str):
    """Get a specific execution"""
    execution = engine.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    return _convert_execution_to_response(execution)

@workflow_router.get("/stats", response_model=WorkflowStatsResponse)
async def get_workflow_stats():
    """Get workflow engine statistics"""
    stats = engine.get_workflow_stats()
    return WorkflowStatsResponse(**stats)

# Advanced endpoints

@workflow_router.post("/validate/{workflow_id}", response_model=Dict[str, Any])
async def validate_workflow(workflow_id: str):
    """Validate a workflow for execution readiness"""
    workflow = engine.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    try:
        # Validate dependencies
        errors = engine._validate_dependencies(workflow)
        
        # Build execution plan
        execution_plan = engine._build_execution_plan(workflow)
        
        # Check for circular dependencies or errors
        has_errors = any(any(task_id.startswith("ERROR:") for task_id in batch) for batch in execution_plan)
        
        return {
            "valid": not bool(errors) and not has_errors,
            "errors": errors,
            "execution_plan": execution_plan,
            "total_tasks": len(workflow.tasks),
            "execution_batches": len(execution_plan)
        }
        
    except Exception as e:
        return {
            "valid": False,
            "errors": [f"Validation failed: {str(e)}"],
            "execution_plan": [],
            "total_tasks": len(workflow.tasks),
            "execution_batches": 0
        }

@workflow_router.get("/executions/{execution_id}/tasks", response_model=List[TaskResponse])
async def get_execution_tasks(execution_id: str):
    """Get tasks for a specific execution"""
    execution = engine.get_execution(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    workflow = engine.get_workflow(execution.workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {execution.workflow_id} not found")
    
    return [_convert_task_to_response(task) for task in workflow.tasks]

@workflow_router.delete("/{workflow_id}", response_model=Dict[str, str])
async def delete_workflow(workflow_id: str):
    """Delete a workflow"""
    workflow = engine.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
    
    # Check if there are active executions
    active_executions = [e for e in engine.list_executions() 
                        if e.workflow_id == workflow_id and e.status == TaskStatus.RUNNING]
    
    if active_executions:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot delete workflow with active executions: {[e.execution_id for e in active_executions]}"
        )
    
    # Remove workflow
    del engine.workflows[workflow_id]
    
    return {"status": "deleted", "workflow_id": workflow_id}

@workflow_router.post("/bulk-create", response_model=List[Dict[str, str]])
async def bulk_create_workflows(requests: List[WorkflowCreateRequest]):
    """Create multiple workflows in bulk"""
    results = []
    
    for i, request in enumerate(requests):
        try:
            # Validate execution mode
            try:
                execution_mode = WorkflowExecutionMode(request.execution_mode)
            except ValueError:
                results.append({
                    "index": i,
                    "status": "error",
                    "error": f"Invalid execution mode: {request.execution_mode}"
                })
                continue
            
            # Create workflow
            workflow_id = engine.create_workflow(
                name=request.name,
                description=request.description,
                execution_mode=execution_mode
            )
            
            # Add tasks
            for task_req in request.tasks:
                try:
                    priority = TaskPriority[task_req.priority]
                except KeyError:
                    results.append({
                        "index": i,
                        "status": "error",
                        "error": f"Invalid task priority: {task_req.priority}"
                    })
                    # Remove partially created workflow
                    if workflow_id in engine.workflows:
                        del engine.workflows[workflow_id]
                    continue
                
                engine.add_task(
                    workflow_id=workflow_id,
                    name=task_req.name,
                    description=task_req.description,
                    agent_type=task_req.agent_type,
                    parameters=task_req.parameters,
                    dependencies=task_req.dependencies,
                    priority=priority
                )
            
            results.append({
                "index": i,
                "status": "created",
                "workflow_id": workflow_id
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "status": "error",
                "error": str(e)
            })
    
    return results