"""
Base Agent Implementation for Sanskriti Setu AI Multi-Agent System
Provides the foundational class that all agents inherit from
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod

from core.shared.interfaces import (
    BaseAgent, TaskData, ExecutionResult, AgentCapability, 
    AgentStatus, AgentType, TaskStatus, SystemEvent
)
from core.shared.config import get_settings, get_feature_flag

logger = logging.getLogger(__name__)


class EnhancedBaseAgent(BaseAgent):
    """Enhanced base agent with dependency injection and event handling"""
    
    def __init__(
        self, 
        agent_id: str, 
        agent_type: AgentType,
        llm_client: Optional[Any] = None,
        memory_store: Optional[Any] = None,
        event_emitter: Optional[Callable] = None
    ):
        super().__init__(agent_id, agent_type)
        
        # Dependency injection
        self.llm_client = llm_client
        self.memory_store = memory_store  
        self.event_emitter = event_emitter or self._default_event_emitter
        
        # Agent state
        self._tasks_processed = 0
        self._total_execution_time = 0.0
        self._errors_count = 0
        self._last_error: Optional[str] = None
        self._initialization_time: Optional[datetime] = None
        
        # Configuration
        self.settings = get_settings()
        self._max_concurrent_tasks = self.settings.max_concurrent_tasks
        
        # Task management
        self._current_tasks: Dict[str, TaskData] = {}
        self._task_lock = asyncio.Lock()
        
        logger.info(f"Initialized agent {self.agent_id} of type {self.agent_type}")
    
    async def initialize(self) -> bool:
        """Initialize the agent - base implementation"""
        try:
            self._initialization_time = datetime.utcnow()
            
            # Initialize capabilities
            await self._initialize_capabilities()
            
            # Set up event handling
            await self.emit_event("agent_initialized", {
                "agent_id": self.agent_id,
                "agent_type": self.agent_type.value,
                "capabilities_count": len(self._capabilities)
            })
            
            logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self._last_error = str(e)
            return False
    
    async def process_task(self, task: TaskData) -> ExecutionResult:
        """Process a task with error handling and metrics"""
        logger.info(f"[DEBUG_BASE_PROCESS_TASK] Starting process_task for agent {self.agent_id}, task {task.id}")
        start_time = time.time()
        
        try:
            # Validate task
            logger.info(f"[DEBUG_BASE_PROCESS_TASK] Validating task {task.id}")
            if not await self.validate_task(task):
                logger.info(f"[DEBUG_BASE_PROCESS_TASK] Task validation FAILED for {task.id}")
                return ExecutionResult(
                    task_id=task.id,
                    success=False,
                    error_message=f"Task validation failed for agent {self.agent_id}"
                )
            logger.info(f"[DEBUG_BASE_PROCESS_TASK] Task validation PASSED for {task.id}")
            
            # Check concurrent task limit
            async with self._task_lock:
                if len(self._current_tasks) >= self._max_concurrent_tasks:
                    logger.info(f"[DEBUG_BASE_PROCESS_TASK] Agent at capacity, rejecting {task.id}")
                    return ExecutionResult(
                        task_id=task.id,
                        success=False,
                        error_message=f"Agent {self.agent_id} at capacity ({self._max_concurrent_tasks} tasks)"
                    )
                
                self._current_tasks[task.id] = task
                logger.info(f"[DEBUG_BASE_PROCESS_TASK] Task {task.id} added to current tasks")
            
            # Update current load
            self._current_load = len(self._current_tasks) / self._max_concurrent_tasks
            
            # Emit task started event
            await self.emit_event("task_started", {
                "agent_id": self.agent_id,
                "task_id": task.id,
                "current_load": self._current_load
            })
            logger.info(f"[DEBUG_BASE_PROCESS_TASK] About to call _execute_task for {task.id}")
            
            # Execute the actual task
            result = await self._execute_task(task)
            logger.info(f"[DEBUG_BASE_PROCESS_TASK] _execute_task completed for {task.id}")
            
            # Update metrics
            execution_time = time.time() - start_time
            self._tasks_processed += 1
            self._total_execution_time += execution_time
            
            result.execution_time = execution_time
            
            # Emit task completed event
            await self.emit_event("task_completed", {
                "agent_id": self.agent_id,
                "task_id": task.id,
                "success": result.success,
                "execution_time": execution_time
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._errors_count += 1
            self._last_error = str(e)
            
            logger.error(f"Error processing task {task.id} in agent {self.agent_id}: {e}")
            
            await self.emit_event("task_failed", {
                "agent_id": self.agent_id,
                "task_id": task.id,
                "error": str(e),
                "execution_time": execution_time
            })
            
            return ExecutionResult(
                task_id=task.id,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
            
        finally:
            # Clean up task tracking
            async with self._task_lock:
                self._current_tasks.pop(task.id, None)
                self._current_load = len(self._current_tasks) / self._max_concurrent_tasks
    
    @abstractmethod
    async def _execute_task(self, task: TaskData) -> ExecutionResult:
        """Execute the actual task - must be implemented by subclasses"""
        pass
    
    async def _initialize_capabilities(self) -> None:
        """Initialize agent capabilities - can be overridden by subclasses"""
        # Default capabilities - subclasses should override this
        self._capabilities = [
            AgentCapability(
                name="basic_processing",
                description="Basic task processing capability",
                input_types=["text"],
                output_types=["text"],
                resource_requirements={"memory": "128MB", "cpu": "0.1"}
            )
        ]
    
    async def validate_task(self, task: TaskData) -> bool:
        """Validate if this agent can handle the task"""
        # Basic validation - check if task type is supported
        task_type = task.metadata.get("type", "general")
        
        for capability in self._capabilities:
            if task_type in capability.input_types:
                return True
        
        # If no specific capability matches, allow general tasks
        return task_type in ["general", "text"]
    
    async def get_status(self) -> AgentStatus:
        """Get enhanced agent status with metrics"""
        base_status = await super().get_status()
        
        # Add enhanced metrics
        base_status.metrics = {
            "tasks_processed": self._tasks_processed,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": (
                self._total_execution_time / self._tasks_processed 
                if self._tasks_processed > 0 else 0.0
            ),
            "errors_count": self._errors_count,
            "error_rate": (
                self._errors_count / self._tasks_processed 
                if self._tasks_processed > 0 else 0.0
            ),
            "last_error": self._last_error,
            "current_tasks_count": len(self._current_tasks),
            "initialization_time": (
                self._initialization_time.isoformat() 
                if self._initialization_time else None
            ),
            "uptime_seconds": (
                (datetime.utcnow() - self._initialization_time).total_seconds()
                if self._initialization_time else 0
            )
        }
        
        return base_status
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit system event"""
        try:
            if self.event_emitter:
                event = SystemEvent(
                    event_type=event_type,
                    source=self.agent_id,
                    data=data
                )
                await self.event_emitter(event)
        except Exception as e:
            logger.warning(f"Failed to emit event {event_type}: {e}")
    
    async def _default_event_emitter(self, event: SystemEvent) -> None:
        """Default event emitter - logs events"""
        logger.info(f"Event: {event.event_type} from {event.source}")
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the agent"""
        try:
            # Wait for current tasks to complete (with timeout)
            max_wait = 30  # 30 seconds
            wait_start = time.time()
            
            while self._current_tasks and (time.time() - wait_start) < max_wait:
                logger.info(f"Waiting for {len(self._current_tasks)} tasks to complete...")
                await asyncio.sleep(1)
            
            if self._current_tasks:
                logger.warning(f"Shutting down with {len(self._current_tasks)} tasks still running")
            
            await self.emit_event("agent_shutdown", {
                "agent_id": self.agent_id,
                "tasks_processed": self._tasks_processed,
                "uptime_seconds": (
                    (datetime.utcnow() - self._initialization_time).total_seconds()
                    if self._initialization_time else 0
                )
            })
            
            logger.info(f"Agent {self.agent_id} shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during agent shutdown: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "tasks_processed": self._tasks_processed,
            "total_execution_time": self._total_execution_time,
            "average_execution_time": (
                self._total_execution_time / self._tasks_processed 
                if self._tasks_processed > 0 else 0.0
            ),
            "errors_count": self._errors_count,
            "error_rate": (
                self._errors_count / self._tasks_processed 
                if self._tasks_processed > 0 else 0.0
            ),
            "current_load": self._current_load,
            "max_concurrent_tasks": self._max_concurrent_tasks,
            "current_tasks_count": len(self._current_tasks),
            "capabilities_count": len(self._capabilities),
            "is_active": self._is_active,
            "initialization_time": (
                self._initialization_time.isoformat() 
                if self._initialization_time else None
            )
        }


# Utility functions for agent management
async def create_agent_with_dependencies(
    agent_class: type,
    agent_id: str,
    agent_type: AgentType,
    **kwargs
) -> EnhancedBaseAgent:
    """Factory function to create agents with proper dependency injection"""
    
    # Set up LLM client based on configuration
    llm_client = None
    if get_feature_flag("llm_integration"):
        llm_client = await _setup_llm_client()
    
    # Set up memory store
    memory_store = None
    if get_feature_flag("agent_memory"):
        memory_store = await _setup_memory_store()
    
    # Set up event emitter
    event_emitter = await _setup_event_emitter()
    
    # Create agent
    agent = agent_class(
        agent_id=agent_id,
        agent_type=agent_type,
        llm_client=llm_client,
        memory_store=memory_store,
        event_emitter=event_emitter,
        **kwargs
    )
    
    # Initialize
    if not await agent.initialize():
        raise RuntimeError(f"Failed to initialize agent {agent_id}")
    
    return agent


async def _setup_llm_client():
    """Set up LLM client (placeholder for Week 2)"""
    # Mock implementation for Week 1
    return None


async def _setup_memory_store():
    """Set up memory store (placeholder for Week 2)"""
    # Mock implementation for Week 1
    return None


async def _setup_event_emitter():
    """Set up event emitter system"""
    # Simple logging-based emitter for Week 1
    async def log_emitter(event: SystemEvent):
        logger.info(f"System Event: {event.event_type} from {event.source} - {event.data}")
    
    return log_emitter