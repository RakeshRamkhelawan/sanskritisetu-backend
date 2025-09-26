"""
Multi-Agent Registry System
Manages discovery, registration, and coordination of all agents in the system
FASE 6: Converted to use database persistence
"""
import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from collections import defaultdict

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update, delete
from sqlalchemy.exc import SQLAlchemyError

from core.shared.interfaces import (
    Agent, AgentType, AgentCapability, AgentStatus, TaskData,
    ExecutionResult, TaskPriority
)
from core.database.database import get_async_session
from core.database.models import RegisteredAgent

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry for all agents in the multi-agent system - Database persistent"""

    def __init__(self, db_session: Optional[AsyncSession] = None):
        # In-memory cache for active agents (for performance)
        self._agents: Dict[str, Agent] = {}
        self._agents_by_type: Dict[AgentType, List[str]] = defaultdict(list)
        self._agent_capabilities: Dict[str, Set[str]] = defaultdict(set)

        # Health monitoring (still in-memory for real-time needs)
        self._agent_health: Dict[str, datetime] = {}
        self._agent_load: Dict[str, float] = defaultdict(float)
        self._agent_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)

        # Task routing (still in-memory for performance)
        self._active_tasks: Dict[str, str] = {}  # task_id -> agent_id
        self._task_queue: List[TaskData] = []

        # Event callbacks
        self._event_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Database session for dependency injection (testing)
        self._db_session = db_session

        logger.info("AgentRegistry initialized with database persistence")
    
    async def register_agent(self, agent: Agent) -> bool:
        """Register a new agent in the system with database persistence"""
        try:
            agent_id = agent.agent_id
            agent_type = agent.agent_type

            # Get capabilities
            capabilities = await agent.get_capabilities()
            capabilities_dict = {cap.name: cap.description for cap in capabilities}

            # Store in database
            async with (self._db_session or get_async_session()) as session:
                # Check if agent already exists
                result = await session.execute(
                    select(RegisteredAgent).where(RegisteredAgent.agent_id == agent_id)
                )
                existing_agent = result.scalar_one_or_none()

                if existing_agent:
                    # Update existing agent
                    stmt = update(RegisteredAgent).where(
                        RegisteredAgent.agent_id == agent_id
                    ).values(
                        address=str(agent.address),
                        capabilities=capabilities_dict,
                        last_heartbeat=datetime.now(),
                        updated_at=datetime.now()
                    )
                else:
                    # Insert new agent
                    stmt = insert(RegisteredAgent).values(
                        agent_id=agent_id,
                        address=str(agent.address),
                        capabilities=capabilities_dict,
                        last_heartbeat=datetime.now()
                    )

                await session.execute(stmt)
                await session.commit()

            # Store in memory cache
            self._agents[agent_id] = agent
            self._agents_by_type[agent_type].append(agent_id)
            self._agent_capabilities[agent_id] = {cap.name for cap in capabilities}

            # Initialize health tracking
            self._agent_health[agent_id] = datetime.now()
            self._agent_load[agent_id] = 0.0

            # Initialize stats
            self._agent_stats[agent_id] = {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "total_execution_time": 0.0,
                "average_response_time": 0.0,
                "registered_at": datetime.now(),
                "last_active": datetime.now()
            }

            logger.info(f"Agent registered in database: {agent_id} (type: {agent_type})")
            await self._emit_event("agent_registered", {
                "agent_id": agent_id,
                "agent_type": agent_type,
                "capabilities": list(self._agent_capabilities[agent_id])
            })

            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent.agent_id}: {e}")
            return False

    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all registered agents from database"""
        try:
            async with (self._db_session or get_async_session()) as session:
                result = await session.execute(select(RegisteredAgent))
                agents = result.scalars().all()

                agent_list = []
                for agent in agents:
                    agent_list.append({
                        "agent_id": agent.agent_id,
                        "address": agent.address,
                        "capabilities": agent.capabilities,
                        "last_heartbeat": agent.last_heartbeat,
                        "created_at": agent.created_at,
                        "updated_at": agent.updated_at
                    })

                return agent_list

        except Exception as e:
            logger.error(f"Failed to get all agents from database: {e}")
            return []

    async def find_best_agent_for_task(self, task: TaskData) -> Optional[Agent]:
        """Find the best available agent for a specific task"""
        try:
            # Get task requirements from metadata
            task_type = task.metadata.get("type", "general")
            required_capabilities = task.metadata.get("capabilities", [])
            preferred_agent_type = task.metadata.get("agent_type")
            
            candidate_agents = []
            
            # If specific agent type is preferred, check those first
            if preferred_agent_type:
                try:
                    agent_type = AgentType(preferred_agent_type)
                    candidate_agents = await self.get_agents_by_type(agent_type)
                except ValueError:
                    logger.warning(f"Unknown agent type requested: {preferred_agent_type}")
            
            # If no candidates yet, check by capabilities
            if not candidate_agents and required_capabilities:
                for capability in required_capabilities:
                    agents_with_cap = await self.get_agents_with_capability(capability)
                    candidate_agents.extend(agents_with_cap)
            
            # If still no candidates, get all specialist agents
            if not candidate_agents:
                candidate_agents = await self.get_agents_by_type(AgentType.SPECIALIST)
                # If no specialists, get CVA as fallback
                if not candidate_agents:
                    candidate_agents = await self.get_agents_by_type(AgentType.CVA)
            
            # Score agents based on load, performance, and availability
            if candidate_agents:
                scored_agents = []
                for agent in candidate_agents:
                    agent_id = agent.agent_id
                    
                    # Check if agent is healthy and active
                    if not await self._is_agent_healthy(agent_id):
                        continue
                    
                    # Calculate score (lower is better)
                    load_score = self._agent_load.get(agent_id, 0.0)
                    stats = self._agent_stats.get(agent_id, {})
                    performance_score = 1.0 - (stats.get("tasks_completed", 0) / 
                                             max(stats.get("tasks_completed", 0) + stats.get("tasks_failed", 0) + 1, 1))
                    
                    total_score = load_score + performance_score
                    scored_agents.append((total_score, agent))
                
                if scored_agents:
                    # Return agent with best (lowest) score
                    scored_agents.sort(key=lambda x: x[0])
                    return scored_agents[0][1]
            
            logger.warning(f"No suitable agent found for task: {task.id}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding agent for task {task.id}: {e}")
            return None
    
    async def distribute_task(self, task: TaskData) -> Optional[ExecutionResult]:
        """Distribute task to the best available agent"""
        try:
            start_time = datetime.now()
            
            # Find best agent
            agent = await self.find_best_agent_for_task(task)
            if not agent:
                logger.error(f"No agent available for task: {task.id}")
                return ExecutionResult(
                task_id=task.id,
                success=False, error_message="No suitable agent available",
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
            
            agent_id = agent.agent_id
            
            # Track task assignment
            self._active_tasks[task.id] = agent_id
            self._agent_load[agent_id] += 0.1  # Increment load
            
            logger.info(f"Distributing task {task.id} to agent {agent_id}")
            
            # Execute task
            try:
                result = await agent.process_task(task)
                
                # Update stats
                execution_time = (datetime.now() - start_time).total_seconds()
                stats = self._agent_stats[agent_id]
                
                if result.success:
                    stats["tasks_completed"] += 1
                else:
                    stats["tasks_failed"] += 1
                
                stats["total_execution_time"] += execution_time
                stats["average_response_time"] = (
                    stats["total_execution_time"] / 
                    (stats["tasks_completed"] + stats["tasks_failed"])
                )
                stats["last_active"] = datetime.now()
                
                await self._emit_event("task_completed", {
                    "task_id": task.id,
                    "agent_id": agent_id,
                    "success": result.success,
                    "execution_time": execution_time
                })
                
                return result
                
            finally:
                # Clean up task tracking
                self._active_tasks.pop(task.id, None)
                self._agent_load[agent_id] = max(0.0, self._agent_load[agent_id] - 0.1)
            
        except Exception as e:
            logger.error(f"Error distributing task {task.id}: {e}")
            return ExecutionResult(
                task_id=task.id,
                success=False, error_message=f"Task distribution failed: {e}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def get_agents_by_type(self, agent_type: AgentType) -> List[Agent]:
        """Get all agents of a specific type"""
        agent_ids = self._agents_by_type[agent_type]
        return [self._agents[aid] for aid in agent_ids if aid in self._agents]
    
    async def get_agents_with_capability(self, capability: str) -> List[Agent]:
        """Get all agents that have a specific capability"""
        matching_agents = []
        for agent_id, capabilities in self._agent_capabilities.items():
            if capability in capabilities:
                if agent_id in self._agents:
                    matching_agents.append(self._agents[agent_id])
        
        return matching_agents
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        total_agents = len(self._agents)
        
        # Calculate healthy agents properly (avoid async generator)
        healthy_agents = 0
        for aid in self._agents.keys():
            if await self._is_agent_healthy(aid):
                healthy_agents += 1
        
        agents_by_type = {}
        for agent_type in AgentType:
            count = len(self._agents_by_type[agent_type])
            agents_by_type[agent_type.value] = count
        
        # Calculate system load
        total_load = sum(self._agent_load.values())
        avg_load = total_load / max(total_agents, 1)
        
        # Aggregate stats
        total_tasks_completed = sum(stats.get("tasks_completed", 0) for stats in self._agent_stats.values())
        total_tasks_failed = sum(stats.get("tasks_failed", 0) for stats in self._agent_stats.values())
        
        return {
            "total_agents": total_agents,
            "healthy_agents": healthy_agents,
            "agents_by_type": agents_by_type,
            "active_tasks": len(self._active_tasks),
            "system_load": avg_load,
            "tasks_completed": total_tasks_completed,
            "tasks_failed": total_tasks_failed,
            "success_rate": total_tasks_completed / max(total_tasks_completed + total_tasks_failed, 1) * 100,
            "uptime": datetime.now().isoformat()
        }
    
    async def _is_agent_healthy(self, agent_id: str) -> bool:
        """Check if agent is healthy (responded recently)"""
        if agent_id not in self._agent_health:
            return False
        
        last_health = self._agent_health[agent_id]
        health_timeout = timedelta(minutes=5)  # 5 minute health timeout
        
        return datetime.now() - last_health < health_timeout
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit system event to registered callbacks"""
        try:
            if event_type in self._event_callbacks:
                for callback in self._event_callbacks[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(data)
                        else:
                            callback(data)
                    except Exception as e:
                        logger.error(f"Event callback error for {event_type}: {e}")
        except Exception as e:
            logger.error(f"Error emitting event {event_type}: {e}")


# Singleton instance
_registry_instance: Optional[AgentRegistry] = None

def get_agent_registry(db_session: Optional[AsyncSession] = None) -> AgentRegistry:
    """Get the global agent registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = AgentRegistry(db_session)
    return _registry_instance