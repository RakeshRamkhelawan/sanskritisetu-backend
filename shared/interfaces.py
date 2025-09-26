"""
Core interfaces and protocols for the Sanskriti Setu AI Multi-Agent System
Defines the foundational contracts that all components must implement
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


# Enums and Constants
class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    APPROVAL_REQUIRED = "approval_required"


class TaskPriority(int, Enum):
    LOW = 1
    MEDIUM = 3
    HIGH = 5
    CRITICAL = 9


class IsolationLevel(str, Enum):
    MOCK = "mock"           # Week 1 - No isolation, direct execution
    PROCESS = "process"     # Week 2 - Process-level isolation
    CONTAINER = "container" # Week 2 - Docker container isolation
    VM = "vm"              # Week 3 - Full VM isolation


class AgentType(str, Enum):
    CVA = "cva"
    ORCHESTRATOR = "orchestrator"
    ETHICS = "ethics"
    SPECIALIST = "specialist"
    SANDBOX = "sandbox"


# Core Data Models
class TaskData(BaseModel):
    """Core task data structure"""
    id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    title: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_to: Optional[str] = None
    estimated_duration: Optional[int] = None  # in minutes
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Result of a task execution"""
    task_id: str
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0  # in seconds
    resource_usage: Dict[str, Any] = Field(default_factory=dict)
    logs: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TaskResult(BaseModel):
    """Result of agent task processing - specialized for agent workflows"""
    task_id: str
    agent_id: str
    result: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    execution_time: float = 0.0  # in seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentCapability(BaseModel):
    """Describes what an agent can do"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    resource_requirements: Dict[str, Any] = Field(default_factory=dict)
    estimated_duration: Optional[int] = None


class AgentStatus(BaseModel):
    """Current status of an agent"""
    agent_id: str
    agent_type: AgentType
    is_active: bool = True
    current_load: float = 0.0  # 0.0 to 1.0
    capabilities: List[AgentCapability] = Field(default_factory=list)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class SandboxEnvironment(BaseModel):
    """Sandbox environment configuration"""
    id: str = Field(default_factory=lambda: f"sandbox_{uuid.uuid4().hex[:8]}")
    isolation_level: IsolationLevel
    resource_limits: Dict[str, Any] = Field(default_factory=dict)
    allowed_capabilities: List[str] = Field(default_factory=list)
    network_access: bool = False
    filesystem_access: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ApprovalRequest(BaseModel):
    """Request for human approval"""
    id: str = Field(default_factory=lambda: f"approval_{uuid.uuid4().hex[:8]}")
    task_id: str
    requesting_agent: str
    approval_type: str
    description: str
    risk_level: int = Field(ge=1, le=10)  # 1=low, 10=critical
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    approved: Optional[bool] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


# Protocol Interfaces
@runtime_checkable
class Agent(Protocol):
    """Core agent protocol - all agents must implement this"""
    
    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent"""
        ...
    
    @property
    def agent_type(self) -> AgentType:
        """Type of this agent"""
        ...
    
    async def process_task(self, task: TaskData) -> ExecutionResult:
        """Process a task and return result"""
        ...
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get list of capabilities this agent supports"""
        ...
    
    async def get_status(self) -> AgentStatus:
        """Get current status of this agent"""
        ...


@runtime_checkable
class SandboxManager(Protocol):
    """Sandbox management protocol"""
    
    async def create_sandbox(self, config: SandboxEnvironment) -> str:
        """Create a new sandbox environment"""
        ...
    
    async def execute_in_sandbox(self, sandbox_id: str, task: TaskData) -> ExecutionResult:
        """Execute task in specified sandbox"""
        ...
    
    async def destroy_sandbox(self, sandbox_id: str) -> bool:
        """Destroy sandbox and cleanup resources"""
        ...
    
    async def get_sandbox_status(self, sandbox_id: str) -> Dict[str, Any]:
        """Get current status of sandbox"""
        ...


@runtime_checkable
class EventEmitter(Protocol):
    """Event emission protocol for system-wide communication"""
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event to the system"""
        ...


# Abstract Base Classes
class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        self._agent_id = agent_id
        self._agent_type = agent_type
        self._is_active = True
        self._current_load = 0.0
        self._capabilities: List[AgentCapability] = []
        
    @property
    def agent_id(self) -> str:
        return self._agent_id
        
    @property
    def agent_type(self) -> AgentType:
        return self._agent_type
    
    @abstractmethod
    async def process_task(self, task: TaskData) -> ExecutionResult:
        """Process a task - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the agent - must be implemented by subclasses"""
        pass
    
    async def get_capabilities(self) -> List[AgentCapability]:
        """Get agent capabilities - can be overridden"""
        return self._capabilities
    
    async def get_status(self) -> AgentStatus:
        """Get current agent status"""
        return AgentStatus(
            agent_id=self._agent_id,
            agent_type=self._agent_type,
            is_active=self._is_active,
            current_load=self._current_load,
            capabilities=self._capabilities,
            last_heartbeat=datetime.utcnow()
        )
    
    async def validate_task(self, task: TaskData) -> bool:
        """Validate if this agent can handle the task"""
        return True  # Default implementation
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event - default no-op implementation"""
        pass


# System Events
class SystemEvent(BaseModel):
    """System event data structure"""
    id: str = Field(default_factory=lambda: f"event_{uuid.uuid4().hex[:8]}")
    event_type: str
    source: str
    data: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: str = "info"  # debug, info, warning, error, critical


# Database Repository Protocols
@runtime_checkable
class TaskRepository(Protocol):
    """Task data repository protocol"""
    
    async def create_task(self, task: TaskData) -> str:
        """Create a new task"""
        ...
    
    async def get_task(self, task_id: str) -> Optional[TaskData]:
        """Get task by ID"""
        ...
    
    async def update_task(self, task: TaskData) -> bool:
        """Update existing task"""
        ...
    
    async def list_tasks(self, status: Optional[TaskStatus] = None) -> List[TaskData]:
        """List tasks, optionally filtered by status"""
        ...


@runtime_checkable
class ExecutionRepository(Protocol):
    """Execution result repository protocol"""
    
    async def store_result(self, result: ExecutionResult) -> str:
        """Store execution result"""
        ...
    
    async def get_results(self, task_id: str) -> List[ExecutionResult]:
        """Get all results for a task"""
        ...


# Configuration
class SystemConfig(BaseModel):
    """System-wide configuration"""
    environment: str = "development"
    debug: bool = True
    
    # Database
    database_url: str = "sqlite:///./sanskriti_setu.db"
    database_pool_size: int = 5
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Security
    secret_key: str = "dev-secret-key-change-in-production"
    access_token_expire_minutes: int = 30
    
    # Feature Flags (Week-based rollout)
    feature_flags: Dict[str, bool] = Field(default_factory=lambda: {
        "real_docker_sandbox": True,       # Week 2 - Production enabled
        "multi_agent_orchestration": False, # Week 2
        "ml_ethics_engine": False,          # Week 3
        "knowledge_evolution": False,       # Week 3
        "advanced_ui": False,               # Week 3
    })
    
    # Sandbox
    default_isolation_level: IsolationLevel = IsolationLevel.MOCK
    sandbox_timeout: int = 300  # 5 minutes
    
    # Agent Configuration  
    max_concurrent_tasks: int = 10
    agent_heartbeat_interval: int = 30  # seconds
    
    # External Services
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None