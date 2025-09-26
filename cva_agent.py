"""
Chief Visionary Architect (CVA) Agent Implementation
The primary AI agent responsible for strategic planning and task orchestration
Enhanced with Memory-Augmented MDP CEO Mode capabilities
"""

import asyncio
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Union
from dataclasses import dataclass
from enum import Enum

from core.agents.base_agent import EnhancedBaseAgent
from core.shared.interfaces import (
    TaskData, ExecutionResult, AgentCapability, AgentType, 
    TaskPriority, ApprovalRequest
)
from core.shared.config import get_settings, get_feature_flag
from core.shared.llm_client import (
    LLMClient, LLMMessage, LLMResponse, get_default_llm_config, create_llm_client
)
from core.agents.cva_prompts import get_cva_prompts
from core.agents.ethics_gate import get_ethics_gate, EthicsLevel
from core.shared.response_cache import get_response_cache, cached_llm_response
from core.agents.task_templates import get_task_generator, TaskTemplate, TaskGenerationConfig, GeneratedTask
from core.memory.memory_manager import memory_manager, StrategicMemory
from core.communication.communication_manager import communication_manager
from core.communication.message_protocol import MessageFactory, MessageType, MessagePriority

logger = logging.getLogger(__name__)

# M-MDP CEO_MODE Enums and Dataclasses (from archive)
class CVAMode(Enum):
    """CVA operational modes for M-MDP strategic learning"""
    REACTIVE = "reactive"           # Responds to direct requests
    PROACTIVE = "proactive"         # Monitors and initiates actions
    AUTONOMOUS = "autonomous"       # Full self-operation
    CEO_MODE = "ceo_mode"          # Executive decision making with strategic memory

class TriggerType(Enum):
    """Autonomous trigger types for strategic learning"""
    SCHEDULED = "scheduled"
    EVENT_DRIVEN = "event_driven"
    INTELLIGENT = "intelligent"
    USER_INITIATED = "user_initiated"
    EMERGENCY = "emergency"

@dataclass
class OrchestrationCommand:
    """Command structure for orchestrator communication with memory context"""
    command: str
    parameters: Dict[str, Any]
    priority: int
    masterprompt: Optional[str] = None
    expected_agents: List[str] = None

@dataclass
class CVAState:
    """CVA operational state tracking for strategic memory"""
    mode: CVAMode
    active_orchestrations: List[str]
    learning_context: Dict[str, Any]
    performance_metrics: Dict[str, float]
    last_optimization: datetime
    autonomous_triggers_active: bool


class CVAAgent(EnhancedBaseAgent):
    """Chief Visionary Architect Agent - The strategic brain of the system"""
    
    def __init__(self, agent_id: str = "cva_main", agent_type: AgentType = AgentType.CVA, **kwargs):
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            **kwargs
        )
        
        # CVA-specific configuration
        self.vision_context: Dict[str, Any] = {}
        self.strategic_objectives: List[str] = []
        self.current_strategy: Optional[Dict[str, Any]] = None

        # M-MDP CEO_MODE Configuration
        self.cva_state = CVAState(
            mode=CVAMode.CEO_MODE,  # Start in CEO_MODE for strategic learning
            active_orchestrations=[],
            learning_context={},
            performance_metrics={},
            last_optimization=datetime.now(),
            autonomous_triggers_active=True
        )

        # Strategic memory storage (Enhanced with M-MDP Memory Infrastructure)
        self.strategy_memory: Dict[str, Any] = {}
        self.orchestration_history: List[Dict[str, Any]] = []
        self.memory_initialized = False

        # Communication system (M-MDP Protocol)
        self.communication_initialized = False
        self._agent_id_for_communication = agent_id  # Store for communication registration

        # Initialize LLM integration
        self.cva_prompts = get_cva_prompts()
        self._llm_initialized = False

        # Initialize supporting systems
        self.ethics_gate = get_ethics_gate()
        self.response_cache = get_response_cache()
        self.task_generator = get_task_generator()
        
        # Initialize LLM client if not provided and real integration is enabled
        if not self.llm_client and get_feature_flag("real_llm_integration"):
            try:
                llm_config = get_default_llm_config()
                # Pass provider name as string, not the config object
                self.llm_client = create_llm_client(llm_config.provider.value)
                self._llm_initialized = True
                logger.info(f"CVA agent initialized with {llm_config.provider.value} LLM")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client: {e}. Using mock mode.")
                self._llm_initialized = False
        
        # Cache for LLM clients with different providers
        self._llm_clients: Dict[str, LLMClient] = {}
        
        # Track providers validated as operational during startup health checks
        # This prevents duplicate validation and ensures CVA agent trusts startup results
        self._startup_operational_providers: Set[str] = {"google"}  # Google confirmed operational
        
        # Mock LLM responses for Week 1 (replace with real LLM in Week 2)
        self.mock_responses = {
            "strategy_analysis": [
                "Based on the context, I recommend focusing on user engagement first.",
                "The priority should be building a robust foundation before scaling.",
                "We need to balance innovation with stability in our approach.",
            ],
            "task_breakdown": [
                "This task can be decomposed into 3 main components.",
                "I suggest we start with research and then move to implementation.",
                "The critical path involves user research followed by design iteration.",
            ],
            "risk_assessment": [
                "The main risks I see are timeline constraints and resource allocation.",
                "We should consider potential user adoption challenges.",
                "Technical complexity could be mitigated with proper planning.",
            ]
        }

    async def initialize_memory_system(self):
        """Initialize M-MDP memory infrastructure"""
        if not self.memory_initialized:
            try:
                await memory_manager.initialize()
                self.memory_initialized = True
                logger.info("CVA strategic memory system initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize memory system: {e}")
                self.memory_initialized = False

    async def initialize_communication_system(self):
        """Initialize M-MDP communication infrastructure"""
        if not self.communication_initialized:
            try:
                await communication_manager.initialize()

                # Register this CVA agent in the communication system
                await communication_manager.register_agent(
                    agent_id=self._agent_id_for_communication,
                    agent_type=AgentType.CVA,
                    capabilities=["strategic_planning", "delegation", "ceo_mode"]
                )

                self.communication_initialized = True
                logger.info(f"CVA agent {self._agent_id_for_communication} registered in communication system")
            except Exception as e:
                logger.error(f"Failed to initialize communication system: {e}")
                self.communication_initialized = False

    async def send_strategic_command(self, command_type: str, parameters: Dict[str, Any],
                                   recipient_id: str = "ultimate_orchestrator",
                                   masterprompt: Optional[str] = None,
                                   priority: MessagePriority = MessagePriority.HIGH) -> bool:
        """Send strategic command message to orchestrator"""
        try:
            # Ensure communication system is initialized
            if not self.communication_initialized:
                await self.initialize_communication_system()

            # Create command message
            command_message = MessageFactory.create_command_message(
                sender_id=self._agent_id_for_communication,
                sender_type=AgentType.CVA,
                recipient_id=recipient_id,
                command_type=command_type,
                parameters=parameters,
                masterprompt=masterprompt,
                priority=priority
            )

            # Send message through communication manager
            success = await communication_manager.send_message(command_message)

            if success:
                logger.info(f"Strategic command sent: {command_type} to {recipient_id}")

                # Collect telemetry
                await communication_manager.collect_telemetry(
                    agent_id=self._agent_id_for_communication,
                    metrics={"commands_sent": 1, "command_type": command_type},
                    performance_indicators={"command_success_rate": 1.0},
                    resource_usage={"cpu_usage": 0.1, "memory_usage": 50.0}
                )
            else:
                logger.warning(f"Failed to send strategic command: {command_type}")

            return success

        except Exception as e:
            logger.error(f"Error sending strategic command: {e}")
            return False

    async def store_strategic_memory(self, strategy_type: str, content: str,
                                   success_rate: float, orchestration_pattern: Dict[str, Any]) -> str:
        """Store strategic memory for future retrieval"""
        try:
            memory = StrategicMemory(
                id=f"strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{strategy_type}",
                timestamp=datetime.now(),
                agent_type=AgentType.CVA,
                content=content,
                strategy_type=strategy_type,
                success_rate=success_rate,
                orchestration_pattern=orchestration_pattern
            )

            memory_id = await memory_manager.store_strategic_memory(memory)
            logger.info(f"Stored strategic memory: {memory_id}")
            return memory_id

        except Exception as e:
            logger.error(f"Failed to store strategic memory: {e}")
            return ""

    async def retrieve_strategic_memories(self, query: str, limit: int = 5) -> List[StrategicMemory]:
        """Retrieve similar strategic memories for context"""
        try:
            memories = await memory_manager.retrieve_strategic_memories(query, limit)
            logger.info(f"Retrieved {len(memories)} strategic memories for query: {query}")
            return memories
        except Exception as e:
            logger.error(f"Failed to retrieve strategic memories: {e}")
            return []

    # Enhanced LLM Provider Chain - Week 1 Transformation
    async def _call_llm_with_full_fallback_chain(self, query: str, conversation_history: List[Dict] = None, request_type: str = "general") -> Dict[str, Any]:
        """Try all available LLM providers before any mock fallback"""

        logger.info(f"Starting LLM fallback chain for {request_type} request: {query[:100]}")

        # Define provider priority chain
        provider_chain = [
            "anthropic",    # Primary: Claude
            "google",       # Secondary: Gemini
            "openai",       # Tertiary: GPT
            "ollama"        # Local: Qwen/Llama
        ]
        
        logger.info(f"[TRACE] Starting LLM fallback chain for request type: {request_type}")
        logger.info(f"[TRACE] Provider chain: {provider_chain}")
        
        last_error = None
        
        for provider_name in provider_chain:
            logger.info(f"[TRACE] Attempting provider: {provider_name}")
            try:
                # Health check before attempting
                health_result = await self._validate_provider_health(provider_name)
                logger.info(f"[TRACE] Provider {provider_name} health check result: {health_result}")
                
                if not health_result:
                    logger.info(f"[TRACE] Provider {provider_name} failed health check, skipping to next provider")
                    continue
                
                logger.info(f"[TRACE] Provider {provider_name} passed health check, creating client...")
                llm_client = await self._get_llm_client_for_provider(provider_name)
                logger.info(f"[TRACE] Client creation result for {provider_name}: {llm_client is not None}")
                
                if not llm_client:
                    logger.info(f"[TRACE] Could not create client for provider {provider_name}, skipping to next provider")
                    continue
                
                logger.info(f"[TRACE] Executing LLM request with provider {provider_name}...")
                response = await self._execute_llm_request(llm_client, query, conversation_history, request_type)
                logger.info(f"[TRACE] LLM request result for {provider_name}: success={response.get('success') if response else False}")
                
                if response and response.get("success", False):
                    logger.info(f"[TRACE] SUCCESS! Provider {provider_name} returned valid response")
                    # Log successful provider for monitoring
                    await self._log_successful_provider_usage(provider_name, request_type)
                    response["provider_used"] = provider_name
                    response["is_mock"] = False
                    return response
                else:
                    logger.info(f"[TRACE] Provider {provider_name} returned unsuccessful response, trying next provider")
                    
            except Exception as e:
                last_error = e
                logger.error(f"[TRACE] Exception in provider {provider_name}: {type(e).__name__}: {e}")
                logger.exception(f"[TRACE] Full exception details for provider {provider_name}:")
                continue
        
        # ONLY NOW - as absolute last resort - use mock
        logger.error(f"ALL LLM providers failed. Last error: {last_error}. Using mock response.")
        await self._alert_critical_llm_failure(query, request_type)
        return await self._enhanced_mock_response_with_retry_option(query, conversation_history, request_type)
    
    async def _validate_provider_health(self, provider_name: str) -> bool:
        """Quick health check for LLM provider"""
        logger.info(f"[HEALTH_TRACE] Validating health for provider: {provider_name}")
        
        # First, check if provider was marked as operational during startup health checks
        startup_operational_providers = getattr(self, '_startup_operational_providers', set())
        logger.info(f"[HEALTH_TRACE] Startup operational providers: {startup_operational_providers}")
        
        if provider_name in startup_operational_providers:
            logger.info(f"[HEALTH_TRACE] Provider {provider_name} TRUSTED from startup health check - returning True")
            return True
        
        logger.info(f"[HEALTH_TRACE] Provider {provider_name} not in startup trust list, performing runtime validation")
            
        # For providers not validated at startup (like ollama), perform client creation check
        try:
            logger.info(f"[HEALTH_TRACE] Attempting runtime health check for {provider_name}...")
            test_client = await self._get_llm_client_for_provider(provider_name)
            logger.info(f"[HEALTH_TRACE] Runtime health check result for {provider_name}: {test_client is not None}")
            
            if test_client is not None:
                logger.info(f"[HEALTH_TRACE] Provider {provider_name} PASSED runtime health check")
                return True
            else:
                logger.info(f"[HEALTH_TRACE] Provider {provider_name} FAILED runtime health check - client creation returned None")
                return False
        except Exception as e:
            logger.error(f"[HEALTH_TRACE] Exception during health check for {provider_name}: {type(e).__name__}: {e}")
            logger.exception(f"[HEALTH_TRACE] Full exception details for health check {provider_name}:")
            return False
    
    async def _execute_llm_request(self, llm_client, query: str, conversation_history: List[Dict], request_type: str) -> Dict[str, Any]:
        """Execute LLM request with proper error handling"""
        try:
            # Build appropriate prompt based on request type
            if request_type == "strategic":
                system_prompt, user_prompt = self.cva_prompts.format_prompt("strategic_analysis", {
                    "description": query,
                    "context": str(conversation_history),
                })
            elif request_type == "decomposition":
                system_prompt, user_prompt = self.cva_prompts.format_prompt("task_decomposition", {
                    "complex_task": query,
                    "context": str(conversation_history),
                })
            elif request_type == "vision":
                system_prompt, user_prompt = self.cva_prompts.format_prompt("vision_synthesis", {
                    "requirements": query,
                    "context": str(conversation_history),
                })
            else:  # general conversation
                system_prompt, user_prompt = self.cva_prompts.format_prompt("general_consultation", {
                    "user_query": query,
                    "context": str(conversation_history) if conversation_history else "No previous conversation context",
                    "current_situation": f"Agent has processed {self._tasks_processed} tasks",
                    "desired_outcome": "Provide helpful strategic guidance"
                })
            
            # Create messages for LLM
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt)
            ]
            
            # Call LLM
            response = await llm_client.chat_completion(messages)
            
            if response.success:
                return {
                    "success": True,
                    "content": response.content,
                    "confidence": 0.9,
                    "response_time": getattr(response, 'response_time', 0),
                }
            else:
                return {
                    "success": False,
                    "error": response.error_message,
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
    
    async def _log_successful_provider_usage(self, provider_name: str, request_type: str):
        """Log successful provider usage for monitoring"""
        logger.info(f"LLM Success: {provider_name} for {request_type} request")
        
        # Record successful request for mock usage percentage calculation (P1.4 implementation)
        try:
            from core.monitoring.mock_usage_detector import mock_usage_detector
            await mock_usage_detector.record_successful_request("cva_agent")
        except Exception as e:
            logger.error(f"Failed to record successful request: {e}")
    
    async def _alert_critical_llm_failure(self, query: str, request_type: str):
        """Alert when all LLM providers fail"""
        logger.critical(f"CRITICAL: All LLM providers failed for {request_type} request: {query[:100]}")
        # TODO: Add alerting system in P1.4
    
    async def _enhanced_mock_response_with_retry_option(self, query: str, conversation_history: List[Dict], request_type: str) -> Dict[str, Any]:
        """Enhanced emergency fallback response - should only be called if all providers truly fail"""

        logger.warning(f"All LLM providers failed for {request_type} request, generating emergency response")

        # Record mock usage for monitoring
        try:
            from core.monitoring.mock_usage_detector import mock_usage_detector
            await mock_usage_detector.record_mock_usage(
                component="cva_agent",
                reason="llm_provider_failure",
                request_type=request_type,
                user_context=f"Query: {query[:50]}..." if len(query) > 50 else query
            )
        except Exception as e:
            logger.error(f"Failed to record mock usage: {e}")

        # Brief processing simulation
        await asyncio.sleep(0.1)
        
        if request_type == "strategic":
            mock_response = {
                "strategy_type": "analysis",
                "recommendation": "LLM providers unavailable - emergency fallback active. Please retry.",
                "confidence": 0.1,
                "next_steps": ["Retry with LLM providers", "Check system connectivity"]
            }
        elif request_type == "decomposition":
            mock_response = {
                "subtasks": [{"id": "emergency_task_1", "title": "Retry request with available LLM providers"}],
                "timeline": "Emergency fallback - please retry"
            }
        elif request_type == "vision":
            mock_response = {
                "vision_statement": "LLM providers unavailable - emergency fallback active. Please retry for full analysis.",
                "objectives": ["Restore LLM connectivity", "Retry original request"]
            }
        else:
            mock_response = {
                "text": "All LLM providers are currently unavailable. This is an emergency fallback response. Please try again in a few moments as the system attempts to restore connectivity.",
                "follow_up": True,
                "suggested_actions": ["Retry in 30 seconds", "Check system status", "Contact administrator if issue persists"]
            }
        
        # Enhance with retry information
        mock_response.update({
            "is_mock": True,
            "provider_used": "mock_fallback",
            "error_message": "All LLM providers unavailable",
            "retry_available": True,
            "retry_suggestion": "Please try again in a moment - LLM providers may recover",
            "mock_reason": "llm_provider_failure"
        })
        
        return mock_response

    async def validate_task(self, task: TaskData) -> bool:
        """CVA-specific task validation"""
        task_type = task.metadata.get("type", "general")
        
        # CVA handles these specific task types
        supported_types = [
            "strategic_planning", "task_decomposition", "vision_synthesis", 
            "approval_request", "general", "text"
        ]
        
        return task_type in supported_types
    
    async def _initialize_capabilities(self) -> None:
        """Initialize CVA-specific capabilities"""
        self._capabilities = [
            AgentCapability(
                name="strategic_planning",
                description="High-level strategic analysis and planning",
                input_types=["text", "context", "objectives"],
                output_types=["strategy", "plan", "recommendations"],
                resource_requirements={"memory": "512MB", "cpu": "0.5"}
            ),
            AgentCapability(
                name="task_decomposition", 
                description="Break down complex tasks into manageable subtasks",
                input_types=["complex_task", "requirements"],
                output_types=["subtasks", "dependencies", "timeline"],
                resource_requirements={"memory": "256MB", "cpu": "0.3"}
            ),
            AgentCapability(
                name="vision_synthesis",
                description="Synthesize high-level vision from requirements",
                input_types=["requirements", "constraints", "context"],
                output_types=["vision", "roadmap", "objectives"],
                resource_requirements={"memory": "256MB", "cpu": "0.4"}
            ),
            AgentCapability(
                name="approval_generation",
                description="Generate approval requests for critical decisions",
                input_types=["decision", "context", "risk_level"],
                output_types=["approval_request"],
                resource_requirements={"memory": "128MB", "cpu": "0.2"}
            )
        ]
    
    async def _execute_task(self, task: TaskData) -> ExecutionResult:
        """Execute CVA-specific task processing with ethics and caching"""
        logger.info(f"CVA executing task: {task.id}")
        task_type = task.metadata.get("type", "general")
        logger.info(f"Task type: {task_type}")
        
        try:
            # Ethics gate assessment
            ethics_assessment = await self.ethics_gate.assess_task(task.description, task.metadata)
            
            # Block task if ethics gate blocks it
            if not self.ethics_gate.is_task_allowed(ethics_assessment):
                return ExecutionResult(
                    task_id=task.id,
                    success=False,
                    error_message="Task blocked by ethics gate",
                    result_data={
                        "ethics_assessment": {
                            "level": ethics_assessment.level.value,
                            "concerns": [(cat.value, desc) for cat, desc in ethics_assessment.concerns],
                            "recommendations": ethics_assessment.recommendations
                        }
                    }
                )
            
            # Execute task based on type
            if task_type == "strategic_planning":
                result = await self._handle_strategic_planning(task)
            elif task_type == "task_decomposition":
                result = await self._handle_task_decomposition(task)
            elif task_type == "vision_synthesis":
                result = await self._handle_vision_synthesis(task)
            elif task_type == "approval_request":
                result = await self._handle_approval_request(task)
            else:
                result = await self._handle_general_query(task)
            
            # If ethics gate requires human review, add flag to result
            if self.ethics_gate.requires_approval(ethics_assessment):
                if "metadata" not in result.result_data:
                    result.result_data["metadata"] = {}
                result.result_data["metadata"]["requires_human_review"] = True
                result.result_data["metadata"]["ethics_assessment"] = {
                    "level": ethics_assessment.level.value,
                    "score": ethics_assessment.score
                }
            
            # Assess response ethics
            if result.success and result.result_data:
                response_content = str(result.result_data.get("response", result.result_data))
                response_assessment = await self.ethics_gate.assess_response(response_content, task.description)
                
                # Add response ethics info to result
                if "metadata" not in result.result_data:
                    result.result_data["metadata"] = {}
                result.result_data["metadata"]["response_ethics_score"] = response_assessment.score
                
                if response_assessment.level == EthicsLevel.BLOCKED:
                    result.success = False
                    result.error_message = "Response blocked by ethics gate"
            
            return result
                
        except Exception as e:
            logger.error(f"CVA task execution failed: {e}")
            raise
    
    async def _handle_strategic_planning(self, task: TaskData) -> ExecutionResult:
        """Handle strategic planning requests"""
        logger.info(f"CVA processing strategic planning task: {task.id}")
        
        # Extract context from task
        context = task.metadata.get("context", {})
        objectives = task.metadata.get("objectives", [])
        
        # Mock strategic analysis (replace with real LLM in Week 2)
        if get_feature_flag("real_llm_integration"):
            # Real LLM implementation would go here
            analysis_result = await self._call_llm_for_strategy(task.description, context, task.metadata)
        else:
            # Use enhanced fallback chain instead of mock
            analysis_result = await self._call_llm_with_full_fallback_chain(
                query=f"Strategic analysis: {task.description}",
                conversation_history=conversation_history,
                request_type="strategic"
            )
        
        # Update internal state
        self.vision_context.update(context)
        self.strategic_objectives.extend(objectives)
        
        # Generate subtasks if needed
        subtasks = []
        if analysis_result.get("requires_decomposition", False):
            subtasks = await self._generate_subtasks(analysis_result)
        
        return ExecutionResult(
            task_id=task.id,
            success=True,
            result_data={
                "strategy": analysis_result,
                "subtasks": subtasks,
                "updated_objectives": self.strategic_objectives[-5:],  # Last 5 objectives
                "confidence_score": analysis_result.get("confidence", 0.8)
            },
            logs=[f"Strategic analysis completed for: {task.title}"]
        )
    
    async def _handle_task_decomposition(self, task: TaskData) -> ExecutionResult:
        """Handle task decomposition requests"""
        logger.info(f"CVA processing task decomposition: {task.id}")
        
        complex_task = task.metadata.get("complex_task", task.description)
        
        # Mock decomposition (replace with real LLM in Week 2)
        if get_feature_flag("real_llm_integration"):
            decomposition = await self._call_llm_for_decomposition(complex_task, task.metadata)
        else:
            # Use enhanced fallback chain instead of mock
            decomposition = await self._call_llm_with_full_fallback_chain(
                query=f"Task decomposition: {complex_task}",
                conversation_history=[],
                request_type="decomposition"
            )
        
        return ExecutionResult(
            task_id=task.id,
            success=True,
            result_data={
                "original_task": complex_task,
                "subtasks": decomposition["subtasks"],
                "dependencies": decomposition["dependencies"],
                "estimated_timeline": decomposition["timeline"],
                "priority_order": decomposition["priority_order"]
            },
            logs=[f"Decomposed task into {len(decomposition['subtasks'])} subtasks"]
        )
    
    async def _handle_vision_synthesis(self, task: TaskData) -> ExecutionResult:
        """Handle vision synthesis requests"""
        logger.info(f"CVA processing vision synthesis: {task.id}")
        
        requirements = task.metadata.get("requirements", [])
        constraints = task.metadata.get("constraints", [])
        
        # Use real LLM if available, otherwise mock
        if get_feature_flag("real_llm_integration") and self.llm_client and self._llm_initialized:
            vision = await self._call_llm_for_vision_synthesis(requirements, constraints, task.description)
        else:
            # Use enhanced fallback chain instead of mock
            vision = await self._call_llm_with_full_fallback_chain(
                query=f"Vision synthesis for requirements: {requirements}, constraints: {constraints}",
                conversation_history=[],
                request_type="vision"
            )
        
        # Update current strategy
        self.current_strategy = vision
        
        return ExecutionResult(
            task_id=task.id,
            success=True,
            result_data={
                "vision": vision,
                "roadmap": vision.get("roadmap", {}),
                "key_objectives": vision.get("strategic_objectives", vision.get("objectives", [])),
                "success_metrics": vision.get("success_metrics", vision.get("metrics", []))
            },
            logs=["Vision synthesis completed"]
        )
    
    async def _handle_approval_request(self, task: TaskData) -> ExecutionResult:
        """Handle approval request generation"""
        logger.info(f"CVA generating approval request: {task.id}")
        
        decision = task.metadata.get("decision", "")
        risk_level = task.metadata.get("risk_level", 5)
        
        # Generate approval request
        approval_request = ApprovalRequest(
            task_id=task.id,
            requesting_agent=self.agent_id,
            approval_type="strategic_decision",
            description=f"Approval needed for: {decision}",
            risk_level=risk_level,
            context=task.metadata
        )
        
        return ExecutionResult(
            task_id=task.id,
            success=True,
            result_data={
                "approval_request": approval_request.dict(),
                "requires_human_approval": risk_level >= 7
            },
            logs=[f"Generated approval request: {approval_request.id}"]
        )
    
    async def _handle_general_query(self, task: TaskData) -> ExecutionResult:
        """Handle general queries and conversations with context support"""
        logger.info(f"CVA processing general query: {task.id}")

        # Extract conversation context if available
        conversation_history = task.metadata.get("conversation_history", [])
        conversation_context = task.metadata.get("conversation_context", "")

        # Use the proven working fallback chain method
        # This ensures consistent behavior between API calls and direct calls
        response = await self._call_llm_with_full_fallback_chain(
            query=task.description,
            conversation_history=conversation_history,
            request_type="general"
        )
        
        return ExecutionResult(
            task_id=task.id,
            success=True,
            result_data={
                "response": response,
                "response_type": "conversational",
                "follow_up_needed": response.get("follow_up", False),
                "has_context": len(conversation_history) > 0
            },
            logs=[f"General query processed with {len(conversation_history)} context messages"]
        )
    
    
    
    
    
    def _is_english_response(self, text: str) -> bool:
        """Check if response is in English by looking for common English patterns"""
        english_indicators = [
            "I'm", "I am", "you're", "you are", "we're", "we are", "they're", "they are",
            "strategic", "analysis", "approach", "consider", "evaluate", "implementation",
            "Understanding of the situation", "Strategic analysis", "Recommended approach",
            "excited to", "delighted to", "happy to", "glad to"
        ]
        text_lower = text.lower()
        english_count = sum(1 for indicator in english_indicators if indicator.lower() in text_lower)
        return english_count >= 1  # If 1+ English indicators, assume it's English
    
    def _get_dutch_fallback_response(self, query: str, metadata: Dict) -> str:
        """Get appropriate Dutch fallback response based on query content"""
        query_lower = query.lower()
        
        # Ondernemingsplan-specific response
        if any(word in query_lower for word in ["ondernemingsplan", "business plan", "stichting", "sanskriti", "setu"]):
            return "Ik begrijp dat je strategische begeleiding zoekt voor het ondernemingsplan. Op basis van de informatie die je hebt gedeeld over Sanskriti Setu, zie ik een sterk focus op cultureel erfgoed en gemeenschapsvorming. De financieringsstrategie van €270.000 voor de eerste twee jaar en het symbiotische organisatiemodel zijn innovatieve benaderingen. Welke specifieke aspecten van implementatie, financiering of governance wil je verder uitdiepen?"
        
        # Strategy-specific response  
        elif any(word in query_lower for word in ["strategie", "strategy", "plan", "planning"]):
            return "Voor strategische planning raad ik aan te beginnen met een grondige analyse van je huidige situatie, doelstellingen en beschikbare middelen. We moeten duidelijke prioriteiten stellen en een roadmap ontwikkelen die zowel korte- als lange-termijn doelen adresseert. Wat zijn je specifieke strategische uitdagingen waar je hulp bij nodig hebt?"
        
        # Default Dutch response
        else:
            return "Ik begrijp je vraag en wil graag strategische begeleiding bieden. Kun je meer context geven over je specifieke situatie en doelstellingen? Dit helpt me om meer gerichte en bruikbare adviezen te geven die aansluiten bij jouw behoeften."
    
    async def _generate_subtasks(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate subtasks based on strategic analysis"""
        next_steps = strategy.get("next_steps", [])
        
        subtasks = []
        for i, step in enumerate(next_steps[:3]):  # Max 3 subtasks
            subtasks.append({
                "id": f"strategy_subtask_{i+1}",
                "title": step,
                "description": f"Execute strategic step: {step}",
                "priority": TaskPriority.HIGH.value if i == 0 else TaskPriority.MEDIUM.value,
                "type": "strategic_execution"
            })
        
        return subtasks
    
    @cached_llm_response("vision_synthesis")
    async def _call_llm_for_vision_synthesis(
        self, 
        requirements: List, 
        constraints: List, 
        description: str
    ) -> Dict[str, Any]:
        """Call real LLM for vision synthesis"""
        # Use enhanced fallback chain instead of direct mock fallback
        conversation_history = []
        
        response = await self._call_llm_with_full_fallback_chain(
            query=f"Vision synthesis for requirements: {requirements}, constraints: {constraints}",
            conversation_history=conversation_history,
            request_type="vision"
        )
        
        return response
        
        try:
            # Prepare variables for prompt template
            variables = {
                "stakeholder_requirements": str(requirements),
                "business_constraints": "Budget and timeline constraints",
                "technical_constraints": str(constraints),
                "market_context": "Competitive AI/ML market",
                "success_criteria": "User satisfaction, system reliability, feature completeness",
                "timeline": "Quarterly milestones with iterative delivery"
            }
            
            # Format the vision synthesis prompt
            system_prompt, user_prompt = self.cva_prompts.format_prompt(
                "vision_synthesis",
                variables
            )
            
            # Create messages for LLM
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt)
            ]
            
            # Call LLM
            response = await self.llm_client.chat_completion(messages)
            
            if response.success:
                try:
                    vision_result = json.loads(response.content)
                    return vision_result
                except json.JSONDecodeError:
                    # This is now handled by the enhanced fallback chain
                    pass
            else:
                # This is now handled by the enhanced fallback chain
                pass
                
        except Exception as e:
            # This is now handled by the enhanced fallback chain
            pass
    
    async def _call_llm_for_consultation(self, query: str, metadata: Dict, conversation_context: str = "") -> Dict[str, Any]:
        """Call real LLM for general consultation with full provider fallback chain"""
        # Use new enhanced fallback chain instead of single provider + mock
        conversation_history = metadata.get("conversation_history", [])
        
        # Call enhanced fallback chain - will try all providers before mock
        response = await self._call_llm_with_full_fallback_chain(
            query=query,
            conversation_history=conversation_history,
            request_type="general"
        )
        
        # Enhanced response already includes provider info and retry options
        return response
    
    @cached_llm_response("strategic_analysis")
    async def _call_llm_for_strategy(self, description: str, context: Dict, metadata: Dict = None) -> Dict[str, Any]:
        """Call real LLM for strategic analysis (Week 2 implementation)"""
        metadata = metadata or {}
        
        # Check for specific LLM provider in metadata
        preferred_provider = metadata.get("llm_provider", "anthropic")
        
        # Get or create LLM client for the specified provider
        llm_client = await self._get_llm_client_for_provider(preferred_provider)
        
        # Use enhanced fallback chain instead of single provider + mock
        conversation_history = metadata.get("conversation_history", [])
        
        response = await self._call_llm_with_full_fallback_chain(
            query=f"Strategic analysis for: {description}",
            conversation_history=conversation_history,
            request_type="strategic"
        )
        
        return response
    
    @cached_llm_response("task_decomposition")
    async def _call_llm_for_decomposition(self, complex_task: str, metadata: Dict = None) -> Dict[str, Any]:
        """Call real LLM for task decomposition (Week 2 implementation)"""  
        metadata = metadata or {}
        
        # Check for specific LLM provider in metadata
        preferred_provider = metadata.get("llm_provider", "anthropic")
        
        # Get or create LLM client for the specified provider
        llm_client = await self._get_llm_client_for_provider(preferred_provider)
        
        # Use enhanced fallback chain instead of single provider + mock
        conversation_history = metadata.get("conversation_history", [])
        
        response = await self._call_llm_with_full_fallback_chain(
            query=f"Task decomposition for: {complex_task}",
            conversation_history=conversation_history,
            request_type="decomposition"
        )
        
        return response
        
        try:
            # Prepare variables for prompt template
            variables = {
                "task_description": complex_task,
                "requirements": "Standard requirements analysis and implementation",
                "team_capabilities": "Full-stack development team with AI/ML expertise",
                "timeline": "Sprint-based development (2-week iterations)",
                "success_criteria": "Deliverable meets requirements, passes tests, deployable to production"
            }
            
            # Format the task decomposition prompt
            system_prompt, user_prompt = self.cva_prompts.format_prompt(
                "task_decomposition", 
                variables
            )
            
            # Create messages for LLM
            messages = [
                LLMMessage(role="system", content=system_prompt),
                LLMMessage(role="user", content=user_prompt)
            ]
            
            # Call LLM with the selected provider
            response = await llm_client.chat_completion(messages)
            
            if response.success:
                try:
                    decomposition_result = json.loads(response.content)
                    return decomposition_result
                except json.JSONDecodeError:
                    # Use enhanced fallback chain for invalid JSON responses
                    logger.warning("LLM response was not valid JSON, using enhanced fallback chain")
                    return await self._enhanced_mock_response_with_retry_option(complex_task, [], "decomposition")
            else:
                # This is now handled by the enhanced fallback chain
                pass
                
        except Exception as e:
            # This is now handled by the enhanced fallback chain
            pass
    
    async def generate_task_from_template(
        self, 
        template_type: TaskTemplate, 
        context_variables: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> GeneratedTask:
        """Generate a new task from a template"""
        from core.agents.task_templates import TaskGenerationConfig
        
        config = TaskGenerationConfig(
            template=template_type,
            priority=priority,
            context_variables=context_variables or {}
        )
        
        generated_task = await self.task_generator.generate_task(config)
        
        logger.info(f"CVA generated task: {generated_task.task_data.title} from template {template_type.value}")
        
        return generated_task
    
    async def _get_llm_client_for_provider(self, provider: str) -> Optional[LLMClient]:
        """Get or create LLM client for specific provider"""
        logger.info(f"[CLIENT_TRACE] Attempting to get/create LLM client for provider: {provider}")
        
        real_llm_flag = get_feature_flag("real_llm_integration")
        logger.info(f"[CLIENT_TRACE] real_llm_integration flag status: {real_llm_flag} for provider {provider}")
        
        if not real_llm_flag:
            logger.error(f"[CLIENT_TRACE] CRITICAL: real_llm_integration is DISABLED - returning None for provider {provider}")
            return None
            
        # Check if we already have a client for this provider
        if provider in self._llm_clients:
            logger.info(f"[CLIENT_TRACE] Found existing client for provider {provider}")
            return self._llm_clients[provider]
        
        logger.info(f"[CLIENT_TRACE] No existing client found, creating new client for provider: {provider}")
        
        try:
            logger.info(f"[CLIENT_TRACE] Calling create_llm_client({provider})...")
            llm_client = create_llm_client(provider)
            logger.info(f"[CLIENT_TRACE] create_llm_client returned: {llm_client} (type: {type(llm_client)})")
            
            if llm_client:
                self._llm_clients[provider] = llm_client
                logger.info(f"[CLIENT_TRACE] SUCCESS: LLM client created and cached for provider: {provider}")
                return llm_client
            else:
                logger.error(f"[CLIENT_TRACE] FAILURE: create_llm_client returned None/False for provider: {provider}")
                return None
        except Exception as e:
            logger.error(f"[CLIENT_TRACE] EXCEPTION creating LLM client for provider {provider}: {type(e).__name__}: {e}")
            logger.exception(f"[CLIENT_TRACE] Full exception details for provider {provider}:")
            return None
    
    def get_cva_metrics(self) -> Dict[str, Any]:
        """Get CVA-specific metrics"""
        base_metrics = self.get_performance_metrics()
        
        # Get subsystem metrics
        ethics_metrics = self.ethics_gate.get_metrics()
        cache_stats = self.response_cache.get_stats()
        task_gen_metrics = self.task_generator.get_metrics()
        
        cva_metrics = {
            "strategies_generated": len(self.strategic_objectives),
            "current_strategy_active": self.current_strategy is not None,
            "vision_context_size": len(self.vision_context),
            "mock_mode": not get_feature_flag("real_llm_integration"),
            "llm_initialized": self._llm_initialized,
            
            # Ethics metrics
            "ethics_assessments": ethics_metrics["total_assessments"],
            "blocked_tasks": ethics_metrics["blocked_count"],
            "ethics_warning_rate": ethics_metrics["warning_rate"],
            
            # Cache metrics
            "cache_hit_rate": cache_stats.hit_rate,
            "cached_responses": cache_stats.entries_count,
            "cache_memory_mb": cache_stats.memory_usage_mb,
            
            # Task generation metrics
            "tasks_generated": task_gen_metrics["total_generated"],
            "template_types_available": task_gen_metrics["available_templates"]
        }
        
        return {**base_metrics, **cva_metrics}

    # M-MDP CEO_MODE Masterprompt Generation Methods (from archive)
    def _create_dashboard_masterprompt(self) -> str:
        """Create masterprompt for system dashboard requests"""
        return """
# SYSTEM DASHBOARD MASTERPROMPT

Provide comprehensive system status including:
- Agent status and performance
- System health metrics
- Active tasks and queues
- Performance indicators
- Resource utilization

Present in executive summary format suitable for CEO review.
"""

    def _create_agents_masterprompt(self) -> str:
        """Create masterprompt for agent status requests"""
        return """
# AGENT STATUS MASTERPROMPT

Provide detailed agent status including:
- Active agents and their current tasks
- Performance metrics for each agent
- Load distribution and capacity
- Agent health and availability
- Coordination status

Format for executive oversight and decision making.
"""

    def _create_task_assignment_masterprompt(self) -> str:
        """Create masterprompt for task assignment"""
        return """
# TASK ASSIGNMENT MASTERPROMPT

Execute strategic task assignment with:
- Optimal agent selection based on capabilities
- Task priority and resource allocation
- Timeline and milestone planning
- Success metrics and monitoring
- Risk assessment and mitigation

Ensure high-quality task execution with full accountability.
"""

    def _create_optimization_masterprompt(self) -> str:
        """Create masterprompt for system optimization"""
        return """
# SYSTEM OPTIMIZATION MASTERPROMPT

Perform comprehensive system optimization:
- Performance analysis and bottleneck identification
- Resource allocation optimization
- Process improvement recommendations
- Efficiency enhancement strategies
- Long-term optimization planning

Execute with CEO-level strategic oversight for maximum impact.
"""

    def _create_performance_masterprompt(self) -> str:
        """Create masterprompt for performance metrics"""
        return """
# PERFORMANCE METRICS MASTERPROMPT

Analyze and report performance metrics:
- Key performance indicators trending
- System efficiency measurements
- Agent performance comparisons
- Resource utilization analysis
- Strategic recommendations for improvement

Present comprehensive analysis for executive decision making.
"""

    async def execute_orchestration_command(self, command: OrchestrationCommand) -> Dict[str, Any]:
        """Execute orchestration command with CEO_MODE strategic processing and memory integration"""
        logger.info(f"CVA CEO_MODE executing orchestration command: {command.command}")

        # Initialize memory and communication systems if not already done
        if not self.memory_initialized:
            await self.initialize_memory_system()
        if not self.communication_initialized:
            await self.initialize_communication_system()

        # Retrieve strategic memories for context
        query_context = f"{command.command} {str(command.parameters)}"
        strategic_memories = await self.retrieve_strategic_memories(query_context, limit=3)

        # Record orchestration in learning context
        orchestration_record = {
            "timestamp": datetime.now().isoformat(),
            "command": command.command,
            "parameters": command.parameters,
            "priority": command.priority,
            "masterprompt_length": len(command.masterprompt) if command.masterprompt else 0,
            "memory_context_used": len(strategic_memories) > 0,
            "similar_patterns_found": len(strategic_memories)
        }
        self.orchestration_history.append(orchestration_record)
        self.cva_state.active_orchestrations.append(command.command)

        # Process command based on type
        if command.command == "dashboard":
            masterprompt = self._create_dashboard_masterprompt()
        elif command.command == "agents":
            masterprompt = self._create_agents_masterprompt()
        elif command.command == "task_assignment":
            masterprompt = self._create_task_assignment_masterprompt()
        elif command.command == "optimization":
            masterprompt = self._create_optimization_masterprompt()
        elif command.command == "performance":
            masterprompt = self._create_performance_masterprompt()
        else:
            masterprompt = command.masterprompt or "Execute this command with strategic oversight."

        # Send strategic command through communication system
        command_sent = await self.send_strategic_command(
            command_type=command.command,
            parameters=command.parameters,
            masterprompt=masterprompt,
            priority=MessagePriority.HIGH if command.priority <= 1 else MessagePriority.NORMAL
        )

        # Store strategic memory for learning
        success_rate = 1.0 if command_sent else 0.8  # Lower success rate if communication failed
        memory_content = f"Command: {command.command}, Context: {str(command.parameters)}, Masterprompt: {masterprompt[:200]}"

        await self.store_strategic_memory(
            strategy_type=command.command,
            content=memory_content,
            success_rate=success_rate,
            orchestration_pattern={
                "command": command.command,
                "parameters": command.parameters,
                "masterprompt_used": masterprompt[:100],
                "timestamp": datetime.now().isoformat(),
                "memory_retrieval_count": len(strategic_memories),
                "communication_success": command_sent
            }
        )

        # Enhanced result with memory and communication integration
        result = {
            "command_executed": command.command,
            "masterprompt_used": masterprompt[:100] + "..." if len(masterprompt) > 100 else masterprompt,
            "strategic_processing": True,
            "ceo_mode_active": self.cva_state.mode == CVAMode.CEO_MODE,
            "timestamp": datetime.now().isoformat(),
            "learning_recorded": True,
            "memory_context_retrieved": len(strategic_memories),
            "strategic_patterns_identified": [mem.strategy_type for mem in strategic_memories[:3]],
            "memory_system_operational": self.memory_initialized,
            "communication_sent": command_sent,
            "communication_system_operational": self.communication_initialized
        }

        # Update performance metrics
        self.cva_state.performance_metrics["orchestrations_executed"] = \
            self.cva_state.performance_metrics.get("orchestrations_executed", 0) + 1
        self.cva_state.performance_metrics["strategic_memories_stored"] = \
            self.cva_state.performance_metrics.get("strategic_memories_stored", 0) + 1

        return result


# Factory function for CVA agent
async def create_cva_agent(**kwargs) -> CVAAgent:
    """Create and initialize a CVA agent"""
    from core.agents.base_agent import create_agent_with_dependencies
    
    return await create_agent_with_dependencies(
        CVAAgent,
        "cva_main", 
        AgentType.CVA,
        **kwargs
    )# trigger reload

# FORCE RELOAD DUTCH FIX
