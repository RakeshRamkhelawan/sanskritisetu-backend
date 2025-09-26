"""
LLM Client Interface and Implementations
Provides abstraction for Google Gemini LLM provider
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    GOOGLE = "google"
    MOCK = "mock"


class LLMModel(Enum):
    """Supported LLM models"""
    # Google Models
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_FLASH = "gemini-1.5-flash"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_PRO = "gemini-pro"

    # Mock Model
    MOCK_MODEL = "mock-model"


@dataclass
class LLMMessage:
    """Standard message format for LLM interactions"""
    role: str  # system, user, assistant
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Standard response format from LLM"""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    response_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    provider: LLMProvider
    model: LLMModel
    api_key: str
    max_tokens: int = 2000
    temperature: float = 0.7
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0


class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider
        self.model = config.model
        self._request_count = 0
        self._total_response_time = 0.0
        self._error_count = 0
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Get chat completion from LLM"""
        pass
    
    @abstractmethod
    async def text_completion(
        self, 
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Get text completion from LLM"""
        pass
    
    async def validate_connection(self) -> bool:
        """Validate connection to LLM provider"""
        try:
            test_response = await self.text_completion(
                "Test connection. Respond with 'OK'.",
                max_tokens=10
            )
            return test_response.success
        except Exception as e:
            logger.error(f"LLM connection validation failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics"""
        avg_response_time = (
            self._total_response_time / max(self._request_count, 1)
        )
        
        return {
            "provider": self.provider.value,
            "model": self.model.value,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "average_response_time": avg_response_time,
            "error_rate": self._error_count / max(self._request_count, 1)
        }
    
    def _update_metrics(self, response_time: float, success: bool):
        """Update internal metrics"""
        self._request_count += 1
        self._total_response_time += response_time
        if not success:
            self._error_count += 1


class MockLLMClient(LLMClient):
    """Mock LLM client for testing and development"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        if config is None:
            config = LLMConfig(
                provider=LLMProvider.MOCK,
                model=LLMModel.MOCK_MODEL,
                api_key="mock-key",
                max_tokens=2000
            )
        super().__init__(config)
        
        # Mock responses for different types of requests
        self.mock_responses = {
            "strategic_analysis": [
                "Based on strategic analysis, I recommend focusing on user-centric design and scalable architecture.",
                "The key strategic priorities should be: 1) Market validation, 2) Technical foundation, 3) User experience optimization.",
                "Strategic assessment indicates high potential for growth through systematic execution of core features."
            ],
            "task_decomposition": [
                "This complex task can be broken down into the following components: Research phase, Design phase, Implementation phase, Testing phase.",
                "Task decomposition suggests: 1) Requirements analysis (2 hours), 2) Solution design (4 hours), 3) Implementation (8 hours), 4) Validation (2 hours).",
                "Breaking down this task: Core logic implementation, User interface design, Data integration, Performance optimization."
            ],
            "vision_synthesis": [
                "Vision synthesis: Create an innovative, user-focused solution that balances technical excellence with practical usability.",
                "Strategic vision: Build a scalable, maintainable platform that delivers measurable value to users while maintaining high quality standards.",
                "Synthesized vision: Develop a robust system that prioritizes user experience, technical reliability, and sustainable growth."
            ],
            "general": [
                "I understand your request. Let me provide a thoughtful analysis of the situation and recommendations for moving forward.",
                "Based on the context provided, I can offer strategic insights and practical next steps to address your needs.",
                "Thank you for your question. I'll analyze this from multiple perspectives and provide actionable recommendations."
            ]
        }
    
    async def chat_completion(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Mock chat completion"""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(0.5)
        
        # Analyze messages to determine response type
        last_message = messages[-1] if messages else None
        response_type = self._determine_response_type(last_message.content if last_message else "")
        
        # Get mock response
        responses = self.mock_responses.get(response_type, self.mock_responses["general"])
        response_content = responses[hash(str(messages)) % len(responses)]
        
        response_time = time.time() - start_time
        self._update_metrics(response_time, True)
        
        return LLMResponse(
            content=response_content,
            model=self.model.value,
            provider=self.provider.value,
            usage={"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
            response_time=response_time,
            success=True
        )
    
    async def text_completion(
        self, 
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Mock text completion"""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(0.3)
        
        # Determine response type from prompt
        response_type = self._determine_response_type(prompt)
        
        # Get mock response
        responses = self.mock_responses.get(response_type, self.mock_responses["general"])
        response_content = responses[hash(prompt) % len(responses)]
        
        response_time = time.time() - start_time
        self._update_metrics(response_time, True)
        
        return LLMResponse(
            content=response_content,
            model=self.model.value,
            provider=self.provider.value,
            usage={"prompt_tokens": len(prompt.split()), "completion_tokens": len(response_content.split()), "total_tokens": len(prompt.split()) + len(response_content.split())},
            response_time=response_time,
            success=True
        )
    
    def _determine_response_type(self, text: str) -> str:
        """Determine the type of response needed based on input text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["strategy", "strategic", "planning", "plan"]):
            return "strategic_analysis"
        elif any(word in text_lower for word in ["decompose", "breakdown", "subtask", "break down"]):
            return "task_decomposition"
        elif any(word in text_lower for word in ["vision", "synthesize", "synthesis", "roadmap"]):
            return "vision_synthesis"
        else:
            return "general"





class GoogleClient(LLMClient):
    """Google Gemini LLM client implementation"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        try:
            import google.generativeai as genai
            genai.configure(api_key=config.api_key)
            self.client = genai.GenerativeModel(config.model.value)
            self._available = True
        except ImportError:
            logger.warning("Google generativeai package not available. Install with: pip install google-generativeai")
            self._available = False
        except Exception as e:
            logger.error(f"Failed to initialize Google client: {e}")
            self._available = False
    
    async def chat_completion(
        self, 
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Google Gemini chat completion"""
        if not self._available:
            return LLMResponse(
                content="Google client not available",
                model=self.model.value,
                provider=self.provider.value,
                success=False,
                error_message="Google client not initialized"
            )
        
        start_time = time.time()
        
        try:
            # Convert messages to Google format
            conversation_text = ""
            for msg in messages:
                if msg.role == "system":
                    conversation_text += f"System: {msg.content}\n\n"
                elif msg.role == "user":
                    conversation_text += f"User: {msg.content}\n\n"
                elif msg.role == "assistant":
                    conversation_text += f"Assistant: {msg.content}\n\n"
            
            # Remove the last "User:" and use it as the prompt
            conversation_text = conversation_text.strip()
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.generate_content(
                    conversation_text,
                    generation_config={
                        "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                        "temperature": kwargs.get("temperature", self.config.temperature),
                    }
                )
            )
            
            response_time = time.time() - start_time
            self._update_metrics(response_time, True)
            
            # Handle blocked content from Google safety filters
            try:
                content = response.text
            except Exception as text_error:
                # Check if content was blocked by safety filters
                if hasattr(response, 'candidates') and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        if candidate.finish_reason == 2:  # SAFETY filter block
                            return LLMResponse(
                                content="Content blocked by safety filters",
                                model=self.model.value,
                                provider=self.provider.value,
                                response_time=response_time,
                                success=False,
                                error_message=f"Content blocked by Google safety filters (finish_reason={candidate.finish_reason})"
                            )
                # Other text access errors
                return LLMResponse(
                    content="Response text not accessible",
                    model=self.model.value,
                    provider=self.provider.value,
                    response_time=response_time,
                    success=False,
                    error_message=f"Could not access response text: {str(text_error)}"
                )
            
            return LLMResponse(
                content=content,
                model=self.model.value,
                provider=self.provider.value,
                usage={
                    "input_tokens": len(conversation_text.split()),
                    "output_tokens": len(content.split()) if content else 0,
                    "total_tokens": len(conversation_text.split()) + (len(content.split()) if content else 0)
                },
                response_time=response_time,
                success=True
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_metrics(response_time, False)
            
            logger.error(f"Google chat completion failed: {e}")
            return LLMResponse(
                content="",
                model=self.model.value,
                provider=self.provider.value,
                response_time=response_time,
                success=False,
                error_message=str(e)
            )
    
    async def text_completion(
        self, 
        prompt: str,
        **kwargs
    ) -> LLMResponse:
        """Google text completion"""
        messages = [LLMMessage(role="user", content=prompt)]
        return await self.chat_completion(messages, **kwargs)






def get_default_llm_config() -> LLMConfig:
    """Get default LLM configuration based on available API keys"""
    from core.shared.config import get_settings

    settings = get_settings()

    # Use Google Gemini or fallback to mock
    if settings.google_api_key:
        return LLMConfig(
            provider=LLMProvider.GOOGLE,
            model=LLMModel.GEMINI_1_5_FLASH,
            api_key=settings.google_api_key,
            max_tokens=2000,
            temperature=0.7
        )
    else:
        # Fallback to mock for development
        logger.warning("No Google API key configured, using mock LLM client")
        return LLMConfig(
            provider=LLMProvider.MOCK,
            model=LLMModel.MOCK_MODEL,
            api_key="mock-key"
        )


# Convenience function for backward compatibility
def create_llm_client(provider_name: str) -> LLMClient:
    """Create LLM client from provider name string - CLEAN VERSION (only Google Gemini)"""
    from core.shared.config import get_settings
    settings = get_settings()

    if provider_name.lower() == "google":
        config = LLMConfig(
            provider=LLMProvider.GOOGLE,
            model=LLMModel.GEMINI_1_5_FLASH,
            api_key=settings.google_api_key,
            max_tokens=2000,
            temperature=0.7
        )
    else:
        raise ValueError(f"Only Google Gemini provider is supported. Got: {provider_name}")

    return create_llm_client_from_config(config)


# Clean factory function for supported providers only
def create_llm_client_from_config(config: LLMConfig) -> LLMClient:
    """Factory function to create appropriate LLM client from config - CLEAN VERSION (only Google Gemini)"""
    if config.provider == LLMProvider.GOOGLE:
        return GoogleClient(config)
    elif config.provider == LLMProvider.MOCK:
        return MockLLMClient(config)
    else:
        raise ValueError(f"Only Google Gemini provider is supported. Got: {config.provider}")