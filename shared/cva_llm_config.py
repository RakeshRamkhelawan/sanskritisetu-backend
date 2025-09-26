"""
CVA-specific LLM Configuration
Special configuration for CVA agent to use premium models
"""

from core.shared.llm_client import LLMConfig, LLMProvider, LLMModel, create_llm_client_from_config
from core.shared.config import get_settings

def get_cva_llm_config() -> LLMConfig:
    """Get CVA-specific LLM configuration with premium models"""
    settings = get_settings()
    
    # CVA gets the best available model
    # Priority: Ollama Qwen3:30B > Google Gemini 2.5 Pro > Ollama default > Mock
    
    if hasattr(settings, 'ollama_base_url') and settings.ollama_base_url:
        # Use premium Ollama model for CVA (30B parameters)
        return LLMConfig(
            provider=LLMProvider.OLLAMA,
            model=LLMModel.LLAMA3_LATEST,  # Efficient Llama3 model for CVA
            api_key=settings.ollama_base_url,
            max_tokens=4000,  # Increased for strategic planning
            temperature=0.8   # Slightly higher for creative strategic thinking
        )
    elif settings.google_api_key:
        # Use Gemini 2.5 Pro for CVA (premium Google model)
        return LLMConfig(
            provider=LLMProvider.GOOGLE,
            model=LLMModel.GEMINI_2_5_PRO,  # Premium Gemini model for CVA
            api_key=settings.google_api_key,
            max_tokens=4000,  # Increased for strategic planning
            temperature=0.8   # Slightly higher for creative strategic thinking
        )
    else:
        # Fallback to mock with enhanced responses
        return LLMConfig(
            provider=LLMProvider.MOCK,
            model=LLMModel.MOCK_MODEL,
            api_key="mock-key",
            max_tokens=4000,
            temperature=0.8
        )

def get_agent_llm_config() -> LLMConfig:
    """Get standard agent LLM configuration (cost-effective models)"""
    settings = get_settings()
    
    # Standard agents get efficient models
    # Priority: Ollama default > Google Gemini 2.5 Flash > Mock
    
    if hasattr(settings, 'ollama_base_url') and settings.ollama_base_url:
        # Use efficient Ollama model for standard agents
        return LLMConfig(
            provider=LLMProvider.OLLAMA,
            model=LLMModel.LLAMA3_LATEST,  # Efficient Llama3 model for agents
            api_key=settings.ollama_base_url,
            max_tokens=2000,
            temperature=0.7
        )
    elif settings.google_api_key:
        # Use Gemini 2.5 Flash for standard agents (cost-effective)
        return LLMConfig(
            provider=LLMProvider.GOOGLE,
            model=LLMModel.GEMINI_2_5_FLASH,  # Cost-effective Gemini model
            api_key=settings.google_api_key,
            max_tokens=2000,
            temperature=0.7
        )
    else:
        # Fallback to mock
        return LLMConfig(
            provider=LLMProvider.MOCK,
            model=LLMModel.MOCK_MODEL,
            api_key="mock-key",
            max_tokens=2000,
            temperature=0.7
        )

def create_cva_llm_client():
    """Create LLM client specifically optimized for CVA"""
    config = get_cva_llm_config()
    return create_llm_client_from_config(config)

def create_agent_llm_client():
    """Create LLM client for standard agents"""
    config = get_agent_llm_config()
    return create_llm_client_from_config(config)