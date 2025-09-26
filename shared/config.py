"""
Configuration management for Sanskriti Setu AI Multi-Agent System
Centralized configuration with environment variable support and feature flags
"""

import os
from typing import Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from core.shared.interfaces import IsolationLevel


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./sanskriti_setu.db", 
        env="DATABASE_URL"
    )
    database_pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    
    # Production Database URLs (for Week 2+)
    production_db_url: str = Field(
        default="postgresql://user:pass@localhost:5432/sanskriti_production",
        env="PRODUCTION_DB_URL"
    )
    sandbox_cva_db_url: str = Field(
        default="postgresql://user:pass@localhost:5432/sanskriti_sandbox_cva",
        env="SANDBOX_CVA_DB_URL"  
    )
    sandbox_shared_db_url: str = Field(
        default="postgresql://user:pass@localhost:5432/sanskriti_sandbox_shared",
        env="SANDBOX_SHARED_DB_URL"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_title: str = Field(default="Sanskriti Setu API", env="API_TITLE")
    api_version: str = Field(default="1.0.0", env="API_VERSION")
    
    # Security
    secret_key: str = Field(
        default="dev-secret-key-change-in-production-please", 
        env="SECRET_KEY"
    )
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    password_hash_rounds: int = Field(default=12, env="PASSWORD_HASH_ROUNDS")
    
    # External AI Services
    google_api_key: Optional[str] = Field(default=None, env="GOOGLE_API_KEY")
    
    # Sandbox Configuration
    default_isolation_level: IsolationLevel = Field(
        default=IsolationLevel.MOCK, 
        env="DEFAULT_ISOLATION_LEVEL"
    )
    sandbox_timeout: int = Field(default=300, env="SANDBOX_TIMEOUT")  # 5 minutes
    sandbox_memory_limit: str = Field(default="512m", env="SANDBOX_MEMORY_LIMIT")
    sandbox_cpu_limit: str = Field(default="0.5", env="SANDBOX_CPU_LIMIT")
    
    # Agent Configuration
    max_concurrent_tasks: int = Field(default=10, env="MAX_CONCURRENT_TASKS")
    agent_heartbeat_interval: int = Field(default=30, env="AGENT_HEARTBEAT_INTERVAL")
    agent_task_timeout: int = Field(default=600, env="AGENT_TASK_TIMEOUT")  # 10 minutes
    
    # Redis Configuration (for Week 2+)
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Frontend Configuration
    frontend_url: str = Field(default="http://localhost:8501", env="FRONTEND_URL")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="sanskriti_setu.log", env="LOG_FILE")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"  # Allow extra fields from .env
    }


# Feature Flags Configuration
FEATURE_FLAGS = {
    # Week 1 - Core functionality
    "basic_cva_agent": True,
    "mock_sandbox": False,  # Disabled for production - using real Docker sandbox
    "simple_task_queue": True,
    "basic_approval_queue": True,
    "streamlit_ui": True,
    
    # Week 2 - Enhanced functionality - Enable by default for production
    "real_llm_integration": True,
    "real_docker_sandbox": True,
    "multi_agent_orchestration": True,  # Enable multi-agent features
    "advanced_task_routing": True,
    "websocket_updates": True,  # Enable WebSocket features
    "agent_load_balancing": True,
    
    # Week 3 - Advanced features - Enable by default for production
    "ml_ethics_engine": True,  # Enable ML ethics features
    "knowledge_evolution": True,
    "advanced_ui": True,  # Enable advanced UI features
    "performance_analytics": True,
    "custom_agent_plugins": True,
    
    # Phase 2 - Intelligent Multi-LLM Orchestration
    "multi_llm_integration": True,  # Multi-LLM provider orchestration
    "response_caching": True,  # LLM response caching system
    "ethics_gate_enabled": True,  # AI safety and ethics validation
    "intelligent_routing": True,  # Performance-based LLM routing
    "dynamic_load_balancing": True,  # Dynamic LLM load balancing
    
    # Phase 3 - Advanced Agent Collaboration & Knowledge Evolution
    "agent_mesh_enabled": True,  # Multi-agent communication mesh
    "collaborative_tasks": True,  # Multi-agent task coordination
    "adaptive_learning": True,  # Agent capability evolution
    "knowledge_patterns": True,  # Pattern discovery and learning
    "system_optimization": True,  # Autonomous system optimization
    "intelligent_monitoring": True,  # Advanced system monitoring
    "performance_prediction": True,  # Predictive performance analysis
    "resource_orchestration": True,  # Dynamic resource allocation
    
    # Phase 4 - Autonomous AI Ecosystem & Self-Expansion
    "autonomous_ai_ecosystem": True,  # Overall Phase 4 autonomous systems
    "autonomous_agent_generation": True,  # Dynamic agent creation
    "self_healing_architecture": True,  # Automatic system repair
    "dynamic_code_generation": True,  # Runtime code generation
    "emergent_intelligence": True,  # Emergent behavior facilitation
    "autonomous_system_analysis": True,  # System needs analysis
    "self_expanding_capabilities": True,  # Self-capability expansion
    "meta_learning_systems": True,  # Learning about learning
    "autonomous_optimization": True,  # Self-optimization systems
}


def get_feature_flag(flag_name: str) -> bool:
    """Get feature flag value, with environment variable override"""
    env_var = f"FEATURE_{flag_name.upper()}"
    env_value = os.getenv(env_var)
    
    if env_value is not None:
        return env_value.lower() in ("true", "1", "yes", "on")
    
    return FEATURE_FLAGS.get(flag_name, False)


def set_feature_flag(flag_name: str, enabled: bool) -> None:
    """Set feature flag value (for testing/development)"""
    FEATURE_FLAGS[flag_name] = enabled


def get_database_url(environment: str = "development") -> str:
    """Get appropriate database URL based on environment"""
    settings = get_settings()
    
    if environment == "production":
        return settings.production_db_url
    elif environment == "sandbox_cva":
        return settings.sandbox_cva_db_url  
    elif environment == "sandbox_shared":
        return settings.sandbox_shared_db_url
    else:
        return settings.database_url


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)"""
    global _settings
    _settings = Settings()
    return _settings


# Environment-specific configurations
ENVIRONMENTS = {
    "development": {
        "debug": True,
        "log_level": "DEBUG",
        "database_url": "sqlite:///./dev_sanskriti_setu.db",
        "isolation_level": IsolationLevel.MOCK,
    },
    "testing": {
        "debug": True,
        "log_level": "WARNING", 
        "database_url": "sqlite:///:memory:",
        "isolation_level": IsolationLevel.MOCK,
    },
    "staging": {
        "debug": False,
        "log_level": "INFO",
        "isolation_level": IsolationLevel.PROCESS,
    },
    "production": {
        "debug": False,
        "log_level": "WARNING",
        "isolation_level": IsolationLevel.CONTAINER,
    }
}


def apply_environment_config(env_name: str) -> None:
    """Apply environment-specific configuration overrides"""
    if env_name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {env_name}")
    
    env_config = ENVIRONMENTS[env_name]
    settings = get_settings()
    
    for key, value in env_config.items():
        if hasattr(settings, key):
            setattr(settings, key, value)


# Configuration validation
def validate_configuration() -> Dict[str, str]:
    """Validate configuration and return any errors"""
    errors = []
    settings = get_settings()
    
    # Check required API keys for production
    if settings.environment == "production":
        if not settings.google_api_key:
            errors.append("GOOGLE_API_KEY must be set in production")
        
        if settings.secret_key == "dev-secret-key-change-in-production-please":
            errors.append("SECRET_KEY must be changed in production")
    
    # Check database URL format
    if not settings.database_url:
        errors.append("DATABASE_URL is required")
    
    # Check sandbox configuration
    if get_feature_flag("real_docker_sandbox"):
        if settings.sandbox_timeout <= 0:
            errors.append("SANDBOX_TIMEOUT must be positive when using real Docker sandbox")
    
    return {"errors": errors, "warnings": []}


if __name__ == "__main__":
    # Configuration testing/debugging
    settings = get_settings()
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"Database URL: {settings.database_url}")
    print(f"Feature flags: {FEATURE_FLAGS}")
    
    # Validate configuration
    validation = validate_configuration()
    if validation["errors"]:
        print(f"Configuration errors: {validation['errors']}")
    if validation["warnings"]:
        print(f"Configuration warnings: {validation['warnings']}")