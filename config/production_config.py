"""
Production Configuration Manager for Sanskriti Setu
Advanced configuration management with environment-specific settings
"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class LLMProviderConfig:
    """LLM Provider Configuration"""
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 30
    rate_limit: int = 100  # requests per minute
    priority: int = 1  # lower = higher priority
    fallback_to: Optional[str] = None

@dataclass  
class DatabaseConfig:
    """Database Configuration"""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    echo: bool = False

@dataclass
class CacheConfig:
    """Cache Configuration"""
    redis_url: Optional[str] = None
    default_ttl: int = 3600  # 1 hour
    max_memory_cache: int = 100  # MB
    enable_redis: bool = True
    enable_memory: bool = True

@dataclass
class SecurityConfig:
    """Security Configuration"""
    secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_hours: int = 24
    bcrypt_rounds: int = 12
    rate_limit_per_minute: int = 60
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

@dataclass
class MonitoringConfig:
    """Monitoring and Alerting Configuration"""
    memory_alert_threshold: float = 85.0
    memory_critical_threshold: float = 90.0
    monitoring_interval: float = 60.0  # seconds
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"
    metrics_retention_days: int = 30

@dataclass
class FeatureFlags:
    """Feature flags for production deployment"""
    real_llm_integration: bool = False
    ethics_gate_enabled: bool = False
    response_caching: bool = False
    multi_llm_integration: bool = False
    docker_sandbox: bool = False
    advanced_analytics: bool = False
    ml_ethics_engine: bool = False
    real_database: bool = True
    production_logging: bool = True
    performance_optimization: bool = True

class ProductionConfigManager:
    """
    Production Configuration Manager
    Manages all configuration for production deployment
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or "production_config.yaml")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Initialize configurations
        self.llm_providers: Dict[str, LLMProviderConfig] = {}
        self.database: Optional[DatabaseConfig] = None
        self.cache: CacheConfig = CacheConfig()
        self.security: Optional[SecurityConfig] = None
        self.monitoring: MonitoringConfig = MonitoringConfig()
        self.feature_flags: FeatureFlags = FeatureFlags()
        
        # Load configuration
        self._load_configuration()
        
        logger.info(f"Production Configuration Manager initialized for {self.environment}")
    
    def _load_configuration(self):
        """Load configuration from file and environment"""
        # Load from file if exists
        if self.config_path.exists():
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_environment()
        
        # Set up default providers
        self._setup_default_llm_providers()
        
        # Validate configuration
        self._validate_configuration()
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Load each section
            if 'llm_providers' in config_data:
                for name, provider_data in config_data['llm_providers'].items():
                    self.llm_providers[name] = LLMProviderConfig(name=name, **provider_data)
            
            if 'database' in config_data:
                self.database = DatabaseConfig(**config_data['database'])
            
            if 'cache' in config_data:
                self.cache = CacheConfig(**config_data['cache'])
            
            if 'security' in config_data:
                self.security = SecurityConfig(**config_data['security'])
            
            if 'monitoring' in config_data:
                self.monitoring = MonitoringConfig(**config_data['monitoring'])
            
            if 'feature_flags' in config_data:
                self.feature_flags = FeatureFlags(**config_data['feature_flags'])
            
            logger.info(f"Configuration loaded from {self.config_path}")
            
        except Exception as e:
            logger.warning(f"Could not load configuration file: {e}")
    
    def _load_from_environment(self):
        """Override configuration with environment variables"""
        # Database
        db_url = os.getenv("DATABASE_URL")
        if db_url:
            if not self.database:
                self.database = DatabaseConfig(url=db_url)
            else:
                self.database.url = db_url
        
        # Cache
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            self.cache.redis_url = redis_url
        
        # Security
        secret_key = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
        if not self.security:
            self.security = SecurityConfig(secret_key=secret_key)
        else:
            self.security.secret_key = secret_key
        
        # Feature flags from environment
        env_features = {
            "real_llm_integration": os.getenv("ENABLE_REAL_LLM", "false").lower() == "true",
            "ethics_gate_enabled": os.getenv("ENABLE_ETHICS_GATE", "false").lower() == "true",
            "response_caching": os.getenv("ENABLE_CACHING", "false").lower() == "true",
            "multi_llm_integration": os.getenv("ENABLE_MULTI_LLM", "false").lower() == "true",
            "docker_sandbox": os.getenv("ENABLE_DOCKER_SANDBOX", "false").lower() == "true",
            "advanced_analytics": os.getenv("ENABLE_ANALYTICS", "false").lower() == "true",
            "ml_ethics_engine": os.getenv("ENABLE_ML_ETHICS", "false").lower() == "true",
        }
        
        # Update feature flags
        for flag, value in env_features.items():
            setattr(self.feature_flags, flag, value)
        
        # Environment-specific overrides
        if self.environment == "production":
            self.feature_flags.production_logging = True
            self.feature_flags.performance_optimization = True
            self.monitoring.log_level = "WARNING"
        elif self.environment == "development":
            self.monitoring.log_level = "DEBUG"
    
    def _setup_default_llm_providers(self):
        """Setup default LLM provider configurations"""
        # Anthropic Claude
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key or self.feature_flags.real_llm_integration:
            self.llm_providers["anthropic"] = LLMProviderConfig(
                name="anthropic",
                enabled=bool(anthropic_key),
                api_key=anthropic_key,
                model="claude-3-haiku-20240307",
                timeout=30,
                rate_limit=50,
                priority=1,
                fallback_to="google"
            )
        
        # Google Gemini  
        google_key = os.getenv("GOOGLE_API_KEY")
        if google_key or self.feature_flags.real_llm_integration:
            self.llm_providers["google"] = LLMProviderConfig(
                name="google",
                enabled=bool(google_key),
                api_key=google_key,
                model="gemini-1.5-flash",
                timeout=20,
                rate_limit=60,
                priority=2,
                fallback_to="openai"
            )
        
        # OpenAI GPT
        openai_key = os.getenv("OPENAI_API_KEY") 
        if openai_key or self.feature_flags.real_llm_integration:
            self.llm_providers["openai"] = LLMProviderConfig(
                name="openai",
                enabled=bool(openai_key),
                api_key=openai_key,
                model="gpt-3.5-turbo",
                timeout=25,
                rate_limit=40,
                priority=3,
                fallback_to="mock"
            )
        
        # Mock provider (always available)
        self.llm_providers["mock"] = LLMProviderConfig(
            name="mock",
            enabled=True,
            model="mock-llm-v1",
            timeout=1,
            rate_limit=1000,
            priority=99  # Lowest priority
        )
    
    def _validate_configuration(self):
        """Validate configuration settings"""
        errors = []
        
        # Database validation
        if self.feature_flags.real_database and not self.database:
            errors.append("Database configuration required when real_database is enabled")
        
        # Security validation
        if not self.security or self.security.secret_key == "dev-secret-change-in-production":
            if self.environment == "production":
                errors.append("Production requires a secure SECRET_KEY")
        
        # LLM validation
        if self.feature_flags.real_llm_integration:
            enabled_providers = [p for p in self.llm_providers.values() if p.enabled and p.api_key]
            if not enabled_providers:
                logger.warning("Real LLM integration enabled but no providers have API keys")
        
        # Cache validation
        if self.cache.enable_redis and not self.cache.redis_url:
            logger.warning("Redis caching enabled but no REDIS_URL provided")
        
        if errors:
            error_msg = "Configuration validation failed: " + "; ".join(errors)
            logger.error(error_msg)
            if self.environment == "production":
                raise ValueError(error_msg)
    
    def save_configuration(self) -> bool:
        """Save current configuration to file"""
        try:
            config_data = {
                'environment': self.environment,
                'last_updated': datetime.now().isoformat(),
                'llm_providers': {
                    name: {
                        'enabled': provider.enabled,
                        'model': provider.model,
                        'timeout': provider.timeout,
                        'rate_limit': provider.rate_limit,
                        'priority': provider.priority,
                        'fallback_to': provider.fallback_to
                    } for name, provider in self.llm_providers.items()
                },
                'database': {
                    'pool_size': self.database.pool_size,
                    'max_overflow': self.database.max_overflow,
                    'pool_timeout': self.database.pool_timeout,
                    'pool_recycle': self.database.pool_recycle,
                    'pool_pre_ping': self.database.pool_pre_ping
                } if self.database else {},
                'cache': {
                    'default_ttl': self.cache.default_ttl,
                    'max_memory_cache': self.cache.max_memory_cache,
                    'enable_redis': self.cache.enable_redis,
                    'enable_memory': self.cache.enable_memory
                },
                'monitoring': {
                    'memory_alert_threshold': self.monitoring.memory_alert_threshold,
                    'memory_critical_threshold': self.monitoring.memory_critical_threshold,
                    'monitoring_interval': self.monitoring.monitoring_interval,
                    'enable_performance_monitoring': self.monitoring.enable_performance_monitoring,
                    'log_level': self.monitoring.log_level,
                    'metrics_retention_days': self.monitoring.metrics_retention_days
                },
                'feature_flags': {
                    'real_llm_integration': self.feature_flags.real_llm_integration,
                    'ethics_gate_enabled': self.feature_flags.ethics_gate_enabled,
                    'response_caching': self.feature_flags.response_caching,
                    'multi_llm_integration': self.feature_flags.multi_llm_integration,
                    'docker_sandbox': self.feature_flags.docker_sandbox,
                    'advanced_analytics': self.feature_flags.advanced_analytics,
                    'ml_ethics_engine': self.feature_flags.ml_ethics_engine,
                    'real_database': self.feature_flags.real_database,
                    'production_logging': self.feature_flags.production_logging,
                    'performance_optimization': self.feature_flags.performance_optimization
                }
            }
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {self.config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def get_enabled_llm_providers(self) -> List[LLMProviderConfig]:
        """Get list of enabled LLM providers sorted by priority"""
        enabled = [p for p in self.llm_providers.values() if p.enabled]
        return sorted(enabled, key=lambda x: x.priority)
    
    def enable_feature(self, feature_name: str) -> bool:
        """Enable a specific feature flag"""
        if hasattr(self.feature_flags, feature_name):
            setattr(self.feature_flags, feature_name, True)
            logger.info(f"Feature enabled: {feature_name}")
            return True
        return False
    
    def disable_feature(self, feature_name: str) -> bool:
        """Disable a specific feature flag"""
        if hasattr(self.feature_flags, feature_name):
            setattr(self.feature_flags, feature_name, False)
            logger.info(f"Feature disabled: {feature_name}")
            return True
        return False
    
    def get_production_readiness_score(self) -> Dict[str, Any]:
        """Assess production readiness"""
        score = 0
        max_score = 0
        issues = []
        
        # Database configuration (20 points)
        max_score += 20
        if self.database and self.database.url:
            score += 15
            if self.database.pool_size >= 20:
                score += 5
        else:
            issues.append("Database not configured")
        
        # Security configuration (25 points)
        max_score += 25
        if self.security:
            if self.security.secret_key != "dev-secret-change-in-production":
                score += 15
            else:
                issues.append("Using default secret key")
            
            if self.security.jwt_expiry_hours <= 24:
                score += 5
            if self.security.bcrypt_rounds >= 12:
                score += 5
        else:
            issues.append("Security configuration missing")
        
        # LLM providers (15 points)
        max_score += 15
        enabled_providers = self.get_enabled_llm_providers()
        if len(enabled_providers) >= 2:
            score += 15
        elif len(enabled_providers) == 1:
            score += 10
            issues.append("Only one LLM provider configured")
        else:
            issues.append("No LLM providers configured")
        
        # Monitoring (15 points)
        max_score += 15
        if self.monitoring.enable_performance_monitoring:
            score += 10
        if self.monitoring.memory_alert_threshold <= 85:
            score += 5
        
        # Caching (10 points)
        max_score += 10
        if self.cache.redis_url:
            score += 10
        elif self.cache.enable_memory:
            score += 5
            issues.append("No Redis caching configured")
        
        # Feature completeness (15 points)
        max_score += 15
        production_features = [
            self.feature_flags.production_logging,
            self.feature_flags.performance_optimization,
            self.feature_flags.real_database
        ]
        score += sum(5 for f in production_features if f)
        
        percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        return {
            "score": score,
            "max_score": max_score,
            "percentage": percentage,
            "grade": self._get_readiness_grade(percentage),
            "issues": issues,
            "recommendations": self._get_readiness_recommendations(percentage, issues)
        }
    
    def _get_readiness_grade(self, percentage: float) -> str:
        """Get readiness grade based on percentage"""
        if percentage >= 90:
            return "A+ (Production Ready)"
        elif percentage >= 80:
            return "A (Ready with minor issues)"
        elif percentage >= 70:
            return "B (Needs improvements)"
        elif percentage >= 60:
            return "C (Major issues)"
        else:
            return "F (Not ready for production)"
    
    def _get_readiness_recommendations(self, percentage: float, issues: List[str]) -> List[str]:
        """Get recommendations based on readiness assessment"""
        recommendations = []
        
        if percentage < 70:
            recommendations.append("Address critical configuration issues before deployment")
        
        if "Database not configured" in issues:
            recommendations.append("Configure PostgreSQL database with connection pooling")
        
        if "Using default secret key" in issues:
            recommendations.append("Generate secure SECRET_KEY for production")
        
        if "No Redis caching configured" in issues:
            recommendations.append("Set up Redis for improved performance")
        
        if percentage >= 90:
            recommendations.append("System is production ready - proceed with deployment")
        
        return recommendations
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get comprehensive configuration summary"""
        return {
            "environment": self.environment,
            "feature_flags": {
                flag: getattr(self.feature_flags, flag) 
                for flag in dir(self.feature_flags) 
                if not flag.startswith('_')
            },
            "llm_providers": {
                name: {
                    "enabled": provider.enabled,
                    "has_api_key": bool(provider.api_key),
                    "model": provider.model,
                    "priority": provider.priority
                } for name, provider in self.llm_providers.items()
            },
            "database_configured": bool(self.database),
            "cache_configured": bool(self.cache.redis_url),
            "security_configured": bool(self.security and self.security.secret_key != "dev-secret-change-in-production"),
            "production_readiness": self.get_production_readiness_score()
        }

# Global configuration manager
config_manager: Optional[ProductionConfigManager] = None

def get_production_config() -> ProductionConfigManager:
    """Get global production configuration manager"""
    global config_manager
    if config_manager is None:
        config_manager = ProductionConfigManager()
    return config_manager

def initialize_production_config(config_path: Optional[str] = None) -> ProductionConfigManager:
    """Initialize production configuration manager"""
    global config_manager
    config_manager = ProductionConfigManager(config_path)
    return config_manager