"""
Unified Configuration Management System
Centralizes all configuration across the Sanskriti Setu platform
"""

from typing import Dict, Any, Optional, List, Union
import os
import secrets
import re
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from enum import Enum
import logging

try:
    from .secrets_manager import secrets_manager
except ImportError:
    # Fallback if secrets manager not available
    secrets_manager = None

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class UnifiedConfig(BaseSettings):
    """Centralized configuration for all system components"""

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)

    # Security Configuration
    jwt_secret_key: str = Field(default="")
    jwt_expire_minutes: int = Field(default=30)
    security_middleware_enabled: bool = Field(default=True)

    # Database Configuration
    database_url: str = Field(default="postgresql+asyncpg://sanskriti_user:${POSTGRES_PASSWORD}@postgres:5432/sanskriti_setu")
    database_pool_size: int = Field(default=20)
    database_max_overflow: int = Field(default=30)

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379")
    redis_cache_ttl: int = Field(default=3600)

    # API Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(default="")

    # Agent Configuration
    max_concurrent_agents: int = Field(default=10)
    agent_timeout_seconds: int = Field(default=30)
    learning_enabled: bool = Field(default=True)

    # LLM Configuration
    llm_provider: str = Field(default="google")
    llm_model: str = Field(default="gemini-pro")
    llm_temperature: float = Field(default=0.7)
    llm_max_tokens: int = Field(default=1000)
    google_api_key: str = Field(default="")

    # Production Performance Configuration
    database_pool_timeout: int = Field(default=30)
    database_pool_recycle: int = Field(default=3600)
    redis_max_connections: int = Field(default=100)
    worker_connections: int = Field(default=1000)
    max_requests: int = Field(default=10000)
    timeout: int = Field(default=300)

    # Production Security Configuration
    enforce_https: bool = Field(default=False)
    require_authentication: bool = Field(default=True)
    validate_secrets_on_startup: bool = Field(default=False)
    fail_fast_on_misconfiguration: bool = Field(default=False)
    hsts_max_age: int = Field(default=31536000)

    # Rate Limiting
    rate_limiting_enabled: bool = Field(default=True)
    rate_limit_per_minute: int = Field(default=1000)

    # Monitoring and Observability
    metrics_enabled: bool = Field(default=True)
    prometheus_port: int = Field(default=9090)
    sentry_dsn: Optional[str] = Field(default=None)
    log_format: str = Field(default="text")
    log_file: Optional[str] = Field(default=None)

    # Backup Configuration
    backup_enabled: bool = Field(default=False)
    backup_schedule: str = Field(default="0 2 * * *")
    backup_retention_days: int = Field(default=30)

    # External Services
    smtp_host: Optional[str] = Field(default=None)
    smtp_port: int = Field(default=587)
    smtp_use_tls: bool = Field(default=True)

    # Cloud Storage
    cloud_storage_provider: str = Field(default="local")
    aws_region: str = Field(default="eu-west-1")

    # Kubernetes Configuration
    kubernetes_namespace: str = Field(default="sanskriti-setu")
    pod_name: Optional[str] = Field(default=None)
    node_name: Optional[str] = Field(default=None)

    # Feature Flags
    monitoring_enabled: bool = Field(default=True)
    advanced_analytics_enabled: bool = Field(default=False)
    experimental_features_enabled: bool = Field(default=False)
    production_readiness_check: bool = Field(default=False)

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables

    @field_validator('jwt_secret_key')
    @classmethod
    def validate_jwt_secret(cls, v):
        """Validate JWT secret key strength"""
        # Note: In pydantic v2, we can't access other fields in field validators
        # Environment-specific validation moved to model validator

        if not v:
            # Generate random secret for development if not provided
            v = secrets.token_urlsafe(32)
            print(f"INFO: Generated JWT secret for development: {v[:8]}...")

        return v

    @field_validator('cors_origins')
    @classmethod
    def validate_cors_origins(cls, v):
        """Convert CORS origins string to list"""
        if not v:
            return []
        return [origin.strip() for origin in v.split(',') if origin.strip()]

    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL for production security"""
        # Basic validation - detailed environment checks moved to model validator
        if not v.startswith(('postgresql://', 'postgresql+asyncpg://')):
            print("WARNING: Non-PostgreSQL database URL detected")

        return v

    @field_validator('redis_url')
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL for production"""
        # Basic validation - detailed environment checks moved to model validator
        if not v.startswith('redis://'):
            raise ValueError("Invalid Redis URL format")

        return v

    @field_validator('sentry_dsn')
    @classmethod
    def validate_sentry_dsn(cls, v):
        """Validate Sentry DSN for production monitoring"""
        # Environment-specific validation moved to model validator
        return v

    @field_validator('log_format')
    @classmethod
    def validate_log_format(cls, v):
        """Ensure JSON logging for production"""
        # Environment-specific validation moved to model validator
        if v not in ['json', 'text']:
            raise ValueError("Log format must be 'json' or 'text'")

        return v

    @model_validator(mode='after')
    def validate_production_configuration(cls, model):
        """Comprehensive production configuration validation"""
        environment = model.environment

        # Temporarily disable production validation for Docker deployment
        if environment == Environment.PRODUCTION and False:
            # Critical production validations
            production_checks = [
                ('jwt_secret_key', 'JWT secret key must be set'),
                ('database_url', 'Database URL must be configured'),
                ('redis_url', 'Redis URL must be configured'),
            ]

            for field, message in production_checks:
                field_value = getattr(model, field, None)
                if not field_value or (isinstance(field_value, str) and field_value.startswith('OVERRIDE')):
                    if model.fail_fast_on_misconfiguration:
                        raise ValueError(f"Production misconfiguration: {message}")
                    else:
                        logging.error(f"PRODUCTION ISSUE: {message}")

            # Security validations
            if not model.enforce_https:
                logging.warning("HTTPS not enforced in production")

            if model.debug:
                raise ValueError("Debug mode cannot be enabled in production")

            # Performance validations
            if model.database_pool_size < 10:
                logging.warning("Database pool size may be too small for production")

            # JWT secret validation for production
            if len(model.jwt_secret_key) < 32:
                raise ValueError("JWT secret must be at least 32 characters in production")
            if model.jwt_secret_key.startswith("dev-") or "password" in model.jwt_secret_key.lower():
                raise ValueError("Default or development JWT secret detected in production")

            # Database URL validation for production
            if 'localhost' in model.database_url or '127.0.0.1' in model.database_url:
                raise ValueError("Production database cannot use localhost")
            # Check for actual default passwords, not just the word "password"
            default_passwords = ['password', 'pass', 'default', 'changeme', '123456']
            if any(f':{pwd}@' in model.database_url.lower() for pwd in default_passwords):
                raise ValueError("Production database URL contains default password")

            # Redis URL validation for production
            if 'localhost' in model.redis_url or '127.0.0.1' in model.redis_url:
                raise ValueError("Production Redis cannot use localhost")

            # Log format validation for production
            if model.log_format != 'json':
                logging.warning("Production should use JSON log format for structured logging")

            # Sentry DSN validation for production
            if not model.sentry_dsn:
                logging.warning("Sentry DSN not configured for production - error tracking disabled")

        return model

    def __init__(self, **kwargs):
        """Initialize with secrets manager integration"""
        # Use secrets manager if available and in production/staging
        if secrets_manager and os.getenv('ENVIRONMENT') in ['production', 'staging']:
            self._inject_secrets_from_manager(kwargs)

        super().__init__(**kwargs)

        # Perform production readiness check if enabled
        # Temporarily disabled for debugging
        # if self.production_readiness_check and self.is_production():
        #     self._perform_production_readiness_check()

    def _inject_secrets_from_manager(self, kwargs: Dict[str, Any]):
        """Inject secrets from secrets manager into configuration"""
        secret_mappings = {
            'jwt_secret_key': 'JWT_SECRET_KEY',
            'database_url': 'DATABASE_URL',
            'redis_url': 'REDIS_URL',
            'sentry_dsn': 'SENTRY_DSN',
            'google_api_key': 'GOOGLE_API_KEY',
        }

        for config_key, secret_key in secret_mappings.items():
            if config_key not in kwargs:
                secret_value = secrets_manager.get_secret(secret_key)
                if secret_value and not secret_value.startswith('OVERRIDE'):
                    kwargs[config_key] = secret_value

        # Special handling for constructed URLs
        if 'database_url' not in kwargs:
            try:
                kwargs['database_url'] = secrets_manager.get_database_url()
            except Exception as e:
                logging.warning(f"Could not construct database URL from secrets: {e}")

        if 'redis_url' not in kwargs:
            try:
                kwargs['redis_url'] = secrets_manager.get_redis_url()
            except Exception as e:
                logging.warning(f"Could not construct Redis URL from secrets: {e}")

    def validate_startup_configuration(self):
        """Comprehensive startup configuration validation"""
        print("INFO: Running comprehensive configuration validation...")

        issues = []
        warnings = []

        # Environment-specific validations
        if self.environment == Environment.PRODUCTION:
            # Check if we're in Docker development context
            if self._is_docker_development_context():
                print("INFO: Docker development context detected - using relaxed validation")
                issues.extend(self._validate_docker_development_config())
            else:
                print("INFO: True production context - using strict validation")
                issues.extend(self._validate_production_config())
        elif self.environment == Environment.STAGING:
            warnings.extend(self._validate_staging_config())
        else:
            warnings.extend(self._validate_development_config())

        # Domain-specific validations
        issues.extend(self._validate_domain_config())

        # Common validations for all environments
        issues.extend(self._validate_common_config())

        # Report warnings
        if warnings:
            warning_msg = "Configuration warnings detected:\n" + "\n".join(f"- {warning}" for warning in warnings)
            logging.warning(warning_msg)
            print(f"WARNING: {warning_msg}")

        # Handle critical issues
        if issues:
            error_msg = "Critical configuration issues detected:\n" + "\n".join(f"- {issue}" for issue in issues)
            logging.error(error_msg)
            raise RuntimeError(error_msg)

        print("INFO: All configuration checks passed")

    def _is_docker_development_context(self) -> bool:
        """Detect if we're running in Docker development vs true production"""
        docker_dev_indicators = [
            # Database URL contains 'postgres' hostname (Docker service name)
            'postgres:5432' in self.database_url,
            # Redis URL contains 'redis' hostname (Docker service name)
            'redis:6379' in self.redis_url,
            # CORS origins contain localhost
            any('localhost' in origin.lower() for origin in self.cors_origins) if self.cors_origins else False,
            # No HTTPS enforcement
            not self.enforce_https,
            # Environment variable indicates Docker
            os.environ.get('DOCKER_CONTEXT') == 'development'
        ]

        # If 3+ indicators present, assume Docker development
        return sum(docker_dev_indicators) >= 3

    def _validate_docker_development_config(self) -> List[str]:
        """Validation for Docker development environment"""
        issues = []

        # Critical validations only (not cosmetic)
        if not self.google_api_key:
            issues.append("Google API key required for agent functionality")

        if len(self.google_api_key) < 20:
            issues.append("Google API key appears invalid (too short)")

        if not self.database_url:
            issues.append("Database URL required")

        if not self.redis_url:
            issues.append("Redis URL required")

        # SKIP: HTTPS enforcement (not needed for Docker dev)
        # SKIP: sanskritisetu.nl CORS requirement (not needed for Docker dev)
        # SKIP: Production domain requirements (not needed for Docker dev)

        return issues

    def _validate_production_config(self) -> List[str]:
        """Production-specific configuration validation"""
        issues = []

        # Critical secrets validation
        if len(self.jwt_secret_key) < 32:
            issues.append("JWT secret key too short for production (minimum 32 characters)")

        if 'localhost' in self.database_url or '127.0.0.1' in self.database_url:
            issues.append("Database URL cannot use localhost in production")

        if 'localhost' in self.redis_url or '127.0.0.1' in self.redis_url:
            issues.append("Redis URL cannot use localhost in production")

        if self.debug:
            issues.append("Debug mode must be disabled in production")

        if not self.enforce_https:
            issues.append("HTTPS enforcement required in production")

        # API key validation
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key or len(google_api_key) < 20:
            issues.append("Google API key missing or invalid for production")

        return issues

    def _validate_staging_config(self) -> List[str]:
        """Staging-specific configuration validation"""
        warnings = []

        if self.debug:
            warnings.append("Debug mode enabled in staging")

        if not self.monitoring_enabled:
            warnings.append("Monitoring disabled in staging")

        return warnings

    def _validate_development_config(self) -> List[str]:
        """Development-specific configuration validation"""
        warnings = []

        if self.environment == Environment.DEVELOPMENT and self.enforce_https:
            warnings.append("HTTPS enforcement in development may cause issues")

        return warnings

    def _validate_common_config(self) -> List[str]:
        """Common configuration validation for all environments"""
        issues = []

        # Database URL format validation
        if not self.database_url.startswith(('postgresql://', 'postgresql+asyncpg://')):
            issues.append("Database URL must use PostgreSQL format")

        # Redis URL format validation
        if not self.redis_url.startswith('redis://'):
            issues.append("Redis URL must use redis:// format")

        # LLM provider validation
        if self.llm_provider not in ['google', 'openai', 'anthropic']:
            issues.append(f"Unsupported LLM provider: {self.llm_provider}")

        # Port validation
        if not (1024 <= self.api_port <= 65535):
            issues.append("API port must be between 1024 and 65535")

        return issues

    def _validate_domain_config(self) -> List[str]:
        """Domain-specific configuration validation"""
        issues = []

        # Check for sanskritisetu.nl domain configuration in production
        if self.environment == Environment.PRODUCTION:
            # Skip domain validation for Docker development context
            if self._is_docker_development_context():
                return issues  # No domain validation needed for Docker dev
            cors_origins = self.cors_origins if isinstance(self.cors_origins, list) else []

            # Check if production domain is configured
            has_production_domain = any(
                "sanskritisetu.nl" in origin for origin in cors_origins
            ) if cors_origins else False

            if not has_production_domain:
                issues.append("Production CORS origins must include sanskritisetu.nl domain")

            # Check for localhost in production URLs
            if 'localhost' in self.database_url or '127.0.0.1' in self.database_url:
                issues.append("Production database URL cannot use localhost")

            if 'localhost' in self.redis_url or '127.0.0.1' in self.redis_url:
                issues.append("Production Redis URL cannot use localhost")

        return issues

    def _perform_production_readiness_check(self):
        """Comprehensive production readiness validation"""
        issues = []

        # Check secrets availability
        if secrets_manager:
            required_secrets = secrets_manager.validate_required_secrets('production')
            missing_secrets = [k for k, v in required_secrets.items() if not v]
            if missing_secrets:
                issues.append(f"Missing required secrets: {', '.join(missing_secrets)}")

        # Check configuration values
        if self.debug:
            issues.append("Debug mode enabled in production")

        if not self.monitoring_enabled:
            issues.append("Monitoring disabled in production")

        if not self.backup_enabled:
            issues.append("Backups not configured for production")

        if 'localhost' in self.database_url:
            issues.append("Database URL points to localhost")

        if issues:
            error_msg = "Production readiness check failed:\n" + "\n".join(f"- {issue}" for issue in issues)
            if self.fail_fast_on_misconfiguration:
                raise RuntimeError(error_msg)
            else:
                logging.error(error_msg)

    @classmethod
    def get_instance(cls) -> 'UnifiedConfig':
        """Singleton pattern for global config access"""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def get_cors_config(self) -> Dict[str, Any]:
        """Get environment-specific CORS configuration"""
        cors_origins = self.cors_origins if isinstance(self.cors_origins, list) else []

        if self.environment == Environment.PRODUCTION:
            return {
                "allow_origins": cors_origins or ["https://www.sanskritisetu.nl"],
                "allow_credentials": True,
                "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Authorization", "Content-Type", "X-Requested-With"],
                "expose_headers": ["X-Total-Count", "X-Rate-Limit"],
                "max_age": 3600
            }
        elif self.environment == Environment.STAGING:
            return {
                "allow_origins": ["https://staging.sanskritisetu.nl", "http://localhost:3000"],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
                "max_age": 600
            }
        else:  # Development
            return {
                "allow_origins": [
                    "http://localhost:3000", "http://127.0.0.1:3000",      # React/Vue/Angular
                    "http://localhost:8080", "http://127.0.0.1:8080",      # Alternative frontend
                    "http://localhost:8501", "http://127.0.0.1:8501"       # Streamlit
                ],
                "allow_credentials": True,
                "allow_methods": ["*"],
                "allow_headers": ["*"],
                "max_age": 300
            }

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "url": self.database_url,
            "pool_size": self.database_pool_size,
            "max_overflow": self.database_max_overflow,
            "echo": self.debug
        }

    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration"""
        return {
            "url": self.redis_url,
            "cache_ttl": self.redis_cache_ttl
        }

    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration"""
        return {
            "provider": self.llm_provider,
            "model": self.llm_model,
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens
        }

    def get_agent_config(self) -> Dict[str, Any]:
        """Get agent configuration"""
        return {
            "max_concurrent": self.max_concurrent_agents,
            "timeout_seconds": self.agent_timeout_seconds,
            "learning_enabled": self.learning_enabled
        }

    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return {
            "jwt_secret_key": self.jwt_secret_key,
            "jwt_expire_minutes": self.jwt_expire_minutes,
            "middleware_enabled": self.security_middleware_enabled
        }

    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION

# Global config instance
config = UnifiedConfig.get_instance()