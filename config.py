"""
Configuration Module - Foundation API Integration
ASCII-only, single server policy compliant
"""

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Import configuration manager
from core.config.production_config import ProductionConfigManager

class Config:
    """Simple configuration wrapper for foundation API"""

    def __init__(self):
        self.production_config = None
        self._config_data = {}
        self._load_dotenv_files()
        self._load_environment_variables()

    def _load_dotenv_files(self):
        """Load environment files in order of priority"""
        # Load development config first (if exists)
        if os.path.exists('.env.development'):
            load_dotenv('.env.development', override=False)

        # Load production config (if exists) - can override development
        if os.path.exists('.env'):
            load_dotenv('.env', override=False)

    def _load_environment_variables(self):
        """Load basic environment variables"""
        self._config_data = {
            "DATABASE_URL": os.getenv("DATABASE_URL", "postgresql://sanskriti_user:sanskriti_pass@localhost:5432/sanskriti_production"),
            "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379"),
            "ENVIRONMENT": os.getenv("ENVIRONMENT", "development"),
            "DEBUG": os.getenv("DEBUG", "false").lower() == "true",
            "API_HOST": os.getenv("API_HOST", "0.0.0.0"),
            "API_PORT": int(os.getenv("API_PORT", "8000")),
            "SECRET_KEY": os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", "google")
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config_data.get(key, default)

    def get_database_url(self) -> str:
        """Get database URL"""
        return self.get("DATABASE_URL")

    def get_redis_url(self) -> str:
        """Get Redis URL"""
        return self.get("REDIS_URL")

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.get("ENVIRONMENT") == "development"

    def is_debug(self) -> bool:
        """Check if debug mode enabled"""
        return self.get("DEBUG", False)

    def initialize_production_config(self) -> bool:
        """Initialize production configuration manager"""
        try:
            self.production_config = ProductionConfigManager()
            # Check if production config manager has load_configuration method
            if hasattr(self.production_config, 'load_configuration'):
                if self.production_config.load_configuration():
                    print("Production configuration initialized successfully")
                    return True
                else:
                    print("Production configuration initialization failed - using basic config")
                    return False
            else:
                print("Production configuration using basic config (load_configuration method not available)")
                return True
        except Exception as e:
            print(f"Production configuration error: {e}")
            return False

# Global configuration instance
config = Config()

# Configuration initialization function
def initialize_config() -> bool:
    """Initialize configuration system"""
    try:
        # Try to initialize production config
        config.initialize_production_config()
        print("Configuration system initialized")
        return True
    except Exception as e:
        print(f"Configuration initialization error: {e}")
        return False

# Convenience functions
def get_database_url() -> str:
    return config.get_database_url()

def get_redis_url() -> str:
    return config.get_redis_url()

def is_development() -> bool:
    return config.is_development()