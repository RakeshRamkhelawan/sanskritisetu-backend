"""
Production Logging Framework for Sanskriti Setu
Comprehensive logging system to prevent production build failures
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

def can_display_unicode() -> bool:
    """
    Check if the current environment can display Unicode characters
    Returns True if Unicode is supported, False if ASCII-only
    """
    try:
        "🚀".encode(sys.stdout.encoding or 'utf-8')
        return True
    except (UnicodeEncodeError, AttributeError):
        return False

class SafeLogger:
    """
    Unicode-safe logger that prevents production failures on Windows
    Automatically falls back to ASCII when Unicode is not supported
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.unicode_safe = can_display_unicode()
    
    def _safe_message(self, unicode_msg: str, ascii_fallback: str) -> str:
        """Return Unicode or ASCII version based on environment support"""
        if self.unicode_safe:
            try:
                # Test if we can actually encode the Unicode message
                unicode_msg.encode(sys.stdout.encoding or 'utf-8')
                return unicode_msg
            except UnicodeEncodeError:
                return ascii_fallback
        else:
            return ascii_fallback
    
    def info(self, unicode_msg: str, ascii_fallback: str = None):
        """Info level logging with Unicode safety"""
        if ascii_fallback is None:
            ascii_fallback = unicode_msg  # Use original if no fallback provided
        safe_msg = self._safe_message(unicode_msg, ascii_fallback)
        self.logger.info(safe_msg)
    
    def error(self, unicode_msg: str, ascii_fallback: str = None):
        """Error level logging with Unicode safety"""
        if ascii_fallback is None:
            ascii_fallback = unicode_msg
        safe_msg = self._safe_message(unicode_msg, ascii_fallback)
        self.logger.error(safe_msg)
    
    def warning(self, unicode_msg: str, ascii_fallback: str = None):
        """Warning level logging with Unicode safety"""
        if ascii_fallback is None:
            ascii_fallback = unicode_msg
        safe_msg = self._safe_message(unicode_msg, ascii_fallback)
        self.logger.warning(safe_msg)
    
    def debug(self, unicode_msg: str, ascii_fallback: str = None):
        """Debug level logging with Unicode safety"""
        if ascii_fallback is None:
            ascii_fallback = unicode_msg
        safe_msg = self._safe_message(unicode_msg, ascii_fallback)
        self.logger.debug(safe_msg)

class ProductionLogger:
    """
    Production-ready logging system with comprehensive validation
    Prevents build failures by validating all critical components
    """
    
    def __init__(self, app_name: str = "sanskriti_setu"):
        self.app_name = app_name
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = self._setup_logger()
        # Create Unicode-safe logger wrapper
        self.safe_logger = SafeLogger(self.logger)
        
    def _setup_logger(self) -> logging.Logger:
        """Configure production logging with file and console handlers"""
        logger = logging.getLogger(self.app_name)
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            self.log_dir / f"{self.app_name}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_startup_validation(self) -> bool:
        """
        Comprehensive startup validation with detailed logging
        Returns: True if all validations pass, False otherwise
        """
        self.logger.info("=" * 50)
        self.logger.info("[START] SANSKRITI SETU STARTUP VALIDATION")
        self.logger.info("=" * 50)
        
        validation_results = {}
        overall_success = True
        
        # Environment validation
        env = os.getenv('ENVIRONMENT', 'development')
        self.logger.info(f"Environment: {env}")
        validation_results['environment'] = env
        
        # Initialize missing_configs list
        missing_configs = []
        
        # Development mode bypass
        if env == 'development':
            self.safe_logger.info("🔧 DEVELOPMENT MODE: Skipping critical configuration validation", 
                                "[DEV] DEVELOPMENT MODE: Skipping critical configuration validation")
            overall_success = True  # Allow development mode to proceed
        else:
            # Critical configuration validation (only for production)
            critical_configs = {
                'DATABASE_URL': 'Database connection string',
                'GOOGLE_API_KEY': 'Google Gemini API access'
            }
            
            for config, description in critical_configs.items():
                if os.getenv(config):
                    self.safe_logger.info(f"✅ {config}: CONFIGURED ({description})", 
                                        f"[OK] {config}: CONFIGURED ({description})")
                    validation_results[config.lower()] = True
                else:
                    self.safe_logger.error(f"❌ {config}: MISSING ({description})", 
                                         f"[MISSING] {config}: MISSING ({description})")
                    missing_configs.append(config)
                    validation_results[config.lower()] = False
                    overall_success = False
        
        # Optional configuration validation
        optional_configs = {
            'REDIS_URL': 'Redis caching (performance optimization)'
        }
        
        for config, description in optional_configs.items():
            if os.getenv(config):
                self.safe_logger.info(f"✅ {config}: CONFIGURED ({description})", 
                                    f"[OK] {config}: CONFIGURED ({description})")
                validation_results[config.lower()] = True
            else:
                self.safe_logger.warning(f"⚠️ {config}: NOT SET ({description})", 
                                       f"[WARNING] {config}: NOT SET ({description})")
                validation_results[config.lower()] = False
        
        # Feature flags validation - Import actual config system
        try:
            from .config import get_feature_flag
            
            feature_flags = {
                'real_llm_integration': 'Real LLM provider integration',
                'ethics_gate_enabled': 'AI safety and ethics validation',
                'response_caching': 'LLM response caching system',
                'multi_llm_integration': 'Multi-LLM provider orchestration',
                'intelligent_routing': 'Performance-based LLM routing',
                'dynamic_load_balancing': 'Dynamic LLM load balancing'
            }
            
            for flag, description in feature_flags.items():
                flag_value = get_feature_flag(flag)
                status = "ENABLED" if flag_value else "DISABLED"
                unicode_msg = f"{'🟢' if flag_value else '🔴'} Feature Flag {flag}: {status} ({description})"
                ascii_msg = f"[{'ENABLED' if flag_value else 'DISABLED'}] Feature Flag {flag}: {status} ({description})"
                self.safe_logger.info(unicode_msg, ascii_msg)
                validation_results[f"feature_{flag}"] = flag_value
        except ImportError:
            self.safe_logger.warning("⚠️ Could not import config system for feature flag validation", 
                                   "[WARNING] Could not import config system for feature flag validation")
        
        # Log validation summary
        if missing_configs and env != 'development':
            self.safe_logger.error(f"🚨 PRODUCTION BUILD BLOCKED: Missing {len(missing_configs)} critical configurations", 
                                 f"[CRITICAL] PRODUCTION BUILD BLOCKED: Missing {len(missing_configs)} critical configurations")
            self.logger.error(f"Missing configurations: {missing_configs}")
        elif env == 'development':
            self.safe_logger.info("🔧 DEVELOPMENT MODE: Configuration validation bypassed", 
                                "[DEV] DEVELOPMENT MODE: Configuration validation bypassed")
        
        self.logger.info("=" * 50)
        if overall_success:
            self.safe_logger.info("🎉 STARTUP VALIDATION: ALL CRITICAL CHECKS PASSED", 
                                "[SUCCESS] STARTUP VALIDATION: ALL CRITICAL CHECKS PASSED")
        else:
            self.safe_logger.error("🚨 STARTUP VALIDATION: CRITICAL FAILURES DETECTED", 
                                 "[FAILURE] STARTUP VALIDATION: CRITICAL FAILURES DETECTED")
        self.logger.info("=" * 50)
        
        # Save validation results to file
        self._save_validation_results(validation_results, overall_success)
        
        return overall_success
    
    def _save_validation_results(self, results: Dict[str, Any], success: bool) -> None:
        """Save validation results to JSON file for debugging"""
        validation_file = self.log_dir / "startup_validation.json"
        validation_data = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'results': results,
            'environment': os.getenv('ENVIRONMENT', 'development')
        }
        
        with open(validation_file, 'w') as f:
            json.dump(validation_data, f, indent=2)
    
    def log_component_health(self, component_name: str, status: str, 
                           details: Optional[str] = None, 
                           response_time: Optional[float] = None) -> None:
        """Log component health status with detailed information"""
        icon = "✅" if status == "HEALTHY" else "❌" if status == "FAILED" else "⚠️"
        ascii_icon = "[OK]" if status == "HEALTHY" else "[FAIL]" if status == "FAILED" else "[WARN]"
        
        unicode_message = f"{icon} {component_name}: {status}"
        ascii_message = f"{ascii_icon} {component_name}: {status}"
        
        if response_time:
            unicode_message += f" ({response_time:.2f}s)"
            ascii_message += f" ({response_time:.2f}s)"
        if details:
            unicode_message += f" - {details}"
            ascii_message += f" - {details}"
        
        if status == "HEALTHY":
            self.safe_logger.info(unicode_message, ascii_message)
        elif status == "FAILED":
            self.safe_logger.error(unicode_message, ascii_message)
        else:
            self.safe_logger.warning(unicode_message, ascii_message)
    
    def log_llm_provider_status(self, provider: str, model: str, 
                              status: str, response_time: Optional[float] = None,
                              error: Optional[str] = None) -> None:
        """Log LLM provider specific status with performance metrics"""
        icon = "🟢" if status == "OPERATIONAL" else "🔴" if status == "FAILED" else "🟡"
        ascii_icon = "[OPERATIONAL]" if status == "OPERATIONAL" else "[FAILED]" if status == "FAILED" else "[WARNING]"
        
        unicode_message = f"{icon} LLM Provider {provider} ({model}): {status}"
        ascii_message = f"{ascii_icon} LLM Provider {provider} ({model}): {status}"
        
        if response_time:
            unicode_message += f" - {response_time:.2f}s avg response"
            ascii_message += f" - {response_time:.2f}s avg response"
        if error:
            unicode_message += f" - Error: {error}"
            ascii_message += f" - Error: {error}"
        
        if status == "OPERATIONAL":
            self.safe_logger.info(unicode_message, ascii_message)
        else:
            self.safe_logger.error(unicode_message, ascii_message)
    
    def log_critical_error(self, component: str, error: Exception, 
                          context: Optional[Dict[str, Any]] = None) -> None:
        """Log critical errors with full context for debugging"""
        self.safe_logger.error("🚨 CRITICAL ERROR DETECTED 🚨", 
                             "[CRITICAL] ERROR DETECTED")
        self.logger.error(f"Component: {component}")
        self.logger.error(f"Error Type: {type(error).__name__}")
        self.logger.error(f"Error Message: {str(error)}")
        
        if context:
            self.logger.error("Context Information:")
            for key, value in context.items():
                self.logger.error(f"  {key}: {value}")
        
        # Log full stack trace
        import traceback
        self.logger.error("Full Stack Trace:")
        self.logger.error(traceback.format_exc())
    
    def log_performance_metrics(self, operation: str, execution_time: float, 
                              success: bool, additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metrics for monitoring and optimization"""
        status = "SUCCESS" if success else "FAILED"
        icon = "⚡" if success else "🐌"
        ascii_icon = "[FAST]" if success else "[SLOW]"
        
        unicode_message = f"{icon} Performance - {operation}: {execution_time:.3f}s ({status})"
        ascii_message = f"{ascii_icon} Performance - {operation}: {execution_time:.3f}s ({status})"
        
        if additional_metrics:
            metric_parts = []
            for key, value in additional_metrics.items():
                metric_parts.append(f"{key}={value}")
            metric_info = f" - {', '.join(metric_parts)}"
            unicode_message += metric_info
            ascii_message += metric_info
        
        self.safe_logger.info(unicode_message, ascii_message)

# Global logger instance
production_logger = ProductionLogger()

# Convenience functions for easy import
def get_safe_logger(name: str) -> SafeLogger:
    """Get a safe logger instance"""
    logger = logging.getLogger(name)
    return SafeLogger(logger)

def log_startup_validation() -> bool:
    """Convenience function for startup validation"""
    return production_logger.log_startup_validation()

def log_component_health(component: str, status: str, details: Optional[str] = None, 
                        response_time: Optional[float] = None) -> None:
    """Convenience function for component health logging"""
    production_logger.log_component_health(component, status, details, response_time)

def log_llm_provider_status(provider: str, model: str, status: str, 
                           response_time: Optional[float] = None, error: Optional[str] = None) -> None:
    """Convenience function for LLM provider status logging"""
    production_logger.log_llm_provider_status(provider, model, status, response_time, error)

def log_critical_error(component: str, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function for critical error logging"""
    production_logger.log_critical_error(component, error, context)

def log_performance_metrics(operation: str, execution_time: float, success: bool, 
                           additional_metrics: Optional[Dict[str, Any]] = None) -> None:
    """Convenience function for performance metrics logging"""
    production_logger.log_performance_metrics(operation, execution_time, success, additional_metrics)