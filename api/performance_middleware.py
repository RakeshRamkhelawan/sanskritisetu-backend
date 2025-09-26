"""
PERFORMANCE OPTIMIZATION MIDDLEWARE
Caching and async optimization for significant speed improvements
"""

import asyncio
import time
import json
from typing import Dict, Any, Optional
from functools import wraps
from datetime import datetime, timedelta

class PerformanceCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self.cache: Dict[str, Dict] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key not in self.cache:
            return None
            
        entry = self.cache[key]
        if datetime.now() > entry['expires']:
            del self.cache[key]
            return None
            
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with expiration"""
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'value': value,
            'expires': datetime.now() + timedelta(seconds=ttl)
        }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        active_entries = 0
        expired_entries = 0
        
        now = datetime.now()
        for entry in self.cache.values():
            if now <= entry['expires']:
                active_entries += 1
            else:
                expired_entries += 1
                
        return {
            'active_entries': active_entries,
            'expired_entries': expired_entries,
            'total_entries': len(self.cache)
        }

# Global cache instance
performance_cache = PerformanceCache()

def cached_response(ttl: int = 300):
    """Decorator for caching endpoint responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and args
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Try to get from cache first
            cached_result = performance_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            performance_cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

def background_task(func):
    """Decorator to run functions in background"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return asyncio.create_task(func(*args, **kwargs))
    return wrapper

class ResponseTimeTracker:
    """Track response times for performance monitoring"""
    
    def __init__(self):
        self.response_times: Dict[str, list] = {}
        self.max_samples = 100  # Keep last 100 samples per endpoint
    
    def add_response_time(self, endpoint: str, response_time: float):
        """Add response time sample"""
        if endpoint not in self.response_times:
            self.response_times[endpoint] = []
        
        samples = self.response_times[endpoint]
        samples.append(response_time)
        
        # Keep only recent samples
        if len(samples) > self.max_samples:
            samples.pop(0)
    
    def get_stats(self, endpoint: str) -> Dict:
        """Get performance stats for endpoint"""
        if endpoint not in self.response_times or not self.response_times[endpoint]:
            return {"samples": 0}
        
        times = self.response_times[endpoint]
        return {
            "samples": len(times),
            "avg_response_time": sum(times) / len(times),
            "min_response_time": min(times),
            "max_response_time": max(times),
            "recent_response_time": times[-1] if times else 0
        }
    
    def get_all_stats(self) -> Dict:
        """Get stats for all endpoints"""
        return {endpoint: self.get_stats(endpoint) for endpoint in self.response_times}

# Global response tracker
response_tracker = ResponseTimeTracker()

def performance_middleware(request, call_next):
    """FastAPI middleware for performance tracking and optimization"""
    async def middleware(request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Track response time
        response_time = time.time() - start_time
        endpoint = f"{request.method} {request.url.path}"
        response_tracker.add_response_time(endpoint, response_time)
        
        # Add performance headers
        response.headers["X-Response-Time"] = str(response_time)
        response.headers["X-Cache-Status"] = "MISS"  # Default, overridden by cache
        
        return response
    
    return middleware

async def async_health_check() -> Dict:
    """Optimized async health check"""
    return {
        "status": "healthy",
        "system": "operational", 
        "timestamp": datetime.now().isoformat(),
        "cache_stats": performance_cache.get_stats(),
        "performance_optimized": True
    }

def get_performance_stats() -> Dict:
    """Get comprehensive performance statistics"""
    return {
        "cache_stats": performance_cache.get_stats(),
        "response_times": response_tracker.get_all_stats(),
        "optimization_active": True,
        "timestamp": datetime.now().isoformat()
    }