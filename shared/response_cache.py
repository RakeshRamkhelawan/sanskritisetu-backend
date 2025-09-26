"""
Response Caching System for LLM Responses
Provides intelligent caching to reduce LLM API calls and improve response times
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for different types of responses"""
    AGGRESSIVE = "aggressive"  # Cache everything, long TTL
    CONSERVATIVE = "conservative"  # Cache only stable responses, medium TTL
    MINIMAL = "minimal"  # Cache only expensive operations, short TTL
    DISABLED = "disabled"  # No caching


@dataclass
class CacheEntry:
    """Cached response entry"""
    key: str
    content: Any
    created_at: datetime
    expires_at: datetime
    hit_count: int
    task_type: str
    content_hash: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CacheStats:
    """Cache performance statistics"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    avg_response_time_cached: float
    avg_response_time_uncached: float
    memory_usage_mb: float
    entries_count: int


class LLMResponseCache:
    """Intelligent caching system for LLM responses"""
    
    def __init__(
        self, 
        strategy: CacheStrategy = CacheStrategy.CONSERVATIVE,
        max_entries: int = 1000,
        default_ttl_hours: int = 24
    ):
        self.strategy = strategy
        self.max_entries = max_entries
        self.default_ttl_hours = default_ttl_hours
        
        # In-memory cache (Redis could be used in production)
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU eviction
        
        # Cache configuration by task type
        self._task_ttl_config = self._initialize_ttl_config()
        self._cacheable_tasks = self._initialize_cacheable_tasks()
        
        # Metrics
        self._total_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cached_response_times: List[float] = []
        self._uncached_response_times: List[float] = []
        
        logger.info(f"LLM Response Cache initialized with {strategy.value} strategy")
    
    def _initialize_ttl_config(self) -> Dict[str, int]:
        """Initialize TTL configuration by task type (in hours)"""
        return {
            "strategic_analysis": 48,  # Strategic analysis can be cached longer
            "vision_synthesis": 72,   # Vision synthesis is usually stable
            "task_decomposition": 24,  # Task decomposition may vary more
            "general_consultation": 6,  # Conversational responses expire quickly
            "approval_generation": 12,  # Approval requests have medium stability
            "risk_assessment": 36,     # Risk assessments are fairly stable
        }
    
    def _initialize_cacheable_tasks(self) -> Dict[str, bool]:
        """Initialize which task types are cacheable by strategy"""
        base_cacheable = {
            "strategic_analysis": True,
            "vision_synthesis": True,
            "task_decomposition": True,
            "general_consultation": False,  # Usually personalized
            "approval_generation": False,   # Usually time-sensitive
            "risk_assessment": True,
        }
        
        if self.strategy == CacheStrategy.AGGRESSIVE:
            return {k: True for k in base_cacheable}
        elif self.strategy == CacheStrategy.CONSERVATIVE:
            return base_cacheable
        elif self.strategy == CacheStrategy.MINIMAL:
            return {
                "strategic_analysis": True,
                "vision_synthesis": True,
                "task_decomposition": False,
                "general_consultation": False,
                "approval_generation": False,
                "risk_assessment": False,
            }
        else:  # DISABLED
            return {k: False for k in base_cacheable}
    
    def _generate_cache_key(self, prompt: str, task_type: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique cache key for the request"""
        # Normalize input for consistent caching
        normalized_prompt = prompt.strip().lower()
        
        # Include relevant metadata that affects response
        relevant_metadata = {
            k: v for k, v in metadata.items() 
            if k in ["context", "objectives", "requirements", "constraints"]
        }
        
        # Create hash from prompt, task type, and relevant metadata
        key_data = {
            "prompt": normalized_prompt,
            "task_type": task_type,
            "metadata": relevant_metadata
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()
        
        return f"{task_type}:{key_hash[:16]}"
    
    def _generate_content_hash(self, content: Any) -> str:
        """Generate hash of response content for integrity checking"""
        content_string = json.dumps(content, sort_keys=True) if isinstance(content, dict) else str(content)
        return hashlib.md5(content_string.encode()).hexdigest()[:12]
    
    async def get_cached_response(
        self, 
        prompt: str, 
        task_type: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Retrieve cached response if available and valid"""
        if self.strategy == CacheStrategy.DISABLED:
            return None
        
        if not self._is_cacheable(task_type):
            return None
        
        metadata = metadata or {}
        cache_key = self._generate_cache_key(prompt, task_type, metadata)
        
        self._total_requests += 1
        start_time = time.time()
        
        # Check if entry exists and is valid
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            
            if datetime.utcnow() <= entry.expires_at:
                # Cache hit
                entry.hit_count += 1
                self._update_access_order(cache_key)
                self._cache_hits += 1
                
                response_time = time.time() - start_time
                self._cached_response_times.append(response_time)
                
                logger.debug(f"Cache HIT for key: {cache_key}")
                return entry.content
            else:
                # Expired entry
                del self._cache[cache_key]
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
        
        # Cache miss
        self._cache_misses += 1
        logger.debug(f"Cache MISS for key: {cache_key}")
        return None
    
    async def cache_response(
        self, 
        prompt: str, 
        task_type: str, 
        response: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Cache a response for future use"""
        if self.strategy == CacheStrategy.DISABLED:
            return False
        
        if not self._is_cacheable(task_type):
            return False
        
        metadata = metadata or {}
        cache_key = self._generate_cache_key(prompt, task_type, metadata)
        
        # Calculate expiration time
        ttl_hours = self._task_ttl_config.get(task_type, self.default_ttl_hours)
        expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
        
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            content=response,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            hit_count=0,
            task_type=task_type,
            content_hash=self._generate_content_hash(response),
            metadata=metadata
        )
        
        # Check if we need to evict entries
        await self._ensure_capacity()
        
        # Store entry
        self._cache[cache_key] = entry
        self._update_access_order(cache_key)
        
        logger.debug(f"Cached response for key: {cache_key} (expires: {expires_at})")
        return True
    
    def _is_cacheable(self, task_type: str) -> bool:
        """Check if a task type is cacheable given current strategy"""
        return self._cacheable_tasks.get(task_type, False)
    
    def _update_access_order(self, cache_key: str):
        """Update LRU access order"""
        if cache_key in self._access_order:
            self._access_order.remove(cache_key)
        self._access_order.append(cache_key)
    
    async def _ensure_capacity(self):
        """Ensure cache doesn't exceed maximum capacity"""
        while len(self._cache) >= self.max_entries:
            # Evict least recently used entry
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
                    logger.debug(f"Evicted cache entry: {oldest_key}")
    
    async def clear_cache(self, task_type: Optional[str] = None):
        """Clear all cache entries or entries of specific task type"""
        if task_type:
            keys_to_remove = [
                key for key, entry in self._cache.items() 
                if entry.task_type == task_type
            ]
            for key in keys_to_remove:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            logger.info(f"Cleared {len(keys_to_remove)} cache entries for task type: {task_type}")
        else:
            self._cache.clear()
            self._access_order.clear()
            logger.info("Cleared all cache entries")
    
    async def cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, entry in self._cache.items() 
            if current_time > entry.expires_at
        ]
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> CacheStats:
        """Get cache performance statistics"""
        hit_rate = self._cache_hits / max(self._total_requests, 1)
        
        avg_cached_time = (
            sum(self._cached_response_times) / max(len(self._cached_response_times), 1)
            if self._cached_response_times else 0.0
        )
        
        avg_uncached_time = (
            sum(self._uncached_response_times) / max(len(self._uncached_response_times), 1)
            if self._uncached_response_times else 0.0
        )
        
        # Estimate memory usage (rough approximation)
        cache_data = []
        for entry in self._cache.values():
            entry_dict = asdict(entry)
            # Convert datetime objects to strings for JSON serialization
            entry_dict['created_at'] = entry_dict['created_at'].isoformat()
            entry_dict['expires_at'] = entry_dict['expires_at'].isoformat()
            cache_data.append(entry_dict)
        
        memory_usage_mb = len(json.dumps(cache_data).encode()) / 1024 / 1024
        
        return CacheStats(
            total_requests=self._total_requests,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            hit_rate=hit_rate,
            avg_response_time_cached=avg_cached_time,
            avg_response_time_uncached=avg_uncached_time,
            memory_usage_mb=memory_usage_mb,
            entries_count=len(self._cache)
        )
    
    def get_cached_entries(self, task_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get information about cached entries"""
        entries = []
        
        for entry in self._cache.values():
            if task_type is None or entry.task_type == task_type:
                entries.append({
                    "key": entry.key,
                    "task_type": entry.task_type,
                    "created_at": entry.created_at.isoformat(),
                    "expires_at": entry.expires_at.isoformat(),
                    "hit_count": entry.hit_count,
                    "content_hash": entry.content_hash,
                    "content_size": len(str(entry.content))
                })
        
        return sorted(entries, key=lambda x: x["created_at"], reverse=True)
    
    async def record_uncached_response_time(self, response_time: float):
        """Record response time for uncached request (for metrics)"""
        self._uncached_response_times.append(response_time)
        # Keep only recent times to avoid memory bloat
        if len(self._uncached_response_times) > 1000:
            self._uncached_response_times = self._uncached_response_times[-500:]
    
    def set_strategy(self, strategy: CacheStrategy):
        """Change caching strategy"""
        if strategy != self.strategy:
            logger.info(f"Changing cache strategy from {self.strategy.value} to {strategy.value}")
            self.strategy = strategy
            self._cacheable_tasks = self._initialize_cacheable_tasks()
            
            # Clear cache if moving to disabled
            if strategy == CacheStrategy.DISABLED:
                asyncio.create_task(self.clear_cache())
    
    def is_healthy(self) -> Dict[str, Any]:
        """Check cache health status"""
        stats = self.get_stats()
        current_time = datetime.utcnow()
        
        health_issues = []
        
        # Check hit rate
        if stats.hit_rate < 0.1 and stats.total_requests > 100:
            health_issues.append("Low cache hit rate (<10%)")
        
        # Check memory usage
        if stats.memory_usage_mb > 100:  # 100MB threshold
            health_issues.append("High memory usage (>100MB)")
        
        # Check for too many expired entries
        expired_count = sum(
            1 for entry in self._cache.values() 
            if current_time > entry.expires_at
        )
        
        if expired_count > len(self._cache) * 0.3:  # >30% expired
            health_issues.append("High percentage of expired entries (>30%)")
        
        return {
            "healthy": len(health_issues) == 0,
            "issues": health_issues,
            "stats": asdict(stats),
            "strategy": self.strategy.value
        }


# Global cache instance
_response_cache = LLMResponseCache()

def get_response_cache() -> LLMResponseCache:
    """Get the global response cache instance"""
    return _response_cache


# Decorator for automatic caching
def cached_llm_response(task_type: str):
    """Decorator to automatically cache LLM responses"""
    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Extract prompt from arguments (assumes first arg is prompt-like)
            prompt = str(args[0]) if args else ""
            metadata = kwargs.get("metadata", {})
            
            # Try to get cached response
            cached = await get_response_cache().get_cached_response(prompt, task_type, metadata)
            if cached is not None:
                return cached
            
            # Call original function and cache result
            start_time = time.time()
            result = await func(self, *args, **kwargs)
            response_time = time.time() - start_time
            
            # Record uncached response time
            await get_response_cache().record_uncached_response_time(response_time)
            
            # Cache the result
            await get_response_cache().cache_response(prompt, task_type, result, metadata)
            
            return result
        
        return wrapper
    return decorator