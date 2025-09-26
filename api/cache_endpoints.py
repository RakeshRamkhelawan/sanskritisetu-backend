"""
Cache Test Endpoints - Production Implementation
API endpoints voor caching integration tests zoals gespecificeerd in FASE 1 STAP 1.4
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import time
import asyncio
from core.caching import cached, set_cache, get_cache_value, delete_cache
from core.auth import UserResponse, Permission, require_permission

router = APIRouter(prefix="/cache", tags=["caching"])


@router.get("/test-decorator")
@cached(ttl=60, key_prefix="test_endpoint")
async def cached_test_endpoint(value: str = "test"):
    """
    Test endpoint met @cached decorator zoals gespecificeerd in plan
    Voor integratietest van decorator op async def API-endpoint
    """
    # Simulate expensive operation
    await asyncio.sleep(0.1)

    return {
        "message": f"Expensive operation completed for: {value}",
        "timestamp": time.time(),
        "computed_value": hash(value) % 10000
    }


@router.get("/test-manual")
async def manual_cache_test(key: str = "test_key", value: str = "test_value"):
    """
    Test endpoint voor manual cache operaties
    """
    # Try to get from cache first
    cached_result = await get_cache_value(key)

    if cached_result is not None:
        return {
            "source": "cache",
            "key": key,
            "value": cached_result,
            "cache_hit": True
        }

    # Simulate expensive operation
    await asyncio.sleep(0.1)
    computed_value = f"computed_{value}_{time.time()}"

    # Store in cache
    await set_cache(key, computed_value, ttl=300)

    return {
        "source": "computed",
        "key": key,
        "value": computed_value,
        "cache_hit": False
    }


@router.post("/set")
async def set_cache_value(key: str, value: str, ttl: int = 3600):
    """
    Set cache value endpoint
    """
    result = await set_cache(key, value, ttl)
    return {
        "success": result,
        "key": key,
        "ttl": ttl
    }


@router.get("/get/{key}")
async def get_cache_endpoint(key: str):
    """
    Get cache value endpoint
    """
    value = await get_cache_value(key)
    return {
        "key": key,
        "value": value,
        "found": value is not None
    }


@router.delete("/delete/{key}")
async def delete_cache_endpoint(key: str):
    """
    Delete cache value endpoint
    """
    result = await delete_cache(key)
    return {
        "success": result,
        "key": key
    }


@router.get("/admin/stats", dependencies=[Depends(require_permission(Permission.ADMIN))])
async def get_cache_stats():
    """
    Admin endpoint voor cache statistics
    """
    from core.caching import get_cache

    cache = await get_cache()
    stats = await cache.get_stats()

    return {
        "cache_stats": stats,
        "endpoint": "admin_cache_stats"
    }


@router.get("/performance-test")
async def cache_performance_test(iterations: int = 100):
    """
    Performance test endpoint voor caching
    """
    # Test cache performance
    start_time = time.time()

    # Set multiple values
    for i in range(iterations):
        await set_cache(f"perf_test_{i}", f"value_{i}", ttl=60)

    set_time = time.time() - start_time

    # Get multiple values
    start_time = time.time()
    hits = 0

    for i in range(iterations):
        value = await get_cache_value(f"perf_test_{i}")
        if value is not None:
            hits += 1

    get_time = time.time() - start_time

    return {
        "iterations": iterations,
        "set_time_seconds": round(set_time, 4),
        "get_time_seconds": round(get_time, 4),
        "cache_hits": hits,
        "hit_rate": round(hits / iterations * 100, 2) if iterations > 0 else 0,
        "avg_set_time_ms": round((set_time / iterations) * 1000, 2) if iterations > 0 else 0,
        "avg_get_time_ms": round((get_time / iterations) * 1000, 2) if iterations > 0 else 0
    }