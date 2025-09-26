"""
Admin Endpoints - Production Implementation
Test endpoints voor RBAC verificatie zoals gespecificeerd in FASE 1 STAP 1.3
"""

from fastapi import APIRouter, Depends
from core.auth import UserResponse, Permission, require_permission

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/dashboard", dependencies=[Depends(require_permission(Permission.ADMIN))])
async def admin_dashboard():
    """
    Admin dashboard endpoint - vereist ADMIN permission
    Zoals gespecificeerd in FASE 1 STAP 1.3
    """
    return {
        "message": "Welcome to admin dashboard",
        "status": "success",
        "required_permission": "admin",
        "dashboard_data": {
            "total_users": 42,
            "active_sessions": 8,
            "system_status": "operational"
        }
    }


@router.get("/users", dependencies=[Depends(require_permission(Permission.ADMIN))])
async def list_users():
    """
    List users endpoint - vereist ADMIN permission
    """
    return {
        "message": "User list access granted",
        "users": [
            {"id": 1, "email": "admin@example.com", "role": "admin"},
            {"id": 2, "email": "user@example.com", "role": "user"}
        ]
    }


@router.get("/system", dependencies=[Depends(require_permission(Permission.SYSTEM_ADMIN))])
async def system_admin():
    """
    System admin endpoint - vereist SYSTEM_ADMIN permission
    """
    return {
        "message": "System admin access granted",
        "system_info": {
            "uptime": "24h 30m",
            "memory_usage": "45%",
            "cpu_usage": "12%"
        }
    }


@router.get("/logs", dependencies=[Depends(require_permission(Permission.LOG_ANALYZER))])
async def access_logs():
    """
    Log analyzer endpoint - vereist LOG_ANALYZER permission
    """
    return {
        "message": "Log access granted",
        "recent_logs": [
            {"timestamp": "2025-01-01T10:00:00", "level": "INFO", "message": "User logged in"},
            {"timestamp": "2025-01-01T10:01:00", "level": "WARN", "message": "Failed login attempt"}
        ]
    }


@router.get("/monitor", dependencies=[Depends(require_permission(Permission.SYSTEM_MONITOR))])
async def system_monitor():
    """
    System monitor endpoint - vereist SYSTEM_MONITOR permission
    """
    return {
        "message": "System monitoring access granted",
        "metrics": {
            "response_time": "45ms",
            "error_rate": "0.1%",
            "throughput": "1000 req/min"
        }
    }


@router.get("/public")
async def public_endpoint():
    """
    Public endpoint - geen permission vereist (voor contrast testing)
    """
    return {
        "message": "This is a public endpoint",
        "access": "everyone"
    }