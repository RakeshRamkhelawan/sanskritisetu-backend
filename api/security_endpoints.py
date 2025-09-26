#!/usr/bin/env python3
"""
Security API Endpoints
Provides security management and monitoring endpoints
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
import secrets

from ..security.security_manager import get_security_manager, SecurityLevel, ThreatLevel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/security", tags=["security"])

# Pydantic models
class LoginRequest(BaseModel):
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")

class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry in seconds")
    security_level: str = Field(..., description="Security clearance level")

class SecurityEventResponse(BaseModel):
    event_id: str
    event_type: str
    severity: str
    source_ip: str
    user_id: Optional[str]
    endpoint: str
    description: str
    timestamp: datetime

class SecurityReportResponse(BaseModel):
    total_events: int
    recent_events_24h: int
    blocked_ips: int
    threat_distribution: Dict[str, int]
    top_threat_sources: List[Dict[str, Any]]
    security_status: str
    last_updated: str

class RateLimitStatus(BaseModel):
    endpoint: str
    current_requests_per_minute: int
    limit_per_minute: int
    current_requests_per_hour: int
    limit_per_hour: int
    status: str

@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, http_request: Request):
    """
    Authenticate user and return JWT token
    """
    security_manager = get_security_manager()
    client_ip = http_request.client.host if http_request.client else "unknown"
    
    try:
        # In production, this would validate against a user database
        # For development, we'll use simple validation
        if request.username == "admin" and request.password == "admin":
            user_id = "admin_user"
            security_level = SecurityLevel.ADMIN
        elif request.username == "user" and request.password == "user":
            user_id = "standard_user"
            security_level = SecurityLevel.AUTHORIZED
        else:
            # Log failed attempt
            security_manager._log_security_event(
                "login_failed",
                ThreatLevel.MEDIUM,
                client_ip,
                request.username,
                "/auth/login",
                f"Failed login attempt for user: {request.username}"
            )
            
            # Track failed attempts
            if client_ip not in security_manager.failed_attempts:
                security_manager.failed_attempts[client_ip] = []
            security_manager.failed_attempts[client_ip].append(datetime.utcnow())
            
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Generate token
        token = security_manager.generate_token(user_id, security_level)
        
        # Log successful login
        security_manager._log_security_event(
            "login_successful",
            ThreatLevel.LOW,
            client_ip,
            user_id,
            "/auth/login",
            f"Successful login for user: {user_id}"
        )
        
        return TokenResponse(
            access_token=token,
            expires_in=24 * 3600,  # 24 hours
            security_level=security_level.value
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        security_manager._log_security_event(
            "login_error",
            ThreatLevel.HIGH,
            client_ip,
            request.username,
            "/auth/login",
            f"Login system error: {str(e)}"
        )
        raise HTTPException(status_code=500, detail="Authentication system error")

@router.get("/report", response_model=SecurityReportResponse)
async def get_security_report(request: Request):
    """
    Get comprehensive security report
    Requires admin access
    """
    try:
        security_manager = get_security_manager()
        report = security_manager.get_security_report()
        
        return SecurityReportResponse(**report)
        
    except Exception as e:
        logger.error(f"Failed to generate security report: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate security report")

@router.get("/events", response_model=List[SecurityEventResponse])
async def get_security_events(
    limit: int = 50,
    severity: Optional[str] = None,
    event_type: Optional[str] = None,
    hours: int = 24,
    request: Request = None
):
    """
    Get recent security events
    Requires admin access
    """
    try:
        security_manager = get_security_manager()
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        events = [
            event for event in security_manager.security_events
            if event.timestamp > cutoff
        ]
        
        # Filter by severity
        if severity:
            try:
                severity_level = ThreatLevel[severity.upper()]
                events = [event for event in events if event.severity == severity_level]
            except KeyError:
                raise HTTPException(status_code=400, detail=f"Invalid severity level: {severity}")
        
        # Filter by event type
        if event_type:
            events = [event for event in events if event.event_type == event_type]
        
        # Sort by timestamp (newest first)
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Limit results
        events = events[:limit]
        
        return [
            SecurityEventResponse(
                event_id=event.event_id,
                event_type=event.event_type,
                severity=event.severity.name,
                source_ip=event.source_ip,
                user_id=event.user_id,
                endpoint=event.endpoint,
                description=event.description,
                timestamp=event.timestamp
            )
            for event in events
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get security events: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve security events")

@router.get("/rate-limits", response_model=List[RateLimitStatus])
async def get_rate_limit_status(request: Request):
    """
    Get current rate limit status
    Requires admin access
    """
    try:
        security_manager = get_security_manager()
        client_ip = request.client.host if request.client else "unknown"
        
        status_list = []
        for rule in security_manager.rate_limit_rules:
            key = f"{client_ip}:{rule.endpoint_pattern}"
            
            if key in security_manager.rate_limits:
                now = datetime.utcnow()
                recent_requests = [
                    req for req in security_manager.rate_limits[key]
                    if req > now - timedelta(hours=1)
                ]
                minute_requests = len([
                    req for req in recent_requests
                    if req > now - timedelta(minutes=1)
                ])
                hour_requests = len(recent_requests)
            else:
                minute_requests = 0
                hour_requests = 0
            
            # Determine status
            if minute_requests >= rule.requests_per_minute or hour_requests >= rule.requests_per_hour:
                status = "exceeded"
            elif minute_requests >= rule.requests_per_minute * 0.8:
                status = "warning"
            else:
                status = "ok"
            
            status_list.append(RateLimitStatus(
                endpoint=rule.endpoint_pattern,
                current_requests_per_minute=minute_requests,
                limit_per_minute=rule.requests_per_minute,
                current_requests_per_hour=hour_requests,
                limit_per_hour=rule.requests_per_hour,
                status=status
            ))
        
        return status_list
        
    except Exception as e:
        logger.error(f"Failed to get rate limit status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve rate limit status")

@router.post("/block-ip")
async def block_ip(ip_address: str, reason: str = "Manual block", request: Request = None):
    """
    Manually block an IP address
    Requires admin access
    """
    try:
        security_manager = get_security_manager()
        client_ip = request.client.host if request.client else "unknown"
        
        security_manager.blocked_ips.add(ip_address)
        
        # Log the blocking action
        security_manager._log_security_event(
            "ip_blocked_manual",
            ThreatLevel.HIGH,
            client_ip,
            "admin",  # Assume admin user
            "/security/block-ip",
            f"Manually blocked IP {ip_address}: {reason}"
        )
        
        return {"message": f"IP address {ip_address} has been blocked", "reason": reason}
        
    except Exception as e:
        logger.error(f"Failed to block IP: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to block IP address")

@router.post("/unblock-ip")
async def unblock_ip(ip_address: str, request: Request = None):
    """
    Unblock an IP address
    Requires admin access
    """
    try:
        security_manager = get_security_manager()
        client_ip = request.client.host if request.client else "unknown"
        
        if ip_address in security_manager.blocked_ips:
            security_manager.blocked_ips.remove(ip_address)
            
            # Log the unblocking action
            security_manager._log_security_event(
                "ip_unblocked",
                ThreatLevel.LOW,
                client_ip,
                "admin",  # Assume admin user
                "/security/unblock-ip",
                f"Manually unblocked IP {ip_address}"
            )
            
            return {"message": f"IP address {ip_address} has been unblocked"}
        else:
            return {"message": f"IP address {ip_address} was not blocked"}
        
    except Exception as e:
        logger.error(f"Failed to unblock IP: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to unblock IP address")

@router.get("/health")
async def security_health():
    """
    Security system health check
    Public endpoint
    """
    try:
        security_manager = get_security_manager()
        
        # Basic health metrics
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        
        recent_events = [
            event for event in security_manager.security_events
            if event.timestamp > last_hour
        ]
        
        critical_events = [
            event for event in recent_events
            if event.severity == ThreatLevel.CRITICAL
        ]
        
        health_status = "healthy"
        if len(critical_events) > 0:
            health_status = "critical"
        elif len([e for e in recent_events if e.severity == ThreatLevel.HIGH]) > 5:
            health_status = "warning"
        
        return {
            "status": health_status,
            "timestamp": now.isoformat(),
            "blocked_ips": len(security_manager.blocked_ips),
            "recent_events": len(recent_events),
            "critical_events": len(critical_events),
            "security_features": {
                "authentication": True,
                "rate_limiting": True,
                "threat_detection": True,
                "ip_blocking": True
            }
        }
        
    except Exception as e:
        logger.error(f"Security health check failed: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

@router.post("/cleanup")
async def cleanup_security_data():
    """
    Cleanup old security data
    Requires admin access
    """
    try:
        security_manager = get_security_manager()
        security_manager.cleanup_old_data()
        
        return {"message": "Security data cleanup completed successfully"}
        
    except Exception as e:
        logger.error(f"Security cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Security cleanup failed")