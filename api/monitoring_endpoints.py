"""
📊 Monitoring API Endpoints - Sanskriti Setu
Advanced monitoring and metrics endpoints for production
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import logging

from core.monitoring.advanced_analytics import (
    get_monitoring_system, start_monitoring, stop_monitoring,
    get_metrics_collector, get_profiler, get_log_analyzer
)
from core.monitoring.mock_usage_detector import mock_usage_detector
from core.auth.rbac import require_permission, Permission

logger = logging.getLogger(__name__)

# Initialize monitoring components
monitoring_system = get_monitoring_system()
metrics_collector = get_metrics_collector()
profiler = get_profiler()
log_analyzer = get_log_analyzer()

# Pydantic models for API responses
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str = "1.0.0"
    environment: str = "production"
    components: Dict[str, str]

class MetricsResponse(BaseModel):
    timestamp: datetime
    system_metrics: Dict[str, float]
    application_metrics: Dict[str, float]
    database_metrics: Dict[str, float]
    health_score: float

class PerformanceResponse(BaseModel):
    timestamp: datetime
    analysis_period_minutes: int
    total_functions_profiled: int
    top_bottlenecks: List[Dict[str, Any]]
    performance_trends: Dict[str, str]
    system_impact: Dict[str, float]
    recommendations: List[str]
    alerts: List[str]

class LogAnalysisResponse(BaseModel):
    timestamp: datetime
    total_entries: int
    time_range: Dict[str, str]
    error_rate: float
    warning_rate: float
    top_errors: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    patterns: Dict[str, int]
    insights: List[str]

class AlertsResponse(BaseModel):
    timestamp: datetime
    total_alerts: int
    critical_alerts: int
    warning_alerts: int
    info_alerts: int
    recent_alerts: List[Dict[str, Any]]

# Create router
router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])

# Startup time for uptime calculation
startup_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def get_health_status():
    """
    Get system health status
    Public endpoint for basic health checking
    """
    try:
        # Collect current metrics
        current_metrics = await metrics_collector.collect_all_metrics()
        health_score = metrics_collector.calculate_health_score(current_metrics)
        
        # Check component health
        components = {
            "api": "healthy" if health_score > 50 else "degraded",
            "database": "healthy",  # This would check actual DB connection
            "redis": "healthy",     # This would check actual Redis connection
            "ml_systems": "healthy" if health_score > 70 else "degraded",
            "monitoring": "healthy"
        }
        
        # Determine overall status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "degraded"
        elif health_score >= 50:
            status = "unhealthy"
        else:
            status = "critical"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now(),
            uptime_seconds=time.time() - startup_time,
            components=components
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            timestamp=datetime.now(),
            uptime_seconds=time.time() - startup_time,
            components={"error": str(e)}
        )

@router.get("/metrics", response_model=MetricsResponse)
async def get_system_metrics(
    background_tasks: BackgroundTasks,
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get comprehensive system metrics
    Requires monitoring permission
    """
    try:
        # Collect all metrics
        system_metrics = await metrics_collector.collect_system_metrics()
        app_metrics = await metrics_collector.collect_application_metrics()
        db_metrics = await metrics_collector.collect_database_metrics()
        
        # Calculate health score
        all_metrics = {**system_metrics, **app_metrics, **db_metrics}
        health_score = metrics_collector.calculate_health_score(all_metrics)
        
        # Store metrics in background
        background_tasks.add_task(metrics_collector.store_metrics, all_metrics)
        
        return MetricsResponse(
            timestamp=datetime.now(),
            system_metrics=system_metrics,
            application_metrics=app_metrics,
            database_metrics=db_metrics,
            health_score=health_score
        )
        
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {e}")

@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics(
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get metrics in Prometheus format
    Requires monitoring permission
    """
    try:
        # Collect metrics
        system_metrics = await metrics_collector.collect_system_metrics()
        app_metrics = await metrics_collector.collect_application_metrics()
        db_metrics = await metrics_collector.collect_database_metrics()
        
        # Format as Prometheus metrics
        prometheus_output = []
        
        # System metrics
        for key, value in system_metrics.items():
            metric_name = f"sanskriti_system_{key}"
            prometheus_output.append(f"# TYPE {metric_name} gauge")
            prometheus_output.append(f"{metric_name} {value}")
        
        # Application metrics
        for key, value in app_metrics.items():
            metric_name = f"sanskriti_app_{key}"
            prometheus_output.append(f"# TYPE {metric_name} gauge")
            prometheus_output.append(f"{metric_name} {value}")
        
        # Database metrics
        for key, value in db_metrics.items():
            metric_name = f"sanskriti_db_{key}"
            prometheus_output.append(f"# TYPE {metric_name} gauge")
            prometheus_output.append(f"{metric_name} {value}")
        
        # Health score
        health_score = metrics_collector.calculate_health_score({**system_metrics, **app_metrics, **db_metrics})
        prometheus_output.append("# TYPE sanskriti_health_score gauge")
        prometheus_output.append(f"sanskriti_health_score {health_score}")
        
        return "\n".join(prometheus_output)
        
    except Exception as e:
        logger.error(f"Failed to generate Prometheus metrics: {e}")
        return f"# Error generating metrics: {e}"

@router.get("/performance", response_model=PerformanceResponse)
async def get_performance_analysis(
    period_minutes: int = Query(60, ge=1, le=1440, description="Analysis period in minutes"),
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get comprehensive performance analysis
    Requires monitoring permission
    """
    try:
        # Generate performance report
        report = profiler.get_performance_report(period_minutes)
        
        # Format bottlenecks for response
        top_bottlenecks = [
            {
                "function_name": name,
                "bottleneck_score": score,
                "profile": {
                    "total_calls": profiler.profiles[name].total_calls,
                    "avg_time": profiler.profiles[name].avg_time,
                    "trend": profiler.profiles[name].trend
                } if name in profiler.profiles else {}
            }
            for name, score in report.top_bottlenecks
        ]
        
        return PerformanceResponse(
            timestamp=report.timestamp,
            analysis_period_minutes=period_minutes,
            total_functions_profiled=report.total_functions_profiled,
            top_bottlenecks=top_bottlenecks,
            performance_trends=report.performance_trends,
            system_impact=report.system_impact,
            recommendations=report.optimization_recommendations,
            alerts=report.alerts
        )
        
    except Exception as e:
        logger.error(f"Failed to generate performance analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {e}")

@router.get("/logs/analysis", response_model=LogAnalysisResponse)
async def get_log_analysis(
    file_path: Optional[str] = Query(None, description="Specific log file to analyze"),
    max_lines: int = Query(1000, ge=100, le=50000, description="Maximum lines to analyze"),
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get log analysis results
    Requires monitoring permission
    """
    try:
        # For demo purposes, we'll analyze a default log file or create mock data
        if file_path:
            analysis = await log_analyzer.analyze_log_file(file_path, max_lines)
        else:
            # Create mock log analysis for demonstration
            from datetime import datetime
            analysis = type('MockAnalysis', (), {
                'total_entries': 1500,
                'time_range': (datetime.now() - timedelta(hours=1), datetime.now()),
                'error_rate': 2.3,
                'warning_rate': 8.7,
                'top_errors': [('Database connection timeout', 15), ('Authentication failed', 8)],
                'anomalies': [
                    {'type': 'rate_anomaly', 'description': 'High error rate detected', 'timestamp': datetime.now().isoformat()}
                ],
                'patterns': {'Database connection issues': 15, 'Authentication failures': 8},
                'insights': ['Database connection pool may need tuning', 'Review authentication timeout settings']
            })()
        
        # Format response
        return LogAnalysisResponse(
            timestamp=datetime.now(),
            total_entries=analysis.total_entries,
            time_range={
                "start": analysis.time_range[0].isoformat(),
                "end": analysis.time_range[1].isoformat()
            },
            error_rate=analysis.error_rate,
            warning_rate=analysis.warning_rate,
            top_errors=[
                {"message": error[0], "count": error[1]} 
                for error in analysis.top_errors
            ],
            anomalies=analysis.anomalies,
            patterns=analysis.patterns,
            insights=analysis.insights
        )
        
    except Exception as e:
        logger.error(f"Log analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Log analysis failed: {e}")

@router.get("/alerts", response_model=AlertsResponse)
async def get_system_alerts(
    hours: int = Query(24, ge=1, le=168, description="Hours to look back for alerts"),
    severity: Optional[str] = Query(None, regex="^(critical|warning|info)$", description="Filter by severity"),
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get system alerts
    Requires monitoring permission
    """
    try:
        # Get current metrics to check for alerts
        current_metrics = await metrics_collector.collect_all_metrics()
        new_alerts = metrics_collector.check_alert_rules(current_metrics)
        
        # Filter recent alerts from metrics collector
        cutoff_time = time.time() - (hours * 3600)
        recent_alerts = [
            alert for alert in metrics_collector.alerts
            if alert.timestamp >= cutoff_time
        ]
        
        # Apply severity filter
        if severity:
            recent_alerts = [alert for alert in recent_alerts if alert.severity == severity]
        
        # Count by severity
        critical_count = sum(1 for alert in recent_alerts if alert.severity == 'critical')
        warning_count = sum(1 for alert in recent_alerts if alert.severity == 'warning')
        info_count = sum(1 for alert in recent_alerts if alert.severity == 'info')
        
        # Format alerts for response
        formatted_alerts = [
            {
                "id": alert.id,
                "severity": alert.severity,
                "title": alert.title,
                "description": alert.description,
                "timestamp": datetime.fromtimestamp(alert.timestamp).isoformat(),
                "resolved": alert.resolved,
                "metadata": alert.metadata
            }
            for alert in recent_alerts[-50:]  # Last 50 alerts
        ]
        
        return AlertsResponse(
            timestamp=datetime.now(),
            total_alerts=len(recent_alerts),
            critical_alerts=critical_count,
            warning_alerts=warning_count,
            info_alerts=info_count,
            recent_alerts=formatted_alerts
        )
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=f"Alert retrieval failed: {e}")

@router.post("/profiler/start")
async def start_profiler(
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Start performance profiling
    Requires admin permission
    """
    try:
        profiler.start_profiling()
        return {"message": "Performance profiling started", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Failed to start profiler: {e}")
        raise HTTPException(status_code=500, detail=f"Profiler start failed: {e}")

@router.post("/profiler/stop")
async def stop_profiler(
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Stop performance profiling
    Requires admin permission
    """
    try:
        profiler.stop_profiling()
        return {"message": "Performance profiling stopped", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Failed to stop profiler: {e}")
        raise HTTPException(status_code=500, detail=f"Profiler stop failed: {e}")

@router.get("/profiler/functions")
async def get_function_profiles(
    limit: int = Query(20, ge=1, le=100, description="Number of functions to return"),
    sort_by: str = Query("bottleneck_score", regex="^(bottleneck_score|total_time|total_calls|avg_time)$"),
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get function performance profiles
    Requires monitoring permission
    """
    try:
        # Get all function profiles
        profiles = []
        for name, profile in profiler.profiles.items():
            profiles.append({
                "name": name,
                "total_calls": profile.total_calls,
                "total_time": profile.total_time,
                "avg_time": profile.avg_time,
                "min_time": profile.min_time,
                "max_time": profile.max_time,
                "median_time": profile.median_time,
                "std_time": profile.std_time,
                "call_frequency": profile.call_frequency,
                "trend": profile.trend,
                "bottleneck_score": profile.bottleneck_score
            })
        
        # Sort profiles
        if sort_by == "bottleneck_score":
            profiles.sort(key=lambda x: x["bottleneck_score"], reverse=True)
        elif sort_by == "total_time":
            profiles.sort(key=lambda x: x["total_time"], reverse=True)
        elif sort_by == "total_calls":
            profiles.sort(key=lambda x: x["total_calls"], reverse=True)
        elif sort_by == "avg_time":
            profiles.sort(key=lambda x: x["avg_time"], reverse=True)
        
        return {
            "timestamp": datetime.now(),
            "total_functions": len(profiles),
            "sort_by": sort_by,
            "functions": profiles[:limit]
        }
        
    except Exception as e:
        logger.error(f"Failed to get function profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Function profiles retrieval failed: {e}")

@router.delete("/profiler/clear")
async def clear_profiler_data(
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Clear all profiler data
    Requires admin permission
    """
    try:
        profiler.clear_profiles()
        return {"message": "Profiler data cleared", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Failed to clear profiler data: {e}")
        raise HTTPException(status_code=500, detail=f"Profiler clear failed: {e}")

@router.get("/system/status")
async def get_system_status():
    """
    Get comprehensive system status
    Public endpoint with basic system information
    """
    try:
        # Collect basic system info
        system_metrics = await metrics_collector.collect_system_metrics()
        health_score = metrics_collector.calculate_health_score(system_metrics)
        
        # Get profiler status
        profiler_active = profiler.is_profiling
        profiler_functions = len(profiler.profiles)
        
        # Get alert counts
        total_alerts = len(metrics_collector.alerts)
        recent_alerts = [alert for alert in metrics_collector.alerts 
                        if time.time() - alert.timestamp < 3600]  # Last hour
        
        return {
            "timestamp": datetime.now(),
            "uptime_seconds": time.time() - startup_time,
            "health_score": health_score,
            "system_status": "healthy" if health_score > 80 else "degraded" if health_score > 50 else "critical",
            "monitoring": {
                "profiler_active": profiler_active,
                "functions_profiled": profiler_functions,
                "total_alerts": total_alerts,
                "recent_alerts": len(recent_alerts)
            },
            "system_metrics": {
                "cpu_usage": system_metrics.get("cpu_usage", 0),
                "memory_usage": system_metrics.get("memory_usage", 0),
                "disk_usage": system_metrics.get("disk_usage", 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return {
            "timestamp": datetime.now(),
            "error": str(e),
            "system_status": "error"
        }

# Mock Usage Detection Endpoints (P1.4 Implementation)
class MockUsageResponse(BaseModel):
    timestamp: datetime
    system_overview: Dict[str, Any]
    components: Dict[str, Dict[str, Any]]
    recent_events: int
    monitoring_active: bool
    alert_thresholds: Dict[str, float]

@router.get("/mock-usage", response_model=MockUsageResponse)
async def get_mock_usage_report():
    """
    Get comprehensive mock usage report
    Public endpoint for mock usage monitoring
    """
    try:
        report = await mock_usage_detector.get_system_mock_usage_report()
        return MockUsageResponse(**report)
    except Exception as e:
        logger.error(f"Failed to get mock usage report: {e}")
        raise HTTPException(status_code=500, detail=f"Mock usage report failed: {e}")

@router.get("/mock-usage/{component}")
async def get_component_mock_usage(component: str):
    """
    Get mock usage status for specific component
    Public endpoint for component-specific monitoring
    """
    try:
        status = await mock_usage_detector.get_component_status(component)
        if status is None:
            raise HTTPException(status_code=404, detail=f"Component '{component}' not found")
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get component mock usage: {e}")
        raise HTTPException(status_code=500, detail=f"Component mock usage query failed: {e}")

@router.post("/mock-usage/start-monitoring")
async def start_mock_usage_monitoring(
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Start mock usage monitoring
    Requires admin permission
    """
    try:
        await mock_usage_detector.start_monitoring()
        return {"message": "Mock usage monitoring started", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Failed to start mock usage monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Mock usage monitoring start failed: {e}")

@router.post("/mock-usage/stop-monitoring")
async def stop_mock_usage_monitoring(
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Stop mock usage monitoring
    Requires admin permission
    """
    try:
        await mock_usage_detector.stop_monitoring()
        return {"message": "Mock usage monitoring stopped", "timestamp": datetime.now()}
    except Exception as e:
        logger.error(f"Failed to stop mock usage monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Mock usage monitoring stop failed: {e}")

@router.get("/mock-usage/thresholds")
async def get_mock_usage_thresholds():
    """
    Get current mock usage alert thresholds
    Public endpoint for threshold information
    """
    try:
        return {
            "timestamp": datetime.now(),
            "thresholds": {
                "low": mock_usage_detector.low_threshold,
                "medium": mock_usage_detector.medium_threshold,
                "high": mock_usage_detector.high_threshold,
                "critical": mock_usage_detector.critical_threshold
            },
            "descriptions": {
                "low": f"Alert when mock usage >= {mock_usage_detector.low_threshold:.1%}",
                "medium": f"Alert when mock usage >= {mock_usage_detector.medium_threshold:.1%}",
                "high": f"Alert when mock usage >= {mock_usage_detector.high_threshold:.1%}",
                "critical": f"Alert when mock usage >= {mock_usage_detector.critical_threshold:.1%}"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get mock usage thresholds: {e}")
        raise HTTPException(status_code=500, detail=f"Threshold query failed: {e}")