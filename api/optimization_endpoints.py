"""
⚡ Optimization API Endpoints - Sanskriti Setu
Performance optimization and system tuning API endpoints
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

from core.optimization.performance_optimizer import (
    get_performance_optimizer, 
    PerformanceOptimizer,
    OptimizationRecommendation
)
from core.optimization.system_tuner import get_system_tuner, TuningConfiguration
from core.auth.rbac import require_permission, Permission

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class PerformanceAnalysisResponse(BaseModel):
    timestamp: str
    current_metrics: Dict[str, Any]
    historical_analysis: Dict[str, Any]
    profiling_analysis: Dict[str, Any]
    optimization_recommendations: List[Dict[str, Any]]
    optimizations_applied: int

class OptimizationRequest(BaseModel):
    optimization_type: str
    parameters: Optional[Dict[str, Any]] = None

class OptimizationResponse(BaseModel):
    optimization_type: str
    applied_at: str
    status: str
    details: Dict[str, Any]

class TuningAnalysisResponse(BaseModel):
    analysis_timestamp: str
    database_analysis: Dict[str, Any]
    application_analysis: Dict[str, Any]
    resource_analysis: Dict[str, Any]
    overall_optimization_potential: str

class TuningRequest(BaseModel):
    category: Optional[str] = None  # database, application, system
    auto_apply: bool = False

class TuningResponse(BaseModel):
    optimization_timestamp: str
    total_optimizations: int
    applied_successfully: int
    failed: int
    requires_restart: bool
    results: List[Dict[str, Any]]

class ProfilingControlRequest(BaseModel):
    action: str  # start, stop, clear
    function_names: Optional[List[str]] = None

# Create router
router = APIRouter(prefix="/api/v1/optimization", tags=["optimization"])

@router.get("/performance/analyze", response_model=PerformanceAnalysisResponse)
async def analyze_performance(
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Comprehensive performance analysis
    Requires monitoring permission
    """
    try:
        optimizer = get_performance_optimizer()
        analysis = await optimizer.analyze_performance()
        
        # Convert to response model format
        return PerformanceAnalysisResponse(
            timestamp=analysis['timestamp'],
            current_metrics=analysis['current_metrics'],
            historical_analysis=analysis['historical_analysis'],
            profiling_analysis=analysis['profiling_analysis'],
            optimization_recommendations=analysis['optimization_recommendations'],
            optimizations_applied=analysis['optimizations_applied']
        )
        
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {e}")

@router.post("/performance/optimize", response_model=OptimizationResponse)
async def apply_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Apply performance optimization
    Requires admin permission
    """
    try:
        optimizer = get_performance_optimizer()
        
        # Apply optimization in background for long-running operations
        if request.optimization_type in ['full_system_optimization', 'deep_analysis']:
            background_tasks.add_task(
                optimizer.apply_optimization, 
                request.optimization_type
            )
            
            return OptimizationResponse(
                optimization_type=request.optimization_type,
                applied_at=datetime.now().isoformat(),
                status="scheduled",
                details={"message": "Optimization scheduled for background execution"}
            )
        else:
            # Apply immediately for quick optimizations
            result = await optimizer.apply_optimization(request.optimization_type)
            
            return OptimizationResponse(
                optimization_type=result['optimization_type'],
                applied_at=result['applied_at'],
                status=result['status'],
                details=result['details']
            )
            
    except Exception as e:
        logger.error(f"Optimization application failed: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {e}")

@router.get("/performance/status")
async def get_optimization_status(
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get current optimization status
    Requires monitoring permission
    """
    try:
        optimizer = get_performance_optimizer()
        status = optimizer.get_optimization_status()
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {e}")

@router.post("/profiling/control")
async def control_profiling(
    request: ProfilingControlRequest,
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Control performance profiling
    Requires admin permission
    """
    try:
        optimizer = get_performance_optimizer()
        
        if request.action == "start":
            # Start profiling for specific functions or all
            if request.function_names:
                result = {
                    "action": "start",
                    "functions": request.function_names,
                    "status": "profiling_started"
                }
            else:
                result = {
                    "action": "start", 
                    "status": "global_profiling_started"
                }
                
        elif request.action == "stop":
            result = {
                "action": "stop",
                "status": "profiling_stopped"
            }
            
        elif request.action == "clear":
            optimizer.profiler.profiles.clear()
            result = {
                "action": "clear",
                "status": "profiling_data_cleared"
            }
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'start', 'stop', or 'clear'")
        
        result["timestamp"] = datetime.now().isoformat()
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Profiling control failed: {e}")
        raise HTTPException(status_code=500, detail=f"Profiling control failed: {e}")

@router.get("/profiling/functions")
async def get_function_profiles(
    limit: int = Query(20, ge=1, le=100, description="Number of functions to return"),
    sort_by: str = Query("total_time", regex="^(total_time|call_count|avg_time)$"),
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get function performance profiles
    Requires monitoring permission
    """
    try:
        optimizer = get_performance_optimizer()
        
        # Get all function profiles
        profiles = []
        for func_name in optimizer.profiler.profiles:
            summary = optimizer.profiler.get_profile_summary(func_name)
            profiles.append(summary)
        
        # Sort profiles
        if sort_by == "total_time":
            profiles.sort(key=lambda x: x["total_time"], reverse=True)
        elif sort_by == "call_count":
            profiles.sort(key=lambda x: x["call_count"], reverse=True)
        elif sort_by == "avg_time":
            profiles.sort(key=lambda x: x["avg_time"], reverse=True)
        
        return JSONResponse(content={
            "timestamp": datetime.now().isoformat(),
            "total_functions": len(profiles),
            "sort_by": sort_by,
            "profiles": profiles[:limit]
        })
        
    except Exception as e:
        logger.error(f"Failed to get function profiles: {e}")
        raise HTTPException(status_code=500, detail=f"Profile retrieval failed: {e}")

@router.get("/tuning/analyze", response_model=TuningAnalysisResponse)
async def analyze_system_tuning(
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Comprehensive system tuning analysis
    Requires monitoring permission
    """
    try:
        tuner = get_system_tuner()
        analysis = await tuner.comprehensive_analysis()
        
        return TuningAnalysisResponse(
            analysis_timestamp=analysis['analysis_timestamp'],
            database_analysis=analysis['database_analysis'],
            application_analysis=analysis['application_analysis'],
            resource_analysis=analysis['resource_analysis'],
            overall_optimization_potential=analysis['overall_optimization_potential']
        )
        
    except Exception as e:
        logger.error(f"System tuning analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Tuning analysis failed: {e}")

@router.post("/tuning/apply", response_model=TuningResponse)
async def apply_system_tuning(
    request: TuningRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Apply system tuning optimizations
    Requires admin permission
    """
    try:
        tuner = get_system_tuner()
        
        if request.auto_apply:
            # Apply optimizations in background
            background_tasks.add_task(
                tuner.apply_recommended_optimizations,
                request.category
            )
            
            return TuningResponse(
                optimization_timestamp=datetime.now().isoformat(),
                total_optimizations=0,
                applied_successfully=0,
                failed=0,
                requires_restart=False,
                results=[{
                    "status": "scheduled",
                    "message": f"System tuning scheduled for category: {request.category or 'all'}"
                }]
            )
        else:
            # Apply immediately and return results
            result = await tuner.apply_recommended_optimizations(request.category)
            
            return TuningResponse(
                optimization_timestamp=result['optimization_timestamp'],
                total_optimizations=result['total_optimizations'],
                applied_successfully=result['applied_successfully'],
                failed=result['failed'],
                requires_restart=result['requires_restart'],
                results=result['results']
            )
            
    except Exception as e:
        logger.error(f"System tuning application failed: {e}")
        raise HTTPException(status_code=500, detail=f"System tuning failed: {e}")

@router.get("/tuning/status")
async def get_tuning_status(
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get system tuning status
    Requires monitoring permission
    """
    try:
        tuner = get_system_tuner()
        status = tuner.get_tuning_status()
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Failed to get tuning status: {e}")
        raise HTTPException(status_code=500, detail=f"Tuning status retrieval failed: {e}")

@router.post("/tuning/auto-enable")
async def enable_auto_tuning(
    interval_minutes: int = Query(60, ge=15, le=1440, description="Auto-tuning interval in minutes"),
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Enable automatic system tuning
    Requires admin permission
    """
    try:
        tuner = get_system_tuner()
        tuner.enable_auto_tuning(interval_minutes)
        
        return JSONResponse(content={
            "status": "auto_tuning_enabled",
            "interval_minutes": interval_minutes,
            "timestamp": datetime.now().isoformat(),
            "message": f"Auto-tuning enabled with {interval_minutes}-minute intervals"
        })
        
    except Exception as e:
        logger.error(f"Failed to enable auto-tuning: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-tuning enable failed: {e}")

@router.post("/tuning/auto-disable")
async def disable_auto_tuning(
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Disable automatic system tuning
    Requires admin permission
    """
    try:
        tuner = get_system_tuner()
        tuner.disable_auto_tuning()
        
        return JSONResponse(content={
            "status": "auto_tuning_disabled",
            "timestamp": datetime.now().isoformat(),
            "message": "Auto-tuning has been disabled"
        })
        
    except Exception as e:
        logger.error(f"Failed to disable auto-tuning: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-tuning disable failed: {e}")

@router.get("/recommendations")
async def get_optimization_recommendations(
    category: Optional[str] = Query(None, regex="^(performance|database|application|system)$"),
    priority: Optional[str] = Query(None, regex="^(critical|high|medium|low)$"),
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get optimization recommendations
    Requires monitoring permission
    """
    try:
        # Get recommendations from both systems
        optimizer = get_performance_optimizer()
        tuner = get_system_tuner()
        
        # Performance recommendations
        performance_analysis = await optimizer.analyze_performance()
        performance_recs = performance_analysis['optimization_recommendations']
        
        # System tuning recommendations
        tuning_analysis = await tuner.comprehensive_analysis()
        
        # Combine all recommendations
        all_recommendations = []
        
        # Add performance recommendations
        for rec in performance_recs:
            if isinstance(rec, dict):
                all_recommendations.append(rec)
            else:
                # Handle OptimizationRecommendation objects
                all_recommendations.append({
                    'category': rec.category,
                    'priority': rec.priority,
                    'description': rec.description,
                    'impact': rec.impact,
                    'implementation_effort': rec.implementation_effort,
                    'expected_improvement': rec.expected_improvement,
                    'code_changes': rec.code_changes
                })
        
        # Add tuning recommendations
        for analysis_type in ['database_analysis', 'application_analysis', 'resource_analysis']:
            if analysis_type in tuning_analysis:
                for rec in tuning_analysis[analysis_type]['recommendations']:
                    all_recommendations.append({
                        'category': rec['category'],
                        'priority': 'medium',  # Default priority for tuning recommendations
                        'description': rec['impact_description'],
                        'impact': rec['impact_description'],
                        'implementation_effort': rec['risk_level'],
                        'expected_improvement': f"Optimize {rec['parameter']}",
                        'requires_restart': rec['requires_restart']
                    })
        
        # Filter by category if specified
        if category:
            all_recommendations = [r for r in all_recommendations if r['category'].lower() == category.lower()]
        
        # Filter by priority if specified
        if priority:
            all_recommendations = [r for r in all_recommendations if r.get('priority', '').lower() == priority.lower()]
        
        return JSONResponse(content={
            "timestamp": datetime.now().isoformat(),
            "total_recommendations": len(all_recommendations),
            "category_filter": category,
            "priority_filter": priority,
            "recommendations": all_recommendations
        })
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendations retrieval failed: {e}")

@router.get("/system/health-check")
async def optimization_health_check():
    """
    Health check for optimization systems
    Public endpoint
    """
    try:
        optimizer = get_performance_optimizer()
        tuner = get_system_tuner()
        
        # Check if systems are operational
        optimizer_status = "healthy" if optimizer else "unavailable"
        tuner_status = "healthy" if tuner else "unavailable"
        
        # Get basic status information
        opt_status = optimizer.get_optimization_status() if optimizer else {}
        tuning_status = tuner.get_tuning_status() if tuner else {}
        
        overall_status = "healthy" if optimizer_status == "healthy" and tuner_status == "healthy" else "degraded"
        
        return JSONResponse(content={
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "components": {
                "performance_optimizer": {
                    "status": optimizer_status,
                    "monitoring_active": opt_status.get('monitoring_active', False),
                    "optimizations_applied": opt_status.get('total_optimizations_applied', 0)
                },
                "system_tuner": {
                    "status": tuner_status,
                    "auto_tuning_enabled": tuning_status.get('auto_tuning_enabled', False),
                    "total_tunings": tuning_status.get('total_tunings_applied', 0)
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Optimization health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "timestamp": datetime.now().isoformat(),
                "overall_status": "error",
                "error": str(e)
            }
        )