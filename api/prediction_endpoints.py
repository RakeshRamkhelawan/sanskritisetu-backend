"""
Phase 5 API Endpoints: Predictive System Evolution
Advanced endpoints for predictive analysis and system optimization.
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta

from core.learning.predictive_evolution import (
    predictive_evolution_engine,
    PredictionResult,
    PredictionType
)

# API Models
class PredictionRequest(BaseModel):
    hours_ahead: int = 24
    prediction_types: Optional[List[str]] = None
    include_optimization_plan: bool = True

class MetricCollectionResponse(BaseModel):
    status: str
    metrics_collected: int
    timestamp: str

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    optimization_plan: Optional[Dict[str, Any]]
    system_status: Dict[str, Any]

# Router setup
prediction_router = APIRouter(prefix="/api/v1/prediction", tags=["Phase5-Prediction"])

@prediction_router.post("/initialize", response_model=Dict[str, Any])
async def initialize_predictive_system():
    """Initialize the predictive evolution system."""
    try:
        success = await predictive_evolution_engine.initialize()
        
        if success:
            return {
                "status": "success",
                "message": "Predictive evolution system initialized successfully",
                "timestamp": datetime.now().isoformat(),
                "capabilities": [
                    "performance_prediction",
                    "bottleneck_detection", 
                    "resource_forecasting",
                    "optimization_planning"
                ]
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize predictive evolution system"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Initialization error: {str(e)}"
        )

@prediction_router.get("/status", response_model=Dict[str, Any])
async def get_prediction_system_status():
    """Get current status of the predictive evolution system."""
    try:
        status = await predictive_evolution_engine.get_system_evolution_status()
        return {
            "system_status": "operational",
            "prediction_engine": status,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Status retrieval error: {str(e)}"
        )

@prediction_router.post("/collect-metrics", response_model=MetricCollectionResponse)
async def collect_system_metrics():
    """Manually trigger system metrics collection."""
    try:
        metrics = await predictive_evolution_engine.collect_system_metrics()
        
        return MetricCollectionResponse(
            status="success",
            metrics_collected=len(metrics),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Metrics collection error: {str(e)}"
        )

@prediction_router.post("/predict", response_model=PredictionResponse)
async def generate_system_predictions(request: PredictionRequest):
    """Generate system performance predictions and optimization plans."""
    try:
        # Generate predictions
        predictions = await predictive_evolution_engine.predict_system_performance(
            hours_ahead=request.hours_ahead
        )
        
        # Convert predictions to dict format
        predictions_data = []
        for pred in predictions:
            predictions_data.append({
                "prediction_id": pred.prediction_id,
                "type": pred.prediction_type.value,
                "confidence": pred.confidence,
                "predicted_value": pred.predicted_value,
                "time_horizon_hours": pred.time_horizon,
                "reasoning": pred.reasoning,
                "recommended_actions": pred.recommended_actions,
                "impact_assessment": pred.impact_assessment
            })
        
        # Generate optimization plan if requested
        optimization_plan = None
        if request.include_optimization_plan:
            optimization_plan = await predictive_evolution_engine.generate_optimization_plan(predictions)
        
        # Get system status
        system_status = await predictive_evolution_engine.get_system_evolution_status()
        
        return PredictionResponse(
            predictions=predictions_data,
            optimization_plan=optimization_plan,
            system_status=system_status
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction generation error: {str(e)}"
        )

@prediction_router.get("/predictions/recent", response_model=Dict[str, Any])
async def get_recent_predictions(limit: int = 10):
    """Get recent predictions from the system."""
    try:
        all_predictions = predictive_evolution_engine.predictions
        recent_predictions = sorted(
            all_predictions, 
            key=lambda p: p.prediction_id,
            reverse=True
        )[:limit]
        
        predictions_data = []
        for pred in recent_predictions:
            predictions_data.append({
                "prediction_id": pred.prediction_id,
                "type": pred.prediction_type.value,
                "confidence": pred.confidence,
                "predicted_value": pred.predicted_value,
                "time_horizon_hours": pred.time_horizon,
                "reasoning": pred.reasoning,
                "recommended_actions": pred.recommended_actions,
                "impact_assessment": pred.impact_assessment
            })
        
        return {
            "predictions": predictions_data,
            "total_count": len(all_predictions),
            "returned_count": len(predictions_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recent predictions retrieval error: {str(e)}"
        )

@prediction_router.get("/metrics/history", response_model=Dict[str, Any])
async def get_metrics_history(
    hours: int = 6,
    metric_type: Optional[str] = None
):
    """Get historical system metrics."""
    try:
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        filtered_metrics = [
            m for m in predictive_evolution_engine.metrics_history
            if m.timestamp > cutoff_time
        ]
        
        if metric_type:
            filtered_metrics = [
                m for m in filtered_metrics
                if m.metric_type == metric_type
            ]
        
        metrics_data = []
        for metric in filtered_metrics:
            metrics_data.append({
                "timestamp": metric.timestamp.isoformat(),
                "metric_type": metric.metric_type,
                "value": metric.value,
                "component": metric.component,
                "metadata": metric.metadata
            })
        
        return {
            "metrics": metrics_data,
            "count": len(metrics_data),
            "time_range_hours": hours,
            "metric_type_filter": metric_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Metrics history retrieval error: {str(e)}"
        )

@prediction_router.post("/optimize/execute")
async def execute_optimization_plan(
    background_tasks: BackgroundTasks,
    plan_id: Optional[str] = None
):
    """Execute an optimization plan (simulate execution for now)."""
    try:
        # In a real implementation, this would execute actual optimizations
        # For now, we simulate the execution
        
        def simulate_optimization():
            import time
            time.sleep(2)  # Simulate optimization work
            print(f"Optimization plan executed: {plan_id or 'latest'}")
        
        background_tasks.add_task(simulate_optimization)
        
        return {
            "status": "optimization_initiated",
            "plan_id": plan_id or f"auto_{int(datetime.now().timestamp())}",
            "message": "Optimization plan execution started in background",
            "estimated_completion": "2-5 minutes",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Optimization execution error: {str(e)}"
        )

@prediction_router.get("/analytics/trends", response_model=Dict[str, Any])
async def get_system_trends():
    """Get system performance trends and analytics."""
    try:
        from datetime import timedelta
        import numpy as np
        
        # Analyze trends in recent metrics
        recent_metrics = [
            m for m in predictive_evolution_engine.metrics_history
            if m.timestamp > datetime.now() - timedelta(hours=12)
        ]
        
        trends = {}
        metric_types = set(m.metric_type for m in recent_metrics)
        
        for metric_type in metric_types:
            type_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
            values = [m.value for m in type_metrics]
            
            if len(values) >= 2:
                # Simple trend calculation
                trend_direction = "increasing" if values[-1] > values[0] else "decreasing"
                trend_strength = abs(values[-1] - values[0]) / max(values[0], 0.1)
                
                trends[metric_type] = {
                    "current_value": values[-1],
                    "trend_direction": trend_direction,
                    "trend_strength": trend_strength,
                    "average_value": np.mean(values),
                    "data_points": len(values)
                }
        
        return {
            "trends": trends,
            "analysis_period_hours": 12,
            "total_metrics_analyzed": len(recent_metrics),
            "trend_summary": {
                "improving_metrics": len([t for t in trends.values() if t["trend_direction"] == "decreasing"]),
                "degrading_metrics": len([t for t in trends.values() if t["trend_direction"] == "increasing"]),
                "stable_metrics": len(metric_types) - len(trends)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Trends analysis error: {str(e)}"
        )

@prediction_router.get("/health", response_model=Dict[str, str])
async def prediction_system_health():
    """Health check for prediction system."""
    try:
        status = await predictive_evolution_engine.get_system_evolution_status()
        
        return {
            "status": "healthy" if status["status"] == "operational" else "degraded",
            "prediction_engine": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }