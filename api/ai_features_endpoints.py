"""
🤖 AI Features API Endpoints - Sanskriti Setu
Advanced AI capabilities endpoints for adaptive learning, intelligent routing, and predictive analytics
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

from core.ai.adaptive_learning_engine import create_adaptive_learning_engine, LearningEvent, AdaptationInsight
from core.ai.intelligent_routing import create_intelligent_router, RequestContext, RouteOption, RequestPriority
from core.ai.predictive_analytics import create_predictive_analytics_engine, PredictionRequest, PredictionType, TimeHorizon
from core.auth.rbac import require_permission, Permission

logger = logging.getLogger(__name__)

# Initialize AI components
adaptive_learning_engine = create_adaptive_learning_engine()
intelligent_router = create_intelligent_router()
predictive_analytics = create_predictive_analytics_engine()

# Pydantic models for API requests/responses
class LearningEventRequest(BaseModel):
    event_type: str = Field(..., description="Type of learning event")
    context: Dict[str, Any] = Field(..., description="Event context")
    outcome: Any = Field(None, description="Event outcome")
    confidence: float = Field(0.8, ge=0.0, le=1.0, description="Confidence in the event")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}

class RoutingRequest(BaseModel):
    request_id: str = Field(..., description="Unique request identifier")
    user_id: Optional[str] = None
    request_type: str = Field(..., description="Type of request")
    complexity_score: float = Field(0.5, ge=0.0, le=1.0, description="Request complexity")
    priority: str = Field("medium", description="Request priority")
    estimated_compute_time: float = Field(1.0, description="Estimated compute time in seconds")
    required_capabilities: List[str] = Field([], description="Required capabilities")
    user_preferences: Dict[str, Any] = {}
    session_context: Dict[str, Any] = {}

class PredictionRequestModel(BaseModel):
    prediction_type: str = Field(..., description="Type of prediction")
    time_horizon: str = Field("short_term", description="Prediction time horizon")
    context: Dict[str, Any] = Field({}, description="Prediction context")
    features: Dict[str, float] = Field(..., description="Input features for prediction")

class UserPreferencesUpdate(BaseModel):
    user_id: str = Field(..., description="User identifier")
    preferences: Dict[str, Any] = Field(..., description="User preferences")

class PerformanceMetricRequest(BaseModel):
    route_id: str = Field(..., description="Route identifier")
    request_id: str = Field(..., description="Request identifier")
    actual_response_time: float = Field(..., description="Actual response time")
    actual_error_occurred: bool = Field(False, description="Whether an error occurred")
    user_satisfaction: Optional[float] = Field(None, ge=1.0, le=5.0, description="User satisfaction rating")
    resource_utilization: float = Field(0.5, description="Resource utilization")
    cost: float = Field(1.0, description="Cost of processing")

class OutcomeRecordingRequest(BaseModel):
    prediction_id: str = Field(..., description="Prediction identifier")
    actual_value: Union[float, str] = Field(..., description="Actual outcome value")
    metadata: Optional[Dict[str, Any]] = None

# API Response models
class LearningInsightResponse(BaseModel):
    insight_id: str
    category: str
    description: str
    confidence: float
    impact_score: float
    recommended_actions: List[str]
    evidence: Dict[str, Any]
    timestamp: str

class RoutingDecisionResponse(BaseModel):
    request_id: str
    selected_route: Dict[str, Any]
    confidence: float
    reasoning: List[str]
    alternative_routes: List[Dict[str, Any]]
    decision_time_ms: float
    predicted_performance: Dict[str, float]

class PredictionResponse(BaseModel):
    prediction_id: str
    prediction_type: str
    time_horizon: str
    predicted_value: Union[float, str, Dict]
    confidence: float
    uncertainty_bounds: List[float]
    feature_importance: Dict[str, float]
    model_used: str
    reasoning: List[str]
    timestamp: str
    expires_at: str

# Create router
router = APIRouter(prefix="/api/v1/ai", tags=["ai-features"])

# Adaptive Learning Endpoints
@router.post("/learning/event")
async def record_learning_event(
    event: LearningEventRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_permission(Permission.AI_TRAINING))
):
    """
    Record a learning event for the adaptive learning system
    Requires AI training permission
    """
    try:
        learning_event = LearningEvent(
            timestamp=time.time(),
            event_type=event.event_type,
            context=event.context,
            outcome=event.outcome,
            confidence=event.confidence,
            user_id=event.user_id,
            session_id=event.session_id,
            metadata=event.metadata
        )
        
        # Record event in background
        background_tasks.add_task(adaptive_learning_engine.record_learning_event, learning_event)
        
        return {
            "message": "Learning event recorded successfully",
            "event_type": event.event_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recording learning event: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record learning event: {e}")

@router.get("/learning/insights", response_model=List[LearningInsightResponse])
async def get_learning_insights(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of insights to return"),
    category: Optional[str] = Query(None, description="Filter by insight category"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    current_user = Depends(require_permission(Permission.AI_INSIGHTS))
):
    """
    Get adaptive learning insights
    Requires AI insights permission
    """
    try:
        # Generate fresh insights
        insights = await adaptive_learning_engine.generate_insights()
        
        # Filter insights
        filtered_insights = insights
        
        if category:
            filtered_insights = [i for i in filtered_insights if i.category == category]
        
        if min_confidence > 0:
            filtered_insights = [i for i in filtered_insights if i.confidence >= min_confidence]
        
        # Sort by impact score and limit
        filtered_insights.sort(key=lambda x: x.impact_score, reverse=True)
        filtered_insights = filtered_insights[:limit]
        
        # Convert to response format
        response_insights = []
        for insight in filtered_insights:
            response_insights.append(LearningInsightResponse(
                insight_id=insight.insight_id,
                category=insight.category,
                description=insight.description,
                confidence=insight.confidence,
                impact_score=insight.impact_score,
                recommended_actions=insight.recommended_actions,
                evidence=insight.evidence,
                timestamp=insight.timestamp.isoformat()
            ))
        
        return response_insights
        
    except Exception as e:
        logger.error(f"Error getting learning insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning insights: {e}")

@router.post("/learning/train")
async def trigger_model_training(
    background_tasks: BackgroundTasks,
    force_retrain: bool = Query(False, description="Force retraining regardless of thresholds"),
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Trigger model training for the adaptive learning system
    Requires admin permission
    """
    try:
        # Trigger training in background
        background_tasks.add_task(adaptive_learning_engine.train_models, force_retrain)
        
        return {
            "message": "Model training initiated",
            "force_retrain": force_retrain,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error triggering model training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger training: {e}")

@router.get("/learning/summary")
async def get_learning_summary(
    current_user = Depends(require_permission(Permission.AI_INSIGHTS))
):
    """
    Get adaptive learning system summary
    Requires AI insights permission
    """
    try:
        summary = adaptive_learning_engine.get_learning_summary()
        return {
            "timestamp": datetime.now().isoformat(),
            "learning_system": summary
        }
        
    except Exception as e:
        logger.error(f"Error getting learning summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning summary: {e}")

# Intelligent Routing Endpoints
@router.post("/routing/route", response_model=RoutingDecisionResponse)
async def route_request(
    routing_request: RoutingRequest,
    strategy: Optional[str] = Query(None, description="Routing strategy to use"),
    current_user = Depends(require_permission(Permission.API_EXECUTE))
):
    """
    Route a request using intelligent routing
    Requires API execution permission
    """
    try:
        # Convert priority string to enum
        try:
            priority = RequestPriority(routing_request.priority.lower())
        except ValueError:
            priority = RequestPriority.MEDIUM
        
        # Create request context
        request_context = RequestContext(
            request_id=routing_request.request_id,
            user_id=routing_request.user_id,
            request_type=routing_request.request_type,
            complexity_score=routing_request.complexity_score,
            priority=priority,
            estimated_compute_time=routing_request.estimated_compute_time,
            required_capabilities=routing_request.required_capabilities,
            user_preferences=routing_request.user_preferences,
            session_context=routing_request.session_context
        )
        
        # Route the request
        decision = await intelligent_router.route_request(request_context, strategy)
        
        # Convert to response format
        return RoutingDecisionResponse(
            request_id=decision.request_id,
            selected_route={
                "route_id": decision.selected_route.route_id,
                "name": decision.selected_route.name,
                "type": decision.selected_route.route_type.value,
                "capabilities": decision.selected_route.capabilities,
                "current_load": decision.selected_route.current_load,
                "performance_score": decision.selected_route.performance_score
            },
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            alternative_routes=[
                {
                    "route_id": route.route_id,
                    "name": route.name,
                    "performance_score": route.performance_score
                } for route in decision.alternative_routes[:3]
            ],
            decision_time_ms=decision.decision_time_ms,
            predicted_performance=decision.predicted_performance
        )
        
    except Exception as e:
        logger.error(f"Error routing request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to route request: {e}")

@router.post("/routing/performance")
async def record_routing_performance(
    metric: PerformanceMetricRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Record routing performance for learning
    Requires monitoring permission
    """
    try:
        from core.ai.intelligent_routing import PerformanceMetric
        
        performance_metric = PerformanceMetric(
            route_id=metric.route_id,
            request_id=metric.request_id,
            actual_response_time=metric.actual_response_time,
            actual_error_occurred=metric.actual_error_occurred,
            user_satisfaction=metric.user_satisfaction,
            resource_utilization=metric.resource_utilization,
            cost=metric.cost,
            timestamp=time.time()
        )
        
        # Record performance in background
        background_tasks.add_task(intelligent_router.record_performance, performance_metric)
        
        return {
            "message": "Performance metric recorded successfully",
            "route_id": metric.route_id,
            "request_id": metric.request_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recording routing performance: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record performance: {e}")

@router.put("/routing/user-preferences")
async def update_user_routing_preferences(
    preferences_update: UserPreferencesUpdate,
    current_user = Depends(require_permission(Permission.USER_UPDATE))
):
    """
    Update user routing preferences
    Requires user update permission
    """
    try:
        intelligent_router.update_user_preferences(
            preferences_update.user_id,
            preferences_update.preferences
        )
        
        return {
            "message": "User preferences updated successfully",
            "user_id": preferences_update.user_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update preferences: {e}")

@router.get("/routing/statistics")
async def get_routing_statistics(
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get routing system statistics
    Requires monitoring permission
    """
    try:
        stats = intelligent_router.get_route_statistics()
        return {
            "timestamp": datetime.now().isoformat(),
            "routing_statistics": stats
        }
        
    except Exception as e:
        logger.error(f"Error getting routing statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {e}")

@router.get("/routing/routes")
async def get_available_routes(
    route_type: Optional[str] = Query(None, description="Filter by route type"),
    available_only: bool = Query(True, description="Show only available routes"),
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get available routing options
    Requires monitoring permission
    """
    try:
        routes = list(intelligent_router.routes.values())
        
        # Apply filters
        if available_only:
            routes = [r for r in routes if r.availability]
        
        if route_type:
            routes = [r for r in routes if r.route_type.value == route_type]
        
        # Convert to response format
        route_info = []
        for route in routes:
            route_info.append({
                "route_id": route.route_id,
                "name": route.name,
                "type": route.route_type.value,
                "capabilities": route.capabilities,
                "current_load": route.current_load,
                "performance_score": route.performance_score,
                "cost_factor": route.cost_factor,
                "availability": route.availability,
                "response_time_avg": route.response_time_avg,
                "error_rate": route.error_rate
            })
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_routes": len(route_info),
            "routes": route_info
        }
        
    except Exception as e:
        logger.error(f"Error getting available routes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get routes: {e}")

# Predictive Analytics Endpoints
@router.post("/predictions/predict", response_model=PredictionResponse)
async def make_prediction(
    prediction_request: PredictionRequestModel,
    current_user = Depends(require_permission(Permission.AI_PREDICTION))
):
    """
    Make a prediction using the predictive analytics engine
    Requires AI prediction permission
    """
    try:
        # Convert string enums to actual enums
        try:
            pred_type = PredictionType(prediction_request.prediction_type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid prediction type: {prediction_request.prediction_type}")
        
        try:
            time_horizon = TimeHorizon(prediction_request.time_horizon.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid time horizon: {prediction_request.time_horizon}")
        
        # Create prediction request
        pred_request = PredictionRequest(
            prediction_type=pred_type,
            time_horizon=time_horizon,
            context=prediction_request.context,
            features=prediction_request.features
        )
        
        # Make prediction
        prediction = await predictive_analytics.make_prediction(pred_request)
        
        # Convert to response format
        return PredictionResponse(
            prediction_id=prediction.prediction_id,
            prediction_type=prediction.prediction_type.value,
            time_horizon=prediction.time_horizon.value,
            predicted_value=prediction.predicted_value,
            confidence=prediction.confidence,
            uncertainty_bounds=list(prediction.uncertainty_bounds),
            feature_importance=prediction.feature_importance,
            model_used=prediction.model_used,
            reasoning=prediction.reasoning,
            timestamp=prediction.timestamp.isoformat(),
            expires_at=prediction.expires_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to make prediction: {e}")

@router.post("/predictions/record-outcome")
async def record_prediction_outcome(
    outcome_request: OutcomeRecordingRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_permission(Permission.AI_TRAINING))
):
    """
    Record actual outcome for a prediction to improve future models
    Requires AI training permission
    """
    try:
        # Record outcome in background
        background_tasks.add_task(
            predictive_analytics.record_actual_outcome,
            outcome_request.prediction_id,
            outcome_request.actual_value,
            outcome_request.metadata
        )
        
        return {
            "message": "Prediction outcome recorded successfully",
            "prediction_id": outcome_request.prediction_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error recording prediction outcome: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record outcome: {e}")

@router.post("/predictions/train")
async def train_prediction_models(
    background_tasks: BackgroundTasks,
    prediction_type: Optional[str] = Query(None, description="Specific prediction type to train"),
    current_user = Depends(require_permission(Permission.SYSTEM_ADMIN))
):
    """
    Trigger training of prediction models
    Requires admin permission
    """
    try:
        # Convert prediction type if specified
        pred_type = None
        if prediction_type:
            try:
                pred_type = PredictionType(prediction_type.lower())
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid prediction type: {prediction_type}")
        
        # Trigger training in background
        background_tasks.add_task(predictive_analytics.train_models, pred_type)
        
        return {
            "message": "Prediction model training initiated",
            "prediction_type": prediction_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering prediction training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger training: {e}")

@router.get("/predictions/analytics-summary")
async def get_predictive_analytics_summary(
    current_user = Depends(require_permission(Permission.AI_INSIGHTS))
):
    """
    Get predictive analytics system summary
    Requires AI insights permission
    """
    try:
        summary = predictive_analytics.get_analytics_summary()
        return {
            "timestamp": datetime.now().isoformat(),
            "predictive_analytics": summary
        }
        
    except Exception as e:
        logger.error(f"Error getting analytics summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics summary: {e}")

# Combined AI System Status
@router.get("/status")
async def get_ai_system_status(
    current_user = Depends(require_permission(Permission.SYSTEM_MONITOR))
):
    """
    Get comprehensive AI system status
    Requires monitoring permission
    """
    try:
        # Get status from all AI components
        learning_summary = adaptive_learning_engine.get_learning_summary()
        routing_stats = intelligent_router.get_route_statistics()
        analytics_summary = predictive_analytics.get_analytics_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "ai_system_status": {
                "adaptive_learning": {
                    "total_events": learning_summary["total_events"],
                    "trained_models": learning_summary["models"],
                    "recent_insights": learning_summary["recent_insights"],
                    "status": "operational" if learning_summary["total_events"] > 0 else "initializing"
                },
                "intelligent_routing": {
                    "total_routes": routing_stats["total_routes"],
                    "total_requests": routing_stats["total_requests_routed"],
                    "performance_records": routing_stats["total_performance_records"],
                    "status": "operational" if routing_stats["total_requests_routed"] > 0 else "ready"
                },
                "predictive_analytics": {
                    "cached_predictions": analytics_summary["total_predictions_cached"],
                    "cache_hit_rate": analytics_summary["cache_hit_rate"],
                    "trained_models": list(analytics_summary["trained_models"].keys()),
                    "status": "operational" if analytics_summary["total_predictions_cached"] > 0 else "ready"
                },
                "overall_status": "operational"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting AI system status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get AI system status: {e}")

@router.get("/capabilities")
async def get_ai_capabilities():
    """
    Get AI system capabilities (public endpoint)
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "ai_capabilities": {
            "adaptive_learning": {
                "description": "Self-improving AI that learns from user interactions and system performance",
                "features": [
                    "Real-time learning from user feedback",
                    "System performance optimization",
                    "Automated insight generation",
                    "Model retraining and adaptation"
                ]
            },
            "intelligent_routing": {
                "description": "AI-powered request routing with performance optimization",
                "features": [
                    "Multiple routing strategies",
                    "User preference learning",
                    "Performance-based optimization",
                    "Load balancing and cost optimization"
                ]
            },
            "predictive_analytics": {
                "description": "ML-powered prediction system for various aspects of system behavior",
                "prediction_types": [pt.value for pt in PredictionType],
                "time_horizons": [th.value for th in TimeHorizon],
                "features": [
                    "Multi-model ensemble predictions",
                    "Uncertainty quantification",
                    "Feature importance analysis",
                    "Continuous model improvement"
                ]
            }
        }
    }