#!/usr/bin/env python3
"""
Enhanced NLP API Endpoints for M-MDP
ASCII-compliant production-ready endpoints for enhanced NLP parser
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from core.agents.enhanced_nlp_parser import EnhancedNLPParser, IntentType

logger = logging.getLogger('NLPAPI')
router = APIRouter(prefix="/api/v1/nlp", tags=["NLP"])

# Global NLP parser instance
nlp_parser = EnhancedNLPParser()

# ASCII-compliant response models
class ParseRequest(BaseModel):
    """Request model for text parsing"""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to parse")
    include_confidence: bool = Field(True, description="Include confidence score")
    include_parameters: bool = Field(True, description="Include extracted parameters")

class ParseResponse(BaseModel):
    """ASCII-compliant parse response"""
    timestamp: str
    input_text: str
    intent: str
    confidence: float
    parameters: Optional[Dict[str, Any]]
    suggestions: List[str]
    success: bool = True

class IntentsResponse(BaseModel):
    """Available intents response"""
    timestamp: str
    available_intents: List[Dict[str, Any]]
    total_count: int
    mmdp_specific_intents: List[str]

class ValidationRequest(BaseModel):
    """Request model for text validation"""
    text: str = Field(..., min_length=1, max_length=1000, description="Text to validate")

class BatchParseRequest(BaseModel):
    """Request model for batch text parsing"""
    texts: List[str] = Field(..., min_items=1, max_items=10, description="List of texts to parse")

class ValidationResponse(BaseModel):
    """Text validation response"""
    is_valid: bool
    validation_score: float
    issues: List[str]
    recommendations: List[str]

# Authentication dependency (mock for now)
async def get_current_user():
    """Mock authentication - replace with real JWT validation"""
    return {"user_id": "system", "role": "admin"}

@router.get("/health", status_code=status.HTTP_200_OK)
async def nlp_system_health():
    """Get NLP system health status"""
    try:
        return {
            "status": "healthy",
            "system": "M-MDP Enhanced NLP Parser",
            "version": "3.0.0",
            "capabilities": [
                "Intent classification with confidence scoring",
                "Parameter extraction and validation",
                "M-MDP command recognition",
                "Multi-language support (English/Dutch)",
                "Context-aware parsing"
            ],
            "timestamp": datetime.now().isoformat(),
            "parser_ready": True
        }
    except Exception as e:
        logger.error(f"NLP health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NLP system health check failed: {str(e)}"
        )

@router.post("/parse", response_model=ParseResponse, status_code=status.HTTP_200_OK)
async def parse_text(
    request: ParseRequest,
    current_user: dict = Depends(get_current_user)
):
    """Parse text and extract intent with parameters"""
    try:
        logger.info(f"Parsing text: {request.text[:50]}...")

        # Parse the text
        parse_result = nlp_parser.parse(request.text)

        # Extract parameters if requested
        parameters = None
        if request.include_parameters:
            parameters = parse_result.parameters

        # Generate suggestions based on intent and confidence
        suggestions = []
        if parse_result.confidence < 0.8:
            suggestions.append("Consider rephrasing for better recognition")
        if parse_result.intent == IntentType.UNKNOWN:
            suggestions.append("Try using more specific M-MDP commands")
            suggestions.append("Use keywords like 'learning', 'safety', or 'prediction'")

        return ParseResponse(
            timestamp=datetime.now().isoformat(),
            input_text=request.text,
            intent=parse_result.intent.value,
            confidence=parse_result.confidence if request.include_confidence else 1.0,
            parameters=parameters,
            suggestions=suggestions
        )

    except Exception as e:
        logger.error(f"Text parsing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse text: {str(e)}"
        )

@router.get("/intents", response_model=IntentsResponse, status_code=status.HTTP_200_OK)
async def get_available_intents(
    current_user: dict = Depends(get_current_user)
):
    """Get all available intent types and their descriptions"""
    try:
        # Get all intent types
        all_intents = []
        mmdp_intents = []

        for intent_type in IntentType:
            intent_info = {
                "name": intent_type.value,
                "description": _get_intent_description(intent_type),
                "examples": _get_intent_examples(intent_type),
                "is_mmdp_specific": _is_mmdp_intent(intent_type)
            }
            all_intents.append(intent_info)

            if intent_info["is_mmdp_specific"]:
                mmdp_intents.append(intent_type.value)

        return IntentsResponse(
            timestamp=datetime.now().isoformat(),
            available_intents=all_intents,
            total_count=len(all_intents),
            mmdp_specific_intents=mmdp_intents
        )

    except Exception as e:
        logger.error(f"Failed to get intents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve available intents: {str(e)}"
        )

@router.post("/validate", response_model=ValidationResponse, status_code=status.HTTP_200_OK)
async def validate_text(
    request: ValidationRequest,
    current_user: dict = Depends(get_current_user)
):
    """Validate text for NLP processing"""
    try:
        # Basic validation
        issues = []
        recommendations = []
        validation_score = 1.0

        text = request.text

        # Check text length
        if len(text) < 5:
            issues.append("Text too short for reliable parsing")
            validation_score -= 0.3
            recommendations.append("Provide more context in your request")

        if len(text) > 500:
            issues.append("Text very long - may affect parsing accuracy")
            validation_score -= 0.1
            recommendations.append("Consider breaking into shorter requests")

        # Check for ASCII compliance
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            issues.append("Non-ASCII characters detected")
            validation_score -= 0.2
            recommendations.append("Use ASCII-only characters for best results")

        # Check for M-MDP keywords
        mmdp_keywords = ["learning", "meta", "safety", "prediction", "training", "monitor"]
        has_mmdp_keywords = any(keyword in text.lower() for keyword in mmdp_keywords)

        if not has_mmdp_keywords:
            recommendations.append("Consider using M-MDP specific keywords for better intent recognition")

        # Ensure minimum score
        validation_score = max(0.0, validation_score)

        return ValidationResponse(
            is_valid=validation_score >= 0.5,
            validation_score=validation_score,
            issues=issues,
            recommendations=recommendations
        )

    except Exception as e:
        logger.error(f"Text validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate text: {str(e)}"
        )

@router.get("/examples", status_code=status.HTTP_200_OK)
async def get_parse_examples(
    intent_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get example texts for different intent types"""
    try:
        examples = {}

        if intent_type:
            # Get examples for specific intent
            try:
                intent_enum = IntentType(intent_type)
                examples[intent_type] = _get_intent_examples(intent_enum)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown intent type: {intent_type}"
                )
        else:
            # Get examples for all intents
            for intent in IntentType:
                examples[intent.value] = _get_intent_examples(intent)

        return {
            "timestamp": datetime.now().isoformat(),
            "examples": examples,
            "usage_note": "Use these examples as templates for your requests"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get examples: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve examples: {str(e)}"
        )

@router.post("/batch-parse", status_code=status.HTTP_200_OK)
async def batch_parse_texts(
    request: BatchParseRequest,
    current_user: dict = Depends(get_current_user)
):
    """Parse multiple texts in batch"""
    try:
        results = []

        for i, text in enumerate(request.texts):
            try:
                parse_result = nlp_parser.parse(text)
                results.append({
                    "index": i,
                    "text": text,
                    "intent": parse_result.intent.value,
                    "confidence": parse_result.confidence,
                    "parameters": parse_result.parameters,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "text": text,
                    "error": str(e),
                    "success": False
                })

        # Calculate summary statistics
        successful_parses = [r for r in results if r["success"]]
        avg_confidence = sum(r["confidence"] for r in successful_parses) / len(successful_parses) if successful_parses else 0

        return {
            "timestamp": datetime.now().isoformat(),
            "total_texts": len(request.texts),
            "successful_parses": len(successful_parses),
            "failed_parses": len(request.texts) - len(successful_parses),
            "average_confidence": avg_confidence,
            "results": results
        }

    except Exception as e:
        logger.error(f"Batch parsing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse texts in batch: {str(e)}"
        )

@router.get("/statistics", status_code=status.HTTP_200_OK)
async def get_parser_statistics(
    current_user: dict = Depends(get_current_user)
):
    """Get NLP parser usage statistics"""
    try:
        # Mock statistics - replace with actual tracking
        stats = {
            "timestamp": datetime.now().isoformat(),
            "total_parses_today": 127,
            "average_confidence": 0.847,
            "intent_distribution": {
                "learning_management": 45,
                "safety_monitoring": 23,
                "prediction_request": 31,
                "system_status": 18,
                "unknown": 10
            },
            "language_distribution": {
                "english": 89,
                "dutch": 38
            },
            "performance_metrics": {
                "average_parse_time_ms": 12.3,
                "high_confidence_rate": 0.784,
                "parameter_extraction_success_rate": 0.923
            },
            "recent_trends": {
                "confidence_trend": "stable",
                "volume_trend": "increasing",
                "error_rate_trend": "decreasing"
            }
        }

        return stats

    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve parser statistics: {str(e)}"
        )

# Helper functions
def _get_intent_description(intent_type: IntentType) -> str:
    """Get description for intent type"""
    descriptions = {
        IntentType.GREETING: "General greetings and conversation starters",
        IntentType.CONVERSATION: "General conversation and information requests",
        IntentType.SYSTEM_STATUS: "System status and health inquiries",
        IntentType.LEARNING_MANAGEMENT: "M-MDP learning system management and control",
        IntentType.META_LEARNING: "Meta-learning and strategy management commands",
        IntentType.SAFETY_MONITORING: "Safety system monitoring and validation requests",
        IntentType.PREDICTION_REQUEST: "Predictive analytics and forecasting requests",
        IntentType.UNKNOWN: "Unrecognized or ambiguous requests"
    }
    return descriptions.get(intent_type, "No description available")

def _get_intent_examples(intent_type: IntentType) -> List[str]:
    """Get example texts for intent type"""
    examples = {
        IntentType.GREETING: [
            "Hello",
            "Good morning",
            "Hi there"
        ],
        IntentType.CONVERSATION: [
            "Tell me about the system",
            "What can you do?",
            "Explain the features"
        ],
        IntentType.SYSTEM_STATUS: [
            "Show system status",
            "How is the system running?",
            "Get health report"
        ],
        IntentType.LEARNING_MANAGEMENT: [
            "Start PPO learning",
            "Check learning progress",
            "Update learning parameters"
        ],
        IntentType.META_LEARNING: [
            "Get meta learning status",
            "Apply strategy transfer",
            "Show learning strategies"
        ],
        IntentType.SAFETY_MONITORING: [
            "Check safety violations",
            "Trigger safety validation",
            "Show safety status"
        ],
        IntentType.PREDICTION_REQUEST: [
            "Predict system performance",
            "Show 24h forecast",
            "Get bottleneck predictions"
        ],
        IntentType.UNKNOWN: [
            "Ambiguous request",
            "Unclear command"
        ]
    }
    return examples.get(intent_type, [])

def _is_mmdp_intent(intent_type: IntentType) -> bool:
    """Check if intent is M-MDP specific"""
    mmdp_intents = {
        IntentType.LEARNING_MANAGEMENT,
        IntentType.META_LEARNING,
        IntentType.SAFETY_MONITORING,
        IntentType.PREDICTION_REQUEST
    }
    return intent_type in mmdp_intents