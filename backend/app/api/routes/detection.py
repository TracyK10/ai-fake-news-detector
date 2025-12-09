"""
API routes for fake news detection
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from backend.app.core.schemas import (
    AnalyzeRequest, 
    AnalyzeResponse,
    FeedbackRequest,
    FeedbackResponse
)
from backend.app.database.db import get_db
from backend.app.models.feedback import Feedback
from backend.app.services.classifier_service import get_classifier
from ml.scripts.inference import FakeNewsClassifier
from loguru import logger
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
async def analyze_news(
    request: Request,
    data: AnalyzeRequest,
    classifier: FakeNewsClassifier = Depends(get_classifier)
):
    """
    Analyze news text and predict if it's fake or real
    
    Args:
        data: Request containing text to analyze
        classifier: Injected classifier service
        
    Returns:
        Prediction results with label and confidence score
    """
    try:
        logger.info(f"Analyzing text: {data.text[:100]}...")
        
        # Run prediction
        result = classifier.predict(data.text)
        
        logger.info(f"Prediction: {result['label']} (confidence: {result['confidence_score']:.2%})")
        
        return AnalyzeResponse(**result)
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail="Error analyzing text")


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    data: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    Submit user feedback on prediction accuracy
    
    Args:
        data: Feedback data including predicted and corrected labels
        db: Database session
        
    Returns:
        Confirmation message with feedback ID
    """
    try:
        logger.info(f"Receiving feedback: predicted={data.predicted_label}, correction={data.user_correction}")
        
        # Create feedback entry
        feedback = Feedback(
            text=data.text,
            predicted_label=data.predicted_label,
            confidence_score=data.confidence_score,
            user_correction=data.user_correction
        )
        
        db.add(feedback)
        db.commit()
        db.refresh(feedback)
        
        logger.info(f"Feedback saved with ID: {feedback.id}")
        
        return FeedbackResponse(
            message="Thank you for your feedback!",
            feedback_id=feedback.id
        )
        
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Error saving feedback")
