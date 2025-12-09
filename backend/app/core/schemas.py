"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Optional
from datetime import datetime


class AnalyzeRequest(BaseModel):
    """Request schema for /analyze endpoint"""
    text: str = Field(..., min_length=10, max_length=10000, 
                      description="News text to analyze")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v.strip()


class AnalyzeResponse(BaseModel):
    """Response schema for /analyze endpoint"""
    label: str = Field(..., description="Predicted label: Real or Fake")
    confidence_score: float = Field(..., ge=0.0, le=1.0, 
                                    description="Confidence score between 0 and 1")
    probabilities: Dict[str, float] = Field(..., 
                                            description="Probability distribution")


class FeedbackRequest(BaseModel):
    """Request schema for /feedback endpoint"""
    text: str = Field(..., min_length=10, max_length=10000)
    predicted_label: str = Field(..., pattern="^(Real|Fake)$")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    user_correction: str = Field(..., pattern="^(Real|Fake)$",
                                 description="User's correction: Real or Fake")


class FeedbackResponse(BaseModel):
    """Response schema for /feedback endpoint"""
    message: str
    feedback_id: int


class HealthResponse(BaseModel):
    """Response schema for health check"""
    status: str
    version: str
    model_loaded: bool
