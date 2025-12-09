"""
Database models for AI Fake News Detector
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class Feedback(Base):
    """Model for storing user feedback on predictions"""
    
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    predicted_label = Column(String(10), nullable=False)  # "Real" or "Fake"
    confidence_score = Column(Float, nullable=False)
    user_correction = Column(String(10), nullable=True)  # User's correction if wrong
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Feedback(id={self.id}, predicted={self.predicted_label}, correction={self.user_correction})>"
