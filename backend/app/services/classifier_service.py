"""
Classifier service for dependency injection
"""

from ml.scripts.inference import FakeNewsClassifier
from backend.app.core.config import get_settings
from loguru import logger

# Global classifier instance
_classifier = None


def get_classifier() -> FakeNewsClassifier:
    """
    Get or create classifier instance (singleton pattern)
    
    Returns:
        FakeNewsClassifier instance
    """
    global _classifier
    
    if _classifier is None:
        settings = get_settings()
        logger.info(f"Initializing classifier with model: {settings.MODEL_PATH}")
        _classifier = FakeNewsClassifier(model_path=settings.MODEL_PATH)
    
    return _classifier
