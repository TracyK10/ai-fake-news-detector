"""
Main FastAPI application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.routes import detection
from backend.app.database.db import init_db
from backend.app.core.config import get_settings
from backend.app.core.schemas import HealthResponse
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Initialize settings
settings = get_settings()

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API for detecting fake news using AI/ML models",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiting middleware
app.state.limiter = detection.limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    detection.router,
    prefix="/api/v1",
    tags=["Detection"]
)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up API server...")
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
    # Pre-load model (optional, for faster first request)
    # Uncomment if you want to load model at startup
    # from backend.app.services.classifier_service import get_classifier
    # get_classifier()
    # logger.info("Model pre-loaded")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server...")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model_loaded=True
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    # Check if model is loaded
    try:
        from backend.app.services.classifier_service import _classifier
        model_loaded = _classifier is not None
    except:
        model_loaded = False
    
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model_loaded=model_loaded
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
