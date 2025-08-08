# app.py - Complete Main Application
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import logging
from api.endpoints import register_endpoints
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure log directory exists and add file handler
settings.LOG_DIR.mkdir(exist_ok=True, parents=True)
file_handler = logging.FileHandler(settings.LOG_DIR / "api.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Create FastAPI app
app = FastAPI(
    title="Sense Face Verification API",
    version=settings.API_VERSION,
    description="Face verification service with async processing"
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3026",     
        "http://localhost:3010", "*"     # Optional: some browsers use IP instead of 'localhost'
    ],
    allow_credentials=True,         
    allow_methods=["*"],            
    allow_headers=["*"],         

)

# Register endpoints
register_endpoints(app)

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Sense Face Verification API",
        "version": settings.API_VERSION,
        "status": "running",
        "documentation": "/docs",
        "health_check": "/api/health"
    }

@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("🚀 Starting Face Verification API")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    logger.info("🛑 Shutting down Face Verification API")

