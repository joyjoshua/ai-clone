"""
FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.config import settings
from app.services.database import connect_to_mongo, close_mongo_connection
from app.services.chromadb_service import chromadb_service
from app.api.routes import health
from app.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting AI Clone API...")
    try:
        # Connect to MongoDB
        await connect_to_mongo()
        
        # Initialize ChromaDB
        await chromadb_service.initialize()
        
        logger.info("✅ All services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Clone API...")
    await close_mongo_connection()
    logger.info("✅ Shutdown complete")

app = FastAPI(
    title="AI Clone API",
    description="RAG-based AI Clone application API",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Clone API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }

