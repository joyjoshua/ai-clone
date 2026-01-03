"""
Health Check Endpoints
"""
from fastapi import APIRouter, HTTPException
from app.services.database import test_connection as test_mongo
from app.services.chromadb_service import chromadb_service

router = APIRouter(prefix="/health", tags=["health"])

@router.get("")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "ai-clone-api"
    }

@router.get("/database")
async def database_health():
    """Check MongoDB connection"""
    try:
        mongo_status = await test_mongo()
        return {
            "mongodb": "connected" if mongo_status else "disconnected",
            "status": "healthy" if mongo_status else "unhealthy"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"MongoDB connection failed: {str(e)}")

@router.get("/vector-db")
async def vector_db_health():
    """Check ChromaDB connection"""
    try:
        chromadb_status = await chromadb_service.test_connection()
        stats = await chromadb_service.get_collection_stats()
        return {
            "chromadb": "connected" if chromadb_status else "disconnected",
            "status": "healthy" if chromadb_status else "unhealthy",
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ChromaDB connection failed: {str(e)}")

@router.get("/full")
async def full_health_check():
    """Full health check for all services"""
    results = {
        "status": "healthy",
        "services": {}
    }
    
    # Check MongoDB
    try:
        mongo_status = await test_mongo()
        results["services"]["mongodb"] = "connected" if mongo_status else "disconnected"
    except Exception as e:
        results["services"]["mongodb"] = f"error: {str(e)}"
        results["status"] = "unhealthy"
    
    # Check ChromaDB
    try:
        chromadb_status = await chromadb_service.test_connection()
        stats = await chromadb_service.get_collection_stats()
        results["services"]["chromadb"] = {
            "status": "connected" if chromadb_status else "disconnected",
            "stats": stats
        }
    except Exception as e:
        results["services"]["chromadb"] = {"status": f"error: {str(e)}"}
        results["status"] = "unhealthy"
    
    return results

