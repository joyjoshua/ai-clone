"""
MongoDB Database Connection Service
"""
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class Database:
    client: Optional[AsyncIOMotorClient] = None
    database: Optional[AsyncIOMotorDatabase] = None

db = Database()

async def connect_to_mongo():
    """Create database connection"""
    try:
        db.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            maxPoolSize=10,
            minPoolSize=1
        )
        db.database = db.client[settings.MONGODB_DB_NAME]
        
        # Test connection
        await db.client.admin.command('ping')
        logger.info(f"[OK] Connected to MongoDB: {settings.MONGODB_DB_NAME}")
        
        # Create indexes
        await create_indexes()
        
        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection"""
    if db.client:
        db.client.close()
        logger.info("MongoDB connection closed")

async def create_indexes():
    """Create database indexes for better query performance"""
    try:
        # Conversations collection indexes
        conversations_collection = db.database.conversations
        await conversations_collection.create_index([("user_id", 1), ("created_at", -1)])
        logger.info("[OK] Created indexes for conversations collection")
        
        # Documents collection indexes
        documents_collection = db.database.documents
        await documents_collection.create_index([("user_id", 1), ("created_at", -1)])
        await documents_collection.create_index([("title", "text")])
        logger.info("[OK] Created indexes for documents collection")
        
        # Users collection indexes
        users_collection = db.database.users
        await users_collection.create_index([("user_id", 1)], unique=True)
        logger.info("[OK] Created indexes for users collection")
        
    except Exception as e:
        logger.warning(f"[WARNING] Error creating indexes (may already exist): {e}")

async def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    if db.database is None:
        await connect_to_mongo()
    return db.database

async def test_connection() -> bool:
    """Test MongoDB connection"""
    try:
        if db.client is None:
            await connect_to_mongo()
        await db.client.admin.command('ping')
        return True
    except Exception as e:
        logger.error(f"MongoDB connection test failed: {e}")
        return False

