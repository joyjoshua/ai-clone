"""
Test script to verify database connections
Run this after installing dependencies to test MongoDB and ChromaDB connections
"""
import asyncio
import sys
from app.services.database import connect_to_mongo, test_connection, close_mongo_connection
from app.services.chromadb_service import chromadb_service
from app.utils.logger import logger

async def test_databases():
    """Test all database connections"""
    print("=" * 60)
    print("Database Connection Tests")
    print("=" * 60)
    
    # Test MongoDB
    print("\n[TEST] Testing MongoDB connection...")
    try:
        await connect_to_mongo()
        mongo_status = await test_connection()
        if mongo_status:
            print("[OK] MongoDB: Connected successfully")
        else:
            print("[FAIL] MongoDB: Connection failed")
            return False
    except Exception as e:
        print(f"[ERROR] MongoDB: Error - {e}")
        print("   Make sure MongoDB is running on localhost:27017")
        print("   Or update MONGODB_URL in .env file")
        return False
    
    # Test ChromaDB
    print("\n[TEST] Testing ChromaDB connection...")
    try:
        await chromadb_service.initialize()
        chromadb_status = await chromadb_service.test_connection()
        stats = await chromadb_service.get_collection_stats()
        
        if chromadb_status:
            print("[OK] ChromaDB: Connected successfully")
            print(f"   Collection: {stats.get('collection_name', 'N/A')}")
            print(f"   Total chunks: {stats.get('total_chunks', 0)}")
            print(f"   Path: {stats.get('path', 'N/A')}")
        else:
            print("[FAIL] ChromaDB: Connection failed")
            return False
    except Exception as e:
        print(f"[ERROR] ChromaDB: Error - {e}")
        return False
    
    # Cleanup
    await close_mongo_connection()
    
    print("\n" + "=" * 60)
    print("[SUCCESS] All database connections successful!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = asyncio.run(test_databases())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[WARNING] Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        sys.exit(1)

