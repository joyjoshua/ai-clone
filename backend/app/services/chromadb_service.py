"""
ChromaDB Vector Database Service
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from app.config import settings

logger = logging.getLogger(__name__)

class ChromaDBService:
    """ChromaDB service for vector storage and retrieval"""
    
    def __init__(self):
        self.client: Optional[PersistentClient] = None
        self.collection = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize ChromaDB client and collection"""
        try:
            self.client = PersistentClient(
                path=settings.CHROMADB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMADB_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "description": "AI Clone knowledge base"}
            )
            
            self._initialized = True
            logger.info(f"[OK] ChromaDB initialized: {settings.CHROMADB_COLLECTION_NAME}")
            return True
            
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize ChromaDB: {e}")
            raise
    
    async def ensure_initialized(self):
        """Ensure ChromaDB is initialized"""
        if not self._initialized:
            await self.initialize()
    
    async def store_embeddings(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Store document chunks with embeddings
        
        Args:
            chunks: List of text chunks
            embeddings: List of embedding vectors
            metadata: List of metadata dicts for each chunk
            ids: Optional list of IDs (auto-generated if not provided)
        
        Returns:
            List of stored chunk IDs
        """
        await self.ensure_initialized()
        
        try:
            # Generate IDs if not provided
            if ids is None:
                from uuid import uuid4
                ids = [str(uuid4()) for _ in chunks]
            
            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadata
            )
            
            logger.info(f"[OK] Stored {len(chunks)} chunks in ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"[ERROR] Error storing embeddings: {e}")
            raise
    
    async def query_similar(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query similar vectors
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
        
        Returns:
            List of results with documents, metadata, and distances
        """
        await self.ensure_initialized()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    formatted_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i],
                        "score": 1 - results["distances"][0][i]  # Convert distance to similarity
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"[ERROR] Error querying ChromaDB: {e}")
            raise
    
    async def delete_chunks(self, chunk_ids: List[str]) -> bool:
        """Delete chunks by IDs"""
        await self.ensure_initialized()
        
        try:
            self.collection.delete(ids=chunk_ids)
            logger.info(f"[OK] Deleted {len(chunk_ids)} chunks from ChromaDB")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Error deleting chunks: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        await self.ensure_initialized()
        
        try:
            count = self.collection.count()
            return {
                "collection_name": settings.CHROMADB_COLLECTION_NAME,
                "total_chunks": count,
                "path": settings.CHROMADB_PATH
            }
        except Exception as e:
            logger.error(f"[ERROR] Error getting collection stats: {e}")
            return {}
    
    async def test_connection(self) -> bool:
        """Test ChromaDB connection"""
        try:
            await self.ensure_initialized()
            stats = await self.get_collection_stats()
            return stats.get("total_chunks", 0) >= 0  # Just check if we can access
        except Exception as e:
            logger.error(f"ChromaDB connection test failed: {e}")
            return False

# Global instance
chromadb_service = ChromaDBService()

