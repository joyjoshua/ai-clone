"""
Application Configuration
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # MongoDB Configuration
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "ai_clone_db"
    
    # Vector Database Configuration
    VECTOR_DB: str = "chromadb"  # "chromadb" or "qdrant"
    CHROMADB_PATH: str = "./chroma_db_data"
    CHROMADB_COLLECTION_NAME: str = "ai_clone_vectors"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "ai_clone_vectors"
    
    # Groq Configuration
    GROQ_API_KEY: str = ""  # Load from .env file - DO NOT hardcode API keys
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = "sentence-transformers"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    OPENAI_API_KEY: str = ""
    
    # Server Configuration
    BACKEND_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:5173"
    
    # File Upload Configuration
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_FILE_TYPES: str = "pdf,txt,md"
    
    # RAG Configuration
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TOP_K_RETRIEVAL: int = 4
    SCORE_THRESHOLD: float = 0.7
    MEMORY_HISTORY_LENGTH: int = 5
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

