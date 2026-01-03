"""
Document Model
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from bson import ObjectId
from app.models.user import PyObjectId


class Document(BaseModel):
    """Document model for MongoDB"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User ID who owns this document")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Full document content")
    file_type: Literal["pdf", "txt", "md", "text", "markdown"] = Field(
        ..., 
        description="File type"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (file size, original filename, etc.)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    chunk_ids: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs stored in vector database"
    )
    status: Literal["processing", "completed", "failed"] = Field(
        default="processing",
        description="Processing status"
    )

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str},
        "json_schema_extra": {
            "example": {
                "user_id": "user123",
                "title": "My Document",
                "content": "Document content here...",
                "file_type": "pdf",
                "metadata": {
                    "file_size": 1024,
                    "original_filename": "document.pdf"
                },
                "chunk_ids": ["chunk1", "chunk2"],
                "status": "completed"
            }
        }
    }

