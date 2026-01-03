"""
Chat API Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatMessageRequest(BaseModel):
    """Request schema for chat message"""
    message: str = Field(..., description="User message", min_length=1, max_length=5000)
    conversation_id: Optional[str] = Field(None, description="Existing conversation ID")
    user_id: str = Field(..., description="User ID", min_length=1)

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is artificial intelligence?",
                "conversation_id": None,
                "user_id": "user123"
            }
        }


class Source(BaseModel):
    """Source citation schema"""
    document_id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    text: str = Field(..., description="Relevant chunk text")
    score: float = Field(..., description="Similarity score", ge=0.0, le=1.0)
    chunk_index: Optional[int] = Field(None, description="Chunk index in document")


class ChatMessageResponse(BaseModel):
    """Response schema for chat message"""
    response: str = Field(..., description="AI generated response")
    conversation_id: str = Field(..., description="Conversation ID")
    sources: List[Source] = Field(default_factory=list, description="Source citations")
    metadata: Optional[dict] = Field(None, description="Response metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Artificial Intelligence (AI) is...",
                "conversation_id": "conv123",
                "sources": [
                    {
                        "document_id": "doc123",
                        "title": "AI Basics",
                        "text": "AI is the simulation...",
                        "score": 0.85,
                        "chunk_index": 2
                    }
                ],
                "metadata": {
                    "model": "llama-3.1-70b-versatile",
                    "tokens_used": 150,
                    "response_time_ms": 450,
                    "memory_usage": 5,
                    "evaluation_score": 0.85
                }
            }
        }


class ConversationResponse(BaseModel):
    """Conversation response schema"""
    id: str = Field(..., alias="_id")
    user_id: str
    messages: List[dict]
    created_at: datetime
    updated_at: datetime

    class Config:
        populate_by_name = True

