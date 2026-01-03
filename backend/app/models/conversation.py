"""
Conversation Model
"""
from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from bson import ObjectId
from app.models.user import PyObjectId


class Message(BaseModel):
    """Individual message in a conversation"""
    role: Literal["user", "assistant"] = Field(..., description="Message role")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how are you?",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class Conversation(BaseModel):
    """Conversation model for MongoDB"""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User ID who owns this conversation")
    messages: List[Message] = Field(default_factory=list, description="List of messages")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        json_schema_extra = {
            "example": {
                "user_id": "user123",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is AI?",
                        "timestamp": "2024-01-01T12:00:00Z"
                    },
                    {
                        "role": "assistant",
                        "content": "AI stands for Artificial Intelligence...",
                        "timestamp": "2024-01-01T12:00:01Z"
                    }
                ]
            }
        }

    def add_message(self, role: Literal["user", "assistant"], content: str):
        """Add a message to the conversation"""
        message = Message(role=role, content=content)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

