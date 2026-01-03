"""
MongoDB CRUD Operations
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from app.models.user import User
from app.models.conversation import Conversation, Message
from app.models.document import Document
from app.services.database import get_database
from app.utils.logger import logger


class CRUDService:
    """CRUD operations for MongoDB collections"""
    
    @staticmethod
    async def get_collection(collection_name: str) -> AsyncIOMotorCollection:
        """Get MongoDB collection"""
        db = await get_database()
        return db[collection_name]


class UserCRUD(CRUDService):
    """User CRUD operations"""
    
    @staticmethod
    async def create_user(user: User) -> User:
        """Create a new user"""
        collection = await UserCRUD.get_collection("users")
        user_dict = user.model_dump(by_alias=True, exclude={"id"})
        result = await collection.insert_one(user_dict)
        user.id = result.inserted_id
        return user
    
    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[User]:
        """Get user by user_id"""
        collection = await UserCRUD.get_collection("users")
        user_doc = await collection.find_one({"user_id": user_id})
        if user_doc:
            return User(**user_doc)
        return None
    
    @staticmethod
    async def get_or_create_user(user_id: str) -> User:
        """Get existing user or create new one"""
        user = await UserCRUD.get_user_by_id(user_id)
        if user:
            return user
        # Create new user
        new_user = User(user_id=user_id)
        return await UserCRUD.create_user(new_user)


class ConversationCRUD(CRUDService):
    """Conversation CRUD operations"""
    
    @staticmethod
    async def create_conversation(user_id: str) -> Conversation:
        """Create a new conversation"""
        collection = await ConversationCRUD.get_collection("conversations")
        conversation = Conversation(user_id=user_id)
        conv_dict = conversation.model_dump(by_alias=True, exclude={"id"})
        result = await collection.insert_one(conv_dict)
        conversation.id = result.inserted_id
        return conversation
    
    @staticmethod
    async def get_conversation(conversation_id: str) -> Optional[Conversation]:
        """Get conversation by ID"""
        collection = await ConversationCRUD.get_collection("conversations")
        if not ObjectId.is_valid(conversation_id):
            return None
        conv_doc = await collection.find_one({"_id": ObjectId(conversation_id)})
        if conv_doc:
            return Conversation(**conv_doc)
        return None
    
    @staticmethod
    async def get_user_conversations(
        user_id: str,
        skip: int = 0,
        limit: int = 20
    ) -> List[Conversation]:
        """Get all conversations for a user"""
        collection = await ConversationCRUD.get_collection("conversations")
        cursor = collection.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        conversations = []
        async for doc in cursor:
            conversations.append(Conversation(**doc))
        return conversations
    
    @staticmethod
    async def add_message(
        conversation_id: str,
        role: str,
        content: str
    ) -> Optional[Conversation]:
        """Add a message to a conversation"""
        collection = await ConversationCRUD.get_collection("conversations")
        if not ObjectId.is_valid(conversation_id):
            return None
        
        message = Message(role=role, content=content)
        result = await collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {
                "$push": {"messages": message.model_dump()},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        if result.modified_count > 0:
            return await ConversationCRUD.get_conversation(conversation_id)
        return None


class DocumentCRUD(CRUDService):
    """Document CRUD operations"""
    
    @staticmethod
    async def create_document(document: Document) -> Document:
        """Create a new document"""
        collection = await DocumentCRUD.get_collection("documents")
        doc_dict = document.model_dump(by_alias=True, exclude={"id"})
        result = await collection.insert_one(doc_dict)
        document.id = result.inserted_id
        return document
    
    @staticmethod
    async def get_document(document_id: str) -> Optional[Document]:
        """Get document by ID"""
        collection = await DocumentCRUD.get_collection("documents")
        if not ObjectId.is_valid(document_id):
            return None
        doc = await collection.find_one({"_id": ObjectId(document_id)})
        if doc:
            return Document(**doc)
        return None
    
    @staticmethod
    async def get_user_documents(
        user_id: str,
        skip: int = 0,
        limit: int = 20
    ) -> List[Document]:
        """Get all documents for a user"""
        collection = await DocumentCRUD.get_collection("documents")
        cursor = collection.find({"user_id": user_id}).sort("created_at", -1).skip(skip).limit(limit)
        documents = []
        async for doc in cursor:
            documents.append(Document(**doc))
        return documents
    
    @staticmethod
    async def update_document(
        document_id: str,
        update_data: Dict[str, Any]
    ) -> Optional[Document]:
        """Update a document"""
        collection = await DocumentCRUD.get_collection("documents")
        if not ObjectId.is_valid(document_id):
            return None
        
        result = await collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": update_data}
        )
        
        if result.modified_count > 0:
            return await DocumentCRUD.get_document(document_id)
        return None
    
    @staticmethod
    async def delete_document(document_id: str) -> bool:
        """Delete a document"""
        collection = await DocumentCRUD.get_collection("documents")
        if not ObjectId.is_valid(document_id):
            return False
        
        result = await collection.delete_one({"_id": ObjectId(document_id)})
        return result.deleted_count > 0
    
    @staticmethod
    async def count_user_documents(user_id: str) -> int:
        """Count documents for a user"""
        collection = await DocumentCRUD.get_collection("documents")
        return await collection.count_documents({"user_id": user_id})

