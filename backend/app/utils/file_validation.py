"""
File Validation Utility
"""
import os
from typing import Tuple, Optional
from fastapi import UploadFile, HTTPException
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Try to import python-magic, fallback to extension-based validation
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logger.warning("python-magic not available, using extension-based validation")

# Allowed file types mapping
ALLOWED_MIME_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/x-markdown": "md"
}

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

def validate_file_type(file: UploadFile) -> Tuple[str, bool]:
    """
    Validate file type using magic bytes (if available) or file extension
    
    Returns:
        Tuple of (file_type, is_valid)
    """
    try:
        filename = file.filename or ""
        ext = os.path.splitext(filename.lower())[1]
        
        # Try magic bytes first if available
        if HAS_MAGIC:
            try:
                # Read first chunk to determine MIME type
                file_content = file.file.read(1024)
                file.file.seek(0)  # Reset file pointer
                
                # Use python-magic to detect MIME type
                mime_type = magic.from_buffer(file_content, mime=True)
                
                # Check if MIME type is allowed
                if mime_type in ALLOWED_MIME_TYPES:
                    file_type = ALLOWED_MIME_TYPES[mime_type]
                    return file_type, True
            except Exception as e:
                logger.warning(f"Magic bytes detection failed, using extension: {e}")
        
        # Fallback: check file extension
        if ext in ALLOWED_EXTENSIONS:
            file_type = ext[1:]  # Remove the dot
            if not HAS_MAGIC:
                logger.info(f"File type detected by extension: {file_type}")
            return file_type, True
        
        return None, False
        
    except Exception as e:
        logger.error(f"Error validating file type: {e}")
        return None, False


def validate_file_size(file: UploadFile) -> bool:
    """Validate file size"""
    try:
        # Get file size
        file.file.seek(0, os.SEEK_END)
        file_size = file.file.tell()
        file.file.seek(0)  # Reset file pointer
        
        max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        
        if file_size > max_size_bytes:
            return False
        
        if file_size == 0:
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating file size: {e}")
        return False


def validate_file(file: UploadFile) -> Tuple[str, Optional[str]]:
    """
    Comprehensive file validation
    
    Returns:
        Tuple of (file_type, error_message)
        If valid: (file_type, None)
        If invalid: (None, error_message)
    """
    # Check file size
    if not validate_file_size(file):
        max_size_mb = settings.MAX_FILE_SIZE_MB
        return None, f"File size exceeds maximum allowed size of {max_size_mb}MB"
    
    # Check file type
    file_type, is_valid = validate_file_type(file)
    if not is_valid:
        allowed_types = ", ".join(settings.ALLOWED_FILE_TYPES.split(","))
        return None, f"File type not allowed. Allowed types: {allowed_types}"
    
    return file_type, None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and other issues"""
    # Remove path components
    filename = os.path.basename(filename)
    
    # Remove or replace dangerous characters
    dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename

