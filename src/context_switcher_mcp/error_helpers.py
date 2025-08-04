"""Error handling helper functions to reduce code duplication"""

from typing import Dict, Any, Optional

from .aorp import create_error_response
from .security import sanitize_error_message


def validation_error(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a validation error response
    
    Args:
        message: Error message
        context: Additional context data
        session_id: Optional session ID for tracking
        
    Returns:
        Formatted AORP error response
    """
    return create_error_response(
        f"Validation failed: {sanitize_error_message(message)}",
        "validation_error",
        context or {},
        recoverable=True,
        session_id=session_id,
    )


def session_not_found_error(
    session_id: str,
    hint: Optional[str] = None
) -> Dict[str, Any]:
    """Create a session not found error response
    
    Args:
        session_id: The session ID that wasn't found
        hint: Optional hint for the user
        
    Returns:
        Formatted AORP error response
    """
    message = f"Session '{session_id}' not found or expired"
    if hint:
        message += f". {hint}"
        
    return create_error_response(
        message,
        "session_not_found",
        {"session_id": session_id},
        recoverable=True,
    )


def rate_limit_error(
    message: str,
    retry_after_seconds: int,
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a rate limit error response
    
    Args:
        message: Error message
        retry_after_seconds: How long to wait before retrying
        session_id: Optional session ID
        
    Returns:
        Formatted AORP error response
    """
    context = {"retry_after_seconds": retry_after_seconds}
    if session_id:
        context["session_id"] = session_id
        
    return create_error_response(
        message,
        "rate_limited",  
        context,
        recoverable=True,
    )


def execution_error(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    recoverable: bool = True
) -> Dict[str, Any]:
    """Create an execution error response
    
    Args:
        message: Error message
        context: Additional context data
        session_id: Optional session ID
        recoverable: Whether the error is recoverable
        
    Returns:
        Formatted AORP error response
    """
    return create_error_response(
        f"Execution failed: {sanitize_error_message(message)}",
        "execution_error",
        context or {},
        recoverable=recoverable,
        session_id=session_id,
    )


def security_error(
    message: str,
    details: Dict[str, Any],
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create a security error response
    
    Args:
        message: Error message
        details: Security-related details
        session_id: Optional session ID
        
    Returns:
        Formatted AORP error response
    """
    return create_error_response(
        f"Security check failed: {sanitize_error_message(message)}",
        "security_error",
        details,
        recoverable=False,  # Security errors typically not recoverable
        session_id=session_id,
    )


def resource_limit_error(
    resource_type: str,
    limit: int,
    current: Optional[int] = None
) -> Dict[str, Any]:
    """Create a resource limit error response
    
    Args:
        resource_type: Type of resource (sessions, analyses, etc.)
        limit: The limit that was exceeded
        current: Current usage (optional)
        
    Returns:
        Formatted AORP error response
    """
    message = f"{resource_type.title()} limit reached ({limit})"
    context = {"resource_type": resource_type, "limit": limit}
    
    if current is not None:
        message += f". Current: {current}"
        context["current"] = current
        
    return create_error_response(
        message,
        "resource_limit",
        context,
        recoverable=True,
    )