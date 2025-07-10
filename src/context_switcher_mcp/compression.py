"""Simple text compression for managing token limits in synthesis."""

import re
from typing import Dict


def compress_perspectives(perspectives: Dict[str, str], max_chars_per_perspective: int = 2000) -> Dict[str, str]:
    """Compress perspective responses to fit within token limits.
    
    Args:
        perspectives: Dictionary of perspective name to response text
        max_chars_per_perspective: Maximum characters per perspective
        
    Returns:
        Compressed perspectives dictionary
    """
    compressed = {}
    
    for name, response in perspectives.items():
        if len(response) <= max_chars_per_perspective:
            compressed[name] = response
        else:
            # Smart truncation - try to keep complete sentences
            compressed[name] = truncate_text(response, max_chars_per_perspective)
    
    return compressed


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text intelligently at sentence boundaries.
    
    Args:
        text: Text to truncate
        max_chars: Maximum character count
        
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Keep adding sentences until we exceed the limit
    result = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length + 1 <= max_chars:
            result.append(sentence)
            current_length += sentence_length + 1
        else:
            # If we haven't added any sentences yet, truncate the first one
            if not result:
                result.append(sentence[:max_chars-3] + "...")
            break
    
    truncated = ' '.join(result)
    
    # If still too long (shouldn't happen), hard truncate
    if len(truncated) > max_chars:
        truncated = truncated[:max_chars-3] + "..."
    
    return truncated


def estimate_token_count(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 characters)."""
    return len(text) // 4


def prepare_synthesis_input(perspectives: Dict[str, str], max_total_chars: int = 12000) -> str:
    """Prepare perspectives for synthesis within token limits.
    
    Args:
        perspectives: Dictionary of perspective responses
        max_total_chars: Maximum total characters for synthesis
        
    Returns:
        Formatted text ready for synthesis
    """
    # Calculate per-perspective limit
    num_perspectives = len(perspectives)
    if num_perspectives == 0:
        return ""
    
    # Reserve some space for formatting
    available_chars = max_total_chars - (num_perspectives * 50)  # ~50 chars per header
    chars_per_perspective = available_chars // num_perspectives
    
    # Compress each perspective
    compressed = compress_perspectives(perspectives, chars_per_perspective)
    
    # Format for synthesis
    sections = []
    for name, content in compressed.items():
        sections.append(f"### {name.upper()} PERSPECTIVE\n{content}")
    
    return "\n\n".join(sections)
