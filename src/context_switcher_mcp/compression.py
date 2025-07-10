"""Text compression and token estimation for managing LLM limits"""

import re
from typing import Dict, Tuple


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
    """Estimate token count using Claude-specific heuristics.
    
    Based on empirical analysis:
    - Average English word ≈ 1.3 tokens
    - Average character ≈ 0.25 tokens (4 chars per token)
    - Punctuation and whitespace affect tokenization
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    # Count words (split on whitespace and punctuation)
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    
    # Count special characters that often become separate tokens
    special_chars = len(re.findall(r'[^\w\s]', text))
    
    # Estimate based on multiple factors
    char_estimate = len(text) / 4.0
    word_estimate = word_count * 1.3
    
    # Weight towards word estimate for normal text
    if word_count > 0:
        # Mix of character and word-based estimates
        estimate = (char_estimate * 0.3) + (word_estimate * 0.7) + (special_chars * 0.5)
    else:
        # Fall back to character estimate for non-word content
        estimate = char_estimate
    
    return int(estimate)


def prepare_synthesis_input(perspectives: Dict[str, str], max_total_chars: int = 12000) -> Tuple[str, Dict[str, any]]:
    """Prepare perspectives for synthesis within token limits.
    
    Args:
        perspectives: Dictionary of perspective responses
        max_total_chars: Maximum total characters for synthesis
        
    Returns:
        Tuple of (formatted text, compression stats)
    """
    # Calculate per-perspective limit
    num_perspectives = len(perspectives)
    if num_perspectives == 0:
        return "", {"perspectives": 0, "total_chars": 0, "estimated_tokens": 0}
    
    # Reserve some space for formatting
    available_chars = max_total_chars - (num_perspectives * 50)  # ~50 chars per header
    chars_per_perspective = available_chars // num_perspectives
    
    # Track original sizes
    original_chars = sum(len(content) for content in perspectives.values())
    original_tokens = sum(estimate_token_count(content) for content in perspectives.values())
    
    # Compress each perspective
    compressed = compress_perspectives(perspectives, chars_per_perspective)
    
    # Format for synthesis
    sections = []
    for name, content in compressed.items():
        sections.append(f"### {name.upper()} PERSPECTIVE\n{content}")
    
    result = "\n\n".join(sections)
    
    # Calculate compression stats
    final_chars = len(result)
    final_tokens = estimate_token_count(result)
    
    stats = {
        "perspectives": num_perspectives,
        "original_chars": original_chars,
        "original_tokens": original_tokens,
        "final_chars": final_chars,
        "final_tokens": final_tokens,
        "compression_ratio": f"{(1 - final_chars/original_chars)*100:.1f}%" if original_chars > 0 else "0%",
        "chars_per_perspective": chars_per_perspective
    }
    
    return result, stats
