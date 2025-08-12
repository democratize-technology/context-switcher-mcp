#!/usr/bin/env python3
"""
Comprehensive script to fix all circular import issues by systematically
replacing logging_config imports with logging_base imports where needed.
"""

import os
import re
from pathlib import Path

def fix_logging_imports_in_file(file_path: Path) -> bool:
    """Fix logging imports in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace all logging_config imports with logging_base imports
        # This pattern handles various relative import depths
        patterns_to_replace = [
            (r'from \.logging_config import get_logger', r'from .logging_base import get_logger'),
            (r'from \.\.logging_config import get_logger', r'from ..logging_base import get_logger'),
            (r'from \.\.\.logging_config import get_logger', r'from ...logging_base import get_logger'),
            (r'from \.logging_config import', r'from .logging_base import'),
            (r'from \.\.logging_config import', r'from ..logging_base import'),
            (r'from \.\.\.logging_config import', r'from ...logging_base import'),
        ]
        
        for old_pattern, new_pattern in patterns_to_replace:
            content = re.sub(old_pattern, new_pattern, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed logging imports in: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def should_skip_file(file_path: Path) -> bool:
    """Determine if a file should be skipped."""
    skip_files = {
        'logging_config.py',  # Main logging config - keep as is
        'logging_base.py',    # New base logging - keep as is
        '__pycache__',        # Skip cache directories
    }
    
    return any(skip in str(file_path) for skip in skip_files)

def main():
    """Main function to fix all circular import issues."""
    base_path = Path("src/context_switcher_mcp")
    
    if not base_path.exists():
        print(f"Base path not found: {base_path}")
        return
    
    print("Systematically fixing ALL circular import issues...")
    
    fixed_count = 0
    total_files = 0
    
    # Find all Python files that need fixing
    for py_file in base_path.rglob("*.py"):
        if should_skip_file(py_file):
            continue
            
        # Check if file has logging_config imports
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "from .logging_config import" in content:
                total_files += 1
                if fix_logging_imports_in_file(py_file):
                    fixed_count += 1
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    print(f"\nCompleted: Fixed imports in {fixed_count}/{total_files} files")
    print(f"All circular imports should now be resolved.")

if __name__ == "__main__":
    main()