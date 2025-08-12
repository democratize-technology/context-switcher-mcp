#!/usr/bin/env python3
"""
Script to systematically fix logging_config import paths throughout the codebase.

This script identifies files with incorrect relative imports and fixes them based on
their directory depth relative to the logging_config.py file.
"""

import os
import re
from pathlib import Path

def get_relative_depth(file_path: Path, base_path: Path) -> int:
    """Calculate how many directory levels deep the file is relative to base."""
    try:
        relative_path = file_path.relative_to(base_path)
        return len(relative_path.parent.parts)
    except ValueError:
        return 0

def fix_imports_in_file(file_path: Path, depth: int) -> bool:
    """Fix logging_config imports in a single file based on its depth."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern to match logging_config imports
        patterns_to_fix = [
            (r'from \.logging_config import', f'from {"." * (depth + 1)}logging_config import'),
            (r'from \.logging_config import get_logger', f'from {"." * (depth + 1)}logging_config import get_logger'),
        ]
        
        for old_pattern, new_replacement in patterns_to_fix:
            content = re.sub(old_pattern, new_replacement, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed imports in: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix all import issues."""
    base_path = Path("src/context_switcher_mcp")
    logging_config_path = base_path / "logging_config.py"
    
    if not logging_config_path.exists():
        print(f"logging_config.py not found at {logging_config_path}")
        return
    
    print("Fixing import statements in all Python files...")
    
    fixed_count = 0
    total_files = 0
    
    # Find all Python files with incorrect imports
    for py_file in base_path.rglob("*.py"):
        if py_file.name == "logging_config.py":
            continue  # Skip the main logging_config file itself
            
        # Calculate depth relative to the base directory
        depth = get_relative_depth(py_file, base_path)
        
        # Only process files that have the incorrect import pattern
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "from .logging_config import" in content:
                total_files += 1
                if fix_imports_in_file(py_file, depth):
                    fixed_count += 1
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    print(f"\nCompleted: Fixed imports in {fixed_count}/{total_files} files")

if __name__ == "__main__":
    main()