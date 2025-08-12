#!/usr/bin/env python3
"""
Script to fix circular import issues by replacing logging_config imports
with logging_base imports where appropriate.
"""

import os
import re
from pathlib import Path

def should_use_logging_base(file_path: Path) -> bool:
    """Determine if a file should use logging_base instead of logging_config."""
    
    # Files that should use the simpler logging_base to avoid circular imports
    circular_import_files = [
        "security/secret_key_manager.py",
        "security/secure_logging.py", 
        "security/security_event_tracker.py",
        "security/client_binding_core.py",
        "security/client_validation_service.py",
        "security/path_validator.py",
        "security/security_monitor.py",
        "security/enhanced_validators.py"
    ]
    
    # Convert to string for easier matching
    file_str = str(file_path.relative_to(Path("src/context_switcher_mcp")))
    
    return any(file_str.endswith(pattern) for pattern in circular_import_files)

def fix_circular_imports_in_file(file_path: Path) -> bool:
    """Fix circular imports by switching to logging_base where appropriate."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        if should_use_logging_base(file_path):
            # Replace logging_config imports with logging_base
            content = re.sub(
                r'from (\.+)logging_config import get_logger',
                r'from \1logging_base import get_logger',
                content
            )
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed circular imports in: {file_path}")
            return True
        
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix circular import issues."""
    base_path = Path("src/context_switcher_mcp")
    
    if not base_path.exists():
        print(f"Base path not found: {base_path}")
        return
    
    print("Fixing circular import issues...")
    
    fixed_count = 0
    total_files = 0
    
    # Find all Python files that might need fixing
    for py_file in base_path.rglob("*.py"):
        if py_file.name in ["logging_config.py", "logging_base.py"]:
            continue  # Skip the logging modules themselves
            
        # Check if file has logging imports that might cause circular dependencies
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "from .logging_config import" in content or "from ..logging_config import" in content:
                total_files += 1
                if fix_circular_imports_in_file(py_file):
                    fixed_count += 1
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    print(f"\nCompleted: Fixed circular imports in {fixed_count}/{total_files} files")

if __name__ == "__main__":
    main()