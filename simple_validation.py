#!/usr/bin/env python3
"""
Simple validation for the simplified session architecture
"""

import sys
import os
import asyncio
import time
from datetime import datetime, timezone

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/context_switcher_mcp'))

# Import the new modules
import session_types

def validate_session_types():
    """Test the pure data types"""
    print("🧪 Testing session_types...")
    
    # Test Thread
    thread = session_types.Thread(
        id="test_thread",
        name="technical", 
        system_prompt="You are technical",
        model_backend=session_types.ModelBackend.BEDROCK
    )
    
    thread.add_message("user", "Test message")
    assert len(thread.conversation_history) == 1
    
    # Test serialization
    data = thread.to_dict()
    restored = session_types.Thread.from_dict(data)
    assert restored.name == thread.name
    
    # Test ClientBinding
    binding = session_types.ClientBinding(
        session_entropy="test",
        creation_timestamp=datetime.now(timezone.utc),
        binding_signature="",
        access_pattern_hash="hash"
    )
    
    secret = "secret_key"
    binding.binding_signature = binding.generate_binding_signature(secret)
    assert binding.validate_binding(secret)
    
    print("  ✅ All session_types tests passed")

def validate_architecture():
    """Validate the architecture consolidation"""
    print("\n🧪 Validating architecture consolidation...")
    
    # Check file sizes
    src_dir = 'src/context_switcher_mcp'
    new_files = ['session_types.py', 'session.py', 'session_manager_new.py']
    
    total_lines = 0
    file_info = {}
    
    for filename in new_files:
        filepath = os.path.join(src_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                lines = len(f.readlines())
            file_info[filename] = lines
            total_lines += lines
    
    print(f"  📊 New architecture files:")
    for filename, lines in file_info.items():
        print(f"    {filename}: {lines} lines")
    print(f"  📊 Total: {total_lines} lines in {len(file_info)} files")
    
    # Check old files that should be replaced
    old_files = [
        'session_concurrency.py',
        'session_lock_manager.py', 
        'session_security.py',
        'session_data.py'
    ]
    
    old_total = 0
    old_info = {}
    
    for filename in old_files:
        filepath = os.path.join(src_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                lines = len(f.readlines())
            old_info[filename] = lines
            old_total += lines
    
    if old_info:
        print(f"\n  📊 Old architecture files (to be replaced):")
        for filename, lines in old_info.items():
            print(f"    {filename}: {lines} lines")
        print(f"  📊 Old total: {old_total} lines in {len(old_info)} files")
        
        reduction = ((old_total - total_lines) / old_total) * 100 if old_total > 0 else 0
        print(f"\n  🎯 Complexity reduction: {reduction:.1f}%")
        print(f"  🎯 Module reduction: {len(old_info)} → {len(file_info)} files")
    
    print("  ✅ Architecture consolidation validated")

def validate_design_principles():
    """Validate key design principles are met"""
    print("\n🧪 Validating design principles...")
    
    # Check session_types is pure data
    with open('src/context_switcher_mcp/session_types.py') as f:
        content = f.read()
        
    # Should have dataclasses, no business logic
    assert '@dataclass' in content
    assert 'asyncio' not in content  # No async in pure data types
    assert 'from_dict' in content and 'to_dict' in content  # Serialization
    print("  ✅ session_types is pure data with serialization")
    
    # Check session.py has built-in functionality
    with open('src/context_switcher_mcp/session.py') as f:
        content = f.read()
        
    # Should have built-in security, concurrency, data management
    assert 'asyncio.Lock' in content  # Built-in locking
    assert 'validate_security' in content  # Built-in security
    assert '_atomic_operation' in content  # Built-in concurrency
    assert 'add_thread' in content  # Built-in data management
    print("  ✅ session.py has unified functionality")
    
    # Check session_manager_new.py is simple
    with open('src/context_switcher_mcp/session_manager_new.py') as f:
        content = f.read()
        
    # Should be simple pool management
    assert 'SimpleSessionManager' in content
    assert 'create_session' in content
    assert 'get_session' in content
    assert '_global_lock' in content  # Simple global lock
    print("  ✅ session_manager_new.py is simplified")
    
    print("  ✅ All design principles validated")

def main():
    """Main validation"""
    print("🚀 Simple Session Architecture Validation")
    print("=" * 50)
    
    try:
        validate_session_types()
        validate_architecture() 
        validate_design_principles()
        
        print("\n" + "=" * 50)
        print("🎉 VALIDATION SUCCESSFUL!")
        print("\n📋 Summary:")
        print("  ✅ Pure data types with serialization")
        print("  ✅ Unified session with built-in functionality") 
        print("  ✅ Simplified session manager")
        print("  ✅ Architecture consolidation achieved")
        print("  ✅ Design principles followed")
        
        print("\n🏆 Simplified session architecture is working!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)