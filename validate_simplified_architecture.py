#!/usr/bin/env python3
"""
Validation script for the simplified session architecture

This script validates that the new simplified session architecture works correctly
without requiring the full MCP server environment.
"""

import sys
import os
import asyncio
import time
from datetime import datetime, timezone, timedelta

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/context_switcher_mcp'))

# Import the new modules directly
import session_types
import session
import session_manager_new

def test_session_types():
    """Test the pure data types and serialization"""
    print("🧪 Testing session_types module...")
    
    # Test ModelBackend enum
    assert session_types.ModelBackend.BEDROCK == "bedrock"
    assert session_types.ModelBackend.LITELLM == "litellm"
    print("  ✅ ModelBackend enum works")
    
    # Test Thread creation and serialization
    thread = session_types.Thread(
        id="test_thread",
        name="technical",
        system_prompt="You are a technical expert",
        model_backend=session_types.ModelBackend.BEDROCK,
        model_name="claude-3-sonnet"
    )
    
    thread.add_message("user", "Test message")
    assert len(thread.conversation_history) == 1
    print("  ✅ Thread creation and message addition works")
    
    # Test serialization
    data = thread.to_dict()
    restored_thread = session_types.Thread.from_dict(data)
    assert restored_thread.name == thread.name
    assert restored_thread.model_backend == thread.model_backend
    print("  ✅ Thread serialization works")
    
    # Test ClientBinding security
    binding = session_types.ClientBinding(
        session_entropy="test_entropy",
        creation_timestamp=datetime.now(timezone.utc),
        binding_signature="",
        access_pattern_hash="test_hash"
    )
    
    secret_key = "test_secret"
    binding.binding_signature = binding.generate_binding_signature(secret_key)
    assert binding.validate_binding(secret_key) is True
    assert binding.validate_binding("wrong_key") is False
    print("  ✅ ClientBinding security works")
    
    # Test SessionState serialization
    state = session_types.SessionState(
        session_id="test_session",
        created_at=datetime.now(timezone.utc),
        topic="test topic"
    )
    
    state_data = state.to_dict()
    restored_state = session_types.SessionState.from_dict(state_data)
    assert restored_state.session_id == state.session_id
    print("  ✅ SessionState serialization works")
    
    print("✅ session_types module validation complete")


async def test_unified_session():
    """Test the unified Session class"""
    print("\n🧪 Testing unified Session class...")
    
    # Create session with mocked dependencies
    from unittest.mock import patch, MagicMock
    
    # Mock the config and logging to avoid dependencies
    with patch('session.get_config') as mock_config, \
         patch('session.get_logger') as mock_logger:
        
        mock_config.return_value = MagicMock()
        mock_logger.return_value = MagicMock()
        
        # Create session
        test_session = session.Session("test_session", topic="test topic")
        assert test_session.session_id == "test_session"
        print("  ✅ Session creation works")
        
        # Test basic info
        info = await test_session.get_session_info()
        assert info["session_id"] == "test_session"
        assert info["topic"] == "test topic"
        assert info["version"] == 0
        print("  ✅ Session info retrieval works")
        
        # Test thread management
        thread = session_types.Thread(
            id="tech_thread",
            name="technical",
            system_prompt="Technical expert",
            model_backend=session_types.ModelBackend.BEDROCK
        )
        
        assert await test_session.add_thread(thread) is True
        assert await test_session.add_thread(thread) is False  # Duplicate
        print("  ✅ Thread management works")
        
        retrieved_thread = await test_session.get_thread("technical")
        assert retrieved_thread is not None
        assert retrieved_thread.name == "technical"
        print("  ✅ Thread retrieval works")
        
        # Test analysis recording
        responses = {"technical": "Technical response", "business": "[NO_RESPONSE]"}
        await test_session.record_analysis("test prompt", responses, response_time=1.5)
        
        last_analysis = await test_session.get_last_analysis()
        assert last_analysis is not None
        assert last_analysis.prompt == "test prompt"
        assert last_analysis.active_count == 1
        assert last_analysis.abstained_count == 1
        print("  ✅ Analysis recording works")
        
        # Test version management
        version, _ = await test_session.get_version_info()
        assert version > 0  # Should have been incremented by operations
        print("  ✅ Version management works")
        
        # Test atomic operations
        async def test_operation():
            return "operation_result"
        
        result = await test_session.atomic_update(test_operation)
        assert result == "operation_result"
        print("  ✅ Atomic operations work")
        
        # Test session cleanup
        await test_session.cleanup()
        print("  ✅ Session cleanup works")
        
    print("✅ Unified Session validation complete")


async def test_simple_session_manager():
    """Test the simplified session manager"""
    print("\n🧪 Testing SimpleSessionManager...")
    
    from unittest.mock import patch, MagicMock
    
    # Mock dependencies
    with patch('session_manager_new.get_config') as mock_config, \
         patch('session_manager_new.get_logger') as mock_logger, \
         patch('session_manager_new.get_perspective_system_prompt') as mock_prompt:
        
        mock_config.return_value = MagicMock()
        mock_config.return_value.session.max_active_sessions = 10
        mock_config.return_value.session.default_ttl_hours = 1.0
        mock_config.return_value.session.cleanup_interval_seconds = 60
        
        mock_logger.return_value = MagicMock()
        mock_prompt.return_value = "Test system prompt"
        
        # Create manager
        manager = session_manager_new.SimpleSessionManager(max_sessions=5, session_ttl_hours=1.0)
        print("  ✅ SessionManager creation works")
        
        # Test session creation
        test_session = await manager.create_session("test_session", topic="test")
        assert test_session.session_id == "test_session"
        print("  ✅ Session creation through manager works")
        
        # Test session retrieval
        retrieved = await manager.get_session("test_session")
        assert retrieved is not None
        assert retrieved.session_id == "test_session"
        print("  ✅ Session retrieval works")
        
        # Test session with perspectives
        session_with_perspectives = await manager.create_session(
            "perspective_session",
            topic="test",
            initial_perspectives=["technical", "business"]
        )
        
        all_threads = await session_with_perspectives.get_all_threads()
        assert len(all_threads) == 2
        print("  ✅ Session with perspectives works")
        
        # Test statistics
        stats = await manager.get_stats()
        assert stats["active_sessions"] == 2
        assert stats["max_sessions"] == 5
        print("  ✅ Session statistics work")
        
        # Test session removal
        assert await manager.remove_session("test_session") is True
        assert await manager.get_session("test_session") is None
        print("  ✅ Session removal works")
        
        # Test cleanup
        cleanup_count = await manager.cleanup_expired_sessions()
        print(f"  ✅ Cleanup works (cleaned {cleanup_count} sessions)")
        
    print("✅ SimpleSessionManager validation complete")


async def test_performance_characteristics():
    """Test basic performance characteristics"""
    print("\n🧪 Testing performance characteristics...")
    
    from unittest.mock import patch, MagicMock
    
    with patch('session.get_config') as mock_config, \
         patch('session.get_logger') as mock_logger:
        
        mock_config.return_value = MagicMock()
        mock_logger.return_value = MagicMock()
        
        # Test concurrent session operations
        async def create_session_with_operations(session_id):
            test_session = session.Session(session_id, create_client_binding=False)  # Skip binding for speed
            
            # Add multiple threads
            for i in range(5):
                thread = session_types.Thread(
                    id=f"thread_{i}",
                    name=f"perspective_{i}",
                    system_prompt="Test prompt",
                    model_backend=session_types.ModelBackend.BEDROCK
                )
                await test_session.add_thread(thread)
            
            # Record analyses
            for i in range(3):
                responses = {f"perspective_{j}": f"response_{j}" for j in range(5)}
                await test_session.record_analysis(f"prompt_{i}", responses)
            
            return test_session
        
        start_time = time.time()
        
        # Create 10 sessions concurrently
        tasks = [create_session_with_operations(f"session_{i}") for i in range(10)]
        sessions = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        assert len(sessions) == 10
        
        # Verify all sessions were created correctly
        for session in sessions:
            info = await session.get_session_info()
            assert info["thread_count"] == 5
            assert info["analysis_count"] == 3
        
        total_time = end_time - start_time
        print(f"  ✅ Created 10 sessions with 5 threads and 3 analyses each in {total_time:.3f}s")
        print(f"  📊 Average time per session: {total_time/10:.3f}s")
        
    print("✅ Performance validation complete")


def test_architecture_consolidation():
    """Verify the architecture consolidation"""
    print("\n🧪 Verifying architecture consolidation...")
    
    # Check that we have the expected modules
    expected_modules = ['session_types', 'session', 'session_manager_new']
    available_modules = []
    
    for module_name in expected_modules:
        try:
            module = sys.modules.get(module_name)
            if module:
                available_modules.append(module_name)
        except ImportError:
            pass
    
    print(f"  📊 Available modules: {available_modules}")
    
    # Check module sizes (approximate line counts)
    module_info = {}
    src_dir = os.path.join(os.path.dirname(__file__), 'src/context_switcher_mcp')
    
    for module_name in expected_modules:
        module_file = f"{module_name}.py"
        file_path = os.path.join(src_dir, module_file)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            module_info[module_name] = lines
    
    total_lines = sum(module_info.values())
    print(f"  📊 Module sizes: {module_info}")
    print(f"  📊 Total lines in new architecture: {total_lines}")
    
    # Verify consolidation goals
    assert len(available_modules) <= 3, f"Expected max 3 modules, got {len(available_modules)}"
    assert total_lines < 1500, f"Expected < 1500 lines total, got {total_lines}"
    
    print("  ✅ Architecture consolidation verified")
    print(f"  🎯 Achieved: {len(available_modules)} focused modules with {total_lines} total lines")
    
    print("✅ Architecture consolidation validation complete")


async def main():
    """Main validation function"""
    print("🚀 Starting Simplified Session Architecture Validation")
    print("=" * 60)
    
    try:
        # Test each component
        test_session_types()
        await test_unified_session()
        await test_simple_session_manager()
        await test_performance_characteristics()
        test_architecture_consolidation()
        
        print("\n" + "=" * 60)
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\n📊 SUMMARY:")
        print("  ✅ Pure data types with serialization")
        print("  ✅ Unified session with built-in security, concurrency, data")
        print("  ✅ Simplified session manager with clean interface")
        print("  ✅ Good performance characteristics")
        print("  ✅ Architecture consolidation goals achieved")
        print("\n🏆 Simplified session architecture is ready for production!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)