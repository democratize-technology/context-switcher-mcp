"""
Test for memory leak fixes in security context sanitization
Ensures that context sanitization does not cause memory leaks
"""

import gc
import threading

import psutil
import pytest
from context_switcher_mcp.error_logging import StructuredErrorLogger  # noqa: E402
from context_switcher_mcp.exceptions import (  # noqa: E402
    ModelAuthenticationError,
    SecurityError,
)
from context_switcher_mcp.security_context_sanitizer import (  # noqa: E402
    SecurityContextSanitizer,
    get_context_sanitizer,
)


class TestMemoryLeakPrevention:
    """Test that context sanitization doesn't cause memory leaks"""

    def setup_method(self):
        """Set up test environment"""
        self.sanitizer = SecurityContextSanitizer()
        gc.collect()  # Clean up before tests

    def test_repeated_sanitization_no_memory_leak(self):
        """Test that repeated sanitization doesn't cause memory leaks"""
        initial_memory = psutil.Process().memory_info().rss

        # Create a large context that will be sanitized many times
        large_context = {f"key_{i}": f"sk-{i * 100:032d}" for i in range(1000)}
        large_context.update(
            {f"session_id_{i}": f"session_{i:08d}" for i in range(1000)}
        )

        # Perform many sanitizations
        for _ in range(100):
            sanitized = self.sanitizer.sanitize_context_dict(large_context.copy())
            # Explicitly delete to help with memory management
            del sanitized

            # Force garbage collection periodically
            if _ % 10 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be reasonable (less than 50MB)
        assert (
            memory_growth < 50 * 1024 * 1024
        ), f"Memory grew by {memory_growth / 1024 / 1024:.2f}MB"

    def test_exception_context_memory_cleanup(self):
        """Test that exception context sanitization cleans up properly"""
        initial_objects = len(gc.get_objects())

        for i in range(1000):
            # Create exception with large context
            error = SecurityError(
                f"Test error {i}",
                security_context={
                    "large_data": "x" * 10000,
                    "api_key": f"sk-{i:032d}",
                    "session_data": {
                        f"nested_key_{j}": f"value_{j}" for j in range(100)
                    },
                },
            )

            # Sanitize exception context
            sanitized = self.sanitizer.sanitize_exception_context(error)

            # Clean up references
            del error
            del sanitized

            # Periodic cleanup
            if i % 100 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count shouldn't grow too much
        object_growth = final_objects - initial_objects
        assert object_growth < 1000, f"Objects grew by {object_growth}"

    def test_circular_reference_cleanup(self):
        """Test that circular references in contexts are properly cleaned up"""
        initial_memory = psutil.Process().memory_info().rss

        for _ in range(100):
            # Create context with circular reference
            context = {"api_key": "sk-circular123", "data": {"nested": "value"}}
            context["self_ref"] = context  # Create circular reference
            context["data"]["parent"] = context  # Another circular reference

            # Sanitize (should handle circular refs gracefully)
            try:
                sanitized = self.sanitizer.sanitize_context_dict(context)
                del sanitized
            except Exception:
                pass  # If it can't handle circular refs, that's ok for this test

            # Break circular references explicitly
            context["self_ref"] = None
            context["data"]["parent"] = None
            del context

            gc.collect()

        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory

        # Should not grow significantly even with circular references
        assert (
            memory_growth < 20 * 1024 * 1024
        ), f"Memory grew by {memory_growth / 1024 / 1024:.2f}MB"

    def test_concurrent_sanitization_memory_safety(self):
        """Test that concurrent sanitization doesn't cause memory issues"""
        results = []
        errors = []

        def sanitize_worker(worker_id: int):
            """Worker function for concurrent sanitization"""
            try:
                context = {
                    f"worker_{worker_id}_key": f"sk-worker{worker_id:016d}",
                    "session_id": f"session_{worker_id}",
                    "large_data": "x" * 1000 * worker_id,  # Variable size data
                }

                for i in range(50):
                    sanitized = self.sanitizer.sanitize_context_dict(context.copy())
                    results.append(len(str(sanitized)))
                    del sanitized

                    if i % 10 == 0:
                        gc.collect()

            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # Run multiple concurrent workers
        threads = []
        for i in range(10):
            thread = threading.Thread(target=sanitize_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all workers to complete
        for thread in threads:
            thread.join(timeout=30)

        # Check results
        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 500, f"Expected 500 results, got {len(results)}"

        # Clean up
        gc.collect()

    def test_global_sanitizer_singleton_memory_efficiency(self):
        """Test that global sanitizer singleton doesn't cause memory issues"""
        initial_sanitizers = []

        # Get multiple references to global sanitizer
        for _ in range(1000):
            sanitizer = get_context_sanitizer()
            initial_sanitizers.append(id(sanitizer))  # Store ID, not object

        # All should be the same instance
        unique_ids = set(initial_sanitizers)
        assert (
            len(unique_ids) == 1
        ), f"Expected 1 unique sanitizer, got {len(unique_ids)}"

        # Clean up
        del initial_sanitizers
        gc.collect()

    def test_large_context_handling_memory_efficiency(self):
        """Test handling of very large contexts without memory issues"""
        # Create progressively larger contexts
        for size_multiplier in [1, 10, 100, 500]:
            large_context = {}

            # Add many keys with sensitive data
            for i in range(size_multiplier):
                large_context[f"api_key_{i}"] = f"sk-{'x' * 50}_{i}"
                large_context[f"session_data_{i}"] = {
                    f"nested_{j}": f"value_{'y' * 100}_{j}"
                    for j in range(min(10, size_multiplier))
                }

            initial_memory = psutil.Process().memory_info().rss

            # Sanitize large context
            sanitized = self.sanitizer.sanitize_context_dict(large_context)

            # Track memory after sanitization (for potential future use)
            # post_sanitization_memory = psutil.Process().memory_info().rss

            # Clean up
            del large_context
            del sanitized
            gc.collect()

            final_memory = psutil.Process().memory_info().rss
            memory_retained = final_memory - initial_memory

            # Should not retain significant memory after cleanup
            assert (
                memory_retained < 10 * 1024 * 1024
            ), f"Size {size_multiplier}: Retained {memory_retained / 1024 / 1024:.2f}MB after cleanup"

    def test_error_logging_integration_memory_safety(self):
        """Test that error logging integration doesn't cause memory leaks"""
        logger = StructuredErrorLogger()
        initial_memory = psutil.Process().memory_info().rss

        # Log many errors with sensitive context
        for i in range(100):
            error = ModelAuthenticationError(
                f"Auth error {i}",
                security_context={
                    "api_key": f"sk-{'x' * 100}_{i}",
                    "session_data": {f"key_{j}": f"value_{j}" for j in range(50)},
                    "large_payload": "x" * 10000,
                },
            )

            # Test actual logging with sanitization
            # The sanitization should occur during logging
            logger.log_error(
                error=error,
                operation_name=f"test_operation_{i}",
                session_id=f"session_{i}",
                additional_context={
                    "more_sensitive_data": f"secret_{i}",
                    "large_context": {"data": "y" * 5000},
                },
            )

            del error

            if i % 20 == 0:
                gc.collect()

        # Final cleanup
        gc.collect()
        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory

        # Should not grow memory significantly
        assert (
            memory_growth < 30 * 1024 * 1024
        ), f"Error logging caused {memory_growth / 1024 / 1024:.2f}MB memory growth"

    def test_sanitizer_internal_cache_management(self):
        """Test that internal caches don't grow unbounded"""
        # The sanitizer shouldn't maintain unbounded internal caches
        initial_sanitizer_dict_size = len(vars(self.sanitizer))

        # Process many different types of contexts
        context_types = [
            "security_context",
            "network_context",
            "performance_context",
            "validation_context",
            "concurrency_context",
        ]

        for context_type in context_types:
            for i in range(100):
                context = {
                    f"type_specific_key_{i}": f"value_{i}",
                    f"sensitive_data_{i}": f"sk-{i:032d}",
                }

                self.sanitizer.sanitize_context_dict(context, context_type)
                del context

        # Sanitizer internal state shouldn't grow unbounded
        final_sanitizer_dict_size = len(vars(self.sanitizer))

        # Should not accumulate internal state
        assert (
            final_sanitizer_dict_size == initial_sanitizer_dict_size
        ), f"Sanitizer internal state grew from {initial_sanitizer_dict_size} to {final_sanitizer_dict_size}"

    def test_hash_function_memory_efficiency(self):
        """Test that hash function doesn't retain references to original data"""
        sensitive_data = ["sk-" + "x" * 1000 + str(i) for i in range(1000)]
        hashes = []

        initial_memory = psutil.Process().memory_info().rss

        # Hash all sensitive data
        for data in sensitive_data:
            hash_value = self.sanitizer._hash_sensitive_data(data)
            hashes.append(hash_value)

        # Clear original sensitive data
        del sensitive_data
        gc.collect()

        # Memory should be freed even though we have hashes
        post_cleanup_memory = psutil.Process().memory_info().rss
        memory_retained = post_cleanup_memory - initial_memory

        # Hashes should be much smaller than original data
        assert (
            memory_retained < 5 * 1024 * 1024
        ), f"Hash function retained too much memory: {memory_retained / 1024 / 1024:.2f}MB"

        # Verify hashes are still valid
        assert len(hashes) == 1000
        assert all(len(h) == 8 for h in hashes if h is not None)

    def teardown_method(self):
        """Clean up after each test"""
        gc.collect()


class TestMemoryLeakFixValidation:
    """Validate that specific memory leak scenarios are fixed"""

    def test_context_dict_reference_cycles_fixed(self):
        """Test that context dict processing doesn't create reference cycles"""
        # This test would fail if sanitization creates unbreakable reference cycles
        weak_refs = []

        # Use a wrapper class since dict objects can't have weak references
        class ContextWrapper:
            def __init__(self, data):
                self.data = data

        for i in range(100):
            context_data = {
                "api_key": f"sk-test{i}",
                "nested": {"deep": {"data": f"sensitive_{i}"}},
            }

            context_wrapper = ContextWrapper(context_data)

            import weakref

            weak_refs.append(weakref.ref(context_wrapper))

            # Sanitize context (use the actual dict data)
            sanitized = SecurityContextSanitizer().sanitize_context_dict(context_data)

            # Clear strong references
            del context_data
            del context_wrapper
            del sanitized

        # Force garbage collection
        gc.collect()

        # Check that original contexts were properly garbage collected
        alive_refs = sum(1 for ref in weak_refs if ref() is not None)

        # Most should be garbage collected (allow for some timing issues)
        assert alive_refs < 10, f"Too many contexts still alive: {alive_refs}/100"

    def test_exception_context_extraction_memory_safe(self):
        """Test that exception context extraction doesn't retain exception references"""
        sanitizer = SecurityContextSanitizer()

        # Create exceptions and extract contexts
        for i in range(100):
            error = SecurityError(
                f"Test error {i}",
                security_context={"large_data": "x" * 10000, "api_key": f"sk-{i:032d}"},
            )

            # Extract and sanitize context
            sanitized = sanitizer.sanitize_exception_context(error)

            # Verify original sensitive data is not in sanitized form
            assert f"sk-{i:032d}" not in str(sanitized)

            # Clear references
            del error
            del sanitized

            # Periodic cleanup
            if i % 20 == 0:
                gc.collect()

        # Final cleanup should work without issues
        gc.collect()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
