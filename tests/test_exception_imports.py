"""Simple test to verify exception imports work correctly"""


def test_exception_imports():
    """Test that all custom exceptions can be imported without errors"""
    try:
        from context_switcher_mcp.exceptions import (
            AnalysisError,
            CircuitBreakerError,
            CircuitBreakerOpenError,
            CircuitBreakerStateError,
            ConfigurationError,
            ContextSwitcherError,
            ModelAuthenticationError,
            ModelBackendError,
            ModelConnectionError,
            ModelRateLimitError,
            ModelTimeoutError,
            ModelValidationError,
            OrchestrationError,
            PerspectiveError,
            SerializationError,
            SessionCleanupError,
            SessionError,
            SessionExpiredError,
            SessionNotFoundError,
            StorageError,
        )

        # Test that they can be instantiated
        assert ContextSwitcherError("test")
        assert SessionError("test")
        assert SessionNotFoundError("test")
        assert SessionExpiredError("test")
        assert SessionCleanupError("test")
        assert OrchestrationError("test")
        assert CircuitBreakerError("test")
        assert CircuitBreakerOpenError("test")
        assert CircuitBreakerStateError("test")
        assert ModelBackendError("test")
        assert ModelConnectionError("test")
        assert ModelTimeoutError("test")
        assert ModelRateLimitError("test")
        assert ModelAuthenticationError("test")
        assert ModelValidationError("test")
        assert AnalysisError("test")
        assert PerspectiveError("test")
        assert ConfigurationError("test")
        assert StorageError("test")
        assert SerializationError("test")

        print("✓ All exception imports successful")

    except ImportError as e:
        print(f"✗ Import error: {e}")
        raise AssertionError(f"Import error: {e}") from e
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        raise AssertionError(f"Unexpected error: {e}") from e


if __name__ == "__main__":
    success = test_exception_imports()
    exit(0 if success else 1)
