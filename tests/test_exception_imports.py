"""Simple test to verify exception imports work correctly"""


def test_exception_imports():
    """Test that all custom exceptions can be imported without errors"""
    try:
        from src.context_switcher_mcp.exceptions import (
            ContextSwitcherError,
            SessionError,
            SessionNotFoundError,
            SessionExpiredError,
            SessionCleanupError,
            OrchestrationError,
            CircuitBreakerError,
            CircuitBreakerOpenError,
            CircuitBreakerStateError,
            ModelBackendError,
            ModelConnectionError,
            ModelTimeoutError,
            ModelRateLimitError,
            ModelAuthenticationError,
            ModelValidationError,
            AnalysisError,
            PerspectiveError,
            ConfigurationError,
            StorageError,
            SerializationError,
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
        assert False, f"Import error: {e}"
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        assert False, f"Unexpected error: {e}"


if __name__ == "__main__":
    success = test_exception_imports()
    exit(0 if success else 1)
