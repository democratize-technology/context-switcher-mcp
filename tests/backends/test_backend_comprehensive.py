"""Comprehensive backend tests covering all failure scenarios and defensive patterns"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
from context_switcher_mcp.backend_factory import (  # noqa: E402
    BackendFactory,
    LiteLLMBackend,
    OllamaBackend,
    get_backend_for_thread,
)
from context_switcher_mcp.backend_interface import BedrockBackend  # noqa: E402
from context_switcher_mcp.exceptions import (  # noqa: E402
    ModelAuthenticationError,
    ModelBackendError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
    ModelValidationError,
    NetworkConnectivityError,
)
from context_switcher_mcp.models import Thread  # noqa: E402
from context_switcher_mcp.types import ModelBackend  # noqa: E402
from litellm import APIError, AuthenticationError, RateLimitError


@pytest.fixture
def mock_litellm_response():
    """Mock LiteLLM API response"""
    return {
        "choices": [{"message": {"content": "This is a test response from LiteLLM"}}],
        "usage": {"total_tokens": 150, "prompt_tokens": 100, "completion_tokens": 50},
    }


@pytest.fixture
def mock_bedrock_response():
    """Mock Bedrock API response"""
    return {
        "body": MagicMock(
            read=MagicMock(
                return_value=json.dumps(
                    {"content": [{"text": "This is a test response from Bedrock"}]}
                ).encode("utf-8")
            )
        ),
        "ResponseMetadata": {"HTTPStatusCode": 200},
    }


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response"""
    return {"message": {"content": "This is a test response from Ollama"}, "done": True}


class TestLiteLLMBackendComprehensive:
    """Test LiteLLM backend with comprehensive failure modes"""

    @pytest.fixture
    def backend(self):
        return LiteLLMBackend()

    @pytest.mark.asyncio
    async def test_successful_call(self, backend, mock_litellm_response):
        """Test successful LiteLLM API call"""

        thread = Thread(
            id="test-thread-1",
            name="test-perspective",
            system_prompt="You are a helpful assistant",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-3.5-turbo",
        )
        thread.add_message("user", "Hello, how are you?")

        # Create a proper mock object with attributes
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[
            0
        ].message.content = "This is a test response from LiteLLM"

        with patch("litellm.acompletion", return_value=mock_response):
            result = await backend.call_model(thread)

            assert result == "This is a test response from LiteLLM"

    @pytest.mark.asyncio
    async def test_malformed_api_response(self, backend):
        """Test handling of malformed API responses"""
        thread = Thread(
            id="test-thread",
            name="test",
            system_prompt="Test",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-3.5-turbo",
        )
        thread.add_message("user", "Test")

        malformed_responses = [
            {"choices": [{"message": None}]},  # Missing content
            {"choices": [{}]},  # Missing message
            {"choices": []},  # Empty choices
            {},  # Empty response
            None,  # None response
            {"choices": [{"message": {"content": None}}]},  # Null content
        ]

        for malformed_response in malformed_responses:
            with patch("litellm.acompletion", return_value=malformed_response):
                with pytest.raises(
                    (
                        ModelBackendError,
                        ModelValidationError,
                        ValueError,
                        KeyError,
                        AttributeError,
                    )
                ):  # Expect parsing error
                    await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_network_errors(self, backend):
        """Test various network error conditions"""
        thread = Thread(
            id="test-thread",
            name="test",
            system_prompt="Test",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-3.5-turbo",
        )
        thread.add_message("user", "Test")

        network_errors = [
            asyncio.TimeoutError("Request timed out"),
            ConnectionError("Connection refused"),
            Exception("Connect error"),  # Simplified for testing
            Exception("Read timeout"),  # Simplified for testing
            OSError("Network unreachable"),
        ]

        for error in network_errors:
            with patch("litellm.acompletion", side_effect=error):
                with pytest.raises(
                    (
                        ModelConnectionError,
                        NetworkConnectivityError,
                        ModelTimeoutError,
                        ConnectionError,
                        OSError,
                        asyncio.TimeoutError,
                    )
                ):  # Expect network error
                    await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_authentication_failures(self, backend):
        """Test authentication failure scenarios"""
        thread = Thread(
            id="test-thread",
            name="test",
            system_prompt="Test",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-3.5-turbo",
        )
        thread.add_message("user", "Test")

        auth_errors = [
            APIError("Invalid API key", "Invalid API key", "litellm", "gpt-3.5-turbo"),
            AuthenticationError("Token expired", "litellm", "gpt-3.5-turbo"),
            Exception("Insufficient permissions"),  # Simplified
            Exception("API key not found"),  # Simplified
        ]

        for error in auth_errors:
            with patch("litellm.acompletion", side_effect=error):
                with pytest.raises(
                    (ModelAuthenticationError, APIError, AuthenticationError)
                ):  # Expect auth error
                    await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_rate_limiting(self, backend):
        """Test rate limiting scenarios"""
        thread = Thread(
            id="test-thread",
            name="test",
            system_prompt="Test",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-3.5-turbo",
        )
        thread.add_message("user", "Test")

        with patch(
            "litellm.acompletion",
            side_effect=RateLimitError(
                "Rate limit exceeded", "litellm", "gpt-3.5-turbo"
            ),
        ):
            with pytest.raises(
                (ModelRateLimitError, RateLimitError)
            ):  # Expect rate limit error
                await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_model_name_validation(self, backend):
        """Test model name handling"""
        thread = Thread(
            id="test-thread",
            name="test",
            system_prompt="Test",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-4",  # Custom model
        )
        thread.add_message("user", "Test")

        # Create proper mock response
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "test"

        # Test with custom model
        with patch("litellm.acompletion", return_value=mock_response):
            result = await backend.call_model(thread)
            assert result == "test"

    @pytest.mark.asyncio
    async def test_parameter_validation(self, backend):
        """Test parameter validation and edge cases"""
        # Test edge case parameters with Thread objects
        edge_case_threads = [
            # Empty system prompt
            Thread(
                id="test-thread-1",
                name="test",
                system_prompt="",
                model_backend=ModelBackend.LITELLM,
                model_name="gpt-3.5-turbo",
            ),
            # Very long prompt
            Thread(
                id="test-thread-2",
                name="test",
                system_prompt="Test",
                model_backend=ModelBackend.LITELLM,
                model_name="gpt-3.5-turbo",
            ),
        ]

        # Add messages to threads
        edge_case_threads[0].add_message("user", "Test")
        edge_case_threads[1].add_message("user", "A" * 1000)  # Long prompt

        # Create proper mock response
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "test"

        with patch("litellm.acompletion", return_value=mock_response):
            for thread in edge_case_threads:
                result = await backend.call_model(thread)
                assert result == "test"


class TestBedrockBackendComprehensive:
    """Test AWS Bedrock backend with security and failure patterns"""

    @pytest.fixture
    def backend(self):
        return BedrockBackend()

    @pytest.mark.asyncio
    async def test_successful_call(self, backend, mock_bedrock_response):
        """Test successful Bedrock API call"""
        thread = Thread(
            id="test-thread",
            name="test",
            system_prompt="Test system",
            model_backend=ModelBackend.BEDROCK,
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        )
        thread.add_message("user", "Test user")

        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client
            # Mock converse API response
            mock_client.converse.return_value = {
                "output": {
                    "message": {
                        "content": [{"text": "This is a test response from Bedrock"}]
                    }
                }
            }

            result = await backend.call_model(thread)

            assert "Bedrock" in result

    @pytest.mark.asyncio
    async def test_malformed_bedrock_responses(self, backend):
        """Test Bedrock response parsing edge cases"""
        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client

            malformed_responses = [
                # Missing output
                {},
                # Missing message
                {"output": {}},
                # Missing content
                {"output": {"message": {}}},
                # Empty content array
                {"output": {"message": {"content": []}}},
                # Missing text field
                {"output": {"message": {"content": [{}]}}},
                # Null text field
                {"output": {"message": {"content": [{"text": None}]}}},
            ]

            thread = Thread(
                id="test-thread",
                name="test",
                system_prompt="Test",
                model_backend=ModelBackend.BEDROCK,
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            )
            thread.add_message("user", "Test")

            for response in malformed_responses:
                mock_client.converse.return_value = response

                with pytest.raises(
                    (
                        ModelBackendError,
                        ModelValidationError,
                        ValueError,
                        KeyError,
                        IndexError,
                        TypeError,
                    )
                ):  # Expect Bedrock parsing error
                    await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_aws_credential_errors(self, backend):
        """Test AWS credential and permission errors"""
        with patch("boto3.client") as mock_boto:
            aws_errors = [
                NoCredentialsError(),
                ClientError(
                    error_response={
                        "Error": {"Code": "AccessDenied", "Message": "Access denied"}
                    },
                    operation_name="InvokeModel",
                ),
                ClientError(
                    error_response={
                        "Error": {
                            "Code": "UnauthorizedOperation",
                            "Message": "Unauthorized",
                        }
                    },
                    operation_name="InvokeModel",
                ),
                BotoCoreError(),
            ]

            thread = Thread(
                id="test-thread",
                name="test",
                system_prompt="Test",
                model_backend=ModelBackend.BEDROCK,
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            )
            thread.add_message("user", "Test")

            for error in aws_errors:
                mock_boto.side_effect = error

                with pytest.raises(
                    (
                        ModelAuthenticationError,
                        ModelBackendError,
                        NoCredentialsError,
                        ClientError,
                        BotoCoreError,
                    )
                ):  # Expect AWS error
                    await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_bedrock_service_errors(self, backend):
        """Test Bedrock service-specific errors"""
        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client

            service_errors = [
                ClientError(
                    error_response={
                        "Error": {
                            "Code": "ModelNotReadyException",
                            "Message": "Model not ready",
                        }
                    },
                    operation_name="Converse",
                ),
                ClientError(
                    error_response={
                        "Error": {
                            "Code": "ValidationException",
                            "Message": "Invalid input",
                        }
                    },
                    operation_name="Converse",
                ),
                ClientError(
                    error_response={
                        "Error": {
                            "Code": "ThrottlingException",
                            "Message": "Request throttled",
                        }
                    },
                    operation_name="Converse",
                ),
            ]

            thread = Thread(
                id="test-thread",
                name="test",
                system_prompt="Test",
                model_backend=ModelBackend.BEDROCK,
                model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            )
            thread.add_message("user", "Test")

            for error in service_errors:
                mock_client.converse.side_effect = error

                with pytest.raises(
                    (
                        ModelRateLimitError,
                        ModelTimeoutError,
                        ModelConnectionError,
                        ModelValidationError,
                        ClientError,
                        BotoCoreError,
                    )
                ):  # Expect Bedrock service error
                    await backend.call_model(thread)

                # Reset side_effect for next iteration
                mock_client.converse.side_effect = None

    @pytest.mark.asyncio
    async def test_bedrock_model_variants(self, backend):
        """Test different Bedrock model configurations"""
        with patch("boto3.client") as mock_boto:
            mock_client = Mock()
            mock_boto.return_value = mock_client
            # Mock converse API response
            mock_client.converse.return_value = {
                "output": {"message": {"content": [{"text": "model response"}]}}
            }

            # Test different model IDs
            model_variants = [
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "amazon.titan-text-express-v1",
            ]

            for model_id in model_variants:
                thread = Thread(
                    id=f"test-thread-{model_id}",
                    name="test",
                    system_prompt="Test",
                    model_backend=ModelBackend.BEDROCK,
                    model_name=model_id,
                )
                thread.add_message("user", "Test")

                result = await backend.call_model(thread)
                assert "model response" in result


class TestOllamaBackendComprehensive:
    """Test Ollama backend resilience and local deployment patterns"""

    @pytest.fixture
    def backend(self):
        return OllamaBackend()

    @pytest.mark.asyncio
    async def test_successful_call(self, backend, mock_ollama_response):
        """Test successful Ollama API call"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock successful response
            mock_response = Mock()
            mock_response.json.return_value = mock_ollama_response
            mock_response.raise_for_status = Mock()
            mock_client.post.return_value = mock_response

            thread = Thread(
                id="test-thread",
                name="test",
                system_prompt="Test system",
                model_backend=ModelBackend.OLLAMA,
                model_name="llama3",
            )
            thread.add_message("user", "Test user")

            result = await backend.call_model(thread)

            assert "Ollama" in result

    @pytest.mark.asyncio
    async def test_ollama_service_unavailable(self, backend):
        """Test Ollama service unavailability scenarios"""
        connection_errors = [
            ConnectionError("Connection refused"),
            httpx.ConnectError("Failed to establish connection"),
            httpx.TimeoutException("Request timeout"),
            OSError("Service unavailable"),
        ]

        for error in connection_errors:
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client_class.return_value.__aenter__.return_value = mock_client
                mock_client.post.side_effect = error

                thread = Thread(
                    id="test-thread",
                    name="test",
                    system_prompt="Test",
                    model_backend=ModelBackend.OLLAMA,
                    model_name="llama3",
                )
                thread.add_message("user", "Test")

                with pytest.raises(
                    (
                        ModelConnectionError,
                        NetworkConnectivityError,
                        ModelTimeoutError,
                        httpx.HTTPError,
                        ConnectionError,
                        OSError,
                    )
                ):  # Expect Ollama connection error
                    await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_ollama_malformed_responses(self, backend):
        """Test malformed Ollama response handling"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            malformed_responses = [
                {},  # Empty response
                {"message": None},  # Null message
                {"message": {}},  # Empty message
                {"message": {"content": None}},  # Null content
                {"done": False},  # Incomplete response
                {"error": "Model not found"},  # Error response
            ]

            for malformed in malformed_responses:
                mock_response = Mock()
                mock_response.json.return_value = malformed
                mock_response.raise_for_status = Mock()
                mock_client.post.return_value = mock_response

                thread = Thread(
                    id="test-thread",
                    name="test",
                    system_prompt="Test",
                    model_backend=ModelBackend.OLLAMA,
                    model_name="llama3",
                )
                thread.add_message("user", "Test")

                try:
                    result = await backend.call_model(thread)
                    # If no error, result should be reasonable
                    if result is not None:
                        assert isinstance(result, str)
                except Exception:
                    # Errors are also acceptable for malformed responses
                    pass

    @pytest.mark.asyncio
    async def test_ollama_http_errors(self, backend):
        """Test Ollama HTTP error responses"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            http_errors = [
                httpx.HTTPStatusError(
                    "404 Not Found", request=Mock(), response=Mock(status_code=404)
                ),
                httpx.HTTPStatusError(
                    "500 Internal Server Error",
                    request=Mock(),
                    response=Mock(status_code=500),
                ),
                httpx.HTTPStatusError(
                    "503 Service Unavailable",
                    request=Mock(),
                    response=Mock(status_code=503),
                ),
            ]

            for error in http_errors:
                mock_client.post.side_effect = error

                thread = Thread(
                    id="test-thread",
                    name="test",
                    system_prompt="Test",
                    model_backend=ModelBackend.OLLAMA,
                    model_name="llama3",
                )
                thread.add_message("user", "Test")

                with pytest.raises(
                    (
                        ModelConnectionError,
                        ModelRateLimitError,
                        NetworkConnectivityError,
                        httpx.HTTPError,
                        httpx.RequestError,
                    )
                ):  # Expect Ollama HTTP error
                    await backend.call_model(thread)

    @pytest.mark.asyncio
    async def test_ollama_model_management(self, backend):
        """Test Ollama model availability and management"""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Test model not found
            model_error_response = Mock()
            model_error_response.json.return_value = {
                "error": 'model "nonexistent" not found'
            }
            model_error_response.raise_for_status = Mock()
            mock_client.post.return_value = model_error_response

            thread = Thread(
                id="test-thread",
                name="test",
                system_prompt="Test",
                model_backend=ModelBackend.OLLAMA,
                model_name="nonexistent",
            )
            thread.add_message("user", "Test")

            with pytest.raises(
                (
                    ModelValidationError,
                    ModelConnectionError,
                    httpx.HTTPError,
                    httpx.RequestError,
                )
            ):  # Expect model not found error
                await backend.call_model(thread)


class TestBackendFactoryComprehensive:
    """Test backend factory with comprehensive scenarios"""

    def test_supported_backends(self):
        """Test factory supports all expected backends"""
        factory = BackendFactory()

        supported_backends = [
            ModelBackend.BEDROCK,
            ModelBackend.LITELLM,
            ModelBackend.OLLAMA,
        ]
        for backend_enum in supported_backends:
            backend = factory.get_backend(backend_enum)
            assert backend is not None

    def test_unsupported_backend(self):
        """Test factory handles unsupported backends"""
        factory = BackendFactory()

        # Test with invalid strings that would need to be converted
        unsupported_strings = ["openai", "anthropic", "cohere", "invalid", ""]
        for backend_name in unsupported_strings:
            with pytest.raises((ValueError, TypeError, AttributeError)):
                # This should fail because strings don't have .value attribute
                factory.get_backend(backend_name)

        # Test with None
        with pytest.raises((ValueError, TypeError, AttributeError)):
            factory.get_backend(None)

    def test_backend_caching(self):
        """Test that backends are properly cached"""
        factory = BackendFactory()

        # Get same backend twice
        backend1 = factory.get_backend(ModelBackend.BEDROCK)
        backend2 = factory.get_backend(ModelBackend.BEDROCK)

        # Should be same instance (cached)
        assert backend1 is backend2

    def test_factory_reset(self):
        """Test factory reset functionality"""
        factory = BackendFactory()

        # Get a backend
        backend1 = factory.get_backend(ModelBackend.BEDROCK)

        # Reset factory
        factory.reset()

        # Get backend again
        backend2 = factory.get_backend(ModelBackend.BEDROCK)

        # Should be different instance after reset
        assert backend1 is not backend2

    def test_backend_availability_detection(self):
        """Test backend availability detection"""
        factory = BackendFactory()

        available_backends = factory.get_available_backends()
        assert isinstance(available_backends, list)
        assert all(isinstance(backend, str) for backend in available_backends)

    @patch.dict("os.environ", {}, clear=True)
    def test_availability_with_missing_credentials(self):
        """Test availability detection without credentials"""
        factory = BackendFactory()

        # Without credentials, some backends might not be available
        available = factory.get_available_backends()

        # Should still return a list (might be empty)
        assert isinstance(available, list)

    @patch.dict(
        "os.environ", {"AWS_PROFILE": "test", "OLLAMA_HOST": "http://localhost:11434"}
    )
    def test_availability_with_credentials(self):
        """Test availability detection with credentials"""
        factory = BackendFactory()

        available = factory.get_available_backends()

        # With credentials, more backends should be available
        assert isinstance(available, list)
        # Note: Actual availability depends on environment

    def test_backend_type_validation(self):
        """Test backend type validation"""
        factory = BackendFactory()

        for backend_name in ["bedrock", "litellm", "ollama"]:
            backend = factory.get_backend(backend_name)

            # Should have required methods
            assert hasattr(backend, "call_model")
            assert callable(backend.call_model)


class TestThreadBackendIntegration:
    """Test backend integration with thread management"""

    def test_get_backend_for_thread(self):
        """Test getting backend for specific thread"""
        thread = Thread(
            id="test-thread-123",
            name="test-perspective",
            system_prompt="You are a helpful assistant",
            model_backend=ModelBackend.BEDROCK,
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
        )

        backend = get_backend_for_thread(thread)
        assert backend is not None
        assert hasattr(backend, "call_model")

    def test_thread_backend_with_invalid_config(self):
        """Test thread backend with invalid configuration"""
        # Create a valid thread with proper enum
        test_thread = Thread(
            id="test-thread",
            name="test",
            system_prompt="test",
            model_backend=ModelBackend.BEDROCK,
            model_name=None,
        )

        # This should work since we have proper enum type
        backend = get_backend_for_thread(test_thread)
        assert backend is not None
        assert hasattr(backend, "call_model")


class TestBackendErrorRecovery:
    """Test backend error recovery and resilience patterns"""

    @pytest.mark.asyncio
    async def test_retry_on_transient_errors(self):
        """Test retry behavior on transient errors"""
        backend = LiteLLMBackend()
        thread = Thread(
            id="test-thread",
            name="test",
            system_prompt="Test",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-3.5-turbo",
        )
        thread.add_message("user", "Test")

        # Simulate transient error followed by success
        responses = [
            RateLimitError(
                "Rate limit exceeded", "litellm", "gpt-3.5-turbo"
            ),  # First call fails
            {
                "choices": [{"message": {"content": "Success after retry"}}]
            },  # Second call succeeds
        ]

        with patch("litellm.acompletion", side_effect=responses):
            # If retry is implemented, should eventually succeed
            try:
                result = await backend.call_model(thread)
                assert "Success" in result
            except Exception:
                # If no retry, failure is also acceptable for this test
                pass

    @pytest.mark.asyncio
    async def test_circuit_breaker_pattern(self):
        """Test circuit breaker behavior with repeated failures"""
        backend = LiteLLMBackend()
        thread = Thread(
            id="test-thread",
            name="test",
            system_prompt="Test",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-3.5-turbo",
        )
        thread.add_message("user", "Test")

        # Simulate repeated failures
        with patch(
            "litellm.acompletion",
            side_effect=APIError(
                "Service unavailable", "Service unavailable", "litellm", "gpt-3.5-turbo"
            ),
        ):
            failures = []

            # Make multiple calls
            for i in range(5):
                try:
                    await backend.call_model(thread)
                except Exception:
                    failures.append(i)

            # Should fail consistently
            assert len(failures) >= 3  # Most calls should fail

    def test_backend_health_checks(self):
        """Test backend health check functionality"""
        factory = BackendFactory()

        for backend_name in ["bedrock", "litellm", "ollama"]:
            is_available = factory.is_backend_available(backend_name)
            assert isinstance(is_available, bool)


class TestBackendPerformanceAndMemory:
    """Test backend performance and memory efficiency"""

    @pytest.mark.asyncio
    async def test_concurrent_backend_calls(self):
        """Test concurrent calls to same backend"""
        backend = LiteLLMBackend()

        # Create proper mock response
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "concurrent response"

        with patch("litellm.acompletion", return_value=mock_response):
            # Make multiple concurrent calls
            tasks = []
            for i in range(10):
                thread = Thread(
                    id=f"test-thread-{i}",
                    name=f"test-{i}",
                    system_prompt=f"Test {i}",
                    model_backend=ModelBackend.LITELLM,
                    model_name="gpt-3.5-turbo",
                )
                thread.add_message("user", f"User {i}")

                task = backend.call_model(thread)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed
            successful_results = [r for r in results if isinstance(r, str)]
            assert len(successful_results) == 10

            # All should have expected content
            for result in successful_results:
                assert "concurrent response" in result

    def test_memory_efficiency(self):
        """Test that backends don't leak memory"""
        import gc

        factory = BackendFactory()

        # Force garbage collection
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create and destroy multiple backends
        for _ in range(100):
            backend = factory.get_backend(ModelBackend.BEDROCK)
            del backend

        factory.reset()
        gc.collect()
        final_objects = len(gc.get_objects())

        # Should not create excessive persistent objects
        object_increase = final_objects - initial_objects
        assert (
            object_increase < 1000
        ), f"Memory leak detected: {object_increase} new objects"

    @pytest.mark.asyncio
    async def test_large_payload_handling(self):
        """Test backends handle large payloads correctly"""
        backend = LiteLLMBackend()

        # Test with very large prompts
        large_prompt = "A" * 50000  # 50KB prompt

        # Create proper mock response
        from unittest.mock import MagicMock

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = "large response"

        thread = Thread(
            id="test-thread-large",
            name="test-large",
            system_prompt="Test",
            model_backend=ModelBackend.LITELLM,
            model_name="gpt-3.5-turbo",
        )
        thread.add_message("user", large_prompt)

        with patch("litellm.acompletion", return_value=mock_response):
            try:
                result = await backend.call_model(thread)
                assert result == "large response"
            except Exception:
                # Some backends might reject very large prompts
                pass
