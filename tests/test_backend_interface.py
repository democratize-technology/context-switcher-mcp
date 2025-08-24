"""Comprehensive tests for backend_interface module"""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from context_switcher_mcp.backend_interface import (  # noqa: E402
    BACKEND_REGISTRY,
    BedrockBackend,
    LiteLLMBackend,
    ModelBackendInterface,
    ModelCallConfig,
    OllamaBackend,
    get_backend_interface,
)
from context_switcher_mcp.exceptions import (  # noqa: E402
    ConfigurationError,
    ModelAuthenticationError,
    ModelBackendError,
    ModelConnectionError,
    ModelRateLimitError,
    ModelTimeoutError,
    ModelValidationError,
    NetworkConnectivityError,
    NetworkTimeoutError,
)
from context_switcher_mcp.models import Thread  # noqa: E402


class TestModelCallConfig:
    """Test ModelCallConfig dataclass"""

    def test_model_call_config_initialization(self):
        """Test ModelCallConfig creation with required parameters"""
        config = ModelCallConfig(
            max_tokens=1000,
            temperature=0.7,
            model_name="test-model",
            timeout_seconds=30.0,
        )

        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert config.model_name == "test-model"
        assert config.timeout_seconds == 30.0

    def test_model_call_config_default_timeout(self):
        """Test ModelCallConfig with default timeout"""
        config = ModelCallConfig(
            max_tokens=1000, temperature=0.7, model_name="test-model"
        )

        assert config.timeout_seconds == 60.0


class ConcreteBackend(ModelBackendInterface):
    """Concrete implementation for testing abstract interface"""

    def _get_model_name(self, thread: Thread) -> str:
        return "test-model"

    async def call_model(self, thread: Thread) -> str:
        return "test response"


class TestModelBackendInterface:
    """Test ModelBackendInterface abstract base class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.backend = ConcreteBackend("test")

        # Mock config
        self.mock_config = Mock()
        self.mock_config.model.default_max_tokens = 1000
        self.mock_config.model.default_temperature = 0.7

        self.backend.config = self.mock_config

    def test_backend_initialization(self):
        """Test backend initialization"""
        assert self.backend.backend_name == "test"
        assert hasattr(self.backend, "config")

    def test_get_model_config(self):
        """Test get_model_config method"""
        thread = Mock()
        thread.model_name = "custom-model"

        config = self.backend.get_model_config(thread)

        assert isinstance(config, ModelCallConfig)
        assert config.max_tokens == 1000
        assert config.temperature == 0.7
        assert config.model_name == "test-model"  # From _get_model_name
        assert config.timeout_seconds == 60.0

    @pytest.mark.asyncio
    async def test_call_model_stream_fallback(self):
        """Test call_model_stream fallback to regular call"""
        thread = Mock()
        thread.name = "test-thread"

        responses = []
        async for response in self.backend.call_model_stream(thread):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0]["type"] == "complete"
        assert responses[0]["content"] == "test response"
        assert responses[0]["thread_name"] == "test-thread"

    @pytest.mark.asyncio
    async def test_call_model_stream_model_backend_error(self):
        """Test call_model_stream with ModelBackendError"""
        backend = ConcreteBackend("test")

        async def failing_call_model(thread):
            raise ModelBackendError("Test error")

        backend.call_model = failing_call_model

        thread = Mock()
        thread.name = "test-thread"

        responses = []
        async for response in backend.call_model_stream(thread):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0]["type"] == "error"
        assert "test error" in responses[0]["content"].lower()
        assert responses[0]["thread_name"] == "test-thread"

    @pytest.mark.asyncio
    async def test_call_model_stream_unexpected_error(self):
        """Test call_model_stream with unexpected error"""
        backend = ConcreteBackend("test")

        async def failing_call_model(thread):
            raise ValueError("Unexpected error")

        backend.call_model = failing_call_model

        thread = Mock()
        thread.name = "test-thread"

        responses = []
        async for response in backend.call_model_stream(thread):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0]["type"] == "error"
        assert "unexpected error" in responses[0]["content"].lower()
        assert responses[0]["thread_name"] == "test-thread"

    def test_format_error_response(self):
        """Test _format_error_response method"""
        with (
            patch(
                "context_switcher_mcp.backend_interface.sanitize_error_message"
            ) as mock_sanitize,
            patch(
                "context_switcher_mcp.backend_interface.create_error_response"
            ) as mock_create_error,
        ):
            mock_sanitize.return_value = "sanitized error"
            mock_create_error.return_value = "formatted error"

            result = self.backend._format_error_response(
                "raw error", "test_error", {"key": "value"}
            )

            assert result == "AORP_ERROR: formatted error"
            mock_sanitize.assert_called_once_with("raw error")
            mock_create_error.assert_called_once_with(
                error_message="test error: sanitized error",
                error_type="test_error",
                context={"backend": "test", "key": "value"},
                recoverable=True,
            )

    def test_get_error_type_and_message_credentials(self):
        """Test error type detection for credentials errors"""
        error = Exception("unauthorized api_key invalid")
        error_type, message = self.backend._get_error_type_and_message(error)

        assert error_type == "credentials_error"
        assert message == "Missing or invalid API credentials"

    def test_get_error_type_and_message_connection(self):
        """Test error type detection for connection errors"""
        error = Exception("connection timeout network failure")
        error_type, message = self.backend._get_error_type_and_message(error)

        assert error_type == "connection_error"
        assert message == "Network connection failed"

    def test_get_error_type_and_message_model_not_found(self):
        """Test error type detection for model not found"""
        error = Exception("model xyz not found")
        error_type, message = self.backend._get_error_type_and_message(error)

        assert error_type == "model_not_found"
        assert message == "Model not found or unavailable"

    def test_get_error_type_and_message_inference_profile(self):
        """Test error type detection for inference profile errors"""
        error = Exception("inference profile configuration issue")
        error_type, message = self.backend._get_error_type_and_message(error)

        assert error_type == "model_configuration_error"
        assert message == "Model configuration issue"

    def test_get_error_type_and_message_generic(self):
        """Test error type detection for generic errors"""
        with patch(
            "context_switcher_mcp.backend_interface.sanitize_error_message"
        ) as mock_sanitize:
            mock_sanitize.return_value = "sanitized generic error"

            error = Exception("some generic api failure")
            error_type, message = self.backend._get_error_type_and_message(error)

            assert error_type == "api_error"
            assert "sanitized generic error" in message

    def test_prepare_messages(self):
        """Test _prepare_messages method"""
        thread = Mock()
        thread.system_prompt = "You are a helpful assistant"
        thread.conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        messages = self.backend._prepare_messages(thread)

        expected = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        assert messages == expected


class TestBedrockBackend:
    """Test BedrockBackend implementation"""

    def setup_method(self):
        """Set up test fixtures"""
        self.backend = BedrockBackend()

        # Mock config
        self.mock_config = Mock()
        self.mock_config.model.bedrock_model_id = "default-bedrock-model"
        self.mock_config.model.default_max_tokens = 1000
        self.mock_config.model.default_temperature = 0.7

        self.backend.config = self.mock_config

        # Mock thread
        self.thread = Mock()
        self.thread.name = "test-thread"
        self.thread.model_name = "test-bedrock-model"
        self.thread.system_prompt = "You are helpful"
        self.thread.conversation_history = [{"role": "user", "content": "Hello"}]

    def test_bedrock_backend_initialization(self):
        """Test BedrockBackend initialization"""
        assert self.backend.backend_name == "bedrock"

    def test_get_model_name_with_thread_model(self):
        """Test _get_model_name with thread model name"""
        result = self.backend._get_model_name(self.thread)
        assert result == "test-bedrock-model"

    def test_get_model_name_with_default_model(self):
        """Test _get_model_name with default model"""
        self.thread.model_name = None
        result = self.backend._get_model_name(self.thread)
        assert result == "default-bedrock-model"

    @pytest.mark.asyncio
    async def test_call_model_success(self):
        """Test successful Bedrock model call"""
        with (
            patch("boto3.client") as mock_boto3,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            # Mock boto3 client
            mock_client = Mock()
            mock_boto3.return_value = mock_client

            # Mock model validation
            mock_validate.return_value = (True, None)

            # Mock successful response
            mock_response = {
                "output": {"message": {"content": [{"text": "Test response"}]}}
            }
            mock_client.converse.return_value = mock_response

            result = await self.backend.call_model(self.thread)

            assert result == "Test response"
            mock_boto3.assert_called_once_with(
                "bedrock-runtime", region_name="us-east-1"
            )
            mock_client.converse.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_model_import_error(self):
        """Test Bedrock call_model with ImportError"""

        # Mock the import statement to raise ImportError
        def mock_import(name, *args):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return __builtins__["__import__"](name, *args)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ConfigurationError) as exc_info:
                await self.backend.call_model(self.thread)

            assert "boto3 library not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_model_validation_error(self):
        """Test Bedrock call_model with model validation error"""
        with (
            patch("boto3.client") as mock_boto3,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_boto3.return_value = Mock()
            mock_validate.return_value = (False, "Invalid model")

            with pytest.raises(ModelValidationError) as exc_info:
                await self.backend.call_model(self.thread)

            assert "Invalid model ID" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_model_credentials_error(self):
        """Test Bedrock call_model with credentials error"""
        with (
            patch("boto3.client") as mock_boto3,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_client = Mock()
            mock_boto3.return_value = mock_client
            mock_validate.return_value = (True, None)

            # Mock credentials error
            mock_client.converse.side_effect = Exception("unauthorized api_key")

            with pytest.raises(ModelAuthenticationError):
                await self.backend.call_model(self.thread)

    @pytest.mark.asyncio
    async def test_call_model_connection_error(self):
        """Test Bedrock call_model with connection error"""
        with (
            patch("boto3.client") as mock_boto3,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_client = Mock()
            mock_boto3.return_value = mock_client
            mock_validate.return_value = (True, None)

            # Mock connection error
            mock_client.converse.side_effect = Exception("connection timeout")

            with pytest.raises(NetworkConnectivityError):
                await self.backend.call_model(self.thread)

    @pytest.mark.asyncio
    async def test_call_model_rate_limit_error(self):
        """Test Bedrock call_model with rate limit error"""
        with (
            patch("boto3.client") as mock_boto3,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_client = Mock()
            mock_boto3.return_value = mock_client
            mock_validate.return_value = (True, None)

            # Mock rate limit error
            mock_client.converse.side_effect = Exception("throttling exceeded")

            with pytest.raises(ModelRateLimitError):
                await self.backend.call_model(self.thread)

    @pytest.mark.asyncio
    async def test_call_model_timeout_error(self):
        """Test Bedrock call_model with timeout error"""
        with (
            patch("boto3.client") as mock_boto3,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_client = Mock()
            mock_boto3.return_value = mock_client
            mock_validate.return_value = (True, None)

            # Mock timeout error
            mock_client.converse.side_effect = Exception("request timeout")

            with pytest.raises(NetworkTimeoutError):
                await self.backend.call_model(self.thread)

    @pytest.mark.asyncio
    async def test_call_model_generic_error(self):
        """Test Bedrock call_model with generic error"""
        with (
            patch("boto3.client") as mock_boto3,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_client = Mock()
            mock_boto3.return_value = mock_client
            mock_validate.return_value = (True, None)

            # Mock generic error
            mock_client.converse.side_effect = Exception("generic error")

            with pytest.raises(ModelBackendError):
                await self.backend.call_model(self.thread)

    @pytest.mark.asyncio
    async def test_call_model_stream_success(self):
        """Test successful Bedrock streaming call"""
        with (
            patch("boto3.client") as mock_boto3,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_client = Mock()
            mock_boto3.return_value = mock_client
            mock_validate.return_value = (True, None)

            # Mock streaming response
            mock_stream = [
                {"contentBlockDelta": {"delta": {"text": "Hello"}}},
                {"contentBlockDelta": {"delta": {"text": " world"}}},
                {"messageStop": {}},
            ]
            mock_client.converse_stream.return_value = {"stream": mock_stream}

            responses = []
            async for response in self.backend.call_model_stream(self.thread):
                responses.append(response)

            assert len(responses) == 3
            assert responses[0]["type"] == "chunk"
            assert responses[0]["content"] == "Hello"
            assert responses[1]["type"] == "chunk"
            assert responses[1]["content"] == " world"
            assert responses[2]["type"] == "complete"
            assert responses[2]["content"] == "Hello world"

    @pytest.mark.asyncio
    async def test_call_model_stream_import_error(self):
        """Test Bedrock streaming with ImportError"""

        # Mock the import statement to raise ImportError
        def mock_import(name, *args):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return __builtins__["__import__"](name, *args)

        with patch("builtins.__import__", side_effect=mock_import):
            responses = []
            async for response in self.backend.call_model_stream(self.thread):
                responses.append(response)

            assert len(responses) == 1
            assert responses[0]["type"] == "error"
            # AORP response contains the error message
            assert "boto3 library not installed" in str(responses[0]["content"])

    @pytest.mark.asyncio
    async def test_call_model_stream_validation_error(self):
        """Test Bedrock streaming with validation error"""
        with (
            patch("boto3.client") as mock_boto3,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_boto3.return_value = Mock()
            mock_validate.return_value = (False, "Invalid model")

            responses = []
            async for response in self.backend.call_model_stream(self.thread):
                responses.append(response)

            assert len(responses) == 1
            assert responses[0]["type"] == "error"


class TestLiteLLMBackend:
    """Test LiteLLMBackend implementation"""

    def setup_method(self):
        """Set up test fixtures"""
        self.backend = LiteLLMBackend()

        # Mock config
        self.mock_config = Mock()
        self.mock_config.model.litellm_model = "default-litellm-model"
        self.mock_config.model.default_max_tokens = 1000
        self.mock_config.model.default_temperature = 0.7

        self.backend.config = self.mock_config

        # Mock thread
        self.thread = Mock()
        self.thread.name = "test-thread"
        self.thread.model_name = "test-litellm-model"
        self.thread.system_prompt = "You are helpful"
        self.thread.conversation_history = [{"role": "user", "content": "Hello"}]

    def test_litellm_backend_initialization(self):
        """Test LiteLLMBackend initialization"""
        assert self.backend.backend_name == "litellm"

    def test_get_model_name(self):
        """Test _get_model_name method"""
        result = self.backend._get_model_name(self.thread)
        assert result == "test-litellm-model"

        self.thread.model_name = None
        result = self.backend._get_model_name(self.thread)
        assert result == "default-litellm-model"

    @pytest.mark.asyncio
    async def test_call_model_success(self):
        """Test successful LiteLLM model call"""
        with (
            patch("litellm.acompletion") as mock_acompletion,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_validate.return_value = (True, None)

            # Mock successful response
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Test response"
            mock_acompletion.return_value = mock_response

            result = await self.backend.call_model(self.thread)

            assert result == "Test response"
            mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_model_import_error(self):
        """Test LiteLLM call_model with ImportError"""

        # Mock the import statement by patching builtins.__import__
        def mock_import(name, *args):
            if name == "litellm":
                raise ImportError("No module named 'litellm'")
            return __builtins__["__import__"](name, *args)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ConfigurationError) as exc_info:
                await self.backend.call_model(self.thread)

            assert "litellm library not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_model_validation_error(self):
        """Test LiteLLM call_model with validation error"""
        with (
            patch("litellm.acompletion"),
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_validate.return_value = (False, "Invalid model")

            with pytest.raises(ModelValidationError):
                await self.backend.call_model(self.thread)

    @pytest.mark.asyncio
    async def test_call_model_rate_limit_error(self):
        """Test LiteLLM call_model with rate limit error"""
        with (
            patch("litellm.acompletion") as mock_acompletion,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_validate.return_value = (True, None)
            mock_acompletion.side_effect = Exception("rate limit exceeded")

            with pytest.raises(ModelRateLimitError):
                await self.backend.call_model(self.thread)


class TestOllamaBackend:
    """Test OllamaBackend implementation"""

    def setup_method(self):
        """Set up test fixtures"""
        self.backend = OllamaBackend()

        # Mock config
        self.mock_config = Mock()
        self.mock_config.model.ollama_model = "default-ollama-model"
        self.mock_config.model.ollama_host = "http://localhost:11434"
        self.mock_config.model.default_max_tokens = 1000
        self.mock_config.model.default_temperature = 0.7

        self.backend.config = self.mock_config

        # Mock thread
        self.thread = Mock()
        self.thread.name = "test-thread"
        self.thread.model_name = "test-ollama-model"
        self.thread.system_prompt = "You are helpful"
        self.thread.conversation_history = [{"role": "user", "content": "Hello"}]

    def test_ollama_backend_initialization(self):
        """Test OllamaBackend initialization"""
        assert self.backend.backend_name == "ollama"

    def test_get_model_name(self):
        """Test _get_model_name method"""
        result = self.backend._get_model_name(self.thread)
        assert result == "test-ollama-model"

        self.thread.model_name = None
        result = self.backend._get_model_name(self.thread)
        assert result == "default-ollama-model"

    @pytest.mark.asyncio
    async def test_call_model_success(self):
        """Test successful Ollama model call"""
        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_validate.return_value = (True, None)

            # Mock httpx client and response
            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            mock_response = Mock()
            mock_response.json.return_value = {"message": {"content": "Test response"}}
            mock_client.post.return_value = mock_response

            result = await self.backend.call_model(self.thread)

            assert result == "Test response"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_model_import_error(self):
        """Test Ollama call_model with ImportError"""

        # Mock the import statement to raise ImportError
        def mock_import(name, *args):
            if name == "httpx":
                raise ImportError("No module named 'httpx'")
            return __builtins__["__import__"](name, *args)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ConfigurationError) as exc_info:
                await self.backend.call_model(self.thread)

            assert "httpx library not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_model_connect_error(self):
        """Test Ollama call_model with connection error"""
        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_validate.return_value = (True, None)

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")

            with pytest.raises(ModelConnectionError) as exc_info:
                await self.backend.call_model(self.thread)

            assert "Failed to connect to Ollama" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_model_timeout_error(self):
        """Test Ollama call_model with timeout error"""
        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_validate.return_value = (True, None)

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")

            with pytest.raises(ModelTimeoutError) as exc_info:
                await self.backend.call_model(self.thread)

            assert "timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_model_model_not_found(self):
        """Test Ollama call_model with model not found error"""
        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_validate.return_value = (True, None)

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock 404 HTTP error
            mock_response = Mock()
            mock_response.status_code = 404
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Not found", request=Mock(), response=mock_response
            )

            with pytest.raises(ModelValidationError) as exc_info:
                await self.backend.call_model(self.thread)

            assert "not found in Ollama" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_model_rate_limit(self):
        """Test Ollama call_model with rate limit error"""
        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_validate.return_value = (True, None)

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock 429 HTTP error
            mock_response = Mock()
            mock_response.status_code = 429
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Rate limited", request=Mock(), response=mock_response
            )

            with pytest.raises(ModelRateLimitError):
                await self.backend.call_model(self.thread)

    @pytest.mark.asyncio
    async def test_call_model_http_error(self):
        """Test Ollama call_model with generic HTTP error"""
        with (
            patch("httpx.AsyncClient") as mock_client_class,
            patch("context_switcher_mcp.security.validate_model_id") as mock_validate,
        ):
            mock_validate.return_value = (True, None)

            mock_client = AsyncMock()
            mock_client_class.return_value.__aenter__.return_value = mock_client

            # Mock 500 HTTP error
            mock_response = Mock()
            mock_response.status_code = 500
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Server error", request=Mock(), response=mock_response
            )

            with pytest.raises(ModelBackendError) as exc_info:
                await self.backend.call_model(self.thread)

            assert "500" in str(exc_info.value)


class TestBackendRegistry:
    """Test backend registry and factory functions"""

    def test_backend_registry_contents(self):
        """Test backend registry contains expected backends"""
        expected_backends = ["bedrock", "litellm", "ollama"]

        assert all(backend in BACKEND_REGISTRY for backend in expected_backends)
        assert len(BACKEND_REGISTRY) == len(expected_backends)

    def test_backend_registry_types(self):
        """Test backend registry contains correct types"""
        assert isinstance(BACKEND_REGISTRY["bedrock"], BedrockBackend)
        assert isinstance(BACKEND_REGISTRY["litellm"], LiteLLMBackend)
        assert isinstance(BACKEND_REGISTRY["ollama"], OllamaBackend)

    def test_get_backend_interface_valid(self):
        """Test get_backend_interface with valid backend name"""
        backend = get_backend_interface("bedrock")
        assert isinstance(backend, BedrockBackend)

        backend = get_backend_interface("litellm")
        assert isinstance(backend, LiteLLMBackend)

        backend = get_backend_interface("ollama")
        assert isinstance(backend, OllamaBackend)

    def test_get_backend_interface_invalid(self):
        """Test get_backend_interface with invalid backend name"""
        with pytest.raises(ValueError) as exc_info:
            get_backend_interface("invalid_backend")

        assert "Unknown backend: invalid_backend" in str(exc_info.value)

    def test_get_backend_interface_none(self):
        """Test get_backend_interface with None"""
        with pytest.raises(ValueError):
            get_backend_interface(None)

    def test_get_backend_interface_empty_string(self):
        """Test get_backend_interface with empty string"""
        with pytest.raises(ValueError):
            get_backend_interface("")


class TestErrorHandling:
    """Test comprehensive error handling scenarios"""

    def setup_method(self):
        """Set up test fixtures"""
        self.backend = ConcreteBackend("test")
        self.thread = Mock()
        self.thread.name = "test-thread"
        self.thread.model_name = "test-model"
        self.thread.system_prompt = "Test prompt"
        self.thread.conversation_history = []

    def test_error_message_classification_edge_cases(self):
        """Test edge cases in error message classification"""
        # Test case-insensitive matching
        error = Exception("API_KEY invalid")
        error_type, message = self.backend._get_error_type_and_message(error)
        assert error_type == "credentials_error"

        # Test multiple keywords
        error = Exception("Connection timeout network failure")
        error_type, message = self.backend._get_error_type_and_message(error)
        assert error_type == "connection_error"

        # Test empty error message
        error = Exception("")
        error_type, message = self.backend._get_error_type_and_message(error)
        assert error_type == "api_error"

    def test_prepare_messages_edge_cases(self):
        """Test _prepare_messages with edge cases"""
        # Empty conversation history
        self.thread.conversation_history = []
        messages = self.backend._prepare_messages(self.thread)
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

        # Long conversation history
        long_history = [{"role": "user", "content": f"Message {i}"} for i in range(100)]
        self.thread.conversation_history = long_history
        messages = self.backend._prepare_messages(self.thread)
        assert len(messages) == 101  # system + 100 messages

    def test_model_config_edge_cases(self):
        """Test model configuration edge cases"""
        # Thread with None model name
        self.thread.model_name = None
        config = self.backend.get_model_config(self.thread)
        assert config.model_name == "test-model"  # From concrete implementation

        # Test config values are properly set
        assert isinstance(config.max_tokens, int)
        assert isinstance(config.temperature, float)
        assert isinstance(config.timeout_seconds, float)


class TestIntegrationScenarios:
    """Test integration scenarios and complex workflows"""

    @pytest.mark.asyncio
    async def test_streaming_error_recovery(self):
        """Test error recovery in streaming scenarios"""
        backend = ConcreteBackend("test")

        # Mock a call that starts successfully then fails
        call_count = 0

        async def failing_after_start(thread):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Intermittent failure")
            return "Success after retry"

        backend.call_model = failing_after_start

        thread = Mock()
        thread.name = "test-thread"

        # First call should fail and produce error
        responses = []
        async for response in backend.call_model_stream(thread):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0]["type"] == "error"

    @pytest.mark.asyncio
    async def test_concurrent_backend_calls(self):
        """Test concurrent calls to different backends"""
        backends = [ConcreteBackend(f"backend-{i}") for i in range(3)]
        threads = [Mock() for _ in range(3)]

        for i, thread in enumerate(threads):
            thread.name = f"thread-{i}"

        # Make concurrent calls
        tasks = []
        for backend, thread in zip(backends, threads, strict=False):
            task = backend.call_model(thread)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        assert all(result == "test response" for result in results)

    def test_backend_registry_thread_safety(self):
        """Test that backend registry is thread-safe"""
        import threading
        import time

        results = []

        def access_registry():
            for _ in range(10):
                backend = get_backend_interface("bedrock")
                results.append(isinstance(backend, BedrockBackend))
                time.sleep(0.001)  # Small delay to encourage race conditions

        threads = [threading.Thread(target=access_registry) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert all(results)  # All accesses should succeed


if __name__ == "__main__":
    pytest.main([__file__])
