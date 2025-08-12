"""Comprehensive tests for LLM profiling system"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

from src.context_switcher_mcp.llm_profiler import (
    LLMProfiler,
    ProfilingConfig,
    ProfilingLevel,
    LLMCallMetrics,
    CostCalculator,
    MemoryProfiler,
    get_global_profiler,
)
from src.context_switcher_mcp.profiling_wrapper import (
    ProfilingBackendWrapper,
    EnhancedProfilingWrapper,
    create_profiling_wrapper,
)
from src.context_switcher_mcp.performance_dashboard import PerformanceDashboard
from src.context_switcher_mcp.models import Thread, ModelBackend
from src.context_switcher_mcp.backend_interface import ModelCallConfig


class TestCostCalculator:
    """Test cost calculation functionality"""

    def test_bedrock_cost_calculation(self):
        """Test Bedrock cost calculation"""
        cost = CostCalculator.calculate_cost(
            "bedrock",
            "anthropic.claude-3-haiku-20240307-v1:0",
            1000,  # input tokens
            500,  # output tokens
        )

        # Expected: (1000/1000 * 0.00025) + (500/1000 * 0.00125) = 0.00025 + 0.000625 = 0.000875
        assert cost == pytest.approx(0.000875, rel=1e-6)

    def test_litellm_cost_calculation(self):
        """Test LiteLLM cost calculation"""
        cost = CostCalculator.calculate_cost(
            "litellm",
            "gpt-4",
            2000,  # input tokens
            1000,  # output tokens
        )

        # Expected: (2000/1000 * 0.03) + (1000/1000 * 0.06) = 0.06 + 0.06 = 0.12
        assert cost == pytest.approx(0.12, rel=1e-6)

    def test_ollama_cost_calculation(self):
        """Test Ollama cost calculation (should be free)"""
        cost = CostCalculator.calculate_cost("ollama", "llama3.2", 1000, 500)

        assert cost == 0.0

    def test_unknown_model_cost(self):
        """Test cost calculation for unknown model"""
        cost = CostCalculator.calculate_cost("bedrock", "unknown-model", 1000, 500)

        assert cost is None

    def test_invalid_backend_cost(self):
        """Test cost calculation for invalid backend"""
        cost = CostCalculator.calculate_cost("unknown-backend", "some-model", 1000, 500)

        assert cost is None


class TestLLMCallMetrics:
    """Test LLM call metrics functionality"""

    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = LLMCallMetrics(
            call_id="test-call-1",
            session_id="test-session",
            thread_name="test-thread",
            backend="bedrock",
            model_name="claude-3-haiku",
        )

        assert metrics.call_id == "test-call-1"
        assert metrics.session_id == "test-session"
        assert metrics.thread_name == "test-thread"
        assert metrics.backend == "bedrock"
        assert metrics.model_name == "claude-3-haiku"
        assert not metrics.success  # Default False
        assert metrics.end_time is None

    def test_latency_calculation(self):
        """Test latency calculation"""
        start_time = time.time()
        metrics = LLMCallMetrics(
            call_id="test-call",
            session_id="test-session",
            thread_name="test-thread",
            backend="bedrock",
            model_name="claude-3-haiku",
            start_time=start_time,
        )

        # Simulate call completion
        metrics.end_time = start_time + 2.5

        assert metrics.total_latency == pytest.approx(2.5, rel=1e-3)

    def test_tokens_per_second_calculation(self):
        """Test token rate calculation"""
        start_time = time.time()
        metrics = LLMCallMetrics(
            call_id="test-call",
            session_id="test-session",
            thread_name="test-thread",
            backend="bedrock",
            model_name="claude-3-haiku",
            start_time=start_time,
            output_tokens=100,
        )

        metrics.end_time = start_time + 2.0

        # 100 tokens / 2 seconds = 50 tokens/second
        assert metrics.tokens_per_second == pytest.approx(50.0, rel=1e-3)

    def test_cost_per_token_calculation(self):
        """Test cost efficiency calculation"""
        metrics = LLMCallMetrics(
            call_id="test-call",
            session_id="test-session",
            thread_name="test-thread",
            backend="bedrock",
            model_name="claude-3-haiku",
            total_tokens=1000,
            estimated_cost_usd=0.001,
        )

        # $0.001 / 1000 tokens = $0.000001 per token
        assert metrics.cost_per_token == pytest.approx(0.000001, rel=1e-6)


class TestProfilingConfig:
    """Test profiling configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ProfilingConfig()

        assert config.enabled is True
        assert config.level == ProfilingLevel.STANDARD
        assert config.sampling_rate == 0.1
        assert config.track_tokens is True
        assert config.track_costs is True
        assert config.track_memory is False
        assert config.max_history_size == 10000


@pytest.mark.asyncio
class TestLLMProfiler:
    """Test main LLM profiler functionality"""

    def test_profiler_initialization(self):
        """Test profiler initialization"""
        config = ProfilingConfig(sampling_rate=0.2)
        profiler = LLMProfiler(config)

        assert profiler.config.sampling_rate == 0.2
        assert profiler._total_calls == 0
        assert profiler._profiled_calls == 0

    def test_sampling_decision(self):
        """Test sampling decision logic"""
        config = ProfilingConfig(sampling_rate=0.5)  # 50% sampling
        profiler = LLMProfiler(config)

        # Test multiple calls to verify sampling rate
        sampled_count = 0
        total_calls = 1000

        for i in range(total_calls):
            if profiler.should_profile_call(f"session-{i}", "thread-1", "bedrock"):
                sampled_count += 1

        # Should be approximately 50% with some tolerance
        assert 0.4 * total_calls <= sampled_count <= 0.6 * total_calls

    def test_always_profile_conditions(self):
        """Test conditions that always trigger profiling"""
        config = ProfilingConfig(sampling_rate=0.0)  # 0% normal sampling
        profiler = LLMProfiler(config)

        # Circuit breaker should always be profiled
        should_profile = profiler.should_profile_call(
            "session-1", "thread-1", "bedrock", circuit_breaker_triggered=True
        )
        assert should_profile is True

    async def test_profile_call_context_manager(self):
        """Test profile_call context manager"""
        config = ProfilingConfig(sampling_rate=1.0)  # Always sample
        profiler = LLMProfiler(config)

        async with profiler.profile_call(
            "session-1", "thread-1", "bedrock", "claude-3-haiku"
        ) as metrics:
            assert metrics is not None
            assert metrics.session_id == "session-1"
            assert metrics.thread_name == "thread-1"
            assert metrics.backend == "bedrock"
            assert metrics.model_name == "claude-3-haiku"

            # Simulate work
            await asyncio.sleep(0.1)

        # Should be marked as successful
        assert metrics.success is True
        assert metrics.total_latency is not None
        assert metrics.total_latency > 0

    async def test_profile_call_with_error(self):
        """Test profiling when an error occurs"""
        config = ProfilingConfig(sampling_rate=1.0)
        profiler = LLMProfiler(config)

        try:
            async with profiler.profile_call(
                "session-1", "thread-1", "bedrock", "claude-3-haiku"
            ) as metrics:
                raise ValueError("Test error")
        except ValueError:
            pass

        # Should capture error information
        assert metrics.success is False
        assert metrics.error_type == "ValueError"
        assert metrics.error_message == "Test error"

    def test_token_usage_recording(self):
        """Test token usage recording"""
        config = ProfilingConfig()
        profiler = LLMProfiler(config)

        metrics = LLMCallMetrics(
            call_id="test-call",
            session_id="session-1",
            thread_name="thread-1",
            backend="bedrock",
            model_name="claude-3-haiku",
        )

        profiler.record_token_usage(
            metrics,
            input_tokens=1000,
            output_tokens=500,
            prompt_length=4000,
            response_length=2000,
        )

        assert metrics.input_tokens == 1000
        assert metrics.output_tokens == 500
        assert metrics.total_tokens == 1500
        assert metrics.prompt_length == 4000
        assert metrics.response_length == 2000
        assert metrics.estimated_cost_usd is not None

    def test_network_timing_recording(self):
        """Test network timing recording"""
        config = ProfilingConfig()
        profiler = LLMProfiler(config)

        start_time = time.time()
        metrics = LLMCallMetrics(
            call_id="test-call",
            session_id="session-1",
            thread_name="thread-1",
            backend="bedrock",
            model_name="claude-3-haiku",
            start_time=start_time,
        )

        first_token_time = start_time + 1.0
        metrics.end_time = start_time + 3.0

        profiler.record_network_timing(metrics, first_token_time)

        assert metrics.network_latency == pytest.approx(1.0, rel=1e-3)
        assert metrics.processing_time == pytest.approx(2.0, rel=1e-3)

    async def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        config = ProfilingConfig()
        profiler = LLMProfiler(config)

        # Add some test metrics
        for i in range(10):
            metrics = LLMCallMetrics(
                call_id=f"call-{i}",
                session_id="session-1",
                thread_name=f"thread-{i % 3}",
                backend="bedrock" if i % 2 == 0 else "litellm",
                model_name="claude-3-haiku",
                success=i < 8,  # 2 failures
                input_tokens=1000 + i * 100,
                output_tokens=500 + i * 50,
                estimated_cost_usd=0.001 * (i + 1),
            )
            metrics.end_time = time.time() + i * 0.1
            await profiler._store_metrics(metrics)

        performance_metrics = await profiler.get_performance_metrics(last_n=10)

        assert "profiling_summary" in performance_metrics
        assert "token_usage" in performance_metrics
        assert "cost_analysis" in performance_metrics
        assert "backend_performance" in performance_metrics

        summary = performance_metrics["profiling_summary"]
        assert summary["total_calls_analyzed"] == 10
        assert summary["success_rate_percent"] == 80.0  # 8/10

    def test_configuration_status(self):
        """Test configuration status reporting"""
        config = ProfilingConfig(
            sampling_rate=0.3, track_memory=True, cost_alert_threshold_usd=50.0
        )
        profiler = LLMProfiler(config)

        status = profiler.get_configuration_status()

        assert status["enabled"] is True
        assert status["sampling_rate"] == 0.3
        assert status["features"]["track_memory"] is True
        assert status["thresholds"]["cost_alert_usd"] == 50.0


@pytest.mark.asyncio
class TestProfilingWrapper:
    """Test profiling wrapper functionality"""

    def test_wrapper_creation(self):
        """Test profiling wrapper creation"""
        # Create mock backend
        mock_backend = Mock()
        mock_backend.backend_name = "test-backend"
        mock_backend.config = Mock()

        # Create wrapper
        wrapper = ProfilingBackendWrapper(mock_backend)

        assert wrapper.backend_name == "test-backend"
        assert wrapper.wrapped_backend == mock_backend

    def test_create_profiling_wrapper(self):
        """Test wrapper factory function"""
        mock_backend = Mock()
        mock_backend.backend_name = "bedrock"

        wrapper = create_profiling_wrapper(mock_backend)

        assert isinstance(wrapper, EnhancedProfilingWrapper)
        assert wrapper.wrapped_backend == mock_backend

    async def test_profiled_model_call(self):
        """Test profiled model call"""
        # Create mock backend
        mock_backend = AsyncMock()
        mock_backend.backend_name = "bedrock"
        mock_backend.config = Mock()
        mock_backend.get_model_config.return_value = ModelCallConfig(
            max_tokens=1000,
            temperature=0.7,
            model_name="claude-3-haiku",
            timeout_seconds=60.0,
        )
        mock_backend.call_model.return_value = "Test response"

        # Create wrapper
        wrapper = ProfilingBackendWrapper(mock_backend)

        # Create test thread
        thread = Thread(
            id="thread-1",
            name="test-thread",
            system_prompt="Test system prompt",
            model_backend=ModelBackend.BEDROCK,
            model_name="claude-3-haiku",
        )
        thread.session_id = "session-1"
        thread.add_message("user", "Test message")

        # Call with profiling
        with patch(
            "src.context_switcher_mcp.llm_profiler.get_global_profiler"
        ) as mock_get_profiler:
            mock_profiler = Mock()
            mock_profiler.config.enabled = True
            mock_profiler.profile_call = AsyncMock()
            mock_profiler.profile_call.return_value.__aenter__ = AsyncMock(
                return_value=Mock()
            )
            mock_profiler.profile_call.return_value.__aexit__ = AsyncMock(
                return_value=None
            )
            mock_get_profiler.return_value = mock_profiler

            response = await wrapper.call_model(thread)

            assert response == "Test response"
            mock_backend.call_model.assert_called_once_with(thread)

    def test_token_estimation(self):
        """Test token estimation logic"""
        mock_backend = Mock()
        mock_backend.backend_name = "bedrock"
        mock_backend.config = Mock()

        wrapper = ProfilingBackendWrapper(mock_backend)

        # Test simple text
        tokens = wrapper._estimate_tokens("This is a test message")
        assert tokens > 0

        # Test empty text
        tokens = wrapper._estimate_tokens("")
        assert tokens == 0

        # Test longer text should have more tokens
        short_tokens = wrapper._estimate_tokens("Short")
        long_tokens = wrapper._estimate_tokens(
            "This is a much longer text that should have more tokens"
        )
        assert long_tokens > short_tokens


@pytest.mark.asyncio
class TestPerformanceDashboard:
    """Test performance dashboard functionality"""

    async def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        dashboard = PerformanceDashboard()

        assert dashboard.profiler is not None
        assert dashboard._cache == {}
        assert dashboard._cache_ttl == 60.0

    async def test_empty_dashboard(self):
        """Test dashboard with no data"""
        dashboard = PerformanceDashboard()

        # Mock empty profiler
        with patch.object(dashboard, "_get_metrics_for_timeframe", return_value=[]):
            result = await dashboard.get_comprehensive_dashboard()

            assert result["status"] == "no_data"
            assert "No profiling data available" in result["message"]

    async def test_cost_breakdown_calculation(self):
        """Test cost breakdown calculation"""
        dashboard = PerformanceDashboard()

        # Create test metrics
        test_metrics = []
        for i in range(5):
            metrics = LLMCallMetrics(
                call_id=f"call-{i}",
                session_id="session-1",
                thread_name=f"thread-{i}",
                backend="bedrock" if i % 2 == 0 else "litellm",
                model_name="claude-3-haiku",
                estimated_cost_usd=0.01 * (i + 1),
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
            )
            test_metrics.append(metrics)

        cost_breakdown = await dashboard._compute_cost_breakdown(test_metrics)

        assert cost_breakdown.total_cost_usd == pytest.approx(
            0.15, rel=1e-3
        )  # 0.01+0.02+0.03+0.04+0.05
        assert "bedrock" in cost_breakdown.cost_by_backend
        assert "litellm" in cost_breakdown.cost_by_backend
        assert cost_breakdown.most_expensive_call is not None

    async def test_performance_analysis_calculation(self):
        """Test performance analysis calculation"""
        dashboard = PerformanceDashboard()

        # Create test metrics with various latencies
        test_metrics = []
        latencies = [1.0, 2.0, 3.0, 5.0, 10.0]

        for i, latency in enumerate(latencies):
            start_time = time.time()
            metrics = LLMCallMetrics(
                call_id=f"call-{i}",
                session_id="session-1",
                thread_name=f"thread-{i}",
                backend="bedrock",
                model_name="claude-3-haiku",
                start_time=start_time,
                success=True,
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            )
            metrics.end_time = start_time + latency
            test_metrics.append(metrics)

        performance = await dashboard._compute_performance_analysis(test_metrics)

        assert performance.total_calls == 5
        assert performance.success_rate == 100.0
        assert performance.avg_latency == pytest.approx(
            4.2, rel=1e-1
        )  # Mean of latencies
        assert performance.median_latency == pytest.approx(3.0, rel=1e-3)
        assert len(performance.slowest_calls) <= 5

    async def test_optimization_recommendations(self):
        """Test optimization recommendations generation"""
        dashboard = PerformanceDashboard()

        # Create metrics with issues that should trigger recommendations
        test_metrics = []

        # Add expensive calls
        for i in range(3):
            metrics = LLMCallMetrics(
                call_id=f"expensive-{i}",
                session_id="session-1",
                thread_name="expensive-thread",
                backend="bedrock",
                model_name="claude-3-opus",
                estimated_cost_usd=0.5,  # Expensive
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
            )
            test_metrics.append(metrics)

        # Add slow calls
        for i in range(2):
            start_time = time.time()
            metrics = LLMCallMetrics(
                call_id=f"slow-{i}",
                session_id="session-1",
                thread_name="slow-thread",
                backend="bedrock",
                model_name="claude-3-haiku",
                start_time=start_time,
                timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
            )
            metrics.end_time = start_time + 15.0  # Slow call
            test_metrics.append(metrics)

        with patch.object(
            dashboard, "_get_metrics_for_timeframe", return_value=test_metrics
        ):
            recommendations = await dashboard.get_optimization_recommendations()

            assert "recommendations" in recommendations
            recommendations_list = recommendations["recommendations"]

            # Should have cost and performance recommendations
            categories = [r["category"] for r in recommendations_list]
            assert "Cost Optimization" in categories
            assert "Performance" in categories


@pytest.mark.asyncio
class TestMemoryProfiler:
    """Test memory profiling functionality"""

    async def test_memory_monitoring(self):
        """Test memory monitoring start/stop"""
        profiler = MemoryProfiler()

        await profiler.start_monitoring()
        assert profiler._monitoring is True

        # Simulate some work
        await asyncio.sleep(0.1)

        peak_memory = await profiler.stop_monitoring()
        assert peak_memory >= 0.0
        assert profiler._monitoring is False

    async def test_memory_monitoring_cancellation(self):
        """Test memory monitoring task cancellation"""
        profiler = MemoryProfiler()

        await profiler.start_monitoring()

        # Stop immediately
        peak_memory = await profiler.stop_monitoring()
        assert peak_memory >= 0.0


@pytest.mark.asyncio
class TestProfilingIntegration:
    """Integration tests for the complete profiling system"""

    async def test_end_to_end_profiling(self):
        """Test complete profiling flow"""
        # This would test the full integration but requires mocking
        # many components. For now, we verify the main components work together.

        config = ProfilingConfig(sampling_rate=1.0, track_costs=True)
        profiler = LLMProfiler(config)
        dashboard = PerformanceDashboard()

        # Verify components are initialized
        assert profiler.config.sampling_rate == 1.0
        assert dashboard.profiler is not None

        # Test basic integration
        status = profiler.get_configuration_status()
        assert "enabled" in status
        assert "sampling_rate" in status

    def test_global_profiler_singleton(self):
        """Test global profiler singleton behavior"""
        profiler1 = get_global_profiler()
        profiler2 = get_global_profiler()

        # Should return the same instance
        assert profiler1 is profiler2
