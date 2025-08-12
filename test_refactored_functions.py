#!/usr/bin/env python3
"""
Test script to verify refactored functions work correctly
This tests the core functionality without requiring all dependencies
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


# Test 1: ConfidenceCalibrator refactoring
def test_confidence_calculator():
    """Test the refactored confidence calculation"""
    from context_switcher_mcp.confidence_metrics import (
        ConfidenceCalibrator,
        QualityMetrics,
        ConfidenceCalculator,
        QualityDistributionBuilder,
        ResponseQuality,
    )

    print("Testing ConfidenceCalibrator refactoring...")

    # Create test data
    calibrator = ConfidenceCalibrator()
    metrics = {
        "test1": QualityMetrics(
            depth_score=0.8,
            specificity_score=0.7,
            coherence_score=0.9,
            relevance_score=0.8,
            structure_score=0.6,
            actionability_score=0.7,
            evidence_score=0.8,
            perspective_alignment=0.9,
            abstention_appropriateness=0.0,
        ),
        "test2": QualityMetrics(
            depth_score=0.6,
            specificity_score=0.8,
            coherence_score=0.7,
            relevance_score=0.9,
            structure_score=0.8,
            actionability_score=0.6,
            evidence_score=0.7,
            perspective_alignment=0.8,
            abstention_appropriateness=0.0,
        ),
    }

    # Test the refactored function
    confidence, breakdown = calibrator.calculate_enhanced_confidence(
        metrics, error_count=0, abstention_count=0, total_perspectives=2
    )

    assert 0.0 <= confidence <= 1.0, f"Confidence should be 0-1, got {confidence}"
    assert "coverage_factor" in breakdown, "Missing coverage_factor in breakdown"
    assert "final_confidence" in breakdown, "Missing final_confidence in breakdown"

    # Test helper classes work
    calculator = ConfidenceCalculator()
    active_metrics = {"test1": metrics["test1"], "test2": metrics["test2"]}
    components = calculator.calculate_components(active_metrics, 0, 0, 2)
    assert "coverage" in components, "ConfidenceCalculator missing coverage"

    dist_builder = QualityDistributionBuilder()
    distribution = dist_builder.build_distribution(active_metrics)
    assert (
        ResponseQuality.GOOD.value in distribution
    ), "QualityDistributionBuilder failed"

    print("✓ ConfidenceCalibrator refactoring working correctly")


# Test 2: CircuitBreakerStore refactoring
def test_circuit_breaker_store():
    """Test the refactored CircuitBreakerStore initialization"""
    from context_switcher_mcp.circuit_breaker_store import CircuitBreakerStore
    import tempfile
    import os

    print("Testing CircuitBreakerStore refactoring...")

    # Test default path
    store1 = CircuitBreakerStore()
    assert str(store1.storage_path).endswith(
        "circuit_breakers.json"
    ), "Default path incorrect"

    # Test custom valid path
    with tempfile.TemporaryDirectory() as tmpdir:
        custom_path = os.path.join(tmpdir, "test_breakers.json")
        store2 = CircuitBreakerStore(custom_path)
        assert str(store2.storage_path) == custom_path, "Custom path not set correctly"

    # Test security validation works
    try:
        CircuitBreakerStore("../../../etc/passwd")
        assert False, "Should have blocked path traversal"
    except ValueError as e:
        assert "Path traversal" in str(e), "Wrong error for path traversal"

    print("✓ CircuitBreakerStore refactoring working correctly")


# Test 3: Basic functionality of other refactored classes
def test_basic_imports():
    """Test that all refactored classes can be imported and instantiated"""
    print("Testing basic imports of refactored classes...")

    # Test MetricsManager classes
    from context_switcher_mcp.metrics_manager import MetricsManager, ThreadMetrics

    metrics_mgr = MetricsManager()
    thread_metrics = ThreadMetrics("test", "test_op", 1.0, True)
    assert thread_metrics.success == True, "ThreadMetrics creation failed"

    # Test that methods exist
    assert hasattr(
        metrics_mgr, "_calculate_thread_summary"
    ), "Missing refactored method"
    assert hasattr(
        metrics_mgr, "_calculate_backend_performance"
    ), "Missing refactored method"

    print("✓ All refactored classes import and instantiate correctly")


if __name__ == "__main__":
    try:
        test_confidence_calculator()
        test_circuit_breaker_store()
        test_basic_imports()
        print("\n🎉 All refactoring tests passed! Functions work correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
