"""Comprehensive tests for confidence_metrics module"""

import pytest
from unittest.mock import Mock

from context_switcher_mcp.confidence_metrics import (
    ResponseQuality,
    QualityMetrics,
    ConfidenceCalibrator,
    ConfidenceCalculator,
    QualityDistributionBuilder,
    analyze_synthesis_quality,
)


class TestResponseQuality:
    """Test ResponseQuality enum"""

    def test_quality_levels(self):
        """Test quality level values"""
        assert ResponseQuality.EXCELLENT.value == "excellent"
        assert ResponseQuality.GOOD.value == "good"
        assert ResponseQuality.FAIR.value == "fair"
        assert ResponseQuality.POOR.value == "poor"
        assert ResponseQuality.UNACCEPTABLE.value == "unacceptable"

    def test_quality_enum_completeness(self):
        """Test all expected quality levels exist"""
        expected_levels = ["excellent", "good", "fair", "poor", "unacceptable"]
        actual_levels = [level.value for level in ResponseQuality]
        assert sorted(actual_levels) == sorted(expected_levels)


class TestQualityMetrics:
    """Test QualityMetrics dataclass"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_metrics = QualityMetrics(
            depth_score=0.8,
            specificity_score=0.7,
            coherence_score=0.9,
            relevance_score=0.8,
            structure_score=0.6,
            actionability_score=0.7,
            evidence_score=0.8,
            perspective_alignment=0.9,
            abstention_appropriateness=0.0,
        )

    def test_overall_score_calculation(self):
        """Test overall score weighted calculation"""
        score = self.sample_metrics.overall_score

        # Manual calculation for verification
        expected = (
            0.8 * 0.15  # depth
            + 0.7 * 0.15  # specificity
            + 0.9 * 0.10  # coherence
            + 0.8 * 0.15  # relevance
            + 0.6 * 0.10  # structure
            + 0.7 * 0.15  # actionability
            + 0.8 * 0.10  # evidence
            + 0.9 * 0.05  # alignment
            + 0.0 * 0.05  # abstention
        )

        assert abs(score - expected) < 0.001

    def test_quality_level_excellent(self):
        """Test excellent quality level assignment"""
        metrics = QualityMetrics(
            depth_score=0.9,
            specificity_score=0.95,
            coherence_score=0.9,
            relevance_score=0.9,
            structure_score=0.9,
            actionability_score=0.9,
            evidence_score=0.9,
            perspective_alignment=0.9,
            abstention_appropriateness=0.0,
        )
        assert metrics.quality_level == ResponseQuality.EXCELLENT

    def test_quality_level_good(self):
        """Test good quality level assignment"""
        metrics = QualityMetrics(
            depth_score=0.7,
            specificity_score=0.7,
            coherence_score=0.7,
            relevance_score=0.7,
            structure_score=0.7,
            actionability_score=0.7,
            evidence_score=0.7,
            perspective_alignment=0.7,
            abstention_appropriateness=0.0,
        )
        assert metrics.quality_level == ResponseQuality.GOOD

    def test_quality_level_fair(self):
        """Test fair quality level assignment"""
        metrics = QualityMetrics(
            depth_score=0.5,
            specificity_score=0.5,
            coherence_score=0.5,
            relevance_score=0.5,
            structure_score=0.5,
            actionability_score=0.5,
            evidence_score=0.5,
            perspective_alignment=0.5,
            abstention_appropriateness=0.0,
        )
        assert metrics.quality_level == ResponseQuality.FAIR

    def test_quality_level_poor(self):
        """Test poor quality level assignment"""
        metrics = QualityMetrics(
            depth_score=0.3,
            specificity_score=0.3,
            coherence_score=0.3,
            relevance_score=0.3,
            structure_score=0.3,
            actionability_score=0.3,
            evidence_score=0.3,
            perspective_alignment=0.3,
            abstention_appropriateness=0.0,
        )
        assert metrics.quality_level == ResponseQuality.POOR

    def test_quality_level_unacceptable(self):
        """Test unacceptable quality level assignment"""
        metrics = QualityMetrics(
            depth_score=0.1,
            specificity_score=0.1,
            coherence_score=0.1,
            relevance_score=0.1,
            structure_score=0.1,
            actionability_score=0.1,
            evidence_score=0.1,
            perspective_alignment=0.1,
            abstention_appropriateness=0.0,
        )
        assert metrics.quality_level == ResponseQuality.UNACCEPTABLE


class TestConfidenceCalibrator:
    """Test ConfidenceCalibrator class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calibrator = ConfidenceCalibrator()

    def test_init_quality_indicators(self):
        """Test quality indicators initialization"""
        assert "specificity" in self.calibrator.quality_indicators
        assert "evidence" in self.calibrator.quality_indicators
        assert "structure" in self.calibrator.quality_indicators
        assert "actionability" in self.calibrator.quality_indicators
        assert "depth" in self.calibrator.quality_indicators

    def test_init_vague_terms(self):
        """Test vague terms initialization"""
        assert "maybe" in self.calibrator.vague_terms
        assert "possibly" in self.calibrator.vague_terms
        assert "might" in self.calibrator.vague_terms

    def test_analyze_response_quality_abstention(self):
        """Test quality analysis for abstention"""
        metrics = self.calibrator.analyze_response_quality(
            "[NO_RESPONSE]", "technical", "test prompt", is_abstention=True
        )

        assert metrics.depth_score == 0.0
        assert metrics.specificity_score == 0.0
        assert metrics.coherence_score == 1.0
        assert metrics.relevance_score == 0.0
        assert metrics.structure_score == 1.0
        assert metrics.actionability_score == 0.0
        assert metrics.evidence_score == 0.0
        assert metrics.perspective_alignment == 1.0
        assert metrics.abstention_appropriateness == 1.0

    def test_analyze_response_quality_normal(self):
        """Test quality analysis for normal response"""
        response = """
        This is a detailed technical implementation specifically focused on performance.
        First, we should implement caching because it improves response times.
        For example, Redis can reduce database load by 50%.
        I recommend implementing these steps: create cache layer, add monitoring.
        """

        metrics = self.calibrator.analyze_response_quality(
            response, "technical", "how to improve performance"
        )

        assert 0.0 <= metrics.depth_score <= 1.0
        assert 0.0 <= metrics.specificity_score <= 1.0
        assert 0.0 <= metrics.coherence_score <= 1.0
        assert 0.0 <= metrics.relevance_score <= 1.0
        assert 0.0 <= metrics.structure_score <= 1.0
        assert 0.0 <= metrics.actionability_score <= 1.0
        assert 0.0 <= metrics.evidence_score <= 1.0
        assert 0.0 <= metrics.perspective_alignment <= 1.0
        assert metrics.abstention_appropriateness == 0.0

    def test_calculate_depth_score_comprehensive_response(self):
        """Test depth score for comprehensive response"""
        long_response = (
            " ".join(["word"] * 250) + " implication consideration trade-off"
        )
        score = self.calibrator._calculate_depth_score(long_response)
        assert score > 0.5

    def test_calculate_depth_score_short_response(self):
        """Test depth score for short response"""
        short_response = "Yes, that works."
        score = self.calibrator._calculate_depth_score(short_response)
        assert score < 0.5

    def test_calculate_specificity_score_specific_response(self):
        """Test specificity score with specific language"""
        response = "Specifically, we need exactly 50% performance improvement with 100MB memory reduction"
        score = self.calibrator._calculate_specificity_score(response)
        assert score > 0.5

    def test_calculate_specificity_score_vague_response(self):
        """Test specificity score with vague language"""
        response = (
            "Maybe we could possibly improve things somewhat, kind of like generally"
        )
        score = self.calibrator._calculate_specificity_score(response)
        assert score < 0.5

    def test_calculate_specificity_score_empty_response(self):
        """Test specificity score with empty response"""
        score = self.calibrator._calculate_specificity_score("")
        assert score == 0.0

    def test_calculate_coherence_score_coherent_response(self):
        """Test coherence score for coherent response"""
        response = "First, we analyze. Therefore, we implement. As a result, performance improves."
        score = self.calibrator._calculate_coherence_score(response)
        assert score > 0.5

    def test_calculate_coherence_score_contradictory_response(self):
        """Test coherence score for contradictory response"""
        response = "This is good but not effective although it works nevertheless fails"
        score = self.calibrator._calculate_coherence_score(response)
        assert score < 1.0

    def test_calculate_relevance_score_relevant_response(self):
        """Test relevance score for relevant response"""
        prompt = "How to improve database performance optimization"
        response = "Database performance optimization requires indexing and query tuning for better performance"
        score = self.calibrator._calculate_relevance_score(response, prompt)
        assert score > 0.5

    def test_calculate_relevance_score_irrelevant_response(self):
        """Test relevance score for irrelevant response"""
        prompt = "How to improve database performance"
        response = "The weather today is sunny and birds are singing in trees"
        score = self.calibrator._calculate_relevance_score(response, prompt)
        assert score < 0.5

    def test_calculate_structure_score_structured_response(self):
        """Test structure score for well-structured response"""
        response = """First, analyze the problem.

        Second, implement the solution:
        - Create the component
        - Add configuration
        - Test thoroughly

        Additionally, monitor performance."""
        score = self.calibrator._calculate_structure_score(response)
        assert score > 0.7

    def test_calculate_actionability_score_actionable_response(self):
        """Test actionability score for actionable response"""
        response = (
            "Implement caching, create monitoring, develop metrics, establish alerts"
        )
        score = self.calibrator._calculate_actionability_score(response)
        assert score > 0.5

    def test_calculate_evidence_score_evidence_rich_response(self):
        """Test evidence score for evidence-rich response"""
        response = "This works because studies show performance improves. For example, Redis reduces latency."
        score = self.calibrator._calculate_evidence_score(response)
        assert score > 0.5

    def test_calculate_alignment_score_technical_perspective(self):
        """Test alignment score for technical perspective"""
        response = "The architecture implementation requires scalable performance code optimization"
        score = self.calibrator._calculate_alignment_score(response, "technical")
        assert score > 0.5

    def test_calculate_alignment_score_business_perspective(self):
        """Test alignment score for business perspective"""
        response = "This strategy will increase revenue and reduce cost while improving market ROI"
        score = self.calibrator._calculate_alignment_score(response, "business")
        assert score > 0.5

    def test_calculate_alignment_score_unknown_perspective(self):
        """Test alignment score for unknown perspective"""
        response = "Some generic response text"
        score = self.calibrator._calculate_alignment_score(response, "unknown")
        assert score == 0.7  # Default alignment

    def test_calculate_enhanced_confidence_no_perspectives(self):
        """Test enhanced confidence with no perspectives"""
        confidence, breakdown = self.calibrator.calculate_enhanced_confidence(
            {}, 0, 0, 0
        )
        assert confidence == 0.0
        assert breakdown["reason"] == "No perspectives available"

    def test_calculate_enhanced_confidence_all_abstentions(self):
        """Test enhanced confidence with all abstentions"""
        abstention_metrics = QualityMetrics(
            depth_score=0.0,
            specificity_score=0.0,
            coherence_score=1.0,
            relevance_score=0.0,
            structure_score=1.0,
            actionability_score=0.0,
            evidence_score=0.0,
            perspective_alignment=1.0,
            abstention_appropriateness=1.0,
        )

        confidence, breakdown = self.calibrator.calculate_enhanced_confidence(
            {"p1": abstention_metrics}, 0, 0, 1
        )
        assert confidence == 0.0
        assert breakdown["reason"] == "All perspectives abstained"

    def test_calculate_enhanced_confidence_normal_case(self):
        """Test enhanced confidence for normal case"""
        metrics = QualityMetrics(
            depth_score=0.8,
            specificity_score=0.7,
            coherence_score=0.9,
            relevance_score=0.8,
            structure_score=0.6,
            actionability_score=0.7,
            evidence_score=0.8,
            perspective_alignment=0.9,
            abstention_appropriateness=0.0,
        )

        confidence, breakdown = self.calibrator.calculate_enhanced_confidence(
            {"p1": metrics}, 0, 0, 1
        )

        assert 0.0 <= confidence <= 1.0
        assert "coverage_factor" in breakdown
        assert "average_quality" in breakdown
        assert "error_penalty" in breakdown
        assert "abstention_penalty" in breakdown
        assert "consistency_factor" in breakdown
        assert "quality_distribution" in breakdown

    def test_validate_confidence_inputs_zero_perspectives(self):
        """Test input validation with zero perspectives"""
        result = self.calibrator._validate_confidence_inputs({}, 0)
        assert result is not None
        assert result[0] == 0.0
        assert result[1]["reason"] == "No perspectives available"

    def test_validate_confidence_inputs_valid(self):
        """Test input validation with valid inputs"""
        result = self.calibrator._validate_confidence_inputs({"p1": Mock()}, 1)
        assert result is None

    def test_extract_active_metrics(self):
        """Test extraction of active metrics"""
        active_metric = Mock()
        active_metric.abstention_appropriateness = 0.0

        abstention_metric = Mock()
        abstention_metric.abstention_appropriateness = 1.0

        metrics = {"active": active_metric, "abstained": abstention_metric}
        active = self.calibrator._extract_active_metrics(metrics)

        assert len(active) == 1
        assert "active" in active
        assert "abstained" not in active


class TestConfidenceCalculator:
    """Test ConfidenceCalculator class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calculator = ConfidenceCalculator()
        self.sample_metrics = {
            "p1": QualityMetrics(
                depth_score=0.8,
                specificity_score=0.7,
                coherence_score=0.9,
                relevance_score=0.8,
                structure_score=0.6,
                actionability_score=0.7,
                evidence_score=0.8,
                perspective_alignment=0.9,
                abstention_appropriateness=0.0,
            )
        }

    def test_calculate_components(self):
        """Test component calculation"""
        components = self.calculator.calculate_components(self.sample_metrics, 0, 0, 1)

        assert "coverage" in components
        assert "avg_quality" in components
        assert "error_penalty" in components
        assert "abstention_penalty" in components
        assert "consistency" in components
        assert "final_confidence" in components

        # Verify component ranges
        assert 0.0 <= components["coverage"] <= 1.0
        assert 0.0 <= components["avg_quality"] <= 1.0
        assert 0.0 <= components["error_penalty"] <= 1.0
        assert 0.0 <= components["abstention_penalty"] <= 1.0
        assert 0.0 <= components["consistency"] <= 1.0
        assert 0.0 <= components["final_confidence"] <= 1.0

    def test_calculate_coverage(self):
        """Test coverage calculation"""
        coverage = self.calculator._calculate_coverage(self.sample_metrics, 2)
        assert coverage == 0.5  # 1 active out of 2 total

    def test_calculate_average_quality(self):
        """Test average quality calculation"""
        avg_quality = self.calculator._calculate_average_quality(self.sample_metrics)
        expected = self.sample_metrics["p1"].overall_score
        assert abs(avg_quality - expected) < 0.001

    def test_calculate_error_penalty(self):
        """Test error penalty calculation"""
        penalty = self.calculator._calculate_error_penalty(0)
        assert penalty == 1.0

        penalty = self.calculator._calculate_error_penalty(2)
        assert penalty == 0.6  # 1.0 - (2 * 0.2)

        penalty = self.calculator._calculate_error_penalty(10)
        assert penalty == 0.0  # max penalty

    def test_calculate_abstention_penalty(self):
        """Test abstention penalty calculation"""
        penalty = self.calculator._calculate_abstention_penalty(0)
        assert penalty == 1.0

        penalty = self.calculator._calculate_abstention_penalty(10)
        assert penalty == 0.7  # min penalty threshold

    def test_calculate_consistency_single_response(self):
        """Test consistency calculation with single response"""
        consistency = self.calculator._calculate_consistency(self.sample_metrics, 0.8)
        assert consistency == 0.8  # Default for single response

    def test_calculate_consistency_multiple_responses(self):
        """Test consistency calculation with multiple responses"""
        metrics_multi = {
            "p1": QualityMetrics(
                depth_score=0.8,
                specificity_score=0.8,
                coherence_score=0.8,
                relevance_score=0.8,
                structure_score=0.8,
                actionability_score=0.8,
                evidence_score=0.8,
                perspective_alignment=0.8,
                abstention_appropriateness=0.0,
            ),
            "p2": QualityMetrics(
                depth_score=0.6,
                specificity_score=0.6,
                coherence_score=0.6,
                relevance_score=0.6,
                structure_score=0.6,
                actionability_score=0.6,
                evidence_score=0.6,
                perspective_alignment=0.6,
                abstention_appropriateness=0.0,
            ),
        }

        avg_quality = sum(m.overall_score for m in metrics_multi.values()) / len(
            metrics_multi
        )
        consistency = self.calculator._calculate_consistency(metrics_multi, avg_quality)
        assert 0.5 <= consistency <= 1.0

    def test_compute_final_confidence(self):
        """Test final confidence computation"""
        components = {
            "coverage": 0.8,
            "avg_quality": 0.7,
            "error_penalty": 1.0,
            "abstention_penalty": 0.9,
            "consistency": 0.8,
        }

        confidence = self.calculator.compute_final_confidence(components)
        expected = (
            0.8 * 0.25  # coverage
            + 0.7 * 0.35  # avg_quality
            + 1.0 * 0.15  # error_penalty
            + 0.9 * 0.10  # abstention_penalty
            + 0.8 * 0.15  # consistency
        )

        assert abs(confidence - expected) < 0.001


class TestQualityDistributionBuilder:
    """Test QualityDistributionBuilder class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.builder = QualityDistributionBuilder()

    def test_build_distribution(self):
        """Test quality distribution building"""
        metrics = {
            "p1": QualityMetrics(
                depth_score=0.9,
                specificity_score=0.9,
                coherence_score=0.9,
                relevance_score=0.9,
                structure_score=0.9,
                actionability_score=0.9,
                evidence_score=0.9,
                perspective_alignment=0.9,
                abstention_appropriateness=0.0,
            ),
            "p2": QualityMetrics(
                depth_score=0.7,
                specificity_score=0.7,
                coherence_score=0.7,
                relevance_score=0.7,
                structure_score=0.7,
                actionability_score=0.7,
                evidence_score=0.7,
                perspective_alignment=0.7,
                abstention_appropriateness=0.0,
            ),
        }

        distribution = self.builder.build_distribution(metrics)

        assert "excellent" in distribution
        assert "good" in distribution
        assert "fair" in distribution
        assert "poor" in distribution
        assert "unacceptable" in distribution

        # Verify counts
        assert distribution["excellent"] == 1
        assert distribution["good"] == 1
        assert distribution["fair"] == 0
        assert distribution["poor"] == 0
        assert distribution["unacceptable"] == 0

    def test_count_by_quality_level(self):
        """Test counting by quality level"""
        metrics = {"p1": Mock(), "p2": Mock()}
        metrics["p1"].quality_level = ResponseQuality.EXCELLENT
        metrics["p2"].quality_level = ResponseQuality.GOOD

        count = self.builder._count_by_quality_level(metrics, ResponseQuality.EXCELLENT)
        assert count == 1

        count = self.builder._count_by_quality_level(metrics, ResponseQuality.FAIR)
        assert count == 0


class TestAnalyzeSynthesisQuality:
    """Test analyze_synthesis_quality function"""

    def test_synthesis_quality_high_quality(self):
        """Test synthesis quality analysis for high-quality synthesis"""
        synthesis = """
        Multiple perspectives agree on the core implementation approach.
        The technical analysis reveals performance optimizations.
        Business perspective highlights revenue impact.
        
        However, there are tensions between speed and security.
        The trade-off emerges clearly from user perspective.
        
        Key insights discover unexpected scalability concerns.
        """

        confidence, breakdown = analyze_synthesis_quality(synthesis, 3, "test prompt")

        assert 0.0 <= confidence <= 1.0
        assert "base_quality" in breakdown
        assert "pattern_diversity" in breakdown
        assert "structural_clarity" in breakdown
        assert "perspective_integration" in breakdown
        assert "patterns_found" in breakdown
        assert "synthesis_confidence" in breakdown

        # Check pattern detection
        patterns = breakdown["patterns_found"]
        assert "convergence" in patterns
        assert "tensions" in patterns
        assert "insights" in patterns

    def test_synthesis_quality_poor_structure(self):
        """Test synthesis quality for poorly structured synthesis"""
        synthesis = "This is a short synthesis without structure or patterns."

        confidence, breakdown = analyze_synthesis_quality(synthesis, 2, "test prompt")

        assert confidence < 0.7  # Should be lower for poor structure
        assert breakdown["structural_clarity"] < 1.0

    def test_synthesis_quality_no_patterns(self):
        """Test synthesis quality with no recognizable patterns"""
        synthesis = (
            "Generic text without any convergence, tension, or insight patterns."
        )

        confidence, breakdown = analyze_synthesis_quality(synthesis, 1, "test prompt")

        assert breakdown["patterns_found"]["convergence"] == 0
        assert breakdown["patterns_found"]["tensions"] == 0
        assert breakdown["patterns_found"]["insights"] == 0

    def test_synthesis_quality_with_perspective_integration(self):
        """Test synthesis quality with good perspective integration"""
        synthesis = """
        The technical perspective shows performance concerns.
        Business perspective indicates cost implications.
        User perspective reveals usability issues.
        Multiple perspectives converge on the solution.
        """

        confidence, breakdown = analyze_synthesis_quality(synthesis, 3, "test prompt")

        assert breakdown["perspective_integration"] > 0.5


class TestErrorHandling:
    """Test error handling and edge cases"""

    def setup_method(self):
        """Set up test fixtures"""
        self.calibrator = ConfidenceCalibrator()

    def test_empty_response_handling(self):
        """Test handling of empty responses"""
        metrics = self.calibrator.analyze_response_quality("", "technical", "prompt")

        # Verify all scores are valid (0.0 to 1.0)
        assert 0.0 <= metrics.depth_score <= 1.0
        assert 0.0 <= metrics.specificity_score <= 1.0
        assert 0.0 <= metrics.coherence_score <= 1.0
        assert 0.0 <= metrics.relevance_score <= 1.0
        assert 0.0 <= metrics.structure_score <= 1.0
        assert 0.0 <= metrics.actionability_score <= 1.0
        assert 0.0 <= metrics.evidence_score <= 1.0
        assert 0.0 <= metrics.perspective_alignment <= 1.0

    def test_extreme_error_counts(self):
        """Test handling of extreme error counts"""
        calculator = ConfidenceCalculator()

        # Very high error count
        penalty = calculator._calculate_error_penalty(100)
        assert penalty == 0.0

        # Negative error count (should be handled gracefully)
        penalty = calculator._calculate_error_penalty(-1)
        assert penalty >= 0.0

    def test_malformed_input_handling(self):
        """Test handling of malformed inputs"""
        # Test with None values where strings expected
        try:
            self.calibrator.analyze_response_quality(None, "technical", "prompt")
        except (TypeError, AttributeError):
            # Expected to fail, but gracefully
            pass

    def test_unicode_text_handling(self):
        """Test handling of unicode text"""
        unicode_response = "This response contains émojis 🚀 and spëcial characters"
        metrics = self.calibrator.analyze_response_quality(
            unicode_response, "technical", "unicode prompt"
        )

        # Should handle unicode without errors
        assert 0.0 <= metrics.overall_score <= 1.0

    def test_very_long_response_handling(self):
        """Test handling of very long responses"""
        long_response = "word " * 10000  # Very long response
        metrics = self.calibrator.analyze_response_quality(
            long_response, "technical", "long prompt test"
        )

        # Should handle long text without performance issues
        assert 0.0 <= metrics.overall_score <= 1.0

    def test_boundary_score_values(self):
        """Test boundary values for quality scores"""
        # Test with metrics at exact boundaries
        boundary_metrics = QualityMetrics(
            depth_score=0.9,
            specificity_score=0.89999,
            coherence_score=0.9,
            relevance_score=0.9,
            structure_score=0.9,
            actionability_score=0.9,
            evidence_score=0.9,
            perspective_alignment=0.9,
            abstention_appropriateness=0.0,
        )

        # Should correctly classify quality level
        assert boundary_metrics.quality_level in [
            ResponseQuality.GOOD,
            ResponseQuality.EXCELLENT,
        ]


if __name__ == "__main__":
    pytest.main([__file__])
