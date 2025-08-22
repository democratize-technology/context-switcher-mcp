"""Core AORP functionality tests - validating the fundamental building blocks"""

import pytest
from context_switcher_mcp.aorp import (
    AORPBuilder,
    calculate_analysis_confidence,
    calculate_synthesis_confidence,
    create_error_response,
    generate_analysis_next_steps,
    generate_synthesis_next_steps,
)


class TestAORPBuilder:
    """Test the AORPBuilder class - the foundation of AORP responses"""

    def test_builder_initialization(self):
        """Test that builder initializes with correct structure"""
        builder = AORPBuilder()
        assert "immediate" in builder.response
        assert "actionable" in builder.response
        assert "quality" in builder.response
        assert "details" in builder.response

    def test_immediate_section_building(self):
        """Test building the immediate section with key fields"""
        response = (
            AORPBuilder()
            .status("success")
            .key_insight("Multi-perspective analysis reveals key patterns")
            .confidence(0.85)
            .session_id("test-session-123")
            .build()
        )

        immediate = response["immediate"]
        assert immediate["status"] == "success"
        assert (
            immediate["key_insight"]
            == "Multi-perspective analysis reveals key patterns"
        )
        assert immediate["confidence"] == 0.85
        assert immediate["session_id"] == "test-session-123"

    def test_actionable_section_building(self):
        """Test building actionable guidance section"""
        next_steps = [
            "synthesize_perspectives() - Discover patterns",
            "add_perspective('security') - Expand analysis",
        ]

        response = (
            AORPBuilder()
            .status("success")
            .key_insight("Test insight")
            .confidence(0.8)
            .next_steps(next_steps)
            .primary_recommendation("Focus on synthesis for deeper insights")
            .secondary_recommendations(["Add security perspective", "Review findings"])
            .workflow_guidance("Present insights then guide to synthesis")
            .build()
        )

        actionable = response["actionable"]
        assert actionable["next_steps"] == next_steps
        assert (
            actionable["recommendations"]["primary"]
            == "Focus on synthesis for deeper insights"
        )
        assert actionable["recommendations"]["secondary"] == [
            "Add security perspective",
            "Review findings",
        ]
        assert (
            actionable["workflow_guidance"]
            == "Present insights then guide to synthesis"
        )

    def test_quality_section_building(self):
        """Test building quality assessment section"""
        response = (
            AORPBuilder()
            .status("success")
            .key_insight("Test insight")
            .confidence(0.8)
            .completeness(0.9)
            .reliability(0.85)
            .urgency("high")
            .indicators(perspectives_responded=4, error_count=0, abstention_count=1)
            .build()
        )

        quality = response["quality"]
        assert quality["completeness"] == 0.9
        assert quality["reliability"] == 0.85
        assert quality["urgency"] == "high"
        assert quality["indicators"]["perspectives_responded"] == 4
        assert quality["indicators"]["error_count"] == 0
        assert quality["indicators"]["abstention_count"] == 1

    def test_details_section_building(self):
        """Test building details section with metadata"""
        test_data = {
            "perspectives": ["technical", "business"],
            "results": ["insight1", "insight2"],
        }

        response = (
            AORPBuilder()
            .status("success")
            .key_insight("Test insight")
            .confidence(0.8)
            .summary("Comprehensive analysis completed with 4 perspectives")
            .data(test_data)
            .metadata(operation_type="analysis", perspective_count=4)
            .debug({"timing": "2.3s", "tokens_used": 1250})
            .build()
        )

        details = response["details"]
        assert (
            details["summary"] == "Comprehensive analysis completed with 4 perspectives"
        )
        assert details["data"] == test_data
        assert details["metadata"]["operation_type"] == "analysis"
        assert details["metadata"]["perspective_count"] == 4
        assert "timestamp" in details["metadata"]  # Auto-added
        assert details["debug"]["timing"] == "2.3s"

    def test_required_fields_validation(self):
        """Test that required fields are enforced"""
        builder = AORPBuilder()

        # Should fail without required fields
        with pytest.raises(ValueError, match="Status is required"):
            builder.build()

        builder.status("success")
        with pytest.raises(ValueError, match="Key insight is required"):
            builder.build()

        builder.key_insight("Test insight")
        with pytest.raises(ValueError, match="Confidence score is required"):
            builder.build()

        # Should succeed with required fields
        response = builder.confidence(0.8).build()
        assert response["immediate"]["status"] == "success"

    def test_confidence_bounds_validation(self):
        """Test that confidence scores are properly bounded"""
        response = (
            AORPBuilder()
            .status("success")
            .key_insight("Test")
            .confidence(1.5)  # Above max
            .build()
        )
        assert response["immediate"]["confidence"] == 1.0

        response = (
            AORPBuilder()
            .status("success")
            .key_insight("Test")
            .confidence(-0.2)  # Below min
            .build()
        )
        assert response["immediate"]["confidence"] == 0.0

    def test_urgency_validation(self):
        """Test urgency level validation"""
        response = (
            AORPBuilder()
            .status("success")
            .key_insight("Test")
            .confidence(0.8)
            .urgency("invalid_level")  # Invalid urgency
            .build()
        )
        assert response["quality"]["urgency"] == "medium"  # Default fallback

    def test_default_values(self):
        """Test that defaults are set for optional quality fields"""
        response = (
            AORPBuilder()
            .status("success")
            .key_insight("Test insight")
            .confidence(0.8)
            .build()
        )

        quality = response["quality"]
        assert quality["completeness"] == 0.5  # Default
        assert quality["reliability"] == 0.5  # Default
        assert quality["urgency"] == "medium"  # Default

        actionable = response["actionable"]
        assert actionable["next_steps"] == []
        assert actionable["recommendations"]["primary"] == ""
        assert actionable["workflow_guidance"] == ""


class TestConfidenceCalculations:
    """Test confidence calculation algorithms"""

    def test_analysis_confidence_perfect_scenario(self):
        """Test confidence calculation with perfect responses"""
        confidence = calculate_analysis_confidence(
            perspectives_responded=4,
            total_perspectives=4,
            error_count=0,
            abstention_count=0,
            response_lengths=[300, 250, 350, 280],  # Good quality responses
        )
        assert confidence > 0.8  # Should be high confidence
        assert confidence <= 1.0

    def test_analysis_confidence_with_errors(self):
        """Test confidence reduction due to errors"""
        confidence = calculate_analysis_confidence(
            perspectives_responded=4,
            total_perspectives=4,
            error_count=2,  # 2 errors should reduce confidence
            abstention_count=0,
            response_lengths=[300, 250, 350, 280],
        )
        assert confidence < 0.8  # Should be reduced due to errors
        assert confidence > 0.0

    def test_analysis_confidence_with_abstentions(self):
        """Test confidence with abstentions (better than errors in some cases)"""
        confidence_with_abstentions = calculate_analysis_confidence(
            perspectives_responded=4,
            total_perspectives=4,
            error_count=0,
            abstention_count=2,
            response_lengths=[300, 250],  # Only 2 actual responses
        )

        confidence_with_errors = calculate_analysis_confidence(
            perspectives_responded=4,
            total_perspectives=4,
            error_count=2,
            abstention_count=0,
            response_lengths=[300, 250],
        )

        # Both should have reasonable confidence but errors might be penalized differently
        # The key is that both are reasonable but below high confidence
        assert 0.0 <= confidence_with_abstentions <= 1.0
        assert 0.0 <= confidence_with_errors <= 1.0
        assert (
            confidence_with_abstentions < 0.8
        )  # Should be reduced due to coverage loss
        assert confidence_with_errors < 0.8  # Should be reduced due to errors

    def test_analysis_confidence_response_quality(self):
        """Test that response length affects confidence"""
        high_quality = calculate_analysis_confidence(
            perspectives_responded=2,
            total_perspectives=2,
            error_count=0,
            abstention_count=0,
            response_lengths=[500, 450],  # High quality responses
        )

        low_quality = calculate_analysis_confidence(
            perspectives_responded=2,
            total_perspectives=2,
            error_count=0,
            abstention_count=0,
            response_lengths=[50, 60],  # Low quality responses
        )

        assert high_quality > low_quality

    def test_analysis_confidence_edge_cases(self):
        """Test edge cases for analysis confidence"""
        # No perspectives
        assert calculate_analysis_confidence(0, 0, 0, 0, []) == 0.0

        # All errors - should be very low but may not be exactly zero due to algorithm
        confidence = calculate_analysis_confidence(2, 2, 2, 0, [])
        assert confidence < 0.3  # Should be very low with all errors
        assert confidence >= 0.0  # But not negative

        # No responses (all abstentions)
        confidence = calculate_analysis_confidence(2, 2, 0, 2, [])
        assert confidence >= 0.0  # Should handle gracefully
        assert confidence < 0.1  # Should be very low with no actual responses

    def test_synthesis_confidence_ideal_scenario(self):
        """Test synthesis confidence with ideal conditions"""
        confidence = calculate_synthesis_confidence(
            perspectives_analyzed=5,  # Good coverage
            patterns_identified=4,  # Rich patterns
            tensions_mapped=3,  # Good tension analysis
            synthesis_length=1200,  # Substantial synthesis
        )
        assert confidence > 0.8
        assert confidence <= 1.0

    def test_synthesis_confidence_minimal_scenario(self):
        """Test synthesis confidence with minimal input"""
        confidence = calculate_synthesis_confidence(
            perspectives_analyzed=1,  # Poor coverage
            patterns_identified=0,  # No patterns
            tensions_mapped=0,  # No tensions
            synthesis_length=100,  # Brief synthesis
        )
        assert confidence < 0.5

    def test_synthesis_confidence_edge_cases(self):
        """Test synthesis confidence edge cases"""
        # No perspectives analyzed
        assert calculate_synthesis_confidence(0, 0, 0, 0) == 0.0

        # Very long synthesis with good analysis
        confidence = calculate_synthesis_confidence(5, 3, 2, 2000)
        assert confidence > 0.8


class TestNextStepsGeneration:
    """Test next steps generation logic"""

    def test_analysis_next_steps_with_errors(self):
        """Test that error recovery is prioritized in next steps"""
        steps = generate_analysis_next_steps(
            session_state="active",
            perspectives_count=4,
            error_count=2,
            has_synthesis=False,
            confidence=0.6,
        )

        # Error recovery should be first step
        assert any(
            "error" in step.lower() and "retry" in step.lower() for step in steps
        )
        assert len(steps) > 0

    def test_analysis_next_steps_low_confidence(self):
        """Test next steps for low confidence scenarios"""
        steps = generate_analysis_next_steps(
            session_state="active",
            perspectives_count=2,
            error_count=0,
            has_synthesis=False,
            confidence=0.2,  # Below threshold
        )

        # Should suggest adding perspectives and refining question
        assert any("more perspectives" in step.lower() for step in steps)
        assert any("refine question" in step.lower() for step in steps)

    def test_analysis_next_steps_high_confidence_no_synthesis(self):
        """Test next steps when ready for synthesis"""
        steps = generate_analysis_next_steps(
            session_state="active",
            perspectives_count=4,
            error_count=0,
            has_synthesis=False,
            confidence=0.9,  # High confidence
        )

        # Should recommend synthesis
        assert any("synthesize_perspectives" in step for step in steps)

    def test_analysis_next_steps_always_provides_continuation(self):
        """Test that next steps always provide a way forward"""
        # Even in edge cases, should provide steps
        steps = generate_analysis_next_steps(
            session_state="active",
            perspectives_count=10,
            error_count=0,
            has_synthesis=True,
            confidence=1.0,
        )

        assert len(steps) > 0  # Should always have at least one step

    def test_synthesis_next_steps_with_tensions(self):
        """Test synthesis next steps when tensions are identified"""
        steps = generate_synthesis_next_steps(
            tensions_identified=3, emergent_insights=2, confidence=0.8
        )

        # Should focus on tension resolution
        assert any("tension" in step.lower() for step in steps)
        assert any("decision framework" in step.lower() for step in steps)

    def test_synthesis_next_steps_high_confidence(self):
        """Test synthesis next steps with high confidence"""
        steps = generate_synthesis_next_steps(
            tensions_identified=1,
            emergent_insights=2,
            confidence=0.9,  # High confidence
        )

        # Should suggest stakeholder presentation and implementation
        assert any(
            "stakeholder" in step.lower() or "decision" in step.lower()
            for step in steps
        )
        assert any("implementation" in step.lower() for step in steps)

    def test_synthesis_next_steps_low_confidence(self):
        """Test synthesis next steps with low confidence"""
        steps = generate_synthesis_next_steps(
            tensions_identified=0,
            emergent_insights=0,
            confidence=0.3,  # Low confidence
        )

        # Should suggest gathering more perspectives
        assert any("additional perspectives" in step.lower() for step in steps)


class TestErrorResponse:
    """Test error response creation"""

    def test_basic_error_response(self):
        """Test basic error response creation"""
        response = create_error_response(
            error_message="Connection timeout",
            error_type="network_error",
            recoverable=True,
        )

        assert response["immediate"]["status"] == "error"
        assert response["immediate"]["confidence"] == 0.0
        assert "Connection timeout" in response["immediate"]["key_insight"]
        assert response["quality"]["urgency"] == "high"  # Non-critical error
        assert len(response["actionable"]["next_steps"]) > 0

    def test_critical_error_response(self):
        """Test critical error response"""
        response = create_error_response(
            error_message="Authentication failed",
            error_type="auth_failure",  # Critical error type
            recoverable=False,
        )

        assert response["quality"]["urgency"] == "critical"
        assert not response["quality"]["indicators"]["recoverable"]
        assert any(
            "administrator" in step.lower()
            for step in response["actionable"]["next_steps"]
        )

    def test_session_not_found_error(self):
        """Test specific error recovery for session not found"""
        response = create_error_response(
            error_message="Session expired",
            error_type="session_not_found",
            recoverable=True,
        )

        steps = response["actionable"]["next_steps"]
        assert any("start_context_analysis" in step for step in steps)
        assert any("list_sessions" in step for step in steps)

    def test_validation_error_response(self):
        """Test validation error response"""
        response = create_error_response(
            error_message="Invalid topic format",
            error_type="validation_error",
            recoverable=True,
        )

        steps = response["actionable"]["next_steps"]
        assert any("review input parameters" in step.lower() for step in steps)
        assert any("documentation" in step.lower() for step in steps)

    def test_error_response_with_session_id(self):
        """Test error response includes session ID when provided"""
        response = create_error_response(
            error_message="Processing failed", session_id="test-session-123"
        )

        assert response["immediate"]["session_id"] == "test-session-123"

    def test_error_response_structure_compliance(self):
        """Test that error responses follow AORP structure"""
        response = create_error_response("Test error")

        # Should have all required AORP sections
        assert "immediate" in response
        assert "actionable" in response
        assert "quality" in response
        assert "details" in response

        # Should have required immediate fields
        assert "status" in response["immediate"]
        assert "key_insight" in response["immediate"]
        assert "confidence" in response["immediate"]

        # Should have metadata with timestamp
        assert "metadata" in response["details"]
        assert "timestamp" in response["details"]["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
