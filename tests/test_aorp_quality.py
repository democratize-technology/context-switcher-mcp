"""AORP quality validation tests - ensuring UX improvements deliver cognitive load reduction"""

import pytest
from context_switcher_mcp.aorp import (  # noqa: E402
    AORPBuilder,
    calculate_analysis_confidence,
    calculate_synthesis_confidence,
    create_error_response,
    generate_analysis_next_steps,
    generate_synthesis_next_steps,
)


class TestResponseQualityValidation:
    """Test that AORP responses meet quality standards for AI workflow enhancement"""

    def test_key_insight_quality_standards(self):
        """Test that key insights meet cognitive load reduction standards"""
        # Good key insights should be:
        # - Concise (under 150 characters)
        # - Actionable or informative
        # - Status-aware

        test_cases = [
            {
                "status": "success",
                "scenario": "perfect analysis",
                "expected_patterns": ["perspectives", "analysis", "success"],
            },
            {
                "status": "partial",
                "scenario": "mixed results",
                "expected_patterns": ["partial", "error", "perspective"],
            },
            {
                "status": "error",
                "scenario": "failure case",
                "expected_patterns": ["failed", "error"],
            },
        ]

        for case in test_cases:
            # Generate key insight based on scenario
            if case["status"] == "success":
                key_insight = "Analysis completed successfully across 4 perspectives"
            elif case["status"] == "partial":
                key_insight = "Partial analysis: 3 perspectives responded, 1 error"
            else:
                key_insight = "Analysis failed: session not found"

            # Test quality standards
            assert (
                len(key_insight) <= 150
            ), f"Key insight too long: {len(key_insight)} chars"
            assert any(
                pattern in key_insight.lower() for pattern in case["expected_patterns"]
            ), f"Key insight missing expected patterns: {case['expected_patterns']}"
            assert key_insight[
                0
            ].isupper(), "Key insight should start with capital letter"
            assert not key_insight.endswith(
                "."
            ), "Key insight should not end with period (action-oriented)"

    def test_next_steps_actionability(self):
        """Test that next steps are genuinely actionable"""
        # Test various scenarios
        scenarios = [
            {
                "name": "high_confidence_ready_for_synthesis",
                "confidence": 0.9,
                "error_count": 0,
                "has_synthesis": False,
                "expected_actions": ["synthesize_perspectives"],
            },
            {
                "name": "low_confidence_needs_more_data",
                "confidence": 0.2,
                "error_count": 0,
                "has_synthesis": False,
                "expected_actions": ["add_perspective", "refine"],
            },
            {
                "name": "errors_need_recovery",
                "confidence": 0.5,
                "error_count": 2,
                "has_synthesis": False,
                "expected_actions": ["retry", "error"],
            },
        ]

        for scenario in scenarios:
            steps = generate_analysis_next_steps(
                session_state="active",
                perspectives_count=4,
                error_count=scenario["error_count"],
                has_synthesis=scenario["has_synthesis"],
                confidence=scenario["confidence"],
            )

            # All steps should be actionable and not vague
            for step in steps:
                # Should contain action words or function calls
                action_indicators = [
                    "()",
                    "add",
                    "check",
                    "refine",
                    "retry",
                    "analyze",
                    "start",
                    "create",
                    "review",
                    "more",
                ]
                assert any(
                    indicator in step.lower() for indicator in action_indicators
                ), f"Step not actionable: {step}"

                # Should not be vague
                vague_words = ["consider", "maybe", "possibly", "think about"]
                assert not any(
                    vague in step.lower() for vague in vague_words
                ), f"Step too vague: {step}"

            # Should contain expected actions for scenario
            for expected_action in scenario["expected_actions"]:
                assert any(
                    expected_action in step.lower() for step in steps
                ), f"Missing expected action '{expected_action}' in scenario '{scenario['name']}'"

    def test_confidence_score_accuracy(self):
        """Test that confidence scores accurately reflect response quality"""
        # Test confidence calculation accuracy with known scenarios
        test_scenarios = [
            {
                "name": "perfect_response",
                "perspectives_responded": 4,
                "total_perspectives": 4,
                "error_count": 0,
                "abstention_count": 0,
                "response_lengths": [300, 280, 320, 290],
                "expected_confidence_range": (0.7, 1.0),
            },
            {
                "name": "partial_with_errors",
                "perspectives_responded": 4,
                "total_perspectives": 4,
                "error_count": 2,
                "abstention_count": 0,
                "response_lengths": [300, 280],
                "expected_confidence_range": (0.4, 0.7),
            },
            {
                "name": "low_quality_responses",
                "perspectives_responded": 2,
                "total_perspectives": 4,
                "error_count": 0,
                "abstention_count": 2,
                "response_lengths": [50, 60],  # Very short responses
                "expected_confidence_range": (0.0, 0.4),
            },
            {
                "name": "all_errors",
                "perspectives_responded": 4,
                "total_perspectives": 4,
                "error_count": 4,
                "abstention_count": 0,
                "response_lengths": [],
                "expected_confidence_range": (0.0, 0.2),
            },
        ]

        for scenario in test_scenarios:
            confidence = calculate_analysis_confidence(
                scenario["perspectives_responded"],
                scenario["total_perspectives"],
                scenario["error_count"],
                scenario["abstention_count"],
                scenario["response_lengths"],
            )

            min_expected, max_expected = scenario["expected_confidence_range"]
            assert (
                min_expected <= confidence <= max_expected
            ), f"Confidence {confidence} not in expected range {scenario['expected_confidence_range']} for scenario '{scenario['name']}'"

    def test_workflow_guidance_effectiveness(self):
        """Test that workflow guidance effectively helps AI assistants"""
        # Test different response types and their guidance
        guidance_tests = [
            {
                "response_type": "session_creation",
                "guidance": "Guide user to formulate their first analysis question",
                "should_contain": ["guide", "user", "question"],
                "should_not_contain": ["error", "retry", "failed"],
            },
            {
                "response_type": "successful_analysis",
                "guidance": "Present key insights to user, then offer synthesis for deeper patterns",
                "should_contain": ["present", "insights", "synthesis"],
                "should_not_contain": ["error", "failed", "retry"],
            },
            {
                "response_type": "error_recovery",
                "guidance": "Present error clearly and guide user to recovery",
                "should_contain": ["error", "recovery", "guide"],
                "should_not_contain": ["synthesis", "insights", "patterns"],
            },
            {
                "response_type": "synthesis_complete",
                "guidance": "Present synthesis as strategic decision framework",
                "should_contain": ["synthesis", "strategic", "decision"],
                "should_not_contain": ["error", "retry", "failed"],
            },
        ]

        for test in guidance_tests:
            guidance = test["guidance"]

            # Should contain appropriate terms
            for term in test["should_contain"]:
                assert (
                    term.lower() in guidance.lower()
                ), f"Workflow guidance missing '{term}' for {test['response_type']}"

            # Should not contain inappropriate terms
            for term in test["should_not_contain"]:
                assert (
                    term.lower() not in guidance.lower()
                ), f"Workflow guidance inappropriately contains '{term}' for {test['response_type']}"

            # Should be actionable (contain action verbs)
            action_verbs = ["present", "guide", "offer", "show", "help", "enable"]
            assert any(
                verb in guidance.lower() for verb in action_verbs
            ), f"Workflow guidance not actionable for {test['response_type']}"

    def test_progressive_disclosure_hierarchy(self):
        """Test that information hierarchy supports progressive disclosure"""
        # Create a complex response to test hierarchy
        complex_response = (
            AORPBuilder()
            .status("partial")
            .key_insight("Mixed analysis: 3 succeeded, 1 failed, 1 abstained")
            .confidence(0.65)
            .session_id("test-session")
            .next_steps(
                [
                    "Check perspective errors and retry if needed",
                    "synthesize_perspectives() - Find patterns in successful responses",
                    "add_perspective('security') - Fill abstention gap",
                ]
            )
            .primary_recommendation("Address failed perspective before synthesis")
            .secondary_recommendations(
                ["Review abstention reasoning", "Consider alternative question framing"]
            )
            .workflow_guidance(
                "Present successful insights while noting coverage limitations"
            )
            .completeness(0.6)
            .reliability(0.65)
            .urgency("medium")
            .indicators(
                successful_perspectives=3,
                failed_perspectives=1,
                abstained_perspectives=1,
                error_types=["timeout"],
                abstention_reasons=["insufficient_context"],
            )
            .summary(
                "Multi-perspective analysis with mixed results requiring error resolution"
            )
            .data(
                {
                    "perspectives": {
                        "technical": {"response": "Technical analysis..."},
                        "business": {"response": "Business perspective..."},
                        "user": {"response": "User experience insights..."},
                        "risk": {"error": "Connection timeout"},
                        "security": {"abstention": "Insufficient security context"},
                    },
                    "timing": {"total_duration": "4.2s", "timeout_threshold": "3.0s"},
                }
            )
            .metadata(
                analysis_id="analysis-123", retry_count=0, model_backend="bedrock"
            )
            .build()
        )

        # Test progressive disclosure principles

        # Level 1: Immediate - Should provide instant understanding
        immediate = complex_response["immediate"]
        assert (
            len(immediate) <= 4
        ), "Immediate section should have ≤4 fields for quick scanning"
        assert "status" in immediate, "Status must be immediately visible"
        assert len(immediate["key_insight"]) <= 100, "Key insight should be scannable"

        # Level 2: Actionable - Should provide clear next actions
        actionable = complex_response["actionable"]
        assert len(actionable["next_steps"]) >= 1, "Must provide at least one next step"
        assert (
            len(actionable["next_steps"]) <= 5
        ), "Should not overwhelm with too many steps"
        assert actionable["recommendations"][
            "primary"
        ], "Must have primary recommendation"

        # Level 3: Quality - Should provide assessment without detail overload
        quality = complex_response["quality"]
        assert len(quality) <= 6, "Quality section should be concise assessment"
        assert "completeness" in quality, "Must include completeness assessment"
        assert "urgency" in quality, "Must include urgency assessment"

        # Level 4: Details - Can contain full complexity
        details = complex_response["details"]
        assert "data" in details, "Details must preserve full data"
        assert "metadata" in details, "Details must include metadata"

        # Test that critical information bubbles up appropriately
        # Error information should be visible in immediate section
        assert (
            "failed" in immediate["key_insight"] or "error" in immediate["key_insight"]
        )

        # Error recovery should be prioritized in next steps
        first_step = actionable["next_steps"][0]
        assert "error" in first_step.lower() or "retry" in first_step.lower()

    def test_error_response_recovery_guidance(self):
        """Test that error responses provide effective recovery guidance"""
        error_scenarios = [
            {
                "error_type": "session_not_found",
                "error_message": "Session 'missing-123' not found",
                "expected_recovery_actions": [
                    "start_context_analysis",
                    "list_sessions",
                ],
                "expected_guidance_keywords": ["error", "recovery"],
            },
            {
                "error_type": "validation_error",
                "error_message": "Topic exceeds maximum length",
                "expected_recovery_actions": ["review input", "documentation"],
                "expected_guidance_keywords": ["error", "recovery"],
            },
            {
                "error_type": "auth_failure",
                "error_message": "Authentication failed",
                "expected_recovery_actions": ["administrator", "contact"],
                "expected_guidance_keywords": ["error", "recovery"],
            },
            {
                "error_type": "quota_exceeded",
                "error_message": "API quota exceeded",
                "expected_recovery_actions": ["retry", "contact"],
                "expected_guidance_keywords": ["error", "recovery"],
            },
        ]

        for scenario in error_scenarios:
            error_response = create_error_response(
                error_message=scenario["error_message"],
                error_type=scenario["error_type"],
                recoverable=scenario["error_type"] != "auth_failure",
            )

            # Test that error response follows AORP structure
            assert error_response["immediate"]["status"] == "error"
            assert error_response["immediate"]["confidence"] == 0.0

            # Test recovery actions are present (at least some of them)
            next_steps = error_response["actionable"]["next_steps"]
            found_actions = 0
            for expected_action in scenario["expected_recovery_actions"]:
                if any(expected_action in step.lower() for step in next_steps):
                    found_actions += 1
            assert (
                found_actions > 0
            ), f"No recovery actions found for {scenario['error_type']}, steps: {next_steps}"

            # Test workflow guidance is appropriate
            workflow_guidance = error_response["actionable"]["workflow_guidance"]
            for keyword in scenario["expected_guidance_keywords"]:
                assert (
                    keyword.lower() in workflow_guidance.lower()
                ), f"Missing guidance keyword '{keyword}' for {scenario['error_type']}"

            # Test urgency is appropriate
            urgency = error_response["quality"]["urgency"]
            if scenario["error_type"] in ["auth_failure", "quota_exceeded"]:
                assert (
                    urgency == "critical"
                ), "Critical errors should have critical urgency"
            else:
                assert urgency in [
                    "high",
                    "medium",
                ], "Recoverable errors should have high/medium urgency"

    def test_synthesis_quality_indicators(self):
        """Test that synthesis responses provide meaningful quality indicators"""
        synthesis_scenarios = [
            {
                "name": "rich_synthesis",
                "perspectives_analyzed": 5,
                "patterns_identified": 4,
                "tensions_mapped": 3,
                "synthesis_length": 1500,
                "expected_confidence_range": (0.7, 1.0),
                "expected_next_steps": ["stakeholder", "implementation", "decision"],
            },
            {
                "name": "basic_synthesis",
                "perspectives_analyzed": 3,
                "patterns_identified": 2,
                "tensions_mapped": 1,
                "synthesis_length": 800,
                "expected_confidence_range": (0.5, 0.8),
                "expected_next_steps": ["explore", "analyze", "insight"],
            },
            {
                "name": "minimal_synthesis",
                "perspectives_analyzed": 2,
                "patterns_identified": 1,
                "tensions_mapped": 0,
                "synthesis_length": 300,
                "expected_confidence_range": (0.2, 0.5),
                "expected_next_steps": ["additional", "perspectives", "gather"],
            },
        ]

        for scenario in synthesis_scenarios:
            confidence = calculate_synthesis_confidence(
                scenario["perspectives_analyzed"],
                scenario["patterns_identified"],
                scenario["tensions_mapped"],
                scenario["synthesis_length"],
            )

            # Test confidence range
            min_conf, max_conf = scenario["expected_confidence_range"]
            assert (
                min_conf <= confidence <= max_conf
            ), f"Synthesis confidence {confidence} not in expected range for {scenario['name']}"

            # Test next steps appropriateness
            next_steps = generate_synthesis_next_steps(
                scenario["tensions_mapped"], scenario["patterns_identified"], confidence
            )

            for expected_keyword in scenario["expected_next_steps"]:
                assert any(
                    expected_keyword in step.lower() for step in next_steps
                ), f"Missing expected next step keyword '{expected_keyword}' for {scenario['name']}"

    def test_response_completeness_validation(self):
        """Test that AORP responses are complete and well-formed"""
        # Test minimal valid response
        minimal_response = (
            AORPBuilder()
            .status("success")
            .key_insight("Test completed")
            .confidence(0.8)
            .build()
        )

        # Should have all required sections
        assert "immediate" in minimal_response
        assert "actionable" in minimal_response
        assert "quality" in minimal_response
        assert "details" in minimal_response

        # Should have required immediate fields
        immediate = minimal_response["immediate"]
        assert "status" in immediate
        assert "key_insight" in immediate
        assert "confidence" in immediate

        # Should have default actionable structure
        actionable = minimal_response["actionable"]
        assert "next_steps" in actionable
        assert "recommendations" in actionable
        assert "workflow_guidance" in actionable

        # Should have default quality assessments
        quality = minimal_response["quality"]
        assert "completeness" in quality
        assert "reliability" in quality
        assert "urgency" in quality

        # Test complex response completeness
        complex_response = (
            AORPBuilder()
            .status("partial")
            .key_insight("Complex analysis with mixed results")
            .confidence(0.7)
            .session_id("test-123")
            .next_steps(["step1", "step2"])
            .primary_recommendation("Primary rec")
            .secondary_recommendations(["sec1", "sec2"])
            .workflow_guidance("Workflow guidance")
            .completeness(0.8)
            .reliability(0.75)
            .urgency("high")
            .indicators(key1="value1", key2="value2")
            .summary("Test summary")
            .data({"test": "data"})
            .metadata(meta1="value1")
            .debug({"debug": "info"})
            .build()
        )

        # Should preserve all provided information
        assert complex_response["immediate"]["session_id"] == "test-123"
        assert len(complex_response["actionable"]["next_steps"]) == 2
        assert len(complex_response["actionable"]["recommendations"]["secondary"]) == 2
        assert complex_response["quality"]["indicators"]["key1"] == "value1"
        assert complex_response["details"]["data"]["test"] == "data"
        assert complex_response["details"]["debug"]["debug"] == "info"
        assert "timestamp" in complex_response["details"]["metadata"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
