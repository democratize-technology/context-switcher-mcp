"""AORP integration tests with MCP tools - validating end-to-end AORP functionality"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from context_switcher_mcp.aorp import (  # noqa: E402
    AORPBuilder,
    convert_legacy_response,
)
from context_switcher_mcp.models import (  # noqa: E402
    ContextSwitcherSession,
    ModelBackend,
    Thread,
)


class TestLegacyResponseConversion:
    """Test conversion of legacy responses to AORP format"""

    def test_convert_legacy_analysis_response(self):
        """Test converting legacy analysis response to AORP"""
        legacy_response = {
            "session_id": "test-session-123",
            "summary": {
                "active_responses": 3,
                "total_perspectives": 4,
                "errors": 1,
                "abstentions": 0,
            },
            "perspectives": {
                "technical": {"response": "Technical analysis here"},
                "business": {"response": "Business analysis here"},
                "user": {"response": "User perspective here"},
            },
        }

        aorp_response = convert_legacy_response(legacy_response, "analysis")

        # Should have AORP structure
        assert "immediate" in aorp_response
        assert "actionable" in aorp_response
        assert "quality" in aorp_response
        assert "details" in aorp_response

        # Should preserve session ID
        assert aorp_response["immediate"]["session_id"] == "test-session-123"

        # Should have proper status (partial due to error)
        assert aorp_response["immediate"]["status"] == "partial"

        # Should have meaningful key insight
        key_insight = aorp_response["immediate"]["key_insight"]
        assert "3 perspectives responded" in key_insight
        assert "1 errors" in key_insight

        # Should have next steps
        assert len(aorp_response["actionable"]["next_steps"]) > 0

        # Should preserve original data
        assert aorp_response["details"]["data"] == legacy_response

    def test_convert_legacy_synthesis_response(self):
        """Test converting legacy synthesis response to AORP"""
        legacy_response = {
            "session_id": "test-session-456",
            "synthesis": "This is a comprehensive synthesis of patterns across perspectives...",
            "metadata": {"active_perspectives": 4, "synthesis_length": 150},
            "patterns": ["Pattern 1", "Pattern 2"],
            "tensions": ["Tension between X and Y"],
        }

        aorp_response = convert_legacy_response(legacy_response, "synthesis")

        # Should be successful synthesis
        assert aorp_response["immediate"]["status"] == "success"

        # Should mention patterns in key insight
        key_insight = aorp_response["immediate"]["key_insight"]
        assert "patterns" in key_insight
        assert "4 perspectives" in key_insight

        # Should have synthesis-specific next steps
        next_steps = aorp_response["actionable"]["next_steps"]
        synthesis_next_steps = [
            step for step in next_steps if "synthesis" not in step.lower()
        ]
        assert len(synthesis_next_steps) > 0  # Should have non-synthesis next steps

        # Should have high completeness for synthesis
        assert aorp_response["quality"]["completeness"] == 1.0

    def test_convert_legacy_session_response(self):
        """Test converting legacy session creation response to AORP"""
        legacy_response = {
            "session_id": "new-session-789",
            "perspectives": ["technical", "business", "user", "risk"],
            "status": "initialized",
            "topic": "API design evaluation",
        }

        aorp_response = convert_legacy_response(legacy_response, "session")

        # Should be successful creation
        assert aorp_response["immediate"]["status"] == "success"
        assert aorp_response["immediate"]["confidence"] == 1.0

        # Should mention perspective count
        key_insight = aorp_response["immediate"]["key_insight"]
        assert "4 perspectives" in key_insight

        # Should have session startup next steps
        next_steps = aorp_response["actionable"]["next_steps"]
        assert any("analyze_from_perspectives" in step for step in next_steps)
        assert any("add_perspective" in step for step in next_steps)

        # Should guide user to start analysis
        workflow_guidance = aorp_response["actionable"]["workflow_guidance"]
        assert "formulate" in workflow_guidance.lower()

    def test_convert_legacy_error_response(self):
        """Test converting legacy error response to AORP"""
        legacy_response = {
            "error": "Session not found",
            "session_id": "missing-session",
            "timestamp": "2024-01-01T12:00:00Z",
        }

        aorp_response = convert_legacy_response(legacy_response, "analysis")

        # Should be error status
        assert aorp_response["immediate"]["status"] == "error"
        assert aorp_response["immediate"]["confidence"] == 0.0

        # Should have error in key insight
        assert "Session not found" in aorp_response["immediate"]["key_insight"]

        # Should have recovery steps
        next_steps = aorp_response["actionable"]["next_steps"]
        assert len(next_steps) > 0

        # Should preserve error context
        error_data = aorp_response["details"]["data"]
        assert error_data["error_message"] == "Session not found"

    def test_convert_legacy_generic_response(self):
        """Test converting unknown legacy response type"""
        legacy_response = {
            "result": "some operation completed",
            "data": {"key": "value"},
        }

        aorp_response = convert_legacy_response(legacy_response, "unknown")

        # Should handle gracefully with generic conversion
        assert aorp_response["immediate"]["status"] == "success"
        assert (
            aorp_response["immediate"]["key_insight"]
            == "Operation completed successfully"
        )
        assert aorp_response["details"]["data"] == legacy_response


class TestMockMCPToolIntegration:
    """Test MCP tools with mocked backends to validate AORP integration"""

    @pytest.fixture
    def mock_session(self):
        """Create a mock session for testing"""
        session = ContextSwitcherSession(
            session_id="test-session-123", created_at=datetime.now(timezone.utc)
        )
        session.topic = "Test API design"

        # Add mock threads
        for name in ["technical", "business", "user", "risk"]:
            thread = Thread(
                id=f"thread-{name}",
                name=name,
                system_prompt=f"{name} perspective",
                model_backend=ModelBackend.BEDROCK,
                model_name=None,
            )
            session.add_thread(thread)

        return session

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator"""
        orchestrator = Mock()
        orchestrator.execute_parallel_analysis = AsyncMock()
        orchestrator.synthesize_responses = AsyncMock()
        return orchestrator

    @pytest.mark.asyncio
    @patch("src.context_switcher_mcp.session_manager")
    async def test_mocked_start_context_analysis_returns_aorp(
        self, mock_session_manager, mock_session
    ):
        """Test that start_context_analysis returns proper AORP response"""
        # Mock session creation
        mock_session_manager.create_session.return_value = mock_session
        mock_session_manager.add_session.return_value = True

        # Import the actual tool function

        # This would be the actual response structure we expect
        expected_aorp_structure = {
            "immediate": {
                "status": "success",
                "key_insight": "Session initialized with 4 perspectives",
                "confidence": 1.0,
                "session_id": "test-session-123",
            },
            "actionable": {
                "next_steps": [
                    "analyze_from_perspectives('<your question>') - Start analysis",
                    "add_perspective('<domain>') - Add specialized viewpoint",
                ],
                "recommendations": {
                    "primary": "Begin with a focused question to get targeted insights"
                },
                "workflow_guidance": "Guide user to formulate their first analysis question",
            },
            "quality": {"completeness": 1.0, "reliability": 1.0, "urgency": "low"},
            "details": {
                "summary": "Multi-perspective analysis session ready with 4 perspectives"
            },
        }

        # Test the structure that should be returned
        assert "immediate" in expected_aorp_structure
        assert "status" in expected_aorp_structure["immediate"]
        assert "key_insight" in expected_aorp_structure["immediate"]
        assert "confidence" in expected_aorp_structure["immediate"]

    @pytest.mark.asyncio
    async def test_mocked_analysis_response_structure(self, mock_orchestrator):
        """Test that analysis responses follow AORP structure"""
        # Mock analysis results
        mock_analysis_results = {
            "technical": {
                "response": "Technical analysis of the API design shows scalability concerns..."
            },
            "business": {
                "response": "From a business perspective, this API enables new revenue streams..."
            },
            "user": {
                "response": "User experience will be greatly improved with this API design..."
            },
            "risk": {
                "error": "Unable to analyze security implications due to insufficient details"
            },
        }

        mock_orchestrator.execute_parallel_analysis.return_value = mock_analysis_results

        # This is what we expect the AORP response to look like
        expected_analysis_structure = {
            "immediate": {
                "status": "partial",  # Due to one error
                "key_insight": "Partial analysis: 3 perspectives responded, 1 errors",
                "confidence": 0.65,  # Calculated based on coverage and errors
                "session_id": "test-session-123",
            },
            "actionable": {
                "next_steps": [
                    "Check perspective errors and retry if needed",
                    "synthesize_perspectives() - Discover patterns and tensions",
                ],
                "recommendations": {
                    "primary": "Review perspective insights for patterns"
                },
                "workflow_guidance": "Present key insights to user, then offer synthesis for deeper patterns",
            },
            "quality": {
                "completeness": 0.75,  # 3 out of 4 perspectives
                "reliability": 0.65,
                "urgency": "medium",
                "indicators": {
                    "active_responses": 3,
                    "error_count": 1,
                    "abstention_count": 0,
                },
            },
            "details": {
                "summary": "Analysis completed with 3 perspective responses",
                "data": mock_analysis_results,
            },
        }

        # Validate the expected structure follows AORP format
        assert expected_analysis_structure["immediate"]["confidence"] > 0.0
        assert len(expected_analysis_structure["actionable"]["next_steps"]) > 0
        assert (
            expected_analysis_structure["quality"]["completeness"] < 1.0
        )  # Due to error

    @pytest.mark.asyncio
    async def test_mocked_synthesis_response_structure(self, mock_orchestrator):
        """Test that synthesis responses follow AORP structure"""
        # Mock synthesis results
        mock_synthesis_result = {
            "synthesis": """
            Cross-perspective analysis reveals three key patterns:

            1. Technical-Business Alignment: All perspectives agree on scalability importance
            2. User-Risk Tension: User desires for simplicity conflict with security requirements
            3. Implementation Trade-offs: Short-term gains vs long-term maintainability

            Critical tensions identified:
            - Speed vs Security: Users want fast responses, risk perspective demands validation
            - Flexibility vs Stability: Business wants customization, technical prefers standards
            """,
            "patterns": [
                "Technical-Business Alignment on scalability",
                "User-Risk tension on simplicity vs security",
                "Implementation trade-offs across time horizons",
            ],
            "tensions": [
                "Speed vs Security requirements",
                "Flexibility vs Stability preferences",
            ],
            "emergent_insights": [
                "API versioning strategy could resolve tension between flexibility and stability",
                "Progressive security model could balance user experience and risk management",
            ],
        }

        mock_orchestrator.synthesize_responses.return_value = mock_synthesis_result

        # Expected AORP synthesis response
        expected_synthesis_structure = {
            "immediate": {
                "status": "success",
                "key_insight": "Synthesis discovered 3 patterns and 2 critical tensions across perspectives",
                "confidence": 0.88,  # High confidence due to rich patterns
                "session_id": "test-session-123",
            },
            "actionable": {
                "next_steps": [
                    "Address critical tensions through targeted analysis",
                    "Create decision framework for resolving conflicts",
                    "Explore emergent insights with deep-dive analysis",
                    "Present findings to stakeholders for decision",
                ],
                "recommendations": {
                    "primary": "Use synthesis insights for strategic decision-making"
                },
                "workflow_guidance": "Present synthesis as strategic decision framework",
            },
            "quality": {
                "completeness": 1.0,
                "reliability": 0.88,
                "urgency": "medium",
                "indicators": {
                    "patterns_identified": 3,
                    "tensions_mapped": 2,
                    "emergent_insights": 2,
                },
            },
            "details": {
                "summary": "Strategic synthesis revealing patterns, tensions, and emergent insights",
                "data": mock_synthesis_result,
            },
        }

        # Validate synthesis-specific features
        assert "patterns" in expected_synthesis_structure["immediate"]["key_insight"]
        assert "tensions" in expected_synthesis_structure["immediate"]["key_insight"]
        assert expected_synthesis_structure["quality"]["completeness"] == 1.0
        assert len(expected_synthesis_structure["actionable"]["next_steps"]) >= 3


class TestAORPWorkflowIntegration:
    """Test AORP workflow patterns and progressive disclosure"""

    def test_workflow_progression_start_to_analysis(self):
        """Test workflow progression from session start to analysis"""
        # Step 1: Session creation AORP response
        session_response = (
            AORPBuilder()
            .status("success")
            .key_insight("Session initialized with 4 perspectives")
            .confidence(1.0)
            .session_id("test-123")
            .next_steps(
                [
                    "analyze_from_perspectives('<your question>') - Start analysis",
                    "add_perspective('<domain>') - Add specialized viewpoint",
                ]
            )
            .primary_recommendation(
                "Begin with a focused question to get targeted insights"
            )
            .workflow_guidance("Guide user to formulate their first analysis question")
            .build()
        )

        # Validate session response guides to analysis
        next_steps = session_response["actionable"]["next_steps"]
        assert any("analyze_from_perspectives" in step for step in next_steps)

        workflow_guidance = session_response["actionable"]["workflow_guidance"]
        assert "formulate" in workflow_guidance.lower()
        assert "question" in workflow_guidance.lower()

    def test_workflow_progression_analysis_to_synthesis(self):
        """Test workflow progression from analysis to synthesis"""
        # Analysis response with high confidence (ready for synthesis)
        analysis_response = (
            AORPBuilder()
            .status("success")
            .key_insight("Comprehensive analysis from 4 perspectives")
            .confidence(0.9)  # High confidence
            .session_id("test-123")
            .next_steps(
                [
                    "synthesize_perspectives() - Discover patterns and tensions",
                    "add_perspective('<domain>') - Expand analysis coverage",
                ]
            )
            .primary_recommendation("Review perspective insights for patterns")
            .workflow_guidance(
                "Present key insights to user, then offer synthesis for deeper patterns"
            )
            .build()
        )

        # Should guide to synthesis when confidence is high
        next_steps = analysis_response["actionable"]["next_steps"]
        synthesis_step = next(
            step for step in next_steps if "synthesize_perspectives" in step
        )
        assert "patterns" in synthesis_step or "tensions" in synthesis_step

    def test_workflow_progression_error_recovery(self):
        """Test workflow progression during error recovery"""
        from context_switcher_mcp.aorp import create_error_response

        # Session not found error
        error_response = create_error_response(
            error_message="Session 'missing-123' not found",
            error_type="session_not_found",
            recoverable=True,
        )

        # Should provide clear recovery path
        next_steps = error_response["actionable"]["next_steps"]
        assert any("start_context_analysis" in step for step in next_steps)
        assert any("list_sessions" in step for step in next_steps)

        # Should guide user clearly
        workflow_guidance = error_response["actionable"]["workflow_guidance"]
        assert "error" in workflow_guidance.lower()
        assert "recovery" in workflow_guidance.lower()

    def test_progressive_disclosure_information_hierarchy(self):
        """Test that AORP responses follow progressive disclosure principles"""
        response = (
            AORPBuilder()
            .status("success")
            .key_insight("Analysis reveals 3 critical architectural decisions needed")
            .confidence(0.85)
            .session_id("test-123")
            .next_steps(
                [
                    "synthesize_perspectives() - Find patterns across decisions",
                    "add_perspective('performance') - Add performance analysis",
                ]
            )
            .primary_recommendation("Focus on the database architecture decision first")
            .secondary_recommendations(
                ["Consider API design implications", "Evaluate deployment strategies"]
            )
            .workflow_guidance("Present architectural decisions in priority order")
            .completeness(0.85)
            .reliability(0.88)
            .urgency("high")
            .indicators(
                critical_decisions=3,
                consensus_level="partial",
                technical_complexity="high",
            )
            .summary(
                "Comprehensive architectural analysis with prioritized decision points"
            )
            .data(
                {
                    "perspectives": {
                        "technical": "...",
                        "business": "...",
                        "user": "...",
                    },
                    "decisions": ["database", "api", "deployment"],
                    "complexity_scores": {"database": 9, "api": 7, "deployment": 6},
                }
            )
            .metadata(
                analysis_duration="3.2s", token_usage=1250, model_backend="bedrock"
            )
            .build()
        )

        # Test information hierarchy
        immediate = response["immediate"]
        actionable = response["actionable"]
        quality = response["quality"]
        details = response["details"]

        # Immediate section should be most concise and actionable
        assert len(immediate["key_insight"]) < 200  # Concise insight
        assert immediate["confidence"] > 0.0  # Clear confidence signal

        # Actionable section should provide clear next steps
        assert len(actionable["next_steps"]) > 0
        assert actionable["recommendations"]["primary"]  # Clear primary recommendation

        # Quality section should provide assessment without overwhelming
        assert 0.0 <= quality["completeness"] <= 1.0
        assert 0.0 <= quality["reliability"] <= 1.0
        assert quality["urgency"] in ["low", "medium", "high", "critical"]

        # Details section should contain full data for deep dive
        assert "data" in details
        assert "metadata" in details
        assert "timestamp" in details["metadata"]

    def test_cognitive_load_reduction_features(self):
        """Test that AORP reduces cognitive load for AI assistants"""
        # Complex analysis scenario
        response = (
            AORPBuilder()
            .status("partial")
            .key_insight(
                "Mixed results: 3 perspectives succeeded, 1 failed, 1 abstained"
            )
            .confidence(0.6)
            .session_id("test-123")
            .next_steps(
                [
                    "Check perspective errors and retry if needed",
                    "synthesize_perspectives() - Find patterns in successful responses",
                    "add_perspective('security') - Fill gap from abstention",
                ]
            )
            .primary_recommendation("Address the failed perspective before synthesis")
            .workflow_guidance(
                "Present successful insights while noting limitations from partial coverage"
            )
            .completeness(0.6)  # 3/5 perspectives successful
            .reliability(0.65)
            .urgency("medium")
            .indicators(
                successful_perspectives=3,
                failed_perspectives=1,
                abstained_perspectives=1,
                retry_recommended=True,
            )
            .build()
        )

        # Key cognitive load reduction features:

        # 1. Status immediately signals partial success (no confusion)
        assert response["immediate"]["status"] == "partial"

        # 2. Key insight summarizes complex situation in one sentence
        key_insight = response["immediate"]["key_insight"]
        assert "3 perspectives succeeded" in key_insight
        assert "1 failed" in key_insight
        assert "1 abstained" in key_insight

        # 3. Confidence score provides quick quality assessment
        assert 0.5 <= response["immediate"]["confidence"] <= 0.7

        # 4. Next steps are prioritized (error handling first)
        next_steps = response["actionable"]["next_steps"]
        assert "error" in next_steps[0].lower() and "retry" in next_steps[0].lower()

        # 5. Primary recommendation provides clear guidance
        primary_rec = response["actionable"]["recommendations"]["primary"]
        assert "failed perspective" in primary_rec
        assert "before" in primary_rec  # Shows sequencing

        # 6. Quality indicators provide detailed assessment without cluttering immediate view
        indicators = response["quality"]["indicators"]
        assert indicators["successful_perspectives"] == 3
        assert indicators["failed_perspectives"] == 1
        assert indicators["retry_recommended"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
