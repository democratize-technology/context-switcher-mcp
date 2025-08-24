"""Real-world AORP scenario tests - validating practical workflow benefits"""

import pytest
from context_switcher_mcp.aorp import (  # noqa: E402
    AORPBuilder,
    create_error_response,
    generate_analysis_next_steps,
    generate_synthesis_next_steps,
)


class TestRealWorldScenarios:
    """Test AORP with realistic Context Switcher usage scenarios"""

    def test_api_design_evaluation_workflow(self):
        """Test complete workflow for API design evaluation scenario"""

        # Scenario: Developer needs to evaluate REST API design
        topic = "REST API design for user management system"

        # Step 1: Session creation response
        session_response = (
            AORPBuilder()
            .status("success")
            .key_insight(
                "API design evaluation session ready with technical, business, user, and security perspectives"
            )
            .confidence(1.0)
            .session_id("api-design-session-001")
            .next_steps(
                [
                    "analyze_from_perspectives('What are the key design decisions for user management API?')",
                    "add_perspective('performance') - Add performance analysis",
                ]
            )
            .primary_recommendation(
                "Start with broad design questions before diving into specifics"
            )
            .workflow_guidance(
                "Present session creation success and guide to initial question"
            )
            .completeness(1.0)
            .reliability(1.0)
            .urgency("low")
            .indicators(perspectives_configured=4, template_used="api_design")
            .summary("Multi-perspective API design evaluation session initialized")
            .data(
                {
                    "topic": topic,
                    "perspectives": ["technical", "business", "user", "security"],
                    "session_type": "api_design_evaluation",
                }
            )
            .metadata(session_type="api_design", template="api_design")
            .build()
        )

        # Validate session response guides to appropriate first question
        assert (
            "API design evaluation session ready"
            in session_response["immediate"]["key_insight"]
        )
        assert any(
            "analyze_from_perspectives" in step
            for step in session_response["actionable"]["next_steps"]
        )
        assert (
            "design questions"
            in session_response["actionable"]["recommendations"]["primary"]
        )

        # Step 2: Analysis response with mixed results
        analysis_response = (
            AORPBuilder()
            .status("partial")
            .key_insight(
                "API design analysis: 3 perspectives provided insights, 1 abstained"
            )
            .confidence(0.72)
            .session_id("api-design-session-001")
            .next_steps(
                [
                    "add_perspective('performance') - Address performance considerations",
                    "synthesize_perspectives() - Find design pattern consensus",
                    "analyze_from_perspectives('How should we handle authentication?') - Dive deeper",
                ]
            )
            .primary_recommendation(
                "Address performance perspective gap before finalizing design"
            )
            .secondary_recommendations(
                [
                    "Consider API versioning strategy",
                    "Evaluate rate limiting requirements",
                ]
            )
            .workflow_guidance(
                "Present insights from 3 perspectives, note security abstained due to insufficient auth details"
            )
            .completeness(0.75)  # 3/4 perspectives
            .reliability(0.72)
            .urgency("medium")
            .indicators(
                successful_perspectives=3,
                abstained_perspectives=1,
                abstention_reason="insufficient_auth_context",
                technical_consensus="REST+GraphQL hybrid",
                business_priority="fast_iteration",
                user_preference="simple_integration",
            )
            .summary(
                "API design insights from technical, business, and user perspectives"
            )
            .data(
                {
                    "perspectives": {
                        "technical": {
                            "response": "Recommend REST for CRUD operations, GraphQL for complex queries...",
                            "confidence": 0.85,
                        },
                        "business": {
                            "response": "Prioritize rapid development and iteration capabilities...",
                            "confidence": 0.80,
                        },
                        "user": {
                            "response": "Developer experience should emphasize simple integration...",
                            "confidence": 0.75,
                        },
                        "security": {
                            "abstention": "Need authentication and authorization requirements before security analysis",
                            "reason": "insufficient_context",
                        },
                    }
                }
            )
            .metadata(analysis_type="api_design_evaluation")
            .build()
        )

        # Validate analysis response handles abstention appropriately
        assert (
            "3 perspectives provided insights, 1 abstained"
            in analysis_response["immediate"]["key_insight"]
        )
        assert (
            0.7 <= analysis_response["immediate"]["confidence"] <= 0.8
        )  # Good but not perfect
        assert (
            "performance" in analysis_response["actionable"]["next_steps"][0]
        )  # Gap filling prioritized
        assert (
            "insufficient auth details"
            in analysis_response["actionable"]["workflow_guidance"]
        )

        # Step 3: Synthesis response after gathering performance perspective
        synthesis_response = (
            AORPBuilder()
            .status("success")
            .key_insight(
                "API design synthesis reveals 3 key patterns and 2 critical tensions requiring resolution"
            )
            .confidence(0.88)
            .session_id("api-design-session-001")
            .next_steps(
                [
                    "Create decision framework for REST vs GraphQL endpoints",
                    "Define authentication strategy to enable security analysis",
                    "Prototype high-priority endpoints for validation",
                    "Present unified design proposal to stakeholders",
                ]
            )
            .primary_recommendation(
                "Resolve REST/GraphQL tension with hybrid approach based on use case patterns"
            )
            .secondary_recommendations(
                [
                    "Implement authentication framework first",
                    "Design for progressive API evolution",
                ]
            )
            .workflow_guidance(
                "Present synthesis as structured decision framework with clear trade-offs"
            )
            .completeness(1.0)
            .reliability(0.88)
            .urgency("high")
            .indicators(
                patterns_identified=3,
                tensions_mapped=2,
                emergent_insights=2,
                consensus_areas=["developer_experience", "iteration_speed"],
                tension_areas=["architecture_choice", "performance_vs_simplicity"],
            )
            .summary(
                "Strategic API design synthesis with actionable patterns and tension resolution"
            )
            .data(
                {
                    "patterns": [
                        "All perspectives value developer experience",
                        "Business and technical align on iteration speed priority",
                        "Performance and user perspectives converge on simplicity",
                    ],
                    "tensions": [
                        "REST simplicity vs GraphQL flexibility",
                        "Performance optimization vs development speed",
                    ],
                    "emergent_insights": [
                        "Hybrid API approach allows phased evolution",
                        "Authentication-first design enables security validation",
                    ],
                    "recommendations": {
                        "architecture": "REST for CRUD, GraphQL for complex queries",
                        "implementation": "Start with REST, add GraphQL selectively",
                        "security": "OAuth2/JWT with progressive disclosure",
                    },
                }
            )
            .metadata(synthesis_type="api_design_strategic")
            .build()
        )

        # Validate synthesis provides strategic decision framework
        assert (
            "3 key patterns and 2 critical tensions"
            in synthesis_response["immediate"]["key_insight"]
        )
        assert (
            synthesis_response["immediate"]["confidence"] > 0.85
        )  # High confidence synthesis
        assert "decision framework" in synthesis_response["actionable"]["next_steps"][0]
        assert (
            "structured decision framework"
            in synthesis_response["actionable"]["workflow_guidance"]
        )
        assert synthesis_response["quality"]["indicators"]["patterns_identified"] == 3
        assert synthesis_response["quality"]["indicators"]["tensions_mapped"] == 2

    def test_debugging_production_issue_scenario(self):
        """Test AORP workflow for urgent production debugging scenario"""

        # Scenario: Critical production issue needs immediate analysis
        # Step 1: Analysis with time pressure and partial information

        urgent_analysis = (
            AORPBuilder()
            .status("partial")
            .key_insight(
                "Production issue analysis: 2 perspectives responded, 2 failed due to system load"
            )
            .confidence(0.45)  # Low due to missing perspectives and urgency
            .session_id("prod-debug-urgent-001")
            .next_steps(
                [
                    "Retry failed perspectives with timeout extension",
                    "add_perspective('infrastructure') - Critical for production issues",
                    "analyze_from_perspectives('What are immediate containment options?') - Focus on urgent actions",
                ]
            )
            .primary_recommendation(
                "Implement immediate containment while gathering more diagnostic data"
            )
            .secondary_recommendations(
                [
                    "Escalate to infrastructure team",
                    "Enable detailed logging for better analysis",
                ]
            )
            .workflow_guidance(
                "Present available insights immediately, emphasize partial coverage due to urgency"
            )
            .completeness(0.5)  # Only 2/4 perspectives
            .reliability(0.45)
            .urgency("critical")  # Production issue
            .indicators(
                successful_perspectives=2,
                failed_perspectives=2,
                failure_reason="system_overload",
                time_pressure=True,
                partial_analysis_acceptable=True,
            )
            .summary(
                "Urgent production issue analysis with partial perspective coverage"
            )
            .data(
                {
                    "issue_type": "production_outage",
                    "severity": "critical",
                    "affected_systems": ["user_auth", "payment_processing"],
                    "available_perspectives": {
                        "technical": {
                            "response": "Database connection pool exhausted, recommend immediate scale-up...",
                            "confidence": 0.80,
                            "urgency": "critical",
                        },
                        "business": {
                            "response": "Customer impact severe, prioritize payment processing restoration...",
                            "confidence": 0.75,
                            "urgency": "critical",
                        },
                    },
                    "failed_perspectives": {
                        "user": {"error": "Analysis timeout due to system load"},
                        "risk": {"error": "Unable to connect to monitoring systems"},
                    },
                }
            )
            .metadata(
                incident_id="INC-2024-001",
                analysis_mode="emergency",
                time_constraint="immediate",
            )
            .build()
        )

        # Validate urgent response prioritizes immediate action
        assert (
            "Production issue analysis" in urgent_analysis["immediate"]["key_insight"]
        )
        assert (
            urgent_analysis["immediate"]["confidence"] < 0.5
        )  # Appropriately low due to partial info
        assert urgent_analysis["quality"]["urgency"] == "critical"
        assert (
            "immediate containment"
            in urgent_analysis["actionable"]["recommendations"]["primary"]
        )
        assert (
            "partial coverage due to urgency"
            in urgent_analysis["actionable"]["workflow_guidance"]
        )

        # Should prioritize containment over completeness
        next_steps = urgent_analysis["actionable"]["next_steps"]
        assert any(
            "immediate" in step.lower() or "containment" in step.lower()
            for step in next_steps
        )

    def test_feature_evaluation_consensus_scenario(self):
        """Test scenario where all perspectives agree (high consensus)"""

        # Scenario: Feature evaluation with strong cross-perspective agreement
        consensus_analysis = (
            AORPBuilder()
            .status("success")
            .key_insight(
                "Feature evaluation shows strong consensus: all 4 perspectives recommend implementation"
            )
            .confidence(0.95)  # High confidence due to consensus
            .session_id("feature-eval-consensus-001")
            .next_steps(
                [
                    "synthesize_perspectives() - Document consensus rationale",
                    "Proceed to implementation planning",
                    "Set up success metrics and monitoring",
                ]
            )
            .primary_recommendation(
                "Move forward with implementation given strong multi-perspective consensus"
            )
            .secondary_recommendations(
                ["Define clear success criteria", "Plan phased rollout strategy"]
            )
            .workflow_guidance(
                "Present strong consensus as clear go/no-go decision with confidence"
            )
            .completeness(1.0)
            .reliability(0.95)
            .urgency("medium")
            .indicators(
                successful_perspectives=4,
                consensus_level="strong",
                agreement_score=0.92,
                conflicting_perspectives=0,
                decision_clarity="high",
            )
            .summary("Clear feature approval with unanimous perspective support")
            .data(
                {
                    "feature": "real_time_collaboration",
                    "consensus_points": [
                        "High user demand validated",
                        "Technical implementation feasible",
                        "Strong business case with clear ROI",
                        "Low risk with existing infrastructure",
                    ],
                    "perspective_scores": {
                        "technical": {"score": 0.90, "recommendation": "implement"},
                        "business": {"score": 0.95, "recommendation": "implement"},
                        "user": {"score": 0.93, "recommendation": "implement"},
                        "risk": {"score": 0.88, "recommendation": "implement"},
                    },
                }
            )
            .metadata(evaluation_type="feature_approval")
            .build()
        )

        # Validate consensus scenario provides clear decision guidance
        assert "strong consensus" in consensus_analysis["immediate"]["key_insight"]
        assert (
            "all 4 perspectives recommend"
            in consensus_analysis["immediate"]["key_insight"]
        )
        assert consensus_analysis["immediate"]["confidence"] > 0.9
        assert (
            "move forward with implementation"
            in consensus_analysis["actionable"]["recommendations"]["primary"].lower()
        )
        assert (
            "clear go/no-go decision"
            in consensus_analysis["actionable"]["workflow_guidance"]
        )
        assert (
            consensus_analysis["quality"]["indicators"]["consensus_level"] == "strong"
        )

    def test_conflicting_perspectives_scenario(self):
        """Test scenario with high tension between perspectives"""

        # Scenario: Architecture decision with significant perspective conflicts
        conflict_synthesis = (
            AORPBuilder()
            .status("success")
            .key_insight(
                "Architecture analysis reveals 3 major tensions requiring structured resolution framework"
            )
            .confidence(0.78)  # Good confidence in identifying tensions
            .session_id("arch-conflict-resolution-001")
            .next_steps(
                [
                    "Create decision matrix for architecture trade-offs",
                    "Schedule perspective alignment workshop",
                    "Define evaluation criteria and weight priorities",
                    "Prototype conflicting approaches for validation",
                ]
            )
            .primary_recommendation(
                "Use structured decision framework to resolve perspective conflicts systematically"
            )
            .secondary_recommendations(
                [
                    "Consider hybrid approach combining perspectives",
                    "Gather additional data for contentious points",
                ]
            )
            .workflow_guidance(
                "Present tensions as structured decision problem requiring systematic resolution"
            )
            .completeness(1.0)
            .reliability(0.78)
            .urgency("high")  # Tensions block progress
            .indicators(
                patterns_identified=2,
                tensions_mapped=3,
                conflict_intensity="high",
                resolution_complexity="complex",
                stakeholder_alignment_needed=True,
            )
            .summary(
                "Architecture decision tensions identified with systematic resolution approach"
            )
            .data(
                {
                    "decision": "microservices_vs_monolith",
                    "tensions": [
                        {
                            "name": "Development Speed vs Operational Complexity",
                            "perspectives": ["business", "technical"],
                            "business_view": "Need fast feature delivery",
                            "technical_view": "Microservices add operational overhead",
                            "intensity": "high",
                        },
                        {
                            "name": "Scalability vs Simplicity",
                            "perspectives": ["technical", "user"],
                            "technical_view": "Microservices enable independent scaling",
                            "user_view": "Monolith simpler for development team",
                            "intensity": "medium",
                        },
                        {
                            "name": "Cost vs Performance",
                            "perspectives": ["business", "risk"],
                            "business_view": "Minimize infrastructure costs",
                            "risk_view": "Distributed systems increase failure modes",
                            "intensity": "high",
                        },
                    ],
                    "resolution_framework": {
                        "approach": "weighted_decision_matrix",
                        "criteria": [
                            "development_speed",
                            "operational_complexity",
                            "scalability",
                            "cost",
                        ],
                        "stakeholders": [
                            "engineering",
                            "product",
                            "operations",
                            "finance",
                        ],
                    },
                }
            )
            .metadata(decision_type="architecture_conflict_resolution")
            .build()
        )

        # Validate conflict resolution approach
        assert "3 major tensions" in conflict_synthesis["immediate"]["key_insight"]
        assert (
            "structured resolution framework"
            in conflict_synthesis["immediate"]["key_insight"]
        )
        assert (
            conflict_synthesis["quality"]["urgency"] == "high"
        )  # Tensions block progress
        assert "decision matrix" in conflict_synthesis["actionable"]["next_steps"][0]
        assert (
            "systematic resolution"
            in conflict_synthesis["actionable"]["workflow_guidance"]
        )
        assert conflict_synthesis["quality"]["indicators"]["tensions_mapped"] == 3
        assert (
            conflict_synthesis["quality"]["indicators"]["conflict_intensity"] == "high"
        )

    def test_error_recovery_workflow_scenario(self):
        """Test complete error recovery workflow"""

        # Scenario: Session expires during critical analysis

        # Step 1: Session not found error
        session_error = create_error_response(
            error_message="Session 'critical-analysis-001' not found or expired",
            error_type="session_not_found",
            recoverable=True,
            context={
                "requested_session": "critical-analysis-001",
                "user_context": "mid_analysis",
                "last_action": "synthesize_perspectives",
            },
        )

        # Validate error provides clear recovery path
        assert session_error["immediate"]["status"] == "error"
        assert "not found or expired" in session_error["immediate"]["key_insight"]
        assert session_error["immediate"]["confidence"] == 0.0

        next_steps = session_error["actionable"]["next_steps"]
        assert any("start_context_analysis" in step for step in next_steps)
        assert any("list_sessions" in step for step in next_steps)

        # Step 2: Validation error during session recreation
        validation_error = create_error_response(
            error_message="Topic exceeds maximum length (1500 chars, max 1000)",
            error_type="validation_error",
            recoverable=True,
            context={"field": "topic", "provided_length": 1500, "max_length": 1000},
        )

        # Validate validation error provides specific guidance
        assert "exceeds maximum length" in validation_error["immediate"]["key_insight"]
        validation_steps = validation_error["actionable"]["next_steps"]
        assert any(
            "review input parameters" in step.lower() for step in validation_steps
        )
        assert any("documentation" in step.lower() for step in validation_steps)

        # Step 3: Successful session recreation
        recovery_success = (
            AORPBuilder()
            .status("success")
            .key_insight("Session recreated successfully - ready to resume analysis")
            .confidence(1.0)
            .session_id("critical-analysis-002")  # New session ID
            .next_steps(
                [
                    "analyze_from_perspectives('<previous question>') - Resume analysis",
                    "Import previous context if needed",
                ]
            )
            .primary_recommendation("Resume analysis from where you left off")
            .workflow_guidance(
                "Present successful recovery and guide to resume workflow"
            )
            .completeness(1.0)
            .reliability(1.0)
            .urgency("medium")
            .indicators(
                recovery_successful=True,
                session_recreated=True,
                context_preserved=False,
            )
            .summary("Error recovery completed, analysis can resume")
            .data(
                {
                    "recovery_type": "session_recreation",
                    "original_session": "critical-analysis-001",
                    "new_session": "critical-analysis-002",
                }
            )
            .metadata(operation_type="error_recovery")
            .build()
        )

        # Validate recovery success guides resumption
        assert "recreated successfully" in recovery_success["immediate"]["key_insight"]
        assert (
            "Resume analysis"
            in recovery_success["actionable"]["recommendations"]["primary"]
        )
        assert any(
            "resume analysis" in step.lower()
            for step in recovery_success["actionable"]["next_steps"]
        )


class TestWorkflowTransitions:
    """Test transitions between different AORP response types"""

    def test_session_to_analysis_transition(self):
        """Test smooth transition from session creation to analysis"""

        # Session creation suggests starting analysis
        session_response = (
            AORPBuilder()
            .status("success")
            .key_insight("Multi-perspective session ready")
            .confidence(1.0)
            .next_steps(["analyze_from_perspectives('<question>') - Start analysis"])
            .workflow_guidance("Guide user to formulate their first analysis question")
            .build()
        )

        # Analysis response builds on session context
        analysis_response = (
            AORPBuilder()
            .status("success")
            .key_insight("Analysis completed across all perspectives")
            .confidence(0.85)
            .next_steps(["synthesize_perspectives() - Find patterns"])
            .workflow_guidance("Present insights, then offer synthesis")
            .build()
        )

        # Validate transition guidance
        session_next = session_response["actionable"]["next_steps"][0]
        assert "analyze_from_perspectives" in session_next

        analysis_next = analysis_response["actionable"]["next_steps"][0]
        assert "synthesize_perspectives" in analysis_next

    def test_analysis_to_synthesis_transition(self):
        """Test transition readiness from analysis to synthesis"""

        # High confidence analysis should suggest synthesis
        ready_for_synthesis = generate_analysis_next_steps(
            session_state="active",
            perspectives_count=4,
            error_count=0,
            has_synthesis=False,
            confidence=0.9,  # High confidence
        )

        synthesis_step = next(
            step for step in ready_for_synthesis if "synthesize_perspectives" in step
        )
        assert "patterns" in synthesis_step or "tensions" in synthesis_step

        # Low confidence should not suggest synthesis
        not_ready_for_synthesis = generate_analysis_next_steps(
            session_state="active",
            perspectives_count=2,
            error_count=1,
            has_synthesis=False,
            confidence=0.3,  # Low confidence
        )

        synthesis_suggestions = [
            step
            for step in not_ready_for_synthesis
            if "synthesize_perspectives" in step
        ]
        assert len(synthesis_suggestions) == 0  # Should not suggest synthesis

    def test_synthesis_to_implementation_transition(self):
        """Test transition from synthesis to implementation planning"""

        # High-confidence synthesis should suggest implementation
        implementation_ready = generate_synthesis_next_steps(
            tensions_identified=2,
            emergent_insights=3,
            confidence=0.9,  # High confidence
        )

        implementation_steps = [
            step
            for step in implementation_ready
            if any(
                keyword in step.lower()
                for keyword in ["implementation", "stakeholder", "decision"]
            )
        ]
        assert len(implementation_steps) > 0

    def test_error_to_recovery_transition(self):
        """Test error recovery provides path back to productive workflow"""

        # Session error should provide path to new session
        session_error = create_error_response(
            error_message="Session expired", error_type="session_not_found"
        )

        recovery_steps = session_error["actionable"]["next_steps"]
        session_creation_steps = [
            step for step in recovery_steps if "start_context_analysis" in step
        ]
        assert len(session_creation_steps) > 0

        # Validation error should provide path to correction
        validation_error = create_error_response(
            error_message="Invalid input", error_type="validation_error"
        )

        correction_steps = validation_error["actionable"]["next_steps"]
        correction_guidance = [
            step
            for step in correction_steps
            if any(
                keyword in step.lower() for keyword in ["review", "correct", "retry"]
            )
        ]
        assert len(correction_guidance) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
