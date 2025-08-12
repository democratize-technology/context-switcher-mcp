# Perspective Templates for common analysis patterns

PERSPECTIVE_TEMPLATES = {
    "architecture_decision": {
        "perspectives": ["technical", "business", "user", "risk"],
        "custom": [
            (
                "scalability",
                "Long-term growth implications, performance at scale, "
                "resource requirements",
            ),
            (
                "team_capability",
                "Current team skills, learning curve, maintenance burden",
            ),
            (
                "migration",
                "Path from current state, backwards compatibility, rollback strategy",
            ),
        ],
    },
    "feature_evaluation": {
        "perspectives": ["user", "business", "technical"],
        "custom": [
            (
                "competitive",
                "Market differentiation, competitor features, unique value proposition",
            ),
            (
                "support_burden",
                "Customer support impact, documentation needs, training requirements",
            ),
            ("timeline", "Development time, dependencies, critical path impact"),
        ],
    },
    "debugging_analysis": {
        "perspectives": ["technical", "risk"],
        "custom": [
            ("performance", "Latency, throughput, resource usage, bottlenecks"),
            (
                "data_flow",
                "Input sources, transformations, output destinations, data integrity",
            ),
            (
                "dependencies",
                "External services, libraries, API calls, version conflicts",
            ),
            ("history", "Recent changes, deployment timeline, similar past issues"),
        ],
    },
    "api_design": {
        "perspectives": ["technical", "user"],
        "custom": [
            (
                "versioning",
                "Breaking changes, deprecation strategy, version maintenance",
            ),
            ("documentation", "Clarity, examples, error messages, discoverability"),
            (
                "consistency",
                "Naming conventions, patterns, principle of least surprise",
            ),
            ("extensibility", "Future additions, plugin points, webhook capabilities"),
        ],
    },
    "security_review": {
        "perspectives": ["risk", "technical", "business"],
        "custom": [
            (
                "attack_surface",
                "Entry points, authentication, authorization, input validation",
            ),
            ("compliance", "Regulatory requirements, audit trails, data protection"),
            (
                "incident_response",
                "Detection, mitigation, recovery, communication plan",
            ),
        ],
    },
}
