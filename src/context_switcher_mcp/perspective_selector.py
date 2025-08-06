"""Smart perspective selection for Context Switcher MCP"""

from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProblemDomain(Enum):
    """Problem domains that can be detected"""

    ARCHITECTURE = "architecture"
    PERFORMANCE = "performance"
    SECURITY = "security"
    USER_EXPERIENCE = "user_experience"
    BUSINESS_STRATEGY = "business_strategy"
    DATA_PROCESSING = "data_processing"
    API_DESIGN = "api_design"
    DEBUGGING = "debugging"
    COMPLIANCE = "compliance"
    MIGRATION = "migration"
    INTEGRATION = "integration"
    SCALABILITY = "scalability"
    ORGANIZATIONAL = "organizational"  # New domain for HR/team topics
    PROCESS = "process"  # New domain for workflow/methodology


@dataclass
class PerspectiveRecommendation:
    """Recommendation for a perspective"""

    name: str
    description: str
    relevance_score: float
    reasoning: str
    custom_prompt: Optional[str] = None


class SmartPerspectiveSelector:
    """Intelligent perspective selection based on problem analysis"""

    def __init__(self):
        # Domain keywords for detection
        self.domain_keywords = {
            ProblemDomain.ARCHITECTURE: [
                "architecture",
                "design pattern",
                "microservice",
                "monolith",
                "component",
                "module",
                "structure",
                "system design",
                "coupling",
            ],
            ProblemDomain.PERFORMANCE: [
                "performance",
                "slow",
                "speed",
                "latency",
                "throughput",
                "optimization",
                "bottleneck",
                "memory",
                "cpu",
                "cache",
            ],
            ProblemDomain.SECURITY: [
                "security",
                "vulnerability",
                "authentication",
                "authorization",
                "encryption",
                "attack",
                "breach",
                "exploit",
                "threat",
                "credential",
            ],
            ProblemDomain.USER_EXPERIENCE: [
                "user experience",
                "ux",
                "ui",
                "interface",
                "usability",
                "accessibility",
                "workflow",
                "navigation",
                "interaction",
            ],
            ProblemDomain.BUSINESS_STRATEGY: [
                "business",
                "revenue",
                "cost",
                "roi",
                "market",
                "strategy",
                "customer",
                "pricing",
                "competitive",
                "growth",
            ],
            ProblemDomain.DATA_PROCESSING: [
                "data",
                "database",
                "sql",
                "etl",
                "pipeline",
                "analytics",
                "storage",
                "query",
                "index",
                "migration",
            ],
            ProblemDomain.API_DESIGN: [
                "api",
                "rest",
                "graphql",
                "endpoint",
                "request",
                "response",
                "documentation",
                "versioning",
                "contract",
                "integration",
            ],
            ProblemDomain.DEBUGGING: [
                "bug",
                "error",
                "crash",
                "debug",
                "fix",
                "issue",
                "problem",
                "broken",
                "failing",
                "exception",
                "stack trace",
            ],
            ProblemDomain.COMPLIANCE: [
                "compliance",
                "regulation",
                "gdpr",
                "hipaa",
                "pci",
                "audit",
                "policy",
                "standard",
                "requirement",
                "certification",
            ],
            ProblemDomain.MIGRATION: [
                "migration",
                "upgrade",
                "legacy",
                "modernization",
                "refactor",
                "transition",
                "port",
                "convert",
                "backwards compatible",
            ],
            ProblemDomain.INTEGRATION: [
                "integration",
                "third-party",
                "external",
                "webhook",
                "plugin",
                "sdk",
                "library",
                "dependency",
                "compatibility",
            ],
            ProblemDomain.SCALABILITY: [
                "scale",
                "scalability",
                "growth",
                "capacity",
                "load balancing",
                "distributed",
                "cluster",
                "horizontal",
                "vertical",
            ],
            ProblemDomain.ORGANIZATIONAL: [
                "team",
                "employee",
                "staff",
                "workforce",
                "culture",
                "morale",
                "productivity",
                "work week",
                "work-life",
                "remote",
                "hybrid",
                "management",
                "leadership",
                "hr",
                "human resources",
                "retention",
                "hiring",
                "burnout",
                "wellbeing",
            ],
            ProblemDomain.PROCESS: [
                "process",
                "workflow",
                "methodology",
                "agile",
                "scrum",
                "kanban",
                "waterfall",
                "devops",
                "ci/cd",
                "automation",
                "efficiency",
                "optimization",
                "standardization",
                "best practices",
            ],
        }

        # Specialized perspectives for each domain
        self.domain_perspectives = {
            ProblemDomain.ARCHITECTURE: [
                ("architect", "System architecture and design patterns expert"),
                ("code_quality", "Code structure and maintainability analyst"),
                ("integration", "System integration and API design specialist"),
            ],
            ProblemDomain.PERFORMANCE: [
                ("performance", "Performance optimization and bottleneck analyst"),
                ("scalability", "Scalability and capacity planning expert"),
                (
                    "infrastructure",
                    "Infrastructure and resource optimization specialist",
                ),
            ],
            ProblemDomain.SECURITY: [
                ("security", "Security vulnerability and threat modeling expert"),
                ("compliance", "Regulatory compliance and audit specialist"),
                ("incident_response", "Security incident and breach response expert"),
            ],
            ProblemDomain.USER_EXPERIENCE: [
                ("ux_designer", "User experience and interaction design expert"),
                ("accessibility", "Accessibility and inclusive design specialist"),
                ("user_researcher", "User behavior and needs analysis expert"),
            ],
            ProblemDomain.BUSINESS_STRATEGY: [
                ("product_manager", "Product strategy and roadmap expert"),
                ("financial", "Financial analysis and ROI calculation specialist"),
                ("market_analyst", "Market trends and competitive analysis expert"),
            ],
            ProblemDomain.DATA_PROCESSING: [
                ("data_engineer", "Data pipeline and ETL specialist"),
                ("database_admin", "Database optimization and management expert"),
                ("data_scientist", "Data analysis and insights expert"),
            ],
            ProblemDomain.API_DESIGN: [
                ("api_architect", "API design and versioning expert"),
                (
                    "developer_experience",
                    "Developer experience and documentation specialist",
                ),
                ("integration", "Third-party integration and compatibility expert"),
            ],
            ProblemDomain.DEBUGGING: [
                ("debugger", "Root cause analysis and debugging expert"),
                ("test_engineer", "Testing strategy and quality assurance specialist"),
                ("reliability", "System reliability and monitoring expert"),
            ],
            ProblemDomain.COMPLIANCE: [
                ("compliance_officer", "Regulatory compliance and policy expert"),
                ("privacy", "Data privacy and protection specialist"),
                ("auditor", "Security and compliance audit expert"),
            ],
            ProblemDomain.MIGRATION: [
                ("migration_architect", "System migration and modernization expert"),
                (
                    "legacy_specialist",
                    "Legacy system understanding and transition expert",
                ),
                ("risk_assessor", "Migration risk and rollback planning specialist"),
            ],
            ProblemDomain.INTEGRATION: [
                ("integration_architect", "Third-party integration design expert"),
                ("compatibility", "Cross-platform compatibility specialist"),
                ("api_consumer", "External API usage and best practices expert"),
            ],
            ProblemDomain.SCALABILITY: [
                ("scale_architect", "Distributed systems and scaling expert"),
                (
                    "capacity_planner",
                    "Capacity planning and growth projection specialist",
                ),
                ("performance", "Load testing and optimization expert"),
            ],
            ProblemDomain.ORGANIZATIONAL: [
                ("hr_specialist", "Human resources and workforce management expert"),
                (
                    "culture_analyst",
                    "Organizational culture and team dynamics specialist",
                ),
                (
                    "productivity_expert",
                    "Workforce productivity and efficiency analyst",
                ),
                (
                    "wellbeing_advocate",
                    "Employee wellbeing and work-life balance specialist",
                ),
                ("change_manager", "Organizational change management expert"),
            ],
            ProblemDomain.PROCESS: [
                ("process_engineer", "Business process optimization specialist"),
                ("agile_coach", "Agile methodology and team transformation expert"),
                ("automation_specialist", "Process automation and efficiency expert"),
                ("quality_analyst", "Process quality and standardization specialist"),
            ],
        }

        # Base perspectives always considered
        self.base_perspectives = ["technical", "business", "user", "risk"]

    def analyze_problem(
        self, topic: str, context: Optional[str] = None
    ) -> Dict[ProblemDomain, float]:
        """Analyze the problem to identify relevant domains"""
        text = topic.lower()
        if context:
            text += " " + context.lower()

        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = 0.0
            for keyword in keywords:
                # Exact match gets higher score
                if keyword in text:
                    score += 1.0
                # Partial match gets lower score
                elif any(word in keyword for word in text.split()):
                    score += 0.5

            # Normalize by number of keywords
            domain_scores[domain] = score / len(keywords)

        return domain_scores

    def recommend_perspectives(
        self,
        topic: str,
        context: Optional[str] = None,
        existing_perspectives: Optional[List[str]] = None,
        max_recommendations: int = 5,
    ) -> List[PerspectiveRecommendation]:
        """Recommend perspectives based on problem analysis"""

        # Analyze problem domains
        domain_scores = self.analyze_problem(topic, context)

        # Sort domains by relevance
        relevant_domains = sorted(
            domain_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Filter domains with meaningful scores
        relevant_domains = [(d, s) for d, s in relevant_domains if s > 0.1]

        recommendations: List[Dict[str, Any]] = []
        used_perspectives = set(existing_perspectives or [])
        used_perspectives.update(self.base_perspectives)

        # Add perspectives from relevant domains
        for domain, domain_score in relevant_domains:
            if len(recommendations) >= max_recommendations:
                break

            domain_perspectives = self.domain_perspectives.get(domain, [])

            for persp_name, persp_desc in domain_perspectives:
                if persp_name in used_perspectives:
                    continue

                # Calculate relevance score
                relevance = domain_score * 0.8  # Domain relevance

                # Boost score if perspective name appears in topic
                if persp_name.replace("_", " ") in topic.lower():
                    relevance += 0.2

                # Create custom prompt for the perspective
                custom_prompt = self._generate_custom_prompt(
                    persp_name, persp_desc, domain, topic
                )

                recommendation = PerspectiveRecommendation(
                    name=persp_name,
                    description=persp_desc,
                    relevance_score=relevance,
                    reasoning=f"Detected {domain.value} concerns in the problem",
                    custom_prompt=custom_prompt,
                )

                recommendations.append(recommendation)
                used_perspectives.add(persp_name)

                if len(recommendations) >= max_recommendations:
                    break

        # Sort by relevance score
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)

        # If we have few recommendations, add some general ones
        if len(recommendations) < 3:
            self._add_fallback_perspectives(recommendations, used_perspectives, topic)

        return recommendations[:max_recommendations]

    def _generate_custom_prompt(
        self, perspective_name: str, description: str, domain: ProblemDomain, topic: str
    ) -> str:
        """Generate a custom system prompt for a perspective"""

        prompt = f"""You are a {description}.

**Your expertise covers:**
- Deep knowledge in {domain.value.replace("_", " ")} domain
- Practical experience with real-world {perspective_name.replace("_", " ")} challenges
- Understanding of industry best practices and common pitfalls

**When analyzing problems, focus on:**
- Specific {perspective_name.replace("_", " ")} considerations
- Actionable recommendations based on your expertise
- Potential risks and mitigation strategies in your domain
- Trade-offs and decision criteria

**For this analysis of "{topic}", consider:**
- How {domain.value.replace("_", " ")} factors impact the solution
- What {perspective_name.replace("_", " ")} patterns or anti-patterns apply
- Concrete steps to address the challenges
- Metrics or criteria to evaluate success

Provide specific, practical insights. If the topic is outside your expertise, respond with [NO_RESPONSE]."""

        return prompt

    def _add_fallback_perspectives(
        self,
        recommendations: List[PerspectiveRecommendation],
        used_perspectives: Set[str],
        topic: str,
    ):
        """Add fallback perspectives when few domain-specific ones are found"""

        fallback_perspectives = [
            (
                "implementation",
                "Practical implementation and coding expert",
                "General implementation concerns",
            ),
            (
                "operations",
                "DevOps and operational excellence expert",
                "Operational and deployment considerations",
            ),
            (
                "future_proofing",
                "Technology trends and future-proofing expert",
                "Long-term sustainability concerns",
            ),
        ]

        for persp_name, persp_desc, reasoning in fallback_perspectives:
            if persp_name not in used_perspectives and len(recommendations) < 5:
                recommendation = PerspectiveRecommendation(
                    name=persp_name,
                    description=persp_desc,
                    relevance_score=0.3,  # Lower score for fallbacks
                    reasoning=reasoning,
                )
                recommendations.append(recommendation)

    def suggest_follow_up_perspectives(
        self, initial_responses: Dict[str, str], original_topic: str
    ) -> List[PerspectiveRecommendation]:
        """Suggest follow-up perspectives based on initial analysis"""

        # Analyze patterns in responses
        all_text = " ".join(initial_responses.values()).lower()

        # Look for emerging themes
        emerging_themes = []

        # Check for cross-cutting concerns mentioned
        if "integration" in all_text and "integration" not in initial_responses:
            emerging_themes.append(
                (
                    "integration_specialist",
                    "System integration expert",
                    "Multiple perspectives mentioned integration challenges",
                )
            )

        if "data" in all_text and "data" not in initial_responses:
            emerging_themes.append(
                (
                    "data_architect",
                    "Data architecture and flow expert",
                    "Data considerations emerged from analysis",
                )
            )

        if any(word in all_text for word in ["cost", "budget", "expense"]):
            if "financial" not in initial_responses:
                emerging_themes.append(
                    (
                        "financial_analyst",
                        "Cost analysis and budgeting expert",
                        "Financial implications identified",
                    )
                )

        # Convert to recommendations
        recommendations: List[Dict[str, Any]] = []
        for name, desc, reasoning in emerging_themes:
            recommendations.append(
                PerspectiveRecommendation(
                    name=name,
                    description=desc,
                    relevance_score=0.7,
                    reasoning=reasoning,
                )
            )

        return recommendations
