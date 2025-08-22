"""Enhanced confidence calibration and quality metrics for Context Switcher MCP"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ResponseQuality(Enum):
    """Response quality levels"""

    EXCELLENT = "excellent"  # 0.9+
    GOOD = "good"  # 0.7-0.89
    FAIR = "fair"  # 0.5-0.69
    POOR = "poor"  # 0.3-0.49
    UNACCEPTABLE = "unacceptable"  # <0.3


@dataclass
class QualityMetrics:
    """Detailed quality metrics for a response"""

    # Content quality
    depth_score: float  # Response comprehensiveness
    specificity_score: float  # Concrete vs vague content
    coherence_score: float  # Internal consistency
    relevance_score: float  # On-topic focus

    # Technical quality
    structure_score: float  # Well-organized response
    actionability_score: float  # Contains practical insights
    evidence_score: float  # Backed by reasoning/examples

    # Meta quality
    perspective_alignment: float  # Matches perspective's expertise
    abstention_appropriateness: float  # Correct use of NO_RESPONSE

    @property
    def overall_score(self) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            "depth": 0.15,
            "specificity": 0.15,
            "coherence": 0.10,
            "relevance": 0.15,
            "structure": 0.10,
            "actionability": 0.15,
            "evidence": 0.10,
            "alignment": 0.05,
            "abstention": 0.05,
        }

        scores = [
            self.depth_score * weights["depth"],
            self.specificity_score * weights["specificity"],
            self.coherence_score * weights["coherence"],
            self.relevance_score * weights["relevance"],
            self.structure_score * weights["structure"],
            self.actionability_score * weights["actionability"],
            self.evidence_score * weights["evidence"],
            self.perspective_alignment * weights["alignment"],
            self.abstention_appropriateness * weights["abstention"],
        ]

        return sum(scores)

    @property
    def quality_level(self) -> ResponseQuality:
        """Determine quality level from overall score"""
        score = self.overall_score
        if score >= 0.9:
            return ResponseQuality.EXCELLENT
        elif score >= 0.7:
            return ResponseQuality.GOOD
        elif score >= 0.5:
            return ResponseQuality.FAIR
        elif score >= 0.3:
            return ResponseQuality.POOR
        else:
            return ResponseQuality.UNACCEPTABLE


class ConfidenceCalibrator:
    """Enhanced confidence calibration with quality metrics"""

    def __init__(self):
        # Keywords indicating high-quality content
        self.quality_indicators = {
            "specificity": [
                "specifically",
                "precisely",
                "exact",
                "detailed",
                "concrete",
            ],
            "evidence": ["because", "due to", "based on", "given that", "evidence"],
            "structure": ["first", "second", "additionally", "furthermore", "however"],
            "actionability": ["recommend", "suggest", "implement", "action", "step"],
            "depth": [
                "implication",
                "consideration",
                "trade-off",
                "impact",
                "consequence",
            ],
        }

        # Vague terms that reduce quality
        self.vague_terms = [
            "maybe",
            "possibly",
            "might",
            "could be",
            "perhaps",
            "somewhat",
            "kind of",
            "sort of",
            "generally",
            "usually",
        ]

    def analyze_response_quality(
        self,
        response: str,
        perspective_name: str,
        prompt: str,
        is_abstention: bool = False,
    ) -> QualityMetrics:
        """Analyze detailed quality metrics for a single response"""

        if is_abstention:
            # Special handling for abstentions
            return QualityMetrics(
                depth_score=0.0,
                specificity_score=0.0,
                coherence_score=1.0,  # Abstaining is coherent
                relevance_score=0.0,
                structure_score=1.0,
                actionability_score=0.0,
                evidence_score=0.0,
                perspective_alignment=1.0,  # Appropriate abstention
                abstention_appropriateness=1.0,
            )

        # Calculate individual metrics
        metrics = QualityMetrics(
            depth_score=self._calculate_depth_score(response),
            specificity_score=self._calculate_specificity_score(response),
            coherence_score=self._calculate_coherence_score(response),
            relevance_score=self._calculate_relevance_score(response, prompt),
            structure_score=self._calculate_structure_score(response),
            actionability_score=self._calculate_actionability_score(response),
            evidence_score=self._calculate_evidence_score(response),
            perspective_alignment=self._calculate_alignment_score(
                response, perspective_name
            ),
            abstention_appropriateness=0.0,  # Should have abstained but didn't
        )

        return metrics

    def _calculate_depth_score(self, response: str) -> float:
        """Measure response depth and comprehensiveness"""
        # Length factor (normalized)
        word_count = len(response.split())
        length_factor = min(1.0, word_count / 200)  # 200+ words is comprehensive

        # Complexity factor (multi-clause sentences)
        sentences = re.split(r"[.!?]+", response)
        complex_sentences = sum(1 for s in sentences if "," in s or ";" in s)
        complexity_factor = min(1.0, complex_sentences / max(1, len(sentences)))

        # Depth keywords
        depth_keywords = sum(
            response.lower().count(kw) for kw in self.quality_indicators["depth"]
        )
        depth_factor = min(1.0, depth_keywords / 3)  # 3+ depth keywords is good

        return length_factor * 0.4 + complexity_factor * 0.3 + depth_factor * 0.3

    def _calculate_specificity_score(self, response: str) -> float:
        """Measure concrete vs vague language"""
        words = response.lower().split()
        total_words = len(words)

        if total_words == 0:
            return 0.0

        # Count specific indicators
        specific_count = sum(
            response.lower().count(kw) for kw in self.quality_indicators["specificity"]
        )

        # Count vague terms (penalty)
        vague_count = sum(response.lower().count(term) for term in self.vague_terms)

        # Numbers and percentages indicate specificity
        numbers_count = len(re.findall(r"\b\d+\b", response))

        specificity = (specific_count + numbers_count - vague_count) / max(
            1, total_words / 50
        )
        return min(1.0, max(0.0, specificity))

    def _calculate_coherence_score(self, response: str) -> float:
        """Measure internal consistency and logical flow"""
        # Check for contradictions (simple heuristic)
        contradiction_patterns = [
            (r"\bbut\s+not\b", r"\balthough\b"),
            (r"\bhowever\b", r"\bnevertheless\b"),
        ]

        contradictions = 0
        for pattern1, pattern2 in contradiction_patterns:
            if re.search(pattern1, response, re.I) and re.search(
                pattern2, response, re.I
            ):
                contradictions += 1

        # Logical connectors indicate coherence
        connectors = ["therefore", "thus", "hence", "because", "since", "as a result"]
        connector_count = sum(response.lower().count(conn) for conn in connectors)

        coherence = 1.0 - (contradictions * 0.2) + min(0.3, connector_count * 0.1)
        return min(1.0, max(0.0, coherence))

    def _calculate_relevance_score(self, response: str, prompt: str) -> float:
        """Measure how well response addresses the prompt"""
        # Extract key terms from prompt
        prompt_words = {
            word.lower() for word in prompt.split() if len(word) > 3 and word.isalnum()
        }

        # Count prompt terms in response
        response_words = response.lower().split()
        matches = sum(1 for word in response_words if word in prompt_words)

        relevance = min(1.0, matches / max(1, len(prompt_words) * 0.5))
        return relevance

    def _calculate_structure_score(self, response: str) -> float:
        """Measure response organization"""
        # Check for structure indicators
        structure_count = sum(
            response.lower().count(kw) for kw in self.quality_indicators["structure"]
        )

        # Check for sections/paragraphs
        paragraphs = response.strip().split("\n\n")
        has_sections = len(paragraphs) > 1

        # Check for lists
        has_lists = bool(re.search(r"^\s*[-*•]\s+", response, re.MULTILINE))

        structure = min(
            1.0,
            structure_count * 0.2
            + (0.3 if has_sections else 0)
            + (0.2 if has_lists else 0)
            + 0.3,
        )  # Base structure score

        return structure

    def _calculate_actionability_score(self, response: str) -> float:
        """Measure practical insights and recommendations"""
        action_count = sum(
            response.lower().count(kw)
            for kw in self.quality_indicators["actionability"]
        )

        # Check for imperative mood (action-oriented)
        imperative_patterns = [
            r"\b(implement|create|develop|establish|ensure|add|update|modify)\b"
        ]
        imperative_count = sum(
            len(re.findall(pattern, response, re.I)) for pattern in imperative_patterns
        )

        actionability = min(
            1.0, (action_count + imperative_count) / 5
        )  # 5+ action items is excellent
        return actionability

    def _calculate_evidence_score(self, response: str) -> float:
        """Measure reasoning and evidence quality"""
        evidence_count = sum(
            response.lower().count(kw) for kw in self.quality_indicators["evidence"]
        )

        # Check for examples
        example_patterns = [
            r"\bfor example\b",
            r"\bsuch as\b",
            r"\be\.g\.\b",
            r"\bi\.e\.\b",
        ]
        example_count = sum(
            len(re.findall(pattern, response, re.I)) for pattern in example_patterns
        )

        evidence = min(1.0, (evidence_count + example_count * 2) / 4)
        return evidence

    def _calculate_alignment_score(self, response: str, perspective_name: str) -> float:
        """Measure how well response aligns with perspective's expertise"""
        # Perspective-specific keywords
        perspective_keywords = {
            "technical": [
                "architecture",
                "implementation",
                "performance",
                "scalability",
                "code",
            ],
            "business": ["revenue", "cost", "roi", "market", "strategy"],
            "user": ["experience", "usability", "workflow", "interface", "adoption"],
            "risk": ["security", "compliance", "vulnerability", "mitigation", "threat"],
        }

        keywords = perspective_keywords.get(perspective_name.lower(), [])
        if not keywords:
            return 0.7  # Default alignment for unknown perspectives

        keyword_count = sum(response.lower().count(kw) for kw in keywords)
        alignment = min(
            1.0, keyword_count / 3
        )  # 3+ perspective keywords is good alignment

        return alignment

    def calculate_enhanced_confidence(
        self,
        perspective_metrics: dict[str, QualityMetrics],
        error_count: int,
        abstention_count: int,
        total_perspectives: int,
    ) -> tuple[float, dict[str, Any]]:
        """Calculate enhanced confidence score with detailed breakdown"""
        # Input validation
        validation_result = self._validate_confidence_inputs(
            perspective_metrics, total_perspectives
        )
        if validation_result:
            return validation_result

        # Extract active metrics
        active_metrics = self._extract_active_metrics(perspective_metrics)
        if not active_metrics:
            return 0.0, {"reason": "All perspectives abstained"}

        # Calculate confidence components using strategy pattern
        calculator = ConfidenceCalculator()
        components = calculator.calculate_components(
            active_metrics, error_count, abstention_count, total_perspectives
        )

        # Build final confidence and breakdown
        confidence = components["final_confidence"]
        breakdown = self._build_confidence_breakdown(components, active_metrics)

        return min(1.0, max(0.0, confidence)), breakdown

    def _validate_confidence_inputs(
        self, perspective_metrics: dict[str, QualityMetrics], total_perspectives: int
    ) -> tuple[float, dict[str, str]] | None:
        """Validate inputs for confidence calculation"""
        if total_perspectives == 0:
            return 0.0, {"reason": "No perspectives available"}
        return None

    def _extract_active_metrics(
        self, perspective_metrics: dict[str, QualityMetrics]
    ) -> dict[str, QualityMetrics]:
        """Extract metrics for non-abstaining perspectives"""
        return {
            k: v
            for k, v in perspective_metrics.items()
            if v.abstention_appropriateness < 1.0
        }

    def _build_confidence_breakdown(
        self, components: dict[str, float], active_metrics: dict[str, QualityMetrics]
    ) -> dict[str, Any]:
        """Build detailed breakdown of confidence calculation"""
        quality_dist_builder = QualityDistributionBuilder()
        quality_distribution = quality_dist_builder.build_distribution(active_metrics)

        return {
            "coverage_factor": round(components["coverage"], 3),
            "average_quality": round(components["avg_quality"], 3),
            "error_penalty": round(components["error_penalty"], 3),
            "abstention_penalty": round(components["abstention_penalty"], 3),
            "consistency_factor": round(components["consistency"], 3),
            "final_confidence": round(components["final_confidence"], 3),
            "quality_distribution": quality_distribution,
        }


def analyze_synthesis_quality(
    synthesis_text: str, perspectives_count: int, original_prompt: str
) -> tuple[float, dict[str, Any]]:
    """Analyze synthesis quality with detailed metrics"""

    calibrator = ConfidenceCalibrator()

    # Pattern detection
    convergence_patterns = [
        r"\bmultiple perspectives agree\b",
        r"\bconsensus\b",
        r"\bshared\s+\w+\b",
        r"\ball\s+perspectives\b",
        r"\bunanimous\b",
    ]

    tension_patterns = [
        r"\btension\b",
        r"\bconflict\b",
        r"\bdisagree\b",
        r"\btrade-off\b",
        r"\bcompeting\s+\w+\b",
    ]

    insight_patterns = [
        r"\bemerges?\b",
        r"\breveals?\b",
        r"\binsight\b",
        r"\bdiscover\w*\b",
        r"\bunexpected\b",
    ]

    # Count patterns
    convergence_count = sum(
        len(re.findall(p, synthesis_text, re.I)) for p in convergence_patterns
    )
    tension_count = sum(
        len(re.findall(p, synthesis_text, re.I)) for p in tension_patterns
    )
    insight_count = sum(
        len(re.findall(p, synthesis_text, re.I)) for p in insight_patterns
    )

    # Analyze synthesis structure
    sections = synthesis_text.strip().split("\n\n")
    has_clear_structure = len(sections) >= 3

    # Calculate synthesis-specific metrics
    pattern_diversity = min(
        1.0, (convergence_count + tension_count + insight_count) / 10
    )
    structural_clarity = 1.0 if has_clear_structure else 0.6
    perspective_integration = min(
        1.0, synthesis_text.lower().count("perspective") / perspectives_count
    )

    # Get base quality metrics
    base_metrics = calibrator.analyze_response_quality(
        synthesis_text, "synthesis", original_prompt
    )

    # Synthesis-specific confidence
    confidence = (
        base_metrics.overall_score * 0.4
        + pattern_diversity * 0.3
        + structural_clarity * 0.15
        + perspective_integration * 0.15
    )

    breakdown = {
        "base_quality": round(base_metrics.overall_score, 3),
        "pattern_diversity": round(pattern_diversity, 3),
        "structural_clarity": round(structural_clarity, 3),
        "perspective_integration": round(perspective_integration, 3),
        "patterns_found": {
            "convergence": convergence_count,
            "tensions": tension_count,
            "insights": insight_count,
        },
        "synthesis_confidence": round(confidence, 3),
    }

    return confidence, breakdown


class ConfidenceCalculator:
    """Handles confidence calculation logic using strategy pattern"""

    def calculate_components(
        self,
        active_metrics: dict[str, QualityMetrics],
        error_count: int,
        abstention_count: int,
        total_perspectives: int,
    ) -> dict[str, float]:
        """Calculate all confidence components"""
        coverage = self._calculate_coverage(active_metrics, total_perspectives)
        avg_quality = self._calculate_average_quality(active_metrics)
        error_penalty = self._calculate_error_penalty(error_count)
        abstention_penalty = self._calculate_abstention_penalty(abstention_count)
        consistency = self._calculate_consistency(active_metrics, avg_quality)

        final_confidence = self.compute_final_confidence(
            {
                "coverage": coverage,
                "avg_quality": avg_quality,
                "error_penalty": error_penalty,
                "abstention_penalty": abstention_penalty,
                "consistency": consistency,
            }
        )

        return {
            "coverage": coverage,
            "avg_quality": avg_quality,
            "error_penalty": error_penalty,
            "abstention_penalty": abstention_penalty,
            "consistency": consistency,
            "final_confidence": final_confidence,
        }

    def _calculate_coverage(
        self, active_metrics: dict[str, QualityMetrics], total_perspectives: int
    ) -> float:
        """Calculate coverage factor"""
        return len(active_metrics) / total_perspectives

    def _calculate_average_quality(
        self, active_metrics: dict[str, QualityMetrics]
    ) -> float:
        """Calculate average quality of active responses"""
        return sum(m.overall_score for m in active_metrics.values()) / len(
            active_metrics
        )

    def _calculate_error_penalty(self, error_count: int) -> float:
        """Calculate penalty for errors"""
        return max(0.0, 1.0 - (error_count * 0.2))

    def _calculate_abstention_penalty(self, abstention_count: int) -> float:
        """Calculate penalty for abstentions"""
        return max(0.7, 1.0 - (abstention_count * 0.05))

    def _calculate_consistency(
        self, active_metrics: dict[str, QualityMetrics], avg_quality: float
    ) -> float:
        """Calculate consistency factor based on quality variance"""
        quality_scores = [m.overall_score for m in active_metrics.values()]
        if len(quality_scores) <= 1:
            return 0.8  # Single response has no variance

        variance = sum((s - avg_quality) ** 2 for s in quality_scores) / len(
            quality_scores
        )
        return max(0.5, 1.0 - variance)

    def compute_final_confidence(self, components: dict[str, float]) -> float:
        """Compute final confidence using weighted components"""
        return (
            components["coverage"] * 0.25
            + components["avg_quality"] * 0.35
            + components["error_penalty"] * 0.15
            + components["abstention_penalty"] * 0.10
            + components["consistency"] * 0.15
        )


class QualityDistributionBuilder:
    """Builds quality distribution breakdown"""

    def build_distribution(
        self, active_metrics: dict[str, QualityMetrics]
    ) -> dict[str, int]:
        """Build quality level distribution for active metrics"""
        quality_levels = [
            ResponseQuality.EXCELLENT,
            ResponseQuality.GOOD,
            ResponseQuality.FAIR,
            ResponseQuality.POOR,
            ResponseQuality.UNACCEPTABLE,
        ]

        return {
            level.value: self._count_by_quality_level(active_metrics, level)
            for level in quality_levels
        }

    def _count_by_quality_level(
        self, active_metrics: dict[str, QualityMetrics], quality_level: ResponseQuality
    ) -> int:
        """Count metrics at specific quality level"""
        return sum(
            1 for m in active_metrics.values() if m.quality_level == quality_level
        )
