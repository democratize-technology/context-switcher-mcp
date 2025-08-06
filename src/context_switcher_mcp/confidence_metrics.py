"""Enhanced confidence calibration and quality metrics for Context Switcher MCP"""

import re
from typing import Any, Dict, Tuple
from dataclasses import dataclass
from enum import Enum


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
        prompt_words = set(
            word.lower() for word in prompt.split() if len(word) > 3 and word.isalnum()
        )

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
        perspective_metrics: Dict[str, QualityMetrics],
        error_count: int,
        abstention_count: int,
        total_perspectives: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate enhanced confidence score with detailed breakdown"""

        if total_perspectives == 0:
            return 0.0, {"reason": "No perspectives available"}

        # Calculate active responses
        active_metrics = {
            k: v
            for k, v in perspective_metrics.items()
            if v.abstention_appropriateness < 1.0
        }

        if not active_metrics:
            return 0.0, {"reason": "All perspectives abstained"}

        # Coverage factor
        coverage = len(active_metrics) / total_perspectives

        # Quality factor (average of active response qualities)
        avg_quality = sum(m.overall_score for m in active_metrics.values()) / len(
            active_metrics
        )

        # Error penalty
        error_penalty = max(0.0, 1.0 - (error_count * 0.2))

        # Abstention penalty (mild - abstentions can be appropriate)
        abstention_penalty = max(0.7, 1.0 - (abstention_count * 0.05))

        # Consistency factor (variance in quality scores)
        quality_scores = [m.overall_score for m in active_metrics.values()]
        if len(quality_scores) > 1:
            variance = sum((s - avg_quality) ** 2 for s in quality_scores) / len(
                quality_scores
            )
            consistency = max(0.5, 1.0 - variance)
        else:
            consistency = 0.8  # Single response has no variance

        # Calculate final confidence
        confidence = (
            coverage * 0.25
            + avg_quality * 0.35
            + error_penalty * 0.15
            + abstention_penalty * 0.10
            + consistency * 0.15
        )

        # Detailed breakdown
        breakdown = {
            "coverage_factor": round(coverage, 3),
            "average_quality": round(avg_quality, 3),
            "error_penalty": round(error_penalty, 3),
            "abstention_penalty": round(abstention_penalty, 3),
            "consistency_factor": round(consistency, 3),
            "final_confidence": round(confidence, 3),
            "quality_distribution": {
                ResponseQuality.EXCELLENT.value: sum(
                    1
                    for m in active_metrics.values()
                    if m.quality_level == ResponseQuality.EXCELLENT
                ),
                ResponseQuality.GOOD.value: sum(
                    1
                    for m in active_metrics.values()
                    if m.quality_level == ResponseQuality.GOOD
                ),
                ResponseQuality.FAIR.value: sum(
                    1
                    for m in active_metrics.values()
                    if m.quality_level == ResponseQuality.FAIR
                ),
                ResponseQuality.POOR.value: sum(
                    1
                    for m in active_metrics.values()
                    if m.quality_level == ResponseQuality.POOR
                ),
                ResponseQuality.UNACCEPTABLE.value: sum(
                    1
                    for m in active_metrics.values()
                    if m.quality_level == ResponseQuality.UNACCEPTABLE
                ),
            },
        }

        return min(1.0, max(0.0, confidence)), breakdown


def analyze_synthesis_quality(
    synthesis_text: str, perspectives_count: int, original_prompt: str
) -> Tuple[float, Dict[str, Any]]:
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
