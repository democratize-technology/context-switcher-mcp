"""
Context alignment convergence detection for Context-Switcher MCP.

This module implements convergence detection to stop iterations when different
contexts start aligning. It uses a moderate threshold (0.85) to balance
alignment detection with maintaining healthy diversity across perspectives.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timezone

from .logging_base import get_logger
from .constants import NO_RESPONSE

logger = get_logger(__name__)


@dataclass
class ConvergenceMetrics:
    """Metrics for convergence analysis"""

    session_id: str
    iteration_number: int
    alignment_score: float
    threshold: float
    converged: bool
    perspective_count: int
    valid_responses: int
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return {
            "session_id": self.session_id,
            "iteration": self.iteration_number,
            "alignment_score": self.alignment_score,
            "threshold": self.threshold,
            "converged": self.converged,
            "perspective_count": self.perspective_count,
            "valid_responses": self.valid_responses,
            "timestamp": self.timestamp.isoformat(),
        }


class ContextAlignmentDetector:
    """
    Detects convergence across different contexts/perspectives.

    Uses cosine similarity of response embeddings to measure alignment
    while maintaining diversity by using a moderate threshold.
    """

    def __init__(self, threshold: float = 0.85):
        """
        Initialize the alignment detector.

        Args:
            threshold: Convergence threshold (0.85 = moderate, allows some diversity)
                      - 0.85 balances alignment detection with diversity preservation
                      - Lower values (0.7) allow more diversity but may miss convergence
                      - Higher values (0.95) catch only very tight alignment
        """
        if threshold < 0.5 or threshold > 1.0:
            logger.warning(
                f"Unusual threshold {threshold}, recommended range: 0.7-0.95"
            )

        self.threshold = threshold
        self.embeddings_cache: Dict[str, np.ndarray] = {}

        logger.info(
            f"Context alignment detector initialized with threshold {threshold} (moderate diversity preservation)"
        )

    async def measure_context_alignment(
        self,
        current_responses: Dict[str, str],
        previous_responses: Optional[Dict[str, str]] = None,
        session_id: str = "unknown",
    ) -> ConvergenceMetrics:
        """
        Measure alignment across contexts in current iteration.

        Args:
            current_responses: Current iteration's perspective responses
            previous_responses: Previous iteration's responses (optional)
            session_id: Session identifier for logging

        Returns:
            ConvergenceMetrics with alignment score and convergence status
        """
        logger.debug(f"Measuring context alignment for session {session_id}")

        # Filter out invalid responses
        valid_current = self._filter_valid_responses(current_responses)

        if len(valid_current) < 2:
            logger.warning(
                f"Insufficient valid responses ({len(valid_current)}) for alignment measurement"
            )
            return ConvergenceMetrics(
                session_id=session_id,
                iteration_number=1,
                alignment_score=0.0,
                threshold=self.threshold,
                converged=False,
                perspective_count=len(current_responses),
                valid_responses=len(valid_current),
                timestamp=datetime.now(timezone.utc),
            )

        # Calculate embeddings for current responses
        current_embeddings = await self._get_embeddings(valid_current, session_id)

        # Calculate alignment score across all current perspectives
        alignment_score = self._calculate_cross_perspective_alignment(
            current_embeddings
        )

        # Check convergence
        converged = alignment_score >= self.threshold

        if converged:
            logger.info(
                f"Contexts aligning at {alignment_score:.3f} (threshold: {self.threshold})"
            )
        else:
            logger.debug(
                f"Context alignment: {alignment_score:.3f} (below threshold: {self.threshold})"
            )

        return ConvergenceMetrics(
            session_id=session_id,
            iteration_number=self._estimate_iteration_number(session_id),
            alignment_score=alignment_score,
            threshold=self.threshold,
            converged=converged,
            perspective_count=len(current_responses),
            valid_responses=len(valid_current),
            timestamp=datetime.now(timezone.utc),
        )

    async def check_iteration_convergence(
        self, session_analyses: List[Dict[str, Any]], session_id: str = "unknown"
    ) -> Tuple[bool, float]:
        """
        Check convergence by comparing recent iterations.

        Args:
            session_analyses: List of session analyses from session.analyses
            session_id: Session identifier for logging

        Returns:
            Tuple of (converged: bool, alignment_score: float)
        """
        if len(session_analyses) < 2:
            logger.debug("Need at least 2 iterations for convergence checking")
            return False, 0.0

        # Get the two most recent analyses
        current_analysis = session_analyses[-1]
        previous_analysis = session_analyses[-2]

        current_responses = current_analysis.get("results", {})
        previous_responses = previous_analysis.get("results", {})

        # Measure alignment in current iteration
        metrics = await self.measure_context_alignment(
            current_responses, previous_responses, session_id
        )

        return metrics.converged, metrics.alignment_score

    def _filter_valid_responses(self, responses: Dict[str, str]) -> Dict[str, str]:
        """Filter out errors and abstentions"""
        return {
            name: response
            for name, response in responses.items()
            if (
                response
                and isinstance(response, str)
                and NO_RESPONSE not in response
                and not response.startswith("ERROR:")
            )
        }

    async def _get_embeddings(
        self, responses: Dict[str, str], session_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for responses, with caching.

        Uses a simple hash-based cache to avoid recomputing embeddings
        for identical responses.
        """
        embeddings = {}

        for perspective, response in responses.items():
            # Create cache key
            cache_key = f"{hash(response)}_{len(response)}"

            if cache_key in self.embeddings_cache:
                embeddings[perspective] = self.embeddings_cache[cache_key]
            else:
                # Generate embedding (using a simple approach for now)
                embedding = await self._generate_embedding(
                    response, perspective, session_id
                )
                self.embeddings_cache[cache_key] = embedding
                embeddings[perspective] = embedding

        return embeddings

    async def _generate_embedding(
        self, text: str, perspective: str, session_id: str
    ) -> np.ndarray:
        """
        Generate embedding for text.

        For now, uses a simple TF-IDF-like approach. Could be enhanced
        with actual embedding models later.
        """
        try:
            # Simple word-based embedding as fallback
            words = text.lower().split()

            # Create a basic frequency vector
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            # Convert to a fixed-size vector (using hash for consistency)
            vector_size = 100  # Fixed size for consistency
            embedding = np.zeros(vector_size)

            for word, count in word_counts.items():
                # Use hash to map words to vector positions
                position = hash(word) % vector_size
                embedding[position] += count

            # Normalize the vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding

        except Exception as e:
            logger.warning(f"Failed to generate embedding for {perspective}: {e}")
            # Return a zero vector as fallback
            return np.zeros(100)

    def _calculate_cross_perspective_alignment(
        self, embeddings: Dict[str, np.ndarray]
    ) -> float:
        """
        Calculate alignment score across all perspectives.

        Uses pairwise cosine similarities and returns the average.
        This captures how aligned the perspectives are overall.
        """
        if len(embeddings) < 2:
            return 0.0

        similarities = []
        perspectives = list(embeddings.keys())

        # Calculate pairwise similarities
        for i in range(len(perspectives)):
            for j in range(i + 1, len(perspectives)):
                perspective_a = perspectives[i]
                perspective_b = perspectives[j]

                embedding_a = embeddings[perspective_a]
                embedding_b = embeddings[perspective_b]

                # Calculate cosine similarity
                similarity = self._cosine_similarity(embedding_a, embedding_b)
                similarities.append(similarity)

        # Return average similarity as alignment score
        if similarities:
            alignment_score = np.mean(similarities)
            return float(np.clip(alignment_score, 0.0, 1.0))

        return 0.0

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Handle zero vectors
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            return float(np.clip(similarity, -1.0, 1.0))

        except Exception as e:
            logger.warning(f"Failed to calculate cosine similarity: {e}")
            return 0.0

    def _estimate_iteration_number(self, session_id: str) -> int:
        """Estimate current iteration number (simple counter for now)"""
        # This could be enhanced to track actual iteration counts per session
        return 1

    def clear_cache(self) -> int:
        """Clear embeddings cache and return number of items cleared"""
        cleared_count = len(self.embeddings_cache)
        self.embeddings_cache.clear()
        logger.debug(f"Cleared {cleared_count} cached embeddings")
        return cleared_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cached_embeddings": len(self.embeddings_cache),
            "threshold": self.threshold,
            "cache_memory_estimate": len(self.embeddings_cache)
            * 100
            * 8,  # rough bytes estimate
        }

    def adjust_threshold(
        self, new_threshold: float, reason: str = "manual_adjustment"
    ) -> None:
        """
        Adjust convergence threshold to balance alignment vs diversity.

        Args:
            new_threshold: New threshold value (0.5-1.0)
            reason: Reason for adjustment (for logging)
        """
        if new_threshold < 0.5 or new_threshold > 1.0:
            logger.error(
                f"Invalid threshold {new_threshold}, must be between 0.5 and 1.0"
            )
            return

        old_threshold = self.threshold
        self.threshold = new_threshold

        logger.info(
            f"Threshold adjusted: {old_threshold:.3f} -> {new_threshold:.3f} ({reason})"
        )

        # Provide guidance on the impact
        if new_threshold > old_threshold:
            logger.info(
                "Higher threshold: will preserve more diversity, detect tighter convergence"
            )
        else:
            logger.info(
                "Lower threshold: will detect convergence earlier, may reduce diversity"
            )

    def get_diversity_guidance(self, alignment_score: float) -> str:
        """
        Provide guidance on diversity vs alignment based on current score.

        Args:
            alignment_score: Current alignment score

        Returns:
            Human-readable guidance on diversity status
        """
        if alignment_score >= self.threshold:
            return "High alignment - contexts have converged, consider synthesis"
        elif alignment_score >= 0.7:
            return "Moderate alignment - perspectives showing some convergence, healthy diversity remains"
        elif alignment_score >= 0.5:
            return "Good diversity - perspectives remain distinct, continue exploration"
        else:
            return "High diversity - perspectives very different, ensure they're addressing the same topic"


# Global instance for use across the application
context_alignment_detector = ContextAlignmentDetector(threshold=0.85)


async def check_context_convergence(
    session_analyses: List[Dict[str, Any]], session_id: str = "unknown"
) -> Tuple[bool, float]:
    """
    Convenience function to check if contexts are converging.

    Args:
        session_analyses: List of analyses from session.analyses
        session_id: Session identifier

    Returns:
        Tuple of (converged: bool, alignment_score: float)
    """
    return await context_alignment_detector.check_iteration_convergence(
        session_analyses, session_id
    )


async def measure_current_alignment(
    current_responses: Dict[str, str], session_id: str = "unknown"
) -> ConvergenceMetrics:
    """
    Convenience function to measure alignment in current responses.

    Args:
        current_responses: Current perspective responses
        session_id: Session identifier

    Returns:
        ConvergenceMetrics with alignment details
    """
    return await context_alignment_detector.measure_context_alignment(
        current_responses, session_id=session_id
    )
