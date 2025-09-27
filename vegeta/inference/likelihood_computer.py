"""
Likelihood Computation for Bayesian Active Inference

Computes p(u|v) by comparing actual observations u with predicted observations u'
across three channels:
- Semantic channel: cosine distance between embeddings
- Structural channel: L2 distance between pattern counts
- Terms channel: Jaccard similarity between term sets

Following the specification from attemp1TextOnly.py
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import LikelihoodError
from .types import PredictedObservations, ObservedFeatures

logger = logging.getLogger(__name__)


class LikelihoodComputer:
    """
    Computes likelihood p(u|v) by comparing observed vs predicted features

    Three-channel likelihood with configurable noise parameters σ²
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize likelihood computer

        Args:
            config: Configuration with noise parameters and weights
        """
        self.config = config

        # Noise variances (aleatoric uncertainty)
        self.σ_sem2 = config.get('σ_sem2', 0.3)
        self.σ_struct2 = config.get('σ_struct2', 0.2)
        self.σ_terms2 = config.get('σ_terms2', 0.2)

        # Channel weights
        self.α = config.get('α', 1.0)  # Semantic weight
        self.β = config.get('β', 0.5)  # Structural weight
        self.γ = config.get('γ', 0.3)  # Terms weight

        # Other parameters
        self.small_set_threshold = config.get('small_set_threshold', 3)
        self.small_set_blend = config.get('small_set_blend', 0.5)
        self.max_terms = config.get('max_terms', 20)

        logger.info(f"LikelihoodComputer initialized with noise parameters: "
                    f"σ_sem2={self.σ_sem2:.2f}, σ_struct2={self.σ_struct2:.2f}, "
                    f"σ_terms2={self.σ_terms2:.2f}")
    def compute_likelihood(self,
                          observed: ObservedFeatures,
                          predicted: PredictedObservations,
                          penalties: Optional[Dict[str, float]] = None) -> float:
        """
        Compute log-likelihood log p(u|v)

        Args:
            observed: Actual observations u from user utterance
            predicted: Predicted observations u' from generative model
            penalties: Optional penalty terms (missing slots, hub nodes, etc.)

        Returns:
            Log-likelihood score (higher = better fit)
        """
        try:
            # Compute per-channel distances
            δ_sem = self._semantic_distance(observed.u_sem, predicted.u_sem)
            δ_struct = self._structural_distance(observed.u_struct, predicted.u_struct)
            δ_terms = self._terms_distance(observed.u_terms, predicted.u_terms)

            # Apply penalties
            total_penalties = sum(penalties.values()) if penalties else 0.0

            # Compute log-likelihood
            log_likelihood = - (
                self.α * δ_sem / self.σ_sem2 +
                self.β * δ_struct / self.σ_struct2 +
                self.γ * δ_terms / self.σ_terms2 +
                total_penalties
            )

            logger.debug(f"Likelihood computation: "
                        f"δ_sem={δ_sem:.3f}, δ_struct={δ_struct:.3f}, "
                        f"δ_terms={δ_terms:.3f}, total_penalty={total_penalties:.3f}")

            return log_likelihood

        except Exception as e:
            logger.error(f"❌ Likelihood computation failed: {e}")
            raise LikelihoodError(f"Likelihood computation failed: {e}")

    def _semantic_distance(self,
                          u_sem: NDArray[np.float32],
                          u_sem_pred: NDArray[np.float32]) -> float:
        """
        Compute semantic distance δ_sem

        Args:
            u_sem: Observed utterance embedding
            u_sem_pred: Predicted utterance embedding

        Returns:
            Cosine distance (1 - cosine_similarity) ∈ [0, 2]
        """
        try:
            # Compute cosine similarity
            dot_product = np.dot(u_sem, u_sem_pred)
            norm_u = np.linalg.norm(u_sem)
            norm_pred = np.linalg.norm(u_sem_pred)

            if norm_u == 0 or norm_pred == 0:
                # Handle zero vectors
                cosine_sim = 0.0
            else:
                cosine_sim = dot_product / (norm_u * norm_pred)

            # Clamp to [-1, 1] range (numerical precision issues)
            cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

            # Convert to distance
            δ_sem = 1.0 - cosine_sim

            return δ_sem

        except Exception as e:
            logger.warning(f"⚠ Semantic distance computation failed: {e}")
            return 1.0  # Maximum distance

    def _structural_distance(self,
                           u_struct: Dict[str, float],
                           u_struct_pred: Dict[str, float]) -> float:
        """
        Compute structural distance δ_struct

        Args:
            u_struct: Observed structural patterns
            u_struct_pred: Predicted structural patterns

        Returns:
            L2 distance between normalized count vectors
        """
        try:
            if not u_struct and not u_struct_pred:
                return 0.0

            # Get all unique keys
            all_keys = set(u_struct.keys()) | set(u_struct_pred.keys())

            # Create vectors
            obs_vec = np.array([u_struct.get(key, 0.0) for key in all_keys])
            pred_vec = np.array([u_struct_pred.get(key, 0.0) for key in all_keys])

            # Apply log1p transformation to compress heavy tails
            obs_vec = np.log1p(obs_vec)
            pred_vec = np.log1p(pred_vec)

            # Compute L2 distance
            δ_struct = np.linalg.norm(obs_vec - pred_vec)

            # Normalize by vector dimension
            δ_struct = δ_struct / np.sqrt(len(all_keys))

            return δ_struct

        except Exception as e:
            logger.warning(f"⚠ Structural distance computation failed: {e}")
            return 1.0  # Maximum distance

    def _terms_distance(self,
                       u_terms: List[str],
                       u_terms_pred: List[str]) -> float:
        """
        Compute terms distance δ_terms

        Args:
            u_terms: Observed canonical terms
            u_terms_pred: Predicted key terms

        Returns:
            Terms distance ∈ [0, 1]
        """
        try:
            if not u_terms and not u_terms_pred:
                return 0.0

            # Convert to sets for Jaccard
            obs_set = set(u_terms)
            pred_set = set(u_terms_pred)

            # Check if we need fallback computation for small sets
            min_size = min(len(obs_set), len(pred_set))

            if min_size < self.small_set_threshold and min_size > 0:
                # Use blended approach for small sets
                jaccard_dist = self._jaccard_distance(obs_set, pred_set)
                embedding_dist = self._terms_embedding_distance(u_terms, u_terms_pred)

                δ_terms = (self.small_set_blend * jaccard_dist +
                          (1 - self.small_set_blend) * embedding_dist)
            else:
                # Standard Jaccard distance
                δ_terms = self._jaccard_distance(obs_set, pred_set)

            return δ_terms

        except Exception as e:
            logger.warning(f"⚠ Terms distance computation failed: {e}")
            return 1.0  # Maximum distance

    def _jaccard_distance(self, set1: set, set2: set) -> float:
        """Compute Jaccard distance (1 - Jaccard similarity)"""
        if not set1 and not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 1.0

        jaccard_sim = intersection / union
        return 1.0 - jaccard_sim

    def _terms_embedding_distance(self,
                                terms1: List[str],
                                terms2: List[str]) -> float:
        """
        Compute distance based on term embeddings (fallback for small sets)

        This would compute embeddings for individual terms and compare
        averaged embeddings. For now, return a simple heuristic.
        """
        # Simple heuristic: distance based on term overlap and length difference
        if not terms1 and not terms2:
            return 0.0

        # Character-level distance as proxy
        text1 = ' '.join(terms1)
        text2 = ' '.join(terms2)

        # Simple edit distance approximation
        len_diff = abs(len(text1) - len(text2))
        max_len = max(len(text1), len(text2))

        if max_len == 0:
            return 0.0

        return min(1.0, len_diff / max_len)

    def update_noise_parameters(self, new_params: Dict[str, float]):
        """
        Update noise parameters σ² dynamically

        Args:
            new_params: Dictionary with keys like 'σ_sem²', 'σ_struct²', 'σ_terms²'
        """
        if 'σ_sem2' in new_params:
            self.σ_sem2 = new_params['σ_sem2']
        if 'σ_struct2' in new_params:
            self.σ_struct2 = new_params['σ_struct2']
        if 'σ_terms2' in new_params:
            self.σ_terms2 = new_params['σ_terms2']

        logger.debug(f"Updated noise parameters: "
                    f"σ_sem2={self.σ_sem2:.2f}, σ_struct2={self.σ_struct2:.2f}, "
                    f"σ_terms2={self.σ_terms2:.2f}")
    def get_channel_contributions(self,
                                observed: ObservedFeatures,
                                predicted: PredictedObservations) -> Dict[str, float]:
        """
        Get individual channel contributions to likelihood

        Useful for debugging and understanding model behavior
        """
        δ_sem = self._semantic_distance(observed.u_sem, predicted.u_sem)
        δ_struct = self._structural_distance(observed.u_struct, predicted.u_struct)
        δ_terms = self._terms_distance(observed.u_terms, predicted.u_terms)

        return {
            'semantic': -self.α * δ_sem / self.σ_sem2,
            'structural': -self.β * δ_struct / self.σ_struct2,
            'terms': -self.γ * δ_terms / self.σ_terms2,
            'δ_sem': δ_sem,
            'δ_struct': δ_struct,
            'δ_terms': δ_terms
        }
