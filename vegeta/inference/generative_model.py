"""
Generative Model for Bayesian Active Inference

The generative model g(v) predicts expected observations u' given hidden causes v.
It generates predictions across three channels:
- Semantic: expected utterance embedding
- Structural: expected graph patterns
- Terms: expected key terms/phrases

Following the theory from attemp1TextOnly.py specification.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from numpy.typing import NDArray

from ..core.exceptions import PredictionError
from .types import PredictedObservations, HiddenStates
from .prediction_channels import (
    SemanticPredictionChannel,
    StructuralPredictionChannel,
    TermsPredictionChannel
)

logger = logging.getLogger(__name__)


class GenerativeModel:
    """
    Core generative model g(v) → u' that predicts expected observations.

    Takes hidden states v and generates predictions across three channels:
    - Semantic: pooled node embeddings from subgraph
    - Structural: expected slot/edge patterns from checklist
    - Terms: expected key terms from node names and checklist lexicon
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize generative model with prediction channels

        Args:
            config: Configuration dictionary containing:
                - ollama_base_url: Ollama API endpoint
                - embedding_model: Model name for semantic predictions
                - max_terms: Maximum terms to predict
                - pooling_method: Node embedding pooling strategy
        """
        self.config = config

        # Initialize prediction channels
        self.semantic_channel = SemanticPredictionChannel(config)
        self.structural_channel = StructuralPredictionChannel(config)
        self.terms_channel = TermsPredictionChannel(config)

        logger.info("🎯 GenerativeModel initialized with prediction channels")

    def predict_observations(self, v: HiddenStates, context: Dict[str, Any]) -> PredictedObservations:
        """
        Main prediction method: g(v) → u'

        Args:
            v: Hidden states containing checklist, subgraph, slots, etc.
            context: Additional context (graph access, embeddings, etc.)

        Returns:
            PredictedObservations across all three channels
        """
        try:
            logger.debug(f"🎯 Generating predictions for checklist: {v.z_checklist}, subgraph: {v.z_subgraph}")

            # Generate predictions across all channels
            u_sem = self.semantic_channel.predict(v, context)
            u_struct = self.structural_channel.predict(v, context)
            u_terms = self.terms_channel.predict(v, context)

            # Calculate overall confidence
            confidence = self._calculate_overall_confidence(u_sem, u_struct, u_terms, context)

            # Prepare metadata
            metadata = {
                'checklist': v.z_checklist,
                'subgraph': v.z_subgraph,
                'goal': v.z_goal,
                'step': v.z_step,
                'slots': v.z_slots,
                'prediction_channels': ['semantic', 'structural', 'terms']
            }

            predictions = PredictedObservations(
                u_sem=u_sem,
                u_struct=u_struct,
                u_terms=u_terms,
                confidence=confidence,
                metadata=metadata
            )

            logger.debug(".3f")
            return predictions

        except Exception as e:
            logger.error(f"❌ Prediction failed for {v.z_checklist}: {e}")
            raise PredictionError(f"Generative model prediction failed: {e}")

    def _calculate_overall_confidence(self,
                                    u_sem: NDArray[np.float32],
                                    u_struct: Dict[str, float],
                                    u_terms: List[str],
                                    context: Dict[str, Any]) -> float:
        """
        Calculate overall prediction confidence based on channel predictions

        Args:
            u_sem: Semantic prediction vector
            u_struct: Structural prediction counts
            u_terms: Terms prediction list
            context: Additional context information

        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence calculation - can be made more sophisticated
        confidence_factors = []

        # Semantic confidence based on vector magnitude (normalized vectors should be ~1)
        sem_confidence = min(1.0, np.linalg.norm(u_sem))
        confidence_factors.append(sem_confidence)

        # Structural confidence based on number of predicted patterns
        struct_confidence = min(1.0, len(u_struct) / 10.0)  # Expect at least 10 structural elements
        confidence_factors.append(struct_confidence)

        # Terms confidence based on number of predicted terms
        terms_confidence = min(1.0, len(u_terms) / 5.0)  # Expect at least 5 terms
        confidence_factors.append(terms_confidence)

        # Weighted average confidence
        weights = [0.5, 0.3, 0.2]  # Favor semantic channel
        overall_confidence = sum(w * c for w, c in zip(weights, confidence_factors))

        return overall_confidence

    def update_channel_weights(self, channel_weights: Dict[str, float]):
        """
        Update relative weights between prediction channels

        Args:
            channel_weights: Dictionary with keys 'semantic', 'structural', 'terms'
                           and float values summing to 1.0
        """
        # This would allow dynamic adjustment of channel importance
        # Implementation would update self._channel_weights
        pass

    def get_channel_weights(self) -> Dict[str, float]:
        """Get current channel weights"""
        # Default weights favoring semantic channel
        return {
            'semantic': 0.5,
            'structural': 0.3,
            'terms': 0.2
        }
