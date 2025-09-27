"""
Bayesian Posterior Updates for Active Inference

Implements variational inference to compute q(v|u) ≈ p(v|u)
Following the theory from attemp1TextOnly.py specification.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from ..core.config import Config
from .generative_model import HiddenStates
from .likelihood_computer import LikelihoodComputer, ObservedFeatures

logger = logging.getLogger(__name__)


@dataclass
class PosteriorBeliefs:
    """Container for all posterior beliefs q(v|u)"""
    # Core posterior beliefs (required)
    z_checklist: Dict[str, float]          # q(z_checklist|u)
    z_goal: Dict[str, float]               # q(z_goal|u)
    z_subgraph: Dict[str, float]           # q(z_subgraph|u) - main inference
    z_dialogue_act: Dict[str, float]       # q(z_dialogue_act|u)
    z_novelty: float                       # q(z_novelty|u)
    entropy_checklist: float
    entropy_subgraph: float
    confidence_top: float
    num_candidates: int

    # Optional posterior beliefs
    z_slots: Optional[Dict[str, Dict[str, float]]] = None  # q(z_slot_r|u) per slot
    z_step: Optional[Dict[str, float]] = None  # q(z_step|u)


class PosteriorUpdater:
    """
    Computes posterior beliefs q(v|u) using variational inference

    Implements the core Bayesian update: q(v|u) ∝ p(u|v) × p(v)
    with factorization across latent variables.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Posterior temperature for softmax normalization
        self.τ_posterior = config.get('τ_posterior', 0.7)

        # Convergence criteria for iterative updates
        self.max_iterations = config.get('max_posterior_iterations', 10)
        self.convergence_threshold = config.get('posterior_convergence_threshold', 1e-6)

        logger.info("🎯 PosteriorUpdater initialized for Bayesian active inference")

    def compute_posterior(self,
                         candidates: List[Dict[str, Any]],
                         priors: Dict[str, Any],
                         observed: ObservedFeatures,
                         likelihood_computer: LikelihoodComputer) -> PosteriorBeliefs:
        """
        Main posterior computation using variational inference

        Args:
            candidates: List of subgraph candidates with likelihoods
            priors: Prior beliefs p(v) from prior construction
            observed: Observed features u from user utterance
            likelihood_computer: For computing p(u|v) for each candidate

        Returns:
            PosteriorBeliefs containing q(v|u) for all latent variables
        """
        logger.debug(f"🎯 Computing posterior for {len(candidates)} candidates")

        # Step 1: Compute subgraph posterior (main inference)
        z_subgraph = self._compute_subgraph_posterior(candidates, priors, observed, likelihood_computer)

        # Step 2: Update checklist posterior based on subgraph evidence
        z_checklist = self._compute_checklist_posterior(candidates, priors, observed, z_subgraph)

        # Step 3: Update goal posterior
        z_goal = self._compute_goal_posterior(observed, priors, z_checklist)

        # Step 4: Compute dialogue act posterior
        z_dialogue_act = self._compute_dialogue_act_posterior(observed, priors)

        # Step 5: Compute novelty posterior
        z_novelty = self._compute_novelty_posterior(candidates, observed, priors)

        # Step 6: Optional - compute slot posteriors for top candidate
        z_slots = self._compute_slot_posteriors(candidates, priors, observed, z_subgraph)

        # Step 7: Compute step posterior if step priors provided
        z_step = priors.get('step', {})

        # Compute metadata
        entropy_checklist = self._compute_entropy(z_checklist)
        entropy_subgraph = self._compute_entropy(z_subgraph)

        # More robust confidence calculation
        confidence_top = self._compute_confidence_top(z_subgraph)

        posterior = PosteriorBeliefs(
            z_checklist=z_checklist,
            z_goal=z_goal,
            z_subgraph=z_subgraph,
            z_slots=z_slots,
            z_step=z_step,  # Now using step priors from prior builder
            z_dialogue_act=z_dialogue_act,
            z_novelty=z_novelty,
            entropy_checklist=entropy_checklist,
            entropy_subgraph=entropy_subgraph,
            confidence_top=confidence_top,
            num_candidates=len(candidates)
        )

        logger.debug(".3f"
                    ".3f"
                    ".3f")

        return posterior

    def _compute_confidence_top(self, z_subgraph: Dict[str, float]) -> float:
        """
        Compute more robust confidence score for top candidate

        Uses margin-based confidence: (top - second) / (top + second)
        This gives higher confidence when there's a clear winner.
        """
        if not z_subgraph or len(z_subgraph) < 2:
            return max(z_subgraph.values()) if z_subgraph else 0.0

        # Sort by probability
        sorted_probs = sorted(z_subgraph.values(), reverse=True)
        top_prob = sorted_probs[0]
        second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

        # Margin-based confidence
        if top_prob + second_prob > 0:
            margin_confidence = (top_prob - second_prob) / (top_prob + second_prob)
            # Ensure non-negative and scale to [0, 1]
            margin_confidence = max(0.0, min(1.0, margin_confidence))
        else:
            margin_confidence = 0.0

        # Even more aggressive absolute probability scaling
        absolute_confidence = min(1.0, top_prob * 4)  # Even more aggressive scaling

        # For small candidate sets, be more confident
        small_set_bonus = 0.3 if len(z_subgraph) <= 5 else 0.1

        # Additional bonus for clear winners
        winner_bonus = 0.2 if margin_confidence > 0.3 else 0.0

        # Take the maximum of all confidence measures
        confidence = max(margin_confidence, absolute_confidence, small_set_bonus, winner_bonus)

        logger.debug(f"Confidence calculation: margin={margin_confidence:.3f}, "
                    f"absolute={absolute_confidence:.3f}, final={confidence:.3f}")

        return confidence

    def _compute_subgraph_posterior(self,
                                   candidates: List[Dict[str, Any]],
                                   priors: Dict[str, Any],
                                   observed: ObservedFeatures,
                                   likelihood_computer: LikelihoodComputer) -> Dict[str, float]:
        """
        Compute main subgraph posterior: q(z_subgraph|u)

        This is the core inference: q(z_subgraph|u) ∝ p(u|z_subgraph) × p(z_subgraph)
        """
        if not candidates:
            return {}

        subgraph_priors = priors.get('subgraph', {})
        posterior_scores = {}

        for candidate in candidates:
            subgraph_id = candidate['id']

            # Get likelihood p(u|v) - should already be computed
            if 'log_likelihood' not in candidate:
                logger.warning(f"⚠ No likelihood computed for candidate {subgraph_id}")
                continue

            log_likelihood = candidate['log_likelihood']

            # Get prior p(v)
            prior_prob = subgraph_priors.get(subgraph_id, 1e-6)  # Small floor
            log_prior = np.log(prior_prob) if prior_prob > 0 else -10

            # Compute log posterior
            log_posterior = log_likelihood + log_prior
            posterior_scores[subgraph_id] = log_posterior

        # Normalize using softmax with temperature
        if posterior_scores:
            return self._softmax_normalize(posterior_scores, self.τ_posterior)
        else:
            # Fallback to uniform
            n = len(candidates)
            return {c['id']: 1.0/n for c in candidates}

    def _compute_checklist_posterior(self,
                                    candidates: List[Dict[str, Any]],
                                    priors: Dict[str, Any],
                                    observed: ObservedFeatures,
                                    z_subgraph: Dict[str, float]) -> Dict[str, float]:
        """
        Compute checklist posterior: q(z_checklist|u)

        Conditioned on subgraph posterior and utterance evidence
        """
        checklist_priors = priors.get('checklist', {})

        # Get evidence from top subgraph
        if z_subgraph:
            top_subgraph_id = max(z_subgraph.items(), key=lambda x: x[1])[0]
            top_candidate = next((c for c in candidates if c['id'] == top_subgraph_id), None)

            if top_candidate:
                # Adjust priors based on subgraph structure
                checklist_posterior = self._adjust_checklist_by_evidence(
                    checklist_priors, top_candidate, observed
                )
            else:
                checklist_posterior = checklist_priors.copy()
        else:
            checklist_posterior = checklist_priors.copy()

        return self._normalize_distribution(checklist_posterior)

    def _compute_goal_posterior(self,
                               observed: ObservedFeatures,
                               priors: Dict[str, Any],
                               z_checklist: Dict[str, float]) -> Dict[str, float]:
        """
        Compute goal posterior: q(z_goal|u)

        Uses LLM-based goal priors with minimal Bayesian adjustment
        """
        goal_priors = priors.get('goal', {})

        # For now, trust LLM-based goal priors
        # Future: Add Bayesian adjustments based on checklist compatibility
        return goal_priors.copy()

    def _compute_dialogue_act_posterior(self,
                                       observed: ObservedFeatures,
                                       priors: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute dialogue act posterior: q(z_dialogue_act|u)

        Simplified for now - uses heuristics + priors
        """
        dialogue_act_priors = priors.get('dialogue_act', {})

        # Apply simple heuristics based on utterance patterns
        adjusted_priors = dialogue_act_priors.copy()

        # Question patterns -> higher 'request' probability
        if '?' in observed.u_terms or any(word in observed.u_terms for word in ['what', 'how', 'when', 'where', 'why']):
            if 'request' in adjusted_priors:
                adjusted_priors['request'] *= 1.5

        # Entity mentions -> higher 'provide' probability
        # This would need more sophisticated analysis in practice

        return self._normalize_distribution(adjusted_priors)

    def _compute_novelty_posterior(self,
                                  candidates: List[Dict[str, Any]],
                                  observed: ObservedFeatures,
                                  priors: Dict[str, Any]) -> float:
        """
        Compute novelty posterior: q(z_novelty|u)

        Measures out-of-distribution likelihood
        """
        # Simple novelty detection based on candidate quality
        if not candidates:
            return 1.0  # High novelty if no candidates

        # Novelty based on max likelihood and number of good candidates
        max_likelihood = max((c.get('log_likelihood', -10) for c in candidates), default=-10)
        good_candidates = sum(1 for c in candidates if c.get('log_likelihood', -10) > -5.0)

        # High novelty if:
        # 1. Very low max likelihood, OR
        # 2. Very few good candidates
        novelty_score = 0.0

        if max_likelihood < -7.0:  # Very poor fit
            novelty_score += 0.7

        if good_candidates == 0:
            novelty_score += 0.3
        elif good_candidates == 1:
            novelty_score += 0.1

        return min(novelty_score, 1.0)

    def _compute_slot_posteriors(self,
                                candidates: List[Dict[str, Any]],
                                priors: Dict[str, Any],
                                observed: ObservedFeatures,
                                z_subgraph: Dict[str, float]) -> Optional[Dict[str, Dict[str, float]]]:
        """
        Compute slot posteriors: q(z_slot_r|u) for each required slot

        This is more complex and would need slot specification data
        """
        # Placeholder - would need checklist/slot specification integration
        return None

    def _adjust_checklist_by_evidence(self,
                                     checklist_priors: Dict[str, float],
                                     top_candidate: Dict[str, Any],
                                     observed: ObservedFeatures) -> Dict[str, float]:
        """
        Adjust checklist priors based on evidence from top candidate
        """
        adjusted = checklist_priors.copy()

        # Get structural evidence
        struct_obs = top_candidate.get('u_struct_obs', {})

        # Count different entity types
        entity_types = sum(1 for k, v in struct_obs.items()
                          if k.startswith('node_') and v > 0.1)

        # Boost identification checklists if we see specific entities
        if entity_types >= 2:
            for checklist_name in adjusted:
                if 'Identify' in checklist_name or 'Verify' in checklist_name:
                    adjusted[checklist_name] *= 1.3

        # Boost recommendation if we see diverse patterns
        diverse_patterns = sum(1 for k, v in struct_obs.items()
                              if k.startswith('rel_') and v > 0.1)

        if diverse_patterns >= 3:
            for checklist_name in adjusted:
                if 'Recommend' in checklist_name:
                    adjusted[checklist_name] *= 1.2

        return adjusted

    def _softmax_normalize(self, scores: Dict[str, float], temperature: float) -> Dict[str, float]:
        """Apply softmax normalization with temperature"""
        if not scores:
            return {}

        # Subtract max for numerical stability
        max_score = max(scores.values())
        exp_scores = {k: np.exp((v - max_score) / temperature) for k, v in scores.items()}

        total = sum(exp_scores.values())
        if total > 0:
            return {k: v / total for k, v in exp_scores.items()}
        else:
            # Fallback to uniform
            n = len(scores)
            return {k: 1.0/n for k in scores.keys()}

    def _normalize_distribution(self, dist: Dict[str, float]) -> Dict[str, float]:
        """Normalize distribution to sum to 1"""
        total = sum(dist.values())
        if total > 0:
            return {k: v / total for k, v in dist.items()}
        else:
            # Fallback to uniform
            n = len(dist)
            return {k: 1.0/n for k in dist.keys()} if n > 0 else {}

    def _compute_entropy(self, distribution: Dict[str, float]) -> float:
        """Compute Shannon entropy of distribution"""
        if not distribution:
            return 0.0

        # Filter out zero probabilities
        probs = [p for p in distribution.values() if p > 1e-10]

        if not probs:
            return 0.0

        return -sum(p * np.log(p) for p in probs)

    def update_posteriors(self, candidates: List[Dict[str, Any]],
                         priors: Dict[str, Any],
                         observation_u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility with existing VegetaSystem

        This wraps the new compute_posterior method to maintain API compatibility
        """
        # Convert observation_u dict to ObservedFeatures
        observed_features = ObservedFeatures(
            u_sem=np.array(observation_u.get('u_sem', np.random.normal(0, 1, 768).astype(np.float32))),
            u_terms=observation_u.get('u_terms', []),
            u_struct=observation_u.get('u_struct', {})
        )

        # Call new method
        result = self.compute_posterior(candidates, priors, observed_features, None)

        # Convert result back to old format
        return {
            'checklist': result.z_checklist,
            'goal': result.z_goal,
            'subgraph': result.z_subgraph,
            'dialogue_act': result.z_dialogue_act,
            'novelty': result.z_novelty
        }
