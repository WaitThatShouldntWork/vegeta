"""
Decision Policy for Bayesian Active Inference

Implements Expected Information Gain (EIG) calculations for deciding between
ASK, ANSWER, and SEARCH actions. Uses the mathematical framework from
attemp1TextOnly.py specification.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .posterior_updater import PosteriorBeliefs
from .generative_model import HiddenStates
from .likelihood_computer import LikelihoodComputer, ObservedFeatures

logger = logging.getLogger(__name__)


@dataclass
class ActionCandidate:
    """Represents a potential action with its expected utility"""
    action: str                    # 'ASK', 'ANSWER', 'SEARCH'
    target: Optional[str]         # What to ask about, or what answer to give
    expected_utility: float       # EIG + value - cost
    eig: float                   # Expected Information Gain
    value: float                 # Expected value/reward
    cost: float                  # Action cost (time, API calls, etc.)
    confidence: float            # Confidence in this action being optimal


@dataclass
class DecisionResult:
    """Final decision with reasoning and metrics"""
    action: str
    target: Optional[str]
    confidence: float
    reasoning: str
    eig_analysis: Dict[str, float]
    action_candidates: List[ActionCandidate]


class DecisionPolicy:
    """
    Decision policy using Expected Information Gain (EIG)

    Implements the mathematical framework for deciding between actions:
    - ANSWER: Provide best answer now
    - ASK: Ask targeted question to reduce uncertainty
    - SEARCH: Look up additional information

    Based on the theory from attemp1TextOnly.py specification.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # EIG calculation parameters
        self.eig_discount_factor = config.get('eig_discount_factor', 0.9)
        self.horizon_length = config.get('horizon_length', 2)  # 2-step lookahead

        # Action costs (configurable)
        self.cost_ask = config.get('cost_ask', 0.10)
        self.cost_search = config.get('cost_search', 0.20)
        self.cost_answer = config.get('cost_answer', 0.05)

        # Value/reward parameters
        self.value_answer_correct = config.get('value_answer_correct', 1.0)
        self.value_answer_wrong = config.get('value_answer_wrong', -0.5)
        self.value_search_success = config.get('value_search_success', 0.7)

        # Very aggressive confidence thresholds for current system state
        self.confidence_threshold_answer = config.get('confidence_threshold_answer', 0.15)  # Even lower
        self.confidence_threshold_search = config.get('confidence_threshold_search', 0.10)  # Even lower

        logger.info("🎯 DecisionPolicy initialized with EIG calculations")

    def decide_action(self,
                     posterior: PosteriorBeliefs,
                     observed: ObservedFeatures,
                     candidates: List[Dict[str, Any]],
                     context: Dict[str, Any]) -> DecisionResult:
        """
        Main decision method using Expected Information Gain

        Args:
            posterior: Current beliefs q(v|u)
            observed: Observed features u
            candidates: Candidate subgraphs with likelihoods
            context: Additional context (session, history, etc.)

        Returns:
            DecisionResult with chosen action and analysis
        """
        logger.debug("🎯 Computing Expected Information Gain for decision making")

        # Generate action candidates
        action_candidates = self._generate_action_candidates(posterior, observed, candidates, context)

        # Select best action based on expected utility
        best_action = max(action_candidates, key=lambda x: x.expected_utility)

        # Create detailed analysis
        eig_analysis = {
            'entropy_subgraph': posterior.entropy_subgraph,
            'entropy_checklist': posterior.entropy_checklist,
            'confidence_top': posterior.confidence_top,
            'num_candidates': posterior.num_candidates,
            'novelty_score': posterior.z_novelty
        }

        # Generate reasoning
        reasoning = self._generate_reasoning(best_action, eig_analysis, context)

        result = DecisionResult(
            action=best_action.action,
            target=best_action.target,
            confidence=best_action.confidence,
            reasoning=reasoning,
            eig_analysis=eig_analysis,
            action_candidates=action_candidates
        )

        logger.info(f"🎯 Decision: {result.action} ({result.confidence:.2f}) - {result.reasoning}")

        return result

    def _generate_action_candidates(self,
                                   posterior: PosteriorBeliefs,
                                   observed: ObservedFeatures,
                                   candidates: List[Dict[str, Any]],
                                   context: Dict[str, Any]) -> List[ActionCandidate]:
        """
        Generate and evaluate all possible action candidates

        Returns list of ActionCandidate objects with EIG calculations
        """
        candidates_list = []

        # 1. ANSWER action - use current best candidate
        # Adjust threshold based on step priors and procedure state
        answer_threshold = self._adjust_answer_threshold(posterior, context)

        logger.debug(f"Decision analysis: confidence={posterior.confidence_top:.3f}, "
                    f"threshold={answer_threshold:.3f}, step_priors={posterior.z_step}")

        if posterior.confidence_top > answer_threshold:
            answer_action = self._evaluate_answer_action(posterior, candidates, context)
            candidates_list.append(answer_action)
            logger.debug(f"Added ANSWER action: {answer_action.action} -> {answer_action.target}")
        else:
            logger.debug(f"ANSWER threshold not met: {posterior.confidence_top:.3f} < {answer_threshold:.3f}")

        # 2. ASK actions - for uncertain slots
        ask_actions = self._evaluate_ask_actions(posterior, observed, candidates, context)
        candidates_list.extend(ask_actions)

        # 3. SEARCH action - if high novelty or low confidence
        if (posterior.z_novelty > 0.5 or
            posterior.confidence_top < self.confidence_threshold_search):
            search_action = self._evaluate_search_action(posterior, observed, context)
            candidates_list.append(search_action)

        # Ensure we always have at least one action
        if not candidates_list:
            candidates_list.append(self._create_fallback_action())

        return candidates_list

    def _adjust_answer_threshold(self, posterior: PosteriorBeliefs, context: Dict[str, Any]) -> float:
        """
        Adjust answer threshold based on procedure state and step priors
        """
        base_threshold = self.confidence_threshold_answer

        # If we're in a procedure and approaching completion, lower threshold
        if posterior.z_step:
            step_keys = list(posterior.z_step.keys())
            if any('procedure_complete' in step for step in step_keys):
                procedure_complete_prob = posterior.z_step.get('procedure_complete', 0.0)
                # Lower threshold as we approach completion
                base_threshold = max(0.4, base_threshold - procedure_complete_prob * 0.3)

        # If we have very clear winner (high margin), lower threshold
        if posterior.confidence_top > 0.5:
            base_threshold = max(0.5, base_threshold)

        logger.debug(f"Adjusted answer threshold: {base_threshold:.2f} (base: {self.confidence_threshold_answer:.2f})")
        return base_threshold

    def _evaluate_answer_action(self,
                               posterior: PosteriorBeliefs,
                               candidates: List[Dict[str, Any]],
                               context: Dict[str, Any]) -> ActionCandidate:
        """
        Evaluate ANSWER action using current posterior
        """
        # Get top candidate
        if posterior.z_subgraph:
            top_subgraph_id = max(posterior.z_subgraph.items(), key=lambda x: x[1])[0]
            top_candidate = next((c for c in candidates if c['id'] == top_subgraph_id), None)
        else:
            top_candidate = candidates[0] if candidates else None

        # Calculate expected value (accuracy * reward)
        accuracy = posterior.confidence_top
        expected_value = accuracy * self.value_answer_correct + (1 - accuracy) * self.value_answer_wrong

        # ANSWER has low EIG (no new information) but high immediate value
        eig = 0.1  # Small information gain from confirming answer
        cost = self.cost_answer

        expected_utility = expected_value + eig - cost

        return ActionCandidate(
            action='ANSWER',
            target=top_candidate['id'] if top_candidate else 'best_guess',
            expected_utility=expected_utility,
            eig=eig,
            value=expected_value,
            cost=cost,
            confidence=posterior.confidence_top
        )

    def _evaluate_ask_actions(self,
                             posterior: PosteriorBeliefs,
                             observed: ObservedFeatures,
                             candidates: List[Dict[str, Any]],
                             context: Dict[str, Any]) -> List[ActionCandidate]:
        """
        Evaluate ASK actions for uncertain slots

        This implements the core EIG calculation for questions
        """
        ask_actions = []

        # Get uncertain slots (high entropy)
        uncertain_slots = self._identify_uncertain_slots(posterior, context)

        for slot_name in uncertain_slots[:3]:  # Limit to top 3 slots
            eig = self._calculate_slot_eig(slot_name, posterior, candidates, observed)

            # Expected value of asking (reduced future uncertainty)
            expected_value = eig * 0.8  # Some value from clarification
            cost = self.cost_ask

            expected_utility = expected_value + eig - cost

            # Confidence in this being the right question to ask
            slot_confidence = self._calculate_question_confidence(slot_name, posterior)

            ask_actions.append(ActionCandidate(
                action='ASK',
                target=slot_name,
                expected_utility=expected_utility,
                eig=eig,
                value=expected_value,
                cost=cost,
                confidence=slot_confidence
            ))

        return ask_actions

    def _evaluate_search_action(self,
                               posterior: PosteriorBeliefs,
                               observed: ObservedFeatures,
                               context: Dict[str, Any]) -> ActionCandidate:
        """
        Evaluate SEARCH action for high-novelty situations
        """
        # EIG from search: expected entropy reduction
        current_entropy = posterior.entropy_subgraph
        expected_entropy_after_search = current_entropy * 0.4  # Assume 60% reduction
        eig = current_entropy - expected_entropy_after_search

        # Expected value: success probability * value
        success_prob = 0.7  # Assume 70% search success rate
        expected_value = success_prob * self.value_search_success

        cost = self.cost_search
        expected_utility = expected_value + eig - cost

        return ActionCandidate(
            action='SEARCH',
            target='web_query',  # Could be more specific
            expected_utility=expected_utility,
            eig=eig,
            value=expected_value,
            cost=cost,
            confidence=success_prob
        )

    def _calculate_slot_eig(self,
                           slot_name: str,
                           posterior: PosteriorBeliefs,
                           candidates: List[Dict[str, Any]],
                           observed: ObservedFeatures) -> float:
        """
        Calculate Expected Information Gain for asking about a specific slot

        EIG = H[current] - E[ H[posterior after answer] ]

        This implements the core mathematical formula from the theory.
        """
        # Current entropy over subgraphs
        current_entropy = posterior.entropy_subgraph

        # Simulate possible answers and their effects
        possible_answers = self._get_possible_slot_answers(slot_name, candidates)

        expected_entropy_after = 0.0

        for answer, probability in possible_answers.items():
            # Simulate how this answer would update posteriors
            simulated_posterior = self._simulate_answer_effect(
                slot_name, answer, posterior, candidates, observed
            )

            # Weight by answer probability
            expected_entropy_after += probability * simulated_posterior.entropy_subgraph

        # EIG = entropy reduction
        eig = current_entropy - expected_entropy_after

        return max(eig, 0.0)  # Ensure non-negative

    def _identify_uncertain_slots(self,
                                 posterior: PosteriorBeliefs,
                                 context: Dict[str, Any]) -> List[str]:
        """
        Identify slots with high uncertainty (high entropy or unknown values)
        Now considers step priors and procedure state
        """
        uncertain_slots = []

        # Priority 1: Current step focus from procedure
        if posterior.z_step:
            current_step = max(posterior.z_step.items(), key=lambda x: x[1])[0]
            if current_step.startswith("collect_"):
                step_slot = current_step.replace("collect_", "")
                uncertain_slots.append(step_slot)
                logger.debug(f"Step priority slot: {step_slot}")

        # Priority 2: Check slot posteriors if available
        if posterior.z_slots:
            for slot_name, slot_posterior in posterior.z_slots.items():
                # High entropy indicates uncertainty
                slot_entropy = self._compute_entropy(slot_posterior)
                if slot_entropy > 0.8:  # Threshold for "uncertain"
                    uncertain_slots.append(slot_name)

        # If no slot posteriors, use checklist context to identify potential slots
        if not uncertain_slots and posterior.z_checklist:
            top_checklist = max(posterior.z_checklist.items(), key=lambda x: x[1])[0]
            if 'Verify' in top_checklist:
                # For VerifyMusicRights, these are typical uncertain slots
                uncertain_slots = ['film', 'music_track', 'composer', 'territory_clearance']

        return uncertain_slots

    def _get_possible_slot_answers(self,
                                  slot_name: str,
                                  candidates: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Get possible answers for a slot from candidates
        """
        possible_answers = {}

        for candidate in candidates:
            # Extract slot values from candidate
            slot_values = self._extract_slot_values_from_candidate(slot_name, candidate)

            for value, confidence in slot_values.items():
                if value in possible_answers:
                    possible_answers[value] += confidence * candidate.get('posterior_prob', 0.1)
                else:
                    possible_answers[value] = confidence * candidate.get('posterior_prob', 0.1)

        # Add "unknown" as a possibility
        total_known = sum(possible_answers.values())
        possible_answers['unknown'] = max(0.1, 1.0 - total_known)

        # Normalize
        total = sum(possible_answers.values())
        if total > 0:
            possible_answers = {k: v/total for k, v in possible_answers.items()}

        return possible_answers

    def _simulate_answer_effect(self,
                               slot_name: str,
                               answer: str,
                               posterior: PosteriorBeliefs,
                               candidates: List[Dict[str, Any]],
                               observed: ObservedFeatures) -> PosteriorBeliefs:
        """
        Simulate how answering a question would affect posteriors

        This is a simplified simulation for EIG calculation
        """
        # Simulate posterior update based on hypothetical answer
        simulated_subgraph = posterior.z_subgraph.copy()

        # Boost candidates that would be consistent with this answer
        for candidate_id, prob in simulated_subgraph.items():
            candidate = next((c for c in candidates if c['id'] == candidate_id), None)
            if candidate:
                consistency = self._check_answer_consistency(slot_name, answer, candidate)
                simulated_subgraph[candidate_id] = prob * (1.0 + consistency)

        # Renormalize
        total = sum(simulated_subgraph.values())
        if total > 0:
            simulated_subgraph = {k: v/total for k, v in simulated_subgraph.items()}

        # Create simulated posterior with reduced entropy
        simulated_entropy = self._compute_entropy(simulated_subgraph)

        # Return modified posterior beliefs
        simulated_posterior = PosteriorBeliefs(
            z_checklist=posterior.z_checklist,
            z_goal=posterior.z_goal,
            z_subgraph=simulated_subgraph,
            z_slots=posterior.z_slots,
            z_step=posterior.z_step,
            z_dialogue_act=posterior.z_dialogue_act,
            z_novelty=posterior.z_novelty,
            entropy_checklist=posterior.entropy_checklist,
            entropy_subgraph=simulated_entropy,
            confidence_top=max(simulated_subgraph.values()) if simulated_subgraph else 0.0,
            num_candidates=posterior.num_candidates
        )

        return simulated_posterior

    def _extract_slot_values_from_candidate(self,
                                          slot_name: str,
                                          candidate: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract possible slot values from a candidate subgraph
        """
        # This would analyze the candidate's structure to find possible slot fillers
        # Simplified implementation
        slot_values = {}

        # Mock extraction based on candidate properties
        if slot_name == 'film':
            # Look for film entities in candidate
            candidate_name = candidate.get('name', '')
            if 'film' in candidate_name.lower() or 'movie' in candidate_name.lower():
                slot_values[candidate_name] = 0.8
        elif slot_name == 'music_track':
            candidate_name = candidate.get('name', '')
            if 'music' in candidate_name.lower() or 'track' in candidate_name.lower():
                slot_values[candidate_name] = 0.7

        # Add unknown option
        slot_values['unknown'] = 0.3

        return slot_values

    def _check_answer_consistency(self,
                                 slot_name: str,
                                 answer: str,
                                 candidate: Dict[str, Any]) -> float:
        """
        Check how consistent an answer is with a candidate
        """
        if answer == 'unknown':
            return 0.5  # Neutral consistency

        candidate_name = candidate.get('name', '').lower()
        answer_lower = answer.lower()

        # Simple consistency check
        if answer_lower in candidate_name or candidate_name in answer_lower:
            return 0.8  # High consistency
        else:
            return 0.2  # Low consistency

    def _compute_entropy(self, distribution: Dict[str, float]) -> float:
        """Compute Shannon entropy of distribution"""
        if not distribution:
            return 0.0

        probs = [p for p in distribution.values() if p > 1e-10]
        if not probs:
            return 0.0

        return -sum(p * np.log(p) for p in probs)

    def _calculate_question_confidence(self, slot_name: str, posterior: PosteriorBeliefs) -> float:
        """Calculate confidence that asking about this slot is the right decision"""
        # Simplified confidence based on slot uncertainty
        if posterior.z_slots and slot_name in posterior.z_slots:
            slot_entropy = self._compute_entropy(posterior.z_slots[slot_name])
            return min(1.0, slot_entropy / 2.0)  # Convert entropy to confidence
        else:
            return 0.6  # Default confidence

    def _create_fallback_action(self) -> ActionCandidate:
        """Create fallback action when no good options available"""
        return ActionCandidate(
            action='ASK',
            target='clarification',
            expected_utility=0.0,
            eig=0.0,
            value=0.0,
            cost=self.cost_ask,
            confidence=0.0
        )

    def _generate_reasoning(self,
                           best_action: ActionCandidate,
                           eig_analysis: Dict[str, float],
                           context: Dict[str, Any]) -> str:
        """
        Generate human-readable reasoning for the decision
        """
        if best_action.action == 'ANSWER':
            return ".2f"
        elif best_action.action == 'ASK':
            return ".2f"
        elif best_action.action == 'SEARCH':
            return ".2f"
        else:
            return f"Selected {best_action.action} action with confidence {best_action.confidence:.2f}"
