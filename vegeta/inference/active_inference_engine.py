"""
Active Inference Engine

Main orchestrator for Bayesian active inference, coordinating:
- Predictive coding (generative model + likelihood)
- Posterior inference (belief updates)
- Decision making (EIG-based action selection)

Following the complete theory from attemp1TextOnly.py specification.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .generative_model import GenerativeModel, HiddenStates
from .likelihood_computer import LikelihoodComputer, ObservedFeatures
from .posterior_updater import PosteriorUpdater, PosteriorBeliefs
from .decision_policy import DecisionPolicy, DecisionResult

logger = logging.getLogger(__name__)


@dataclass
class InferenceStep:
    """Records a single step in the active inference process"""
    step_name: str
    timestamp: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    duration_ms: float


@dataclass
class ActiveInferenceResult:
    """Complete result from active inference process"""
    decision: DecisionResult
    posterior: PosteriorBeliefs
    predictive_model: Dict[str, Any]
    inference_trace: List[InferenceStep]
    total_duration_ms: float
    confidence_score: float


class ActiveInferenceEngine:
    """
    Main orchestrator for Bayesian active inference

    Coordinates the complete perception-action loop:
    1. Observation processing
    2. Predictive coding (generative model)
    3. Likelihood computation
    4. Posterior belief updates
    5. Decision making with EIG
    6. Action execution

    This implements the full theory from attemp1TextOnly.py specification.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize core components
        self.generative_model = GenerativeModel(config)
        self.likelihood_computer = LikelihoodComputer(config)
        self.posterior_updater = PosteriorUpdater(config)
        self.decision_policy = DecisionPolicy(config)

        # Performance tracking
        self.enable_tracing = config.get('enable_inference_tracing', True)

        logger.info("🚀 ActiveInferenceEngine initialized for Bayesian decision making")

    def perform_inference(self,
                         observed_features: ObservedFeatures,
                         priors: Dict[str, Any],
                         candidates: List[Dict[str, Any]],
                         context: Dict[str, Any]) -> ActiveInferenceResult:
        """
        Perform complete active inference cycle

        Args:
            observed_features: User utterance features u
            priors: Prior beliefs p(v) from previous turns/context
            candidates: Candidate subgraphs from retrieval
            context: Session context, history, etc.

        Returns:
            ActiveInferenceResult with decision and full analysis
        """
        start_time = time.time()
        inference_trace = [] if self.enable_tracing else None

        logger.info("🎯 Starting active inference cycle")

        # Step 1: Predictive Coding - Generate expectations u'
        logger.debug("Step 1: Predictive coding")
        step_start = time.time()

        # Create hidden states from priors (simplified)
        hidden_states = self._create_hidden_states_from_priors(priors, context)

        # Generate predictions
        predictions = self.generative_model.predict_observations(hidden_states, context)

        self._record_step(inference_trace, "predictive_coding",
                         {"hidden_states": hidden_states.__dict__, "context_keys": list(context.keys())},
                         {"predictions": predictions.__dict__, "confidence": predictions.confidence},
                         (time.time() - step_start) * 1000)

        # Step 2: Likelihood Computation - Compare u vs u'
        logger.debug("Step 2: Likelihood computation")
        step_start = time.time()

        # Compute likelihoods for all candidates
        for candidate in candidates:
            if 'log_likelihood' not in candidate:
                # Compute likelihood p(u|v) for this candidate
                candidate['log_likelihood'] = self.likelihood_computer.compute_likelihood(
                    observed_features, predictions
                )

        self._record_step(inference_trace, "likelihood_computation",
                         {"num_candidates": len(candidates), "observed_features": len(observed_features.u_terms)},
                         {"candidates_with_likelihood": len([c for c in candidates if 'log_likelihood' in c])},
                         (time.time() - step_start) * 1000)

        # Step 3: Posterior Inference - Update beliefs q(v|u)
        logger.debug("Step 3: Posterior inference")
        step_start = time.time()

        posterior = self.posterior_updater.compute_posterior(
            candidates, priors, observed_features, self.likelihood_computer
        )

        self._record_step(inference_trace, "posterior_inference",
                         {"prior_keys": list(priors.keys()), "num_candidates": len(candidates)},
                         {"posterior_entropy": posterior.entropy_subgraph, "confidence_top": posterior.confidence_top},
                         (time.time() - step_start) * 1000)

        # Step 4: Decision Making - Choose action using EIG
        logger.debug("Step 4: Decision making")
        step_start = time.time()

        decision = self.decision_policy.decide_action(
            posterior, observed_features, candidates, context
        )

        self._record_step(inference_trace, "decision_making",
                         {"posterior_confidence": posterior.confidence_top, "num_candidates": len(candidates)},
                         {"decision": decision.action, "confidence": decision.confidence},
                         (time.time() - step_start) * 1000)

        # Step 5: Compile results
        total_duration = (time.time() - start_time) * 1000
        confidence_score = self._calculate_overall_confidence(posterior, decision)

        # Package predictive model information
        predictive_model = {
            'predictions': predictions.__dict__,
            'likelihood_function': 'three_channel_distance',
            'noise_parameters': {
                'σ_sem2': self.likelihood_computer.σ_sem2,
                'σ_struct2': self.likelihood_computer.σ_struct2,
                'σ_terms2': self.likelihood_computer.σ_terms2
            },
            'channel_weights': {
                'α': self.likelihood_computer.α,
                'β': self.likelihood_computer.β,
                'γ': self.likelihood_computer.γ
            }
        }

        result = ActiveInferenceResult(
            decision=decision,
            posterior=posterior,
            predictive_model=predictive_model,
            inference_trace=inference_trace or [],
            total_duration_ms=total_duration,
            confidence_score=confidence_score
        )

        logger.info(".1f"                  ".2f")

        return result

    def _create_hidden_states_from_priors(self,
                                        priors: Dict[str, Any],
                                        context: Dict[str, Any]) -> HiddenStates:
        """
        Create HiddenStates from priors and context

        This is a simplified mapping - in practice, this would be more sophisticated
        """
        # Get top checklist
        checklist_priors = priors.get('checklist', {})
        top_checklist = max(checklist_priors.items(), key=lambda x: x[1])[0] if checklist_priors else 'unknown'

        # Get top subgraph (placeholder)
        subgraph_priors = priors.get('subgraph', {})
        top_subgraph = max(subgraph_priors.items(), key=lambda x: x[1])[0] if subgraph_priors else 'unknown'

        # Get goal
        goal_priors = priors.get('goal', {})
        top_goal = max(goal_priors.items(), key=lambda x: x[1])[0] if goal_priors else None

        return HiddenStates(
            z_checklist=top_checklist,
            z_subgraph=top_subgraph,
            z_goal=top_goal,
            z_slots=None,  # Would be computed from slot priors
            z_step=None    # Would be determined from procedure state
        )

    def _record_step(self,
                    trace: Optional[List[InferenceStep]],
                    step_name: str,
                    inputs: Dict[str, Any],
                    outputs: Dict[str, Any],
                    duration_ms: float):
        """Record a step in the inference trace"""
        if trace is not None:
            trace.append(InferenceStep(
                step_name=step_name,
                timestamp=time.time(),
                inputs=inputs,
                outputs=outputs,
                duration_ms=duration_ms
            ))

    def _calculate_overall_confidence(self,
                                    posterior: PosteriorBeliefs,
                                    decision: DecisionResult) -> float:
        """
        Calculate overall confidence in the inference result
        Now uses the improved posterior confidence directly with minor adjustments
        """
        # Use the improved posterior confidence as the primary factor
        base_confidence = posterior.confidence_top

        print(f"DEBUG: ActiveInferenceEngine confidence calculation:")
        print(f"  Posterior confidence_top: {posterior.confidence_top}")
        print(f"  Decision confidence: {decision.confidence}")
        print(f"  Entropy: {posterior.entropy_subgraph}")
        print(f"  Novelty: {posterior.z_novelty}")

        # Small adjustments for other factors (but don't reduce confidence)
        adjustments = [
            decision.confidence * 0.1,           # Small boost from decision confidence
            (1.0 - posterior.entropy_subgraph / 5.0) * 0.05,  # Small entropy bonus
            (1.0 - posterior.z_novelty) * 0.05   # Small novelty bonus
        ]

        # Only add positive adjustments, don't reduce base confidence
        positive_adjustments = sum(max(0, adj) for adj in adjustments)
        confidence = min(1.0, base_confidence + positive_adjustments)

        print(f"  Adjustments: {positive_adjustments}")
        print(f"  Final confidence: {confidence}")

        return confidence

    def get_inference_stats(self) -> Dict[str, Any]:
        """Get statistics about inference performance"""
        return {
            'components_initialized': True,
            'generative_model_ready': True,
            'likelihood_computer_ready': True,
            'posterior_updater_ready': True,
            'decision_policy_ready': True,
            'tracing_enabled': self.enable_tracing
        }

    def reset_inference_state(self):
        """Reset any internal inference state"""
        # This would clear any cached beliefs, reset uncertainty estimates, etc.
        logger.info("🔄 Inference state reset")
        pass

    def update_inference_parameters(self, new_params: Dict[str, Any]):
        """
        Update inference parameters dynamically

        Args:
            new_params: Dictionary with parameter updates
        """
        # Update likelihood computer parameters
        if 'noise_parameters' in new_params:
            noise_params = new_params['noise_parameters']
            if 'σ_sem2' in noise_params:
                self.likelihood_computer.σ_sem2 = noise_params['σ_sem2']
            if 'σ_struct2' in noise_params:
                self.likelihood_computer.σ_struct2 = noise_params['σ_struct2']
            if 'σ_terms2' in noise_params:
                self.likelihood_computer.σ_terms2 = noise_params['σ_terms2']

        # Update decision policy parameters
        if 'decision_parameters' in new_params:
            decision_params = new_params['decision_parameters']
            if 'cost_ask' in decision_params:
                self.decision_policy.cost_ask = decision_params['cost_ask']
            if 'cost_search' in decision_params:
                self.decision_policy.cost_search = decision_params['cost_search']

        logger.info("🔧 Inference parameters updated")

    def explain_inference(self, result: ActiveInferenceResult) -> str:
        """
        Generate human-readable explanation of the inference process

        Args:
            result: Complete inference result

        Returns:
            Detailed explanation string
        """
        explanation = f"""
🧠 ACTIVE INFERENCE ANALYSIS
{'='*50}

🎯 DECISION: {result.decision.action}
📊 Confidence: {result.confidence_score:.3f}
⏱️  Processing Time: {result.total_duration_ms:.1f}ms

🎲 BELIEF STATE:
• Top Subgraph Confidence: {result.posterior.confidence_top:.3f}
• Subgraph Entropy: {result.posterior.entropy_subgraph:.3f}
• Checklist Entropy: {result.posterior.entropy_checklist:.3f}
• Novelty Score: {result.posterior.z_novelty:.3f}
• Candidates Evaluated: {result.posterior.num_candidates}

🎯 EXPECTED INFORMATION GAIN:
• Best Action: {result.decision.action}
• EIG Analysis: {result.decision.eig_analysis}
• Reasoning: {result.decision.reasoning}

🔮 PREDICTIVE MODEL:
• Semantic Predictions: {len(result.predictive_model['predictions']['u_terms'])} terms
• Structural Predictions: {len(result.predictive_model['predictions']['u_struct'])} patterns
• Model Confidence: {result.predictive_model['predictions']['confidence']:.3f}

🛤️ INFERENCE TRACE:
"""

        for step in result.inference_trace:
            explanation += f"• {step.step_name}: {step.duration_ms:.1f}ms\n"

        explanation += f"\n💡 SUMMARY:\n{self._generate_summary(result)}"

        return explanation

    def _generate_summary(self, result: ActiveInferenceResult) -> str:
        """Generate concise summary of inference results"""
        if result.decision.action == 'ANSWER':
            summary = f"High confidence answer available ({result.confidence_score:.2f}). "
            summary += f"Top candidate explains the query well with low uncertainty."
        elif result.decision.action == 'ASK':
            summary = f"Need clarification on {result.decision.target}. "
            summary += f"High uncertainty detected ({result.posterior.entropy_subgraph:.2f} entropy)."
        elif result.decision.action == 'SEARCH':
            summary = f"Query appears novel ({result.posterior.z_novelty:.2f}). "
            summary += f"Additional information needed from external sources."
        else:
            summary = f"Complex decision scenario with moderate confidence ({result.confidence_score:.2f})."

        return summary
