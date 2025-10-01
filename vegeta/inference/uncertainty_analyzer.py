"""
Uncertainty analysis and decision making using Expected Information Gain
"""

import logging
import numpy as np
from typing import Dict, Any, List

from ..utils.llm_client import LLMClient
from ..core.config import Config

logger = logging.getLogger(__name__)

class UncertaintyAnalyzer:
    """
    Analyze uncertainty and make ASK/ANSWER/SEARCH decisions using EIG
    """
    
    def __init__(self, llm_client: LLMClient, config: Config):
        self.llm_client = llm_client
        self.config = config
        self.defaults = config.system_defaults
    
    def make_decision(self, posteriors: Dict[str, Any],
                     candidates: List[Dict[str, Any]],
                     observation_u: Dict[str, Any],
                     retrieval_context: Dict[str, Any],
                     session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Decide between ANSWER, ASK, or SEARCH based on uncertainty and procedure-driven logic"""

        # CRITICAL: Check for procedure-driven checklists FIRST
        # This prevents the system from falling back to clarification for procedure queries
        procedure_state = session_context.get('procedure_state', {})
        checklist_posterior = posteriors.get('checklist', {})

        if procedure_state.get('active_checklist') or self._is_procedure_checklist_top(checklist_posterior):
            return self._handle_procedure_driven_decision(posteriors, session_context)

        # Fall back to standard decision logic for non-procedure queries
        return self._handle_standard_decision(posteriors, candidates, observation_u, retrieval_context, session_context)

    def _is_procedure_checklist_top(self, checklist_posterior: Dict[str, float]) -> bool:
        """Check if top checklist is procedure-driven (not AI-driven)"""

        if not checklist_posterior:
            return False

        top_checklist = max(checklist_posterior.items(), key=lambda x: x[1])[0]

        # Define which checklists are procedure-driven vs AI-driven
        procedure_driven_checklists = {'VerifyMusicRights'}  # Add more as needed

        return top_checklist in procedure_driven_checklists

    def _handle_procedure_driven_decision(self, posteriors: Dict[str, Any],
                                        session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle decision-making for procedure-driven checklists"""

        procedure_state = session_context.get('procedure_state', {})
        checklist_posterior = posteriors.get('checklist', {})
        step_posterior = posteriors.get('step', {})
        slot_priors = posteriors.get('slot', {})

        # Determine active checklist
        if procedure_state.get('active_checklist'):
            active_checklist = procedure_state['active_checklist']
        elif checklist_posterior:
            active_checklist = max(checklist_posterior.items(), key=lambda x: x[1])[0]
        else:
            return self._create_fallback_decision()

        # Get completed slots
        completed_slots = procedure_state.get('completed_slots', [])

        # Find missing required slots
        missing_slots = self._get_missing_required_slots(active_checklist, completed_slots, slot_priors)

        if not missing_slots:
            # All required slots complete - procedure finished
            return {
                'action': 'ANSWER',
                'target': f"{active_checklist}_complete",
                'confidence': 1.0,
                'margin': 1.0,
                'reasoning': f"Procedure {active_checklist} completed successfully",
                'entropy': 0.0
            }

        # Ask for the first missing slot (procedure-driven, not AI choice)
        next_slot = missing_slots[0]
        slot_confidence = self._calculate_slot_confidence(slot_priors.get(next_slot, {}))

        return {
            'action': 'ASK',
            'target': next_slot,
            'confidence': slot_confidence,
            'margin': 0.5,  # Fixed margin for procedure steps
            'reasoning': f"Procedure {active_checklist} requires {next_slot} (step {len(completed_slots) + 1}/{len(completed_slots) + len(missing_slots)})",
            'entropy': 1.0 if slot_confidence < 0.5 else 0.5
        }

    def _get_missing_required_slots(self, checklist_name: str,
                                   completed_slots: List[str],
                                   slot_priors: Dict[str, Dict[str, float]]) -> List[str]:
        """Get required slots that are still missing"""

        missing_slots = []

        try:
            # Query SlotSpecs for this checklist
            slot_specs = self.config.db_manager.execute_query("""
                MATCH (c:Checklist {name: $checklist_name})-[:REQUIRES]->(ss:SlotSpec)
                WHERE ss.required = true
                RETURN ss.name as slot_name
                ORDER BY ss.name
            """, {"checklist_name": checklist_name})

            for spec in slot_specs:
                slot_name = spec['slot_name']
                if slot_name not in completed_slots:
                    # Check if we have a non-unknown value with reasonable confidence
                    slot_prior = slot_priors.get(slot_name, {})
                    if not self._has_confident_slot_value(slot_prior):
                        missing_slots.append(slot_name)

        except Exception as e:
            logger.warning(f"Error getting missing slots for {checklist_name}: {e}")

        return missing_slots

    def _has_confident_slot_value(self, slot_prior: Dict[str, float]) -> bool:
        """Check if slot has a confident (non-unknown) value"""

        if not slot_prior:
            return False

        # Remove unknown and find best alternative
        known_values = {k: v for k, v in slot_prior.items() if k != 'unknown'}

        if not known_values:
            return False

        best_prob = max(known_values.values())
        return best_prob > 0.7  # Confidence threshold for "has value"

    def _calculate_slot_confidence(self, slot_prior: Dict[str, float]) -> float:
        """Calculate overall confidence for a slot"""

        if not slot_prior:
            return 0.0

        # Confidence is 1 - probability of unknown
        unknown_prob = slot_prior.get('unknown', 0.0)
        return 1.0 - unknown_prob

    def _handle_standard_decision(self, posteriors: Dict[str, Any],
                                candidates: List[Dict[str, Any]],
                                observation_u: Dict[str, Any],
                                retrieval_context: Dict[str, Any],
                                session_context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle standard (non-procedure) decision making"""

        # CRITICAL: Check if this is a response to a previous question
        conversation_response = self._check_conversation_context(observation_u, session_context)
        if conversation_response:
            return conversation_response

        # Get top candidate and confidence metrics
        if not posteriors.get('subgraph'):
            return self._create_fallback_decision()

        top_prob = max(posteriors['subgraph'].values())
        top_subgraph_id = max(posteriors['subgraph'].items(), key=lambda x: x[1])[0]
        top_candidate = next(c for c in candidates if c['id'] == top_subgraph_id)

        second_prob = sorted(posteriors['subgraph'].values(), reverse=True)[1] if len(posteriors['subgraph']) > 1 else 0.0
        margin = top_prob - second_prob

                # Calculate entropies
        entropies = self._calculate_entropies(posteriors)

        # CRITICAL: Check for procedure-driven checklists FIRST
        # This must happen before immediate clarification to avoid bypassing procedures
        procedure_state = session_context.get('procedure_state', {})
        checklist_posterior = posteriors.get('checklist', {})

        if procedure_state.get('active_checklist') or self._is_procedure_checklist_top(checklist_posterior):
            return self._handle_procedure_driven_decision(posteriors, session_context)

        # LLM-BASED FAST PATH: Check for immediate clarification needs (only for non-procedure queries)
        extraction = observation_u.get('u_meta', {}).get('extraction', {})
        query_analysis = extraction.get('query_analysis', {})

        # Only trigger immediate clarification if we don't have strong procedure-driven priors
        # Check if any procedure-driven checklist has significant probability
        procedure_driven_checklists = {'VerifyMusicRights'}

        has_procedure_potential = any(
            name in procedure_driven_checklists and checklist_posterior.get(name, 0) > 0.3
            for name in checklist_posterior.keys()
        ) if checklist_posterior else False

        # Only trigger immediate clarification if:
        # 1. No procedure-driven checklist has significant probability (>0.3)
        # 2. The query is truly vague AND needs clarification
        should_clarify_immediately = (
            not has_procedure_potential and
            query_analysis.get('immediate_clarification_needed', False) and
            query_analysis.get('clarity') in ['vague', 'extremely_vague']
        )

        if should_clarify_immediately:
            return self._handle_immediate_clarification(query_analysis)
        
        # EARLY ROUTING: Intent-based routing for recommendations
        if self._is_recommendation_intent(observation_u, posteriors, retrieval_context):
            return self._handle_recommendation_routing(retrieval_context, top_prob, margin)
        
        # Use LLM to assess evidence quality
        evidence_assessment = self._assess_evidence_quality_llm(top_candidate, posteriors, entropies, margin)
        
        # Check if we should ANSWER based on LLM assessment
        if evidence_assessment['decision_ready']:
            return {
                'action': 'ANSWER',
                'target': top_candidate.get('entity_name') or top_candidate.get('anchor_name'),
                'confidence': top_prob,
                'margin': margin,
                'reasoning': f"LLM assessment: {evidence_assessment['reasoning']}",
                'entropy': entropies.get('subgraph', 0.0),
                'evidence_assessment': evidence_assessment
            }
        
        # Otherwise choose between ASK and SEARCH
        return self._choose_ask_or_search(posteriors, candidates, observation_u, top_candidate, margin, entropies)
    
    def _calculate_entropies(self, posteriors: Dict) -> Dict[str, float]:
        """Calculate Shannon entropy for each posterior distribution"""
        
        entropies = {}
        
        for var_name, distribution in posteriors.items():
            if isinstance(distribution, dict):
                # Shannon entropy: H = -Σ p(x) log p(x)
                entropy = 0.0
                for prob in distribution.values():
                    if prob > 1e-10:  # Avoid log(0)
                        entropy -= prob * np.log2(prob)
                entropies[var_name] = entropy
            else:
                # Scalar value (e.g., novelty)
                entropies[var_name] = 0.0
        
        return entropies

    def _check_conversation_context(self, observation_u: Dict[str, Any],
                                  session_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if current input is a response to a previous question.
        This enables proper multi-turn conversation flow using Bayesian inference.
        """
        current_utterance = observation_u.get('u_meta', {}).get('utterance', '').strip().lower()
        recent_turns = session_context.get('recent_turns', [])

        if not recent_turns:
            return None

        # Get the most recent turn (what we just asked)
        previous_turn = recent_turns[-1]
        previous_question = previous_turn.get('system_response', '').strip()
        previous_target = previous_turn.get('target', '')

        logger.debug(f"Checking conversation context: previous='{previous_question}', current='{current_utterance}'")

        # Handle confirmation responses using Bayesian belief update
        if self._is_confirmation(current_utterance):
            return self._handle_bayesian_confirmation(previous_question, previous_target, observation_u)

        return None

    def _is_confirmation(self, utterance: str) -> bool:
        """Check if utterance is a confirmation (yes, yeah, correct, etc.)"""
        confirmations = ['yes', 'yeah', 'yep', 'correct', 'right', 'true', 'sure', 'definitely', 'absolutely', 'exactly']
        return utterance in confirmations

    def _handle_bayesian_confirmation(self, previous_question: str, previous_target: str,
                                    observation_u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle confirmation using proper Bayesian belief updates.
        This maintains the integrity of our graph-based inference approach.
        """
        logger.info(f"User confirmed: {previous_target} - updating belief state")

        # Instead of hardcoding responses, we enhance the observation to bias toward the confirmed entity
        # This allows the normal graph traversal and inference to naturally discover the information

        enhanced_observation = observation_u.copy()
        enhanced_meta = enhanced_observation.get('u_meta', {}).copy()

        # Add confirmation context that will influence downstream inference
        enhanced_meta['confirmation_context'] = {
            'confirmed_entity': previous_target,
            'confirmation_strength': 0.95,  # Very strong confirmation signal
            'previous_question': previous_question,
            'inference_hint': f"User confirmed they mean {previous_target} - strongly bias toward this entity in graph search"
        }

        # Enhance the utterance to include confirmation context
        original_utterance = enhanced_meta.get('utterance', '')
        enhanced_meta['utterance'] = f"{original_utterance} [CONFIRMED_CONTEXT: {previous_target}]"

        # Add semantic bias (this would ideally update embeddings, but we use metadata for now)
        enhanced_observation['u_meta'] = enhanced_meta
        enhanced_observation['confirmation_bias'] = {
            'target_entity': previous_target,
            'bias_strength': 0.95,
            'reason': 'User confirmation of previous clarification'
        }

        # The system will continue with normal inference but with strong bias toward confirmed entity
        # This allows graph-based discovery while respecting user confirmation
        return {
            'action': 'CONTINUE_INFERENCE',
            'target': previous_target,
            'confidence': 0.95,
            'margin': 0.9,
            'reasoning': f'User confirmed {previous_target} - proceeding with biased inference toward this entity',
            'enhanced_observation': enhanced_observation,
            'confirmation_bias': previous_target
        }

    def _handle_immediate_clarification(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle LLM-identified immediate clarification needs"""
        
        clarification_type = query_analysis.get('clarification_type', 'task_domain')
        
        clarification_responses = {
            'task_domain': "I'd be happy to help! Could you tell me what specific task you need assistance with?",
            'intent': "I'm having trouble understanding what you need. Could you provide more details about what you're trying to accomplish?",
            'specifics': "Could you give me more specific details about what you're looking for?"
        }
        
        quick_response = clarification_responses.get(clarification_type, 
                                                   "Could you provide more information about what you need?")
        
        return {
            'action': 'ASK',
            'target': clarification_type,
            'confidence': 0.0,
            'margin': 0.0,
            'reasoning': f'LLM analysis: {query_analysis.get("clarity", "vague")} query needs {clarification_type} clarification',
            'entropy': float('inf'),
            'quick_response': quick_response,
            'llm_analysis': query_analysis
        }
    
    def _is_recommendation_intent(self, observation_u: Dict, posteriors: Dict, retrieval_context: Dict) -> bool:
        """Check if this looks like a recommendation request"""
        
        canonical_terms = observation_u.get('u_meta', {}).get('extraction', {}).get('canonical_terms', [])
        has_recommend_term = any('recommend' in term.lower() for term in canonical_terms)
        has_similar_pattern = any(word in ' '.join(canonical_terms).lower() for word in ['similar', 'like', 'comparable'])
        
        top_checklist = max(posteriors.get('checklist', {"None": 1.0}).items(), key=lambda x: x[1])[0]
        
        return (has_recommend_term or has_similar_pattern or 'Recommend' in top_checklist)
    
    def _handle_recommendation_routing(self, retrieval_context: Dict, top_prob: float, margin: float) -> Dict[str, Any]:
        """Handle recommendation intent routing"""
        
        linked_entity_ids = retrieval_context.get('linked_entity_ids', [])
        
        if linked_entity_ids and top_prob < 0.95:
            linked_entity = linked_entity_ids[0]
            return {
                'action': 'SEARCH',
                'target': 'similar_items',
                'confidence': top_prob,
                'margin': margin,
                'reasoning': f"Recommendation intent with known entity ({linked_entity}) - searching for similar items",
                'entropy': 1.0  # High uncertainty, need search
            }
        
        # Fallback to asking for preferences
        return {
            'action': 'ASK',
            'target': 'preferences',
            'confidence': top_prob,
            'margin': margin,
            'reasoning': "Recommendation intent detected but need to understand user preferences",
            'entropy': 1.5
        }
    
    def _assess_evidence_quality_llm(self, top_candidate: Dict, posteriors: Dict, entropies: Dict, margin: float) -> Dict[str, Any]:
        """Use LLM to assess if current evidence is sufficient for making a decision"""
        
        top_prob = max(posteriors['subgraph'].values())
        second_prob = sorted(posteriors['subgraph'].values(), reverse=True)[1] if len(posteriors['subgraph']) > 1 else 0.0
        
        analysis_prompt = f"""Assess whether we have sufficient evidence to give a confident answer. Return valid JSON only:

{{
    "decision_ready": true/false,
    "confidence_sufficient": true/false,
    "margin_sufficient": true/false,
    "recommended_action": "answer|ask|search",
    "reasoning": "brief explanation of assessment"
}}

Evidence Assessment:
- Top candidate: {top_candidate.get('entity_name', top_candidate.get('anchor_name', 'unknown'))}
- Top probability: {top_prob:.3f}
- Second probability: {second_prob:.3f}
- Margin between top 2: {margin:.3f}
- Total candidates: {len(posteriors['subgraph'])}
- Uncertainty (entropy): {entropies.get('subgraph', 0.0):.3f}

Guidelines:
- decision_ready: true if evidence strongly points to one answer
- confidence_sufficient: true if top probability indicates strong evidence
- margin_sufficient: true if there's clear separation between top candidates
- Consider both the strength of evidence AND the clarity of the winner

JSON:"""
        
        try:
            result = self.llm_client.call_ollama_json(analysis_prompt)
            return {
                'decision_ready': result.get('decision_ready', False),
                'confidence_sufficient': result.get('confidence_sufficient', False), 
                'margin_sufficient': result.get('margin_sufficient', False),
                'recommended_action': result.get('recommended_action', 'ask'),
                'reasoning': result.get('reasoning', 'Evidence assessment completed')
            }
        except Exception as e:
            logger.error(f"Evidence quality assessment failed: {e}")
            # Fallback to conservative thresholds
            return {
                'decision_ready': top_prob >= 0.70 and margin >= 0.20,
                'confidence_sufficient': top_prob >= 0.70,
                'margin_sufficient': margin >= 0.20, 
                'recommended_action': 'answer' if (top_prob >= 0.70 and margin >= 0.20) else 'ask',
                'reasoning': 'LLM assessment failed, using fallback thresholds'
            }
    
    def _choose_ask_or_search(self, posteriors: Dict, candidates: List[Dict], observation_u: Dict, 
                             top_candidate: Dict, margin: float, entropies: Dict) -> Dict[str, Any]:
        """Choose between ASK and SEARCH using simplified EIG"""
        
        # Simplified decision logic - would be more sophisticated in full implementation
        
        # If we have specific entities and low confidence, ASK about specifics
        if top_candidate.get('entity_name') and posteriors['subgraph'][top_candidate['id']] < 0.6:
            return {
                'action': 'ASK',
                'target': 'specifics',
                'confidence': posteriors['subgraph'][top_candidate['id']],
                'margin': margin,
                'reasoning': f"Low confidence in {top_candidate.get('entity_name', 'candidate')}, asking for specifics",
                'entropy': entropies.get('subgraph', 0.0)
            }
        
        # If uncertainty is high and we have multiple candidates, SEARCH for more info
        if entropies.get('subgraph', 0.0) > 2.0 and len(candidates) > 3:
            return {
                'action': 'SEARCH',
                'target': 'missing_facts',
                'confidence': posteriors['subgraph'][top_candidate['id']],
                'margin': margin,
                'reasoning': "High uncertainty with multiple candidates, searching for more facts",
                'entropy': entropies.get('subgraph', 0.0)
            }
        
        # Default to ASK
        return {
            'action': 'ASK',
            'target': 'clarification',
            'confidence': posteriors['subgraph'][top_candidate['id']],
            'margin': margin,
            'reasoning': "Moderate uncertainty, asking for clarification",
            'entropy': entropies.get('subgraph', 0.0)
        }
    
    def _create_fallback_decision(self) -> Dict[str, Any]:
        """Create fallback decision when no candidates available"""
        return {
            'action': 'ASK',
            'target': 'clarification',
            'confidence': 0.0,
            'margin': 0.0,
            'reasoning': 'No candidates found, asking for clarification',
            'entropy': float('inf')
        }
