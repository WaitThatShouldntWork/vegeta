"""
Bayesian prior construction
"""

import logging
from typing import Dict, Any, List

from ..utils.llm_client import LLMClient
from ..utils.database import DatabaseManager
from ..core.config import Config
from ..extraction.query_analyzer import QueryAnalyzer

logger = logging.getLogger(__name__)

class PriorBuilder:
    """
    Build Bayesian priors over hidden states
    """
    
    def __init__(self, llm_client: LLMClient, db_manager: DatabaseManager, config: Config):
        self.llm_client = llm_client
        self.db_manager = db_manager
        self.config = config
        self.query_analyzer = QueryAnalyzer(llm_client, config)
    
    def build_priors(self, observation_u: Dict[str, Any],
                    session_context: Dict[str, Any],
                    retrieval_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build all priors for the current turn using Bayesian filtering from previous beliefs"""

        # Get basic dialogue act classification
        utterance = observation_u.get('u_meta', {}).get('utterance', '')
        dialogue_act_prior = self.query_analyzer.classify_dialogue_act_llm(utterance)

        # Get previous posterior for Bayesian filtering (belief carryover)
        previous_posterior = session_context.get('belief_state', {}).get('posterior', {})
        inertia_rho = session_context.get('inertia_rho', 0.7)  # Belief carryover strength

        # Build checklist prior with belief carryover
        checklist_prior = self._build_checklist_prior_with_carryover(
            observation_u, dialogue_act_prior, previous_posterior, inertia_rho
        )

        # Build goal prior using LLM analysis + belief carryover
        extraction = observation_u.get('u_meta', {}).get('extraction', {})
        goal_prior = self.query_analyzer.classify_user_intent_llm(
            utterance, dialogue_act_prior, extraction
        )
        goal_prior = self._apply_belief_carryover(goal_prior, previous_posterior.get('goal', {}), inertia_rho)

        # Build subgraph prior with belief carryover
        candidates = retrieval_context.get('candidates', [])
        subgraph_prior = self._build_subgraph_prior_with_carryover(
            candidates, previous_posterior, inertia_rho
        )

        # Build step prior (MISSING FROM CURRENT SYSTEM!)
        step_prior = self._build_step_prior(checklist_prior, goal_prior, session_context)

        # Build slot priors (MISSING FROM CURRENT SYSTEM!)
        slot_priors = self._build_slot_priors(candidates, step_prior, checklist_prior)

        # Build novelty prior
        novelty_prior = self._build_novelty_prior(retrieval_context['anchors'], observation_u)

        # Check if we need to initialize procedure state for procedure-driven checklists
        current_procedure_state = session_context.get('procedure_state', {})

        # If VerifyMusicRights becomes the top checklist and no procedure is active, initialize it
        top_checklist = max(checklist_prior.items(), key=lambda x: x[1])[0] if checklist_prior else None
        if (top_checklist == 'VerifyMusicRights' and
            not current_procedure_state.get('active_checklist')):
            logger.info(f"Initializing VerifyMusicRights procedure for session {session_context.get('session_id', 'unknown')}")

        return {
            'checklist': checklist_prior,
            'goal': goal_prior,
            'subgraph': subgraph_prior,
            'step': step_prior,
            'slot': slot_priors,
            'dialogue_act': dialogue_act_prior,
            'novelty': novelty_prior
        }

    def _build_checklist_prior_with_carryover(self, observation_u: Dict[str, Any],
                                             dialogue_act_prior: Dict[str, float],
                                             previous_posterior: Dict[str, Any],
                                             inertia_rho: float) -> Dict[str, float]:
        """Build checklist prior using Bayesian filtering from previous beliefs"""

        # Start with base checklist prior
        base_prior = self._build_checklist_prior(observation_u, dialogue_act_prior)

        # Apply belief carryover from previous posterior
        previous_checklist = previous_posterior.get('checklist', {})

        if previous_checklist:
            # Bayesian filtering: p_{t+1} ∝ (q_t)^ρ × base_prior
            filtered_prior = {}
            for checklist_name in base_prior:
                previous_prob = previous_checklist.get(checklist_name, 0.01)  # Small floor
                carryover_weight = previous_prob ** inertia_rho
                filtered_prior[checklist_name] = carryover_weight * base_prior[checklist_name]

            # Renormalize
            total = sum(filtered_prior.values())
            if total > 0:
                filtered_prior = {k: v/total for k, v in filtered_prior.items()}
            return filtered_prior

        return base_prior

    def _apply_belief_carryover(self, current_prior: Dict[str, float],
                               previous_posterior: Dict[str, float],
                               inertia_rho: float) -> Dict[str, float]:
        """Apply belief carryover to any prior distribution"""

        if not previous_posterior:
            return current_prior

        # Bayesian filtering: p_{t+1} ∝ (q_t)^ρ × current_prior
        filtered_prior = {}
        for key in current_prior:
            previous_prob = previous_posterior.get(key, 0.01)
            carryover_weight = previous_prob ** inertia_rho
            filtered_prior[key] = carryover_weight * current_prior[key]

        # Renormalize
        total = sum(filtered_prior.values())
        if total > 0:
            filtered_prior = {k: v/total for k, v in filtered_prior.items()}

        return filtered_prior

    def _build_subgraph_prior_with_carryover(self, candidates: List[Dict[str, Any]],
                                            previous_posterior: Dict[str, Any],
                                            inertia_rho: float) -> Dict[str, float]:
        """Build subgraph prior with belief carryover and recency/simplicity bonuses"""

        # Start with base subgraph prior
        base_prior = self._build_subgraph_prior(candidates)

        # Apply belief carryover from previous subgraph posterior
        previous_subgraph = previous_posterior.get('subgraph', {})

        if previous_subgraph:
            filtered_prior = {}
            for candidate_id in base_prior:
                previous_prob = previous_subgraph.get(candidate_id, 0.01)
                carryover_weight = previous_prob ** inertia_rho

                # Add recency bonus for recently discussed subgraphs
                candidate = next((c for c in candidates if c['id'] == candidate_id), {})
                recency_bonus = 1.0
                simplicity_bonus = 1.0

                # Simplicity bonus: prefer smaller subgraphs
                subgraph_size = candidate.get('subgraph_size', 10)
                simplicity_bonus = 1.0 / (1.0 + subgraph_size / 10.0)  # Smaller is better

                filtered_prior[candidate_id] = carryover_weight * base_prior[candidate_id] * recency_bonus * simplicity_bonus

            # Renormalize
            total = sum(filtered_prior.values())
            if total > 0:
                filtered_prior = {k: v/total for k, v in filtered_prior.items()}
            return filtered_prior

        return base_prior

    def _build_step_prior(self, checklist_prior: Dict[str, float],
                         goal_prior: Dict[str, float],
                         session_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Build prior over procedure steps p(z_step | z_checklist, z_goal, history)

        Following the theory from attemp1TextOnly.py:
        - Model steps explicitly: (:Process)-[:HAS_STEP]->(:Step) with preconditions
        - Signals: Recently satisfied preconditions, last answered slot, typical step orders
        - Goal ↔ step compatibility
        - Procedure-agnostic fallback for non-procedural checklists
        """

        procedure_state = session_context.get('procedure_state', {})
        recent_turns = session_context.get('recent_turns', [])
        belief_state = session_context.get('belief_state', {})

        # Get top checklist and goal
        top_checklist = max(checklist_prior.items(), key=lambda x: x[1])[0] if checklist_prior else None
        top_goal = max(goal_prior.items(), key=lambda x: x[1])[0] if goal_prior else None

        if procedure_state.get('active_checklist') and procedure_state['active_checklist'] == top_checklist:
            # We're in an active procedure - determine current step
            checklist_name = procedure_state['active_checklist']
            completed_slots = procedure_state.get('completed_slots', [])
            current_step = procedure_state.get('current_step', 'initial')

            try:
                # Query checklist SlotSpecs to understand procedure requirements
                slot_specs = self.db_manager.execute_query("""
                    MATCH (c:Checklist {name: $checklist_name})-[:REQUIRES]->(ss:SlotSpec)
                    RETURN ss.name as slot_name, ss.required as required,
                           ss.expect_labels as expect_labels
                    ORDER BY ss.name
                """, {"checklist_name": checklist_name})

                # Determine which slots are missing and their priority
                missing_required = []
                missing_optional = []

                for spec in slot_specs:
                    slot_name = spec['slot_name']
                    if slot_name not in completed_slots:
                        if spec['required']:
                            missing_required.append({
                                'name': slot_name,
                                'labels': spec['expect_labels']
                            })
                        else:
                            missing_optional.append(slot_name)

                # Build step prior based on procedure logic
                step_prior = {}

                if not missing_required:
                    # All required slots complete - ready for final verification
                    step_prior["procedure_complete"] = 0.9
                    step_prior["final_verification"] = 0.1
                elif current_step == "initial" or not current_step.startswith("collect_"):
                    # Starting procedure - begin with first missing required slot
                    if missing_required:
                        first_slot = missing_required[0]['name']
                        step_prior[f"collect_{first_slot}"] = 0.8
                        step_prior["procedure_complete"] = 0.2
                else:
                    # Continue with current step or move to next
                    current_slot = current_step.replace("collect_", "")
                    if current_slot in [s['name'] for s in missing_required]:
                        # Stay on current step with high probability
                        step_prior[current_step] = 0.7
                        # Small probability of moving to next step
                        current_idx = next(i for i, s in enumerate(missing_required)
                                         if s['name'] == current_slot)
                        if current_idx + 1 < len(missing_required):
                            next_slot = missing_required[current_idx + 1]['name']
                            step_prior[f"collect_{next_slot}"] = 0.2
                        step_prior["procedure_complete"] = 0.1
                    else:
                        # Current step completed, move to next
                        if missing_required:
                            next_slot = missing_required[0]['name']
                            step_prior[f"collect_{next_slot}"] = 0.8
                            step_prior["procedure_complete"] = 0.2

                # Apply goal compatibility (theory: goal ↔ step compatibility)
                if top_goal:
                    goal_step_compatibility = self._compute_goal_step_compatibility(
                        top_goal, step_prior, checklist_name
                    )
                    step_prior = {step: prob * goal_step_compatibility.get(step, 1.0)
                                for step, prob in step_prior.items()}

                # Normalize
                total = sum(step_prior.values())
                if total > 0:
                    step_prior = {step: prob/total for step, prob in step_prior.items()}

            except Exception as e:
                logger.warning(f"Error building procedure step prior: {e}")
                step_prior = {"unknown_procedure_step": 1.0}
        else:
            # No active procedure or checklist changed
            if top_checklist == "VerifyMusicRights":
                # Initialize new procedure
                step_prior = {"collect_film": 0.8, "procedure_complete": 0.2}
            else:
                # Non-procedural checklist or fallback
                step_prior = {"no_procedure": 1.0}

        logger.info(f"🎯 Built step prior: {step_prior}")
        return step_prior

    def _compute_goal_step_compatibility(self, goal: str, step_prior: Dict[str, float],
                                       checklist_name: str) -> Dict[str, float]:
        """Compute compatibility between goal and procedure steps"""

        compatibility = {}

        for step in step_prior.keys():
            if step == "procedure_complete":
                # Final steps always compatible with verification goals
                if "verify" in goal.lower() or "check" in goal.lower():
                    compatibility[step] = 1.2
                else:
                    compatibility[step] = 1.0
            elif step.startswith("collect_"):
                slot_name = step.replace("collect_", "")
                # Information gathering steps compatible with identify/recommend goals
                if goal in ["identify", "recommend", "find"]:
                    compatibility[step] = 1.1
                else:
                    compatibility[step] = 1.0
            else:
                compatibility[step] = 1.0

        return compatibility

    def _build_slot_priors(self, candidates: List[Dict[str, Any]],
                          step_prior: Dict[str, float],
                          checklist_prior: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Build initial slot priors based on candidate neighborhoods (MISSING FROM CURRENT SYSTEM!)"""

        # This is another CRITICAL missing piece
        # We need initial guesses for each required slot based on what's available in candidates

        slot_priors = {}

        # Get top checklist to know which slots we need
        top_checklist = max(checklist_prior.items(), key=lambda x: x[1])[0] if checklist_prior else None

        if not top_checklist:
            return {"unknown": {"unknown": 1.0}}

        try:
            # Query SlotSpecs for this checklist
            slot_specs = self.db_manager.execute_query("""
                MATCH (c:Checklist {name: $checklist_name})-[:REQUIRES]->(ss:SlotSpec)
                RETURN ss.name as slot_name, ss.expect_labels as expect_labels,
                       ss.required as required, ss.cardinality as cardinality
                ORDER BY ss.required DESC, ss.name
            """, {"checklist_name": top_checklist})

            # Get current step to prioritize relevant slots
            current_step = max(step_prior.items(), key=lambda x: x[1])[0] if step_prior else None
            step_focus_slot = None
            if current_step and current_step.startswith("collect_"):
                step_focus_slot = current_step.replace("collect_", "")

            for spec in slot_specs:
                slot_name = spec['slot_name']
                expect_labels = spec.get('expect_labels', [])
                required = spec.get('required', False)
                cardinality = spec.get('cardinality', 'ONE')

                # Build slot prior with sophisticated logic
                slot_prior = self._build_single_slot_prior(
                    slot_name, expect_labels, required, cardinality,
                    candidates, step_focus_slot, current_step
                )

                slot_priors[slot_name] = slot_prior

        except Exception as e:
            logger.warning(f"Error building slot priors: {e}")
            slot_priors = {"unknown": {"unknown": 1.0}}

        logger.debug(f"Built slot priors for {len(slot_priors)} slots")
        return slot_priors

    def _build_single_slot_prior(self, slot_name: str, expect_labels: List[str],
                               required: bool, cardinality: str,
                               candidates: List[Dict[str, Any]],
                               step_focus_slot: str = None,
                               current_step: str = None) -> Dict[str, float]:
        """
        Build prior for a single slot with sophisticated neighborhood analysis

        Following the theory: type constraints, popularity nearby, unknown option
        """

        candidate_fillers = {}
        base_unknown_mass = 0.7  # High uncertainty by default

        # Adjust unknown mass based on context
        if step_focus_slot == slot_name:
            # This slot is the current focus - lower uncertainty
            base_unknown_mass = 0.4
        elif current_step == "procedure_complete":
            # Procedure nearly done - lower uncertainty for remaining slots
            base_unknown_mass = 0.5
        elif not required:
            # Optional slots can remain unknown
            base_unknown_mass = 0.8

        # Analyze candidate subgraphs for potential fillers
        for candidate in candidates[:10]:  # Check top 10 candidates
            candidate_weight = candidate.get('score', 0.1)  # Use candidate score as weight

            # Look for nodes that match expected labels in this candidate
            nodes = candidate.get('nodes', [])
            for node in nodes:
                node_labels = node.get('labels', [])
                if isinstance(node_labels, str):
                    node_labels = [node_labels]

                # Check if node matches expected type
                label_match = any(expected in node_labels for expected in expect_labels)

                if label_match:
                    node_name = node.get('name') or node.get('properties', {}).get('name') or node.get('id', 'unknown')
                    node_id = node.get('id', node_name)

                    # Boost score if this matches the current step focus
                    weight = candidate_weight
                    if step_focus_slot == slot_name and node_name != 'unknown':
                        weight *= 2.0  # Double weight for step-focused slots

                    if node_name not in candidate_fillers:
                        candidate_fillers[node_name] = weight
                    else:
                        candidate_fillers[node_name] += weight

        # Look for existing slot values in the graph (pre-filled slots)
        try:
            existing_slots = self.db_manager.execute_query("""
                MATCH (sv:SlotValue {slot: $slot_name})
                RETURN sv.value as value, count(*) as frequency
                ORDER BY frequency DESC
                LIMIT 5
            """, {"slot_name": slot_name})

            for slot_value in existing_slots:
                value = slot_value['value']
                frequency = slot_value['frequency']
                # Add existing slot values with frequency-based weight
                weight = min(frequency * 0.1, 0.3)  # Cap at 0.3
                if value not in candidate_fillers:
                    candidate_fillers[value] = weight
                else:
                    candidate_fillers[value] += weight

        except Exception as e:
            logger.debug(f"Could not query existing slot values: {e}")

        # Apply cardinality constraints
        if cardinality == "ONE":
            # For single-value slots, we can be more decisive
            if candidate_fillers:
                base_unknown_mass = min(base_unknown_mass, 0.6)
        elif cardinality == "MANY":
            # For multi-value slots, unknown mass should be lower
            base_unknown_mass = min(base_unknown_mass, 0.3)
            # Allow multiple high-confidence fillers
            top_fillers = sorted(candidate_fillers.items(), key=lambda x: x[1], reverse=True)
            if len(top_fillers) > 1:
                # Keep top 3 fillers for MANY cardinality
                candidate_fillers = dict(top_fillers[:3])

        # Normalize known candidates
        total_known = sum(candidate_fillers.values())
        if total_known > 0:
            remaining_mass = 1.0 - base_unknown_mass
            candidate_fillers = {name: (weight/total_known) * remaining_mass
                               for name, weight in candidate_fillers.items()}

        # Add unknown option
        candidate_fillers["unknown"] = base_unknown_mass

        # Ensure we have at least some mass distribution
        total_mass = sum(candidate_fillers.values())
        if total_mass > 0:
            candidate_fillers = {name: mass/total_mass
                               for name, mass in candidate_fillers.items()}

        return candidate_fillers
    
    def _build_checklist_prior(self, observation_u: Dict, dialogue_act_prior: Dict) -> Dict[str, float]:
        """Build prior over checklists based on dialogue acts and extracted terms"""
        
        # Query available checklists
        try:
            checklists = self.db_manager.get_available_checklists()
        except Exception as e:
            logger.error(f"Failed to query checklists: {e}")
            checklists = []
        
        if not checklists:
            # Fallback to generic checklists (domain-agnostic)
            checklists = [
                {'name': 'Identify', 'description': 'Identify a specific target from clues'},
                {'name': 'Recommend', 'description': 'Recommend items based on preferences'},
                {'name': 'Verify', 'description': 'Verify facts or properties'}
            ]
        
        # Extract signals from observation and dialogue acts
        canonical_terms = observation_u.get('u_meta', {}).get('extraction', {}).get('canonical_terms', [])
        utterance_text = observation_u.get('u_meta', {}).get('utterance', '').lower()

        # Intent-driven checklist selection
        checklist_prior = {}

        # Strong signal: "recommendation" in canonical terms
        has_recommend_term = any('recommend' in term.lower() for term in canonical_terms)
        has_similar_pattern = any(word in ' '.join(canonical_terms).lower() for word in ['similar', 'like', 'comparable'])
        request_score = dialogue_act_prior.get('request', 0.0)

        # Apply loose heuristic matching to existing formal checklists
        for checklist in checklists:
            name = checklist['name']
            description = checklist.get('description', '').lower()

            # Enhanced matching logic
            if 'verifymusicrights' in name.lower():
                # Strong match for music rights verification
                has_music_rights_terms = any(term in utterance_text for term in ['verify', 'music', 'rights', 'copyright', 'permission'])
                print(f"DEBUG: VerifyMusicRights - utterance: '{utterance_text}'")
                print(f"DEBUG: VerifyMusicRights - has_music_terms: {has_music_rights_terms}")

                if has_music_rights_terms:
                    checklist_prior[name] = 0.8  # High weight for music rights verification
                    print(f"DEBUG: ✅ VerifyMusicRights set to 0.8")
                else:
                    checklist_prior[name] = 0.1  # Low weight if not music rights related
                    print(f"DEBUG: ❌ VerifyMusicRights set to 0.1")

            elif ('identify' in name.lower() or 'find' in description):
                if has_recommend_term:
                    checklist_prior[name] = 0.1  # Low weight when intent doesn't match
                else:
                    checklist_prior[name] = 0.4  # Moderate weight
            elif 'recommend' in name.lower() or 'suggest' in description:
                if has_recommend_term:
                    checklist_prior[name] = 0.4  # Moderate weight
                else:
                    checklist_prior[name] = 0.1  # Low weight
            else:
                checklist_prior[name] = 0.2  # Neutral weight for unknown checklists
        
        # Always prioritize flexible operation over rigid procedures
        if checklist_prior:
            # Check if any procedure-driven checklist has strong match
            procedure_driven_checklists = {'VerifyMusicRights'}
            has_strong_procedure_match = any(
                name in procedure_driven_checklists and checklist_prior.get(name, 0) > 0.5
                for name in checklist_prior.keys()
            )

            if has_strong_procedure_match:
                # Allow procedure-driven checklists to dominate when there's a strong match
                total_formal_mass = 0.8  # 80% weight for procedure-driven checklists
            else:
                # Normal case: keep formal checklists secondary to flexible operation
                total_formal_mass = 0.3  # Only 30% weight for formal checklists

            current_sum = sum(checklist_prior.values())
            if current_sum > 0:
                for name in checklist_prior:
                    checklist_prior[name] = (checklist_prior[name] / current_sum) * total_formal_mass
        
        # High probability of operating without rigid procedural constraints
        if has_strong_procedure_match:
            checklist_prior['None'] = 0.2  # Lower chance when procedure-driven checklist has strong match
        else:
            checklist_prior['None'] = 0.7  # 70% chance of flexible, intent-driven operation
        
        return checklist_prior
    
    def _build_subgraph_prior(self, candidates: List[Dict]) -> Dict[str, float]:
        """Build prior over candidate subgraphs using retrieval scores and simplicity"""
        
        if not candidates:
            return {}
        
        # Use retrieval scores as base, add simplicity bias
        subgraph_prior = {}
        
        for candidate in candidates:
            cand_id = candidate['id']
            
            # Base score from retrieval
            retrieval_score = candidate.get('retrieval_score', 0.0)
            
            # Simplicity bias (prefer smaller subgraphs)
            size = candidate.get('subgraph_size', 1)
            simplicity_bonus = 1.0 / (1.0 + 0.1 * size)  # Gentle penalty for larger graphs
            
            # Combined prior score
            prior_score = retrieval_score * simplicity_bonus
            subgraph_prior[cand_id] = prior_score
        
        # Apply softmax with temperature for normalization
        tau = self.config.get('system.defaults.tau_retrieval', 0.7)
        
        # Softmax normalization
        import numpy as np
        max_score = max(subgraph_prior.values()) if subgraph_prior else 0
        exp_scores = {k: np.exp((v - max_score) / tau) for k, v in subgraph_prior.items()}
        total_exp = sum(exp_scores.values())
        
        if total_exp > 0:
            subgraph_prior = {k: v / total_exp for k, v in exp_scores.items()}
        
        return subgraph_prior
    
    def _build_novelty_prior(self, anchors: List[Dict], observation_u: Dict) -> float:
        """Build prior over novelty/OOD signals"""
        
        # Check semantic similarity to anchors
        if anchors:
            max_anchor_sim = max(anchor.get('s_combined', 0.0) for anchor in anchors)
        else:
            max_anchor_sim = 0.0
        
        # Novelty indicators
        novelty_signals = []
        
        # Low similarity to all anchors
        if max_anchor_sim < 0.35:
            novelty_signals.append(0.3)
        
        # Few extracted entities linked to graph
        extraction = observation_u.get('u_meta', {}).get('extraction', {})
        entities = extraction.get('entities', [])
        if len(entities) > 0:
            # This would need linked_entity_ids from retrieval context
            # For now, use a placeholder
            novelty_signals.append(0.1)
        
        # Short utterance (hard to interpret)
        utterance = observation_u.get('u_meta', {}).get('utterance', '')
        if len(utterance.split()) < 5:
            novelty_signals.append(0.1)
        
        # Combine novelty signals
        novelty_score = min(0.8, sum(novelty_signals))  # Cap at 0.8
        
        return novelty_score
