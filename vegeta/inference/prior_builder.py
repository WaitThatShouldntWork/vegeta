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
        """Build all priors for the current turn"""
        
        # Get basic dialogue act classification
        utterance = observation_u.get('u_meta', {}).get('utterance', '')
        dialogue_act_prior = self.query_analyzer.classify_dialogue_act_llm(utterance)
        
        # Build checklist prior
        checklist_prior = self._build_checklist_prior(observation_u, dialogue_act_prior)
        
        # Build goal prior using LLM analysis
        extraction = observation_u.get('u_meta', {}).get('extraction', {})
        goal_prior = self.query_analyzer.classify_user_intent_llm(
            utterance, dialogue_act_prior, extraction
        )
        
        # Build subgraph prior
        candidates = retrieval_context.get('candidates', [])
        subgraph_prior = self._build_subgraph_prior(candidates)
        
        # Build novelty prior
        novelty_prior = self._build_novelty_prior(retrieval_context['anchors'], observation_u)
        
        return {
            'checklist': checklist_prior,
            'goal': goal_prior,
            'subgraph': subgraph_prior,
            'dialogue_act': dialogue_act_prior,
            'novelty': novelty_prior
        }
    
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
        
        # Intent-driven checklist selection
        checklist_prior = {}
        
        # Strong signal: "recommendation" in canonical terms
        has_recommend_term = any('recommend' in term.lower() for term in canonical_terms)
        has_similar_pattern = any(word in ' '.join(canonical_terms).lower() for word in ['similar', 'like', 'comparable'])
        request_score = dialogue_act_prior.get('request', 0.0)
        
        logger.info(f"Intent signals: recommend_term={has_recommend_term}, similar_pattern={has_similar_pattern}, request_score={request_score:.3f}")
        
        # Apply loose heuristic matching to existing formal checklists
        for checklist in checklists:
            name = checklist['name']
            description = checklist.get('description', '').lower()
            
            # Heuristic matching - don't force it
            if ('identify' in name.lower() or 'find' in description):
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
            # Normalize existing checklists but keep them secondary to flexible operation
            total_formal_mass = 0.3  # Only 30% weight to formal checklists
            current_sum = sum(checklist_prior.values())
            if current_sum > 0:
                for name in checklist_prior:
                    checklist_prior[name] = (checklist_prior[name] / current_sum) * total_formal_mass
        
        # High probability of operating without rigid procedural constraints
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
