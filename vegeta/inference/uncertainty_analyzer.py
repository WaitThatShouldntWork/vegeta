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
        """Decide between ANSWER, ASK, or SEARCH based on uncertainty and lookahead EIG"""
        
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
        
        # LLM-BASED FAST PATH: Check for immediate clarification needs
        extraction = observation_u.get('u_meta', {}).get('extraction', {})
        query_analysis = extraction.get('query_analysis', {})
        
        if query_analysis.get('immediate_clarification_needed', False):
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
