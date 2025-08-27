"""
Bayesian posterior updates
"""

import logging
import numpy as np
from typing import Dict, Any, List

from ..core.config import Config

logger = logging.getLogger(__name__)

class PosteriorUpdater:
    """
    Update Bayesian posteriors using likelihood and priors
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.defaults = config.system_defaults
    
    def update_posteriors(self, candidates: List[Dict[str, Any]], 
                         priors: Dict[str, Any],
                         observation_u: Dict[str, Any]) -> Dict[str, Any]:
        """Update all posterior distributions"""
        
        # Update subgraph posterior (main inference)
        posterior_subgraph = self._update_posterior_subgraph(candidates, priors, observation_u)
        
        # Get top candidate for conditioning other posteriors
        if posterior_subgraph:
            top_subgraph_id = max(posterior_subgraph.items(), key=lambda x: x[1])[0]
            top_candidate = next(c for c in candidates if c['id'] == top_subgraph_id)
        else:
            top_candidate = candidates[0] if candidates else {}
        
        # Update other posteriors
        posterior_checklist = self._update_posterior_checklist(observation_u, priors, top_candidate)
        posterior_goal = self._update_posterior_goal(observation_u, priors)
        
        # Dialogue act posterior (simplified - just use prior for now)
        posterior_dialogue_act = priors['dialogue_act'].copy()
        
        return {
            'checklist': posterior_checklist,
            'goal': posterior_goal,
            'subgraph': posterior_subgraph,
            'dialogue_act': posterior_dialogue_act,
            'novelty': priors['novelty']  # Static for now
        }
    
    def _update_posterior_subgraph(self, candidates: List[Dict], priors: Dict, observation_u: Dict) -> Dict[str, float]:
        """Update posterior over subgraphs using Bayesian inference"""
        
        # Compute posterior ∝ p(u|v) * p(v) for each candidate
        posterior_scores = {}
        
        for candidate in candidates:
            cand_id = candidate['id']
            
            # Likelihood p(u|v) - already computed in feature generation
            log_likelihood = candidate.get('log_likelihood', 0.0)
            
            # Prior p(v)
            prior_prob = priors['subgraph'].get(cand_id, 1e-6)  # Small floor to avoid log(0)
            log_prior = np.log(prior_prob)
            
            # Posterior score (in log space)
            log_posterior = log_likelihood + log_prior
            posterior_scores[cand_id] = log_posterior
            
            # Store in candidate for tracking
            candidate['log_likelihood_full'] = log_likelihood
            candidate['log_prior'] = log_prior
            candidate['log_posterior'] = log_posterior
        
        # Normalize using softmax with temperature
        tau = self.defaults['tau_posterior']
        max_score = max(posterior_scores.values()) if posterior_scores else 0
        
        exp_scores = {k: np.exp((v - max_score) / tau) for k, v in posterior_scores.items()}
        total_exp = sum(exp_scores.values())
        
        if total_exp > 0:
            posterior_subgraph = {k: v / total_exp for k, v in exp_scores.items()}
        else:
            # Fallback to uniform
            n = len(posterior_scores)
            posterior_subgraph = {k: 1.0/n for k in posterior_scores.keys()} if n > 0 else {}
        
        return posterior_subgraph
    
    def _update_posterior_checklist(self, observation_u: Dict, priors: Dict, top_candidate: Dict) -> Dict[str, float]:
        """Update checklist posterior based on evidence"""
        
        # For simplicity, adjust checklist prior based on subgraph evidence
        checklist_posterior = priors['checklist'].copy()
        
        # Use generic signals: if we observe concentrated evidence for a specific type, prefer Identify-like checklists
        struct_obs = top_candidate.get('u_struct_obs', {})
        has_specific_entity = any(k.startswith('label_') and v > 0 for k, v in struct_obs.items())
        
        if has_specific_entity:
            for key in checklist_posterior:
                if 'Identify' in key:
                    checklist_posterior[key] *= 1.2
        
        # If evidence suggests multiple related entities, slightly boost Recommend-like
        total_labels = sum(v for k, v in struct_obs.items() if k.startswith('label_'))
        if total_labels >= 2:
            for key in checklist_posterior:
                if 'Recommend' in key:
                    checklist_posterior[key] *= 1.1
        
        # Normalize
        total = sum(checklist_posterior.values())
        if total > 0:
            checklist_posterior = {k: v/total for k, v in checklist_posterior.items()}
        
        return checklist_posterior
    
    def _update_posterior_goal(self, observation_u: Dict, priors: Dict) -> Dict[str, float]:
        """Update goal posterior - simplified since LLM already does sophisticated intent analysis"""
        
        # Since LLM-based goal priors already incorporate dialogue acts, entities, and context,
        # we can trust them more and apply minimal adjustments
        goal_posterior = priors['goal'].copy()
        
        # Optional: minor adjustments based on evidence strength (future enhancement)
        # For now, trust the LLM analysis
        
        return goal_posterior
