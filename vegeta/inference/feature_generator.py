"""
Feature generation for candidates
"""

import logging
from typing import Dict, Any, List

from ..utils.database import DatabaseManager
from ..core.config import Config

logger = logging.getLogger(__name__)

class FeatureGenerator:
    """
    Generate features for candidate evaluation
    """
    
    def __init__(self, db_manager: DatabaseManager, config: Config):
        self.db_manager = db_manager
        self.config = config
        self.defaults = config.system_defaults
    
    def generate_features(self, candidates: List[Dict[str, Any]], 
                         observation_u: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate features for all candidates"""
        
        # PERFORMANCE FIX: Batch fetch all slot values at once instead of N+1 queries
        entity_ids = []
        for candidate in candidates:
            if candidate.get('entity_id'):
                entity_ids.append(candidate['entity_id'])
            elif candidate.get('anchor_id'):
                entity_ids.append(candidate['anchor_id'])
        
        # Get all slot values in one batched query
        if entity_ids:
            slot_values_batch = self.db_manager.get_slot_values_for_entities_batch(entity_ids)
        else:
            slot_values_batch = {}
        
        for candidate in candidates:
            # Compute observed structural features
            candidate['u_struct_obs'] = self._compute_u_struct_obs(candidate)
            
            # Generate expected terms (with pre-fetched slot values)
            candidate['expected_terms'] = self._generate_expected_terms(candidate, slot_values_batch)
            
            # Compute distances
            candidate['distances'] = self._compute_delta_distances(observation_u, candidate)
            
            # Compute preliminary likelihood score
            candidate['log_likelihood'] = self._compute_log_likelihood(candidate)
        
        # Sort by likelihood
        candidates.sort(key=lambda x: x['log_likelihood'], reverse=True)
        return candidates
    
    def _compute_u_struct_obs(self, candidate: Dict[str, Any]) -> Dict[str, int]:
        """Compute observed structural features for a candidate"""
        
        try:
            if candidate.get('entity_id'):
                # Candidate-centric
                query = """
                MATCH (c:Entity {id: $cid})
                OPTIONAL MATCH (c)-[r]-(n)
                RETURN 
                    count(DISTINCT n) as node_count,
                    count(DISTINCT type(r)) as edge_type_count,
                    collect(DISTINCT labels(n)) as node_label_groups,
                    collect(DISTINCT type(r)) as edge_types
                """
                params = {"cid": candidate['entity_id']}
            else:
                # Fallback to anchor-based computation
                query = """
                MATCH (anchor:Entity {id: $anchor_id})
                OPTIONAL MATCH (anchor)-[r]-(connected)
                WHERE connected.id IN $connected_ids
                RETURN 
                    count(DISTINCT connected) as node_count,
                    count(DISTINCT type(r)) as edge_type_count,
                    collect(DISTINCT labels(connected)) as node_label_groups,
                    collect(DISTINCT type(r)) as edge_types
                """
                params = {"anchor_id": candidate['anchor_id'], "connected_ids": candidate['connected_ids']}
            
            result = self.db_manager.execute_query_single(query, params)
            
            if result:
                # Flatten label groups and count each label
                all_labels = []
                for label_group in result['node_label_groups']:
                    if label_group:  # Skip None/empty groups
                        all_labels.extend(label_group)
                
                # Count label occurrences
                label_counts = {}
                for label in all_labels:
                    label_counts[f"label_{label}"] = label_counts.get(f"label_{label}", 0) + 1
                
                # Count edge types
                edge_counts = {}
                for edge_type in result['edge_types']:
                    if edge_type:
                        edge_counts[f"edge_{edge_type}"] = edge_counts.get(f"edge_{edge_type}", 0) + 1
                
                # Combine counts
                struct_features = {
                    'total_nodes': result['node_count'] or 0,
                    'total_edge_types': result['edge_type_count'] or 0,
                    **label_counts,
                    **edge_counts
                }
                
                return struct_features
            
        except Exception as e:
            logger.error(f"Failed to compute structural features for {candidate['id']}: {e}")
        
        # Fallback
        return {'total_nodes': candidate.get('subgraph_size', 0)}
    
    def _generate_expected_terms(self, candidate: Dict[str, Any], slot_values_batch: Dict[str, List[Dict[str, Any]]] = None) -> List[str]:
        """Generate expected terms for candidate"""
        
        expected_terms: List[str] = []

        # 1) Candidate entity name words
        if candidate.get('entity_name'):
            words = candidate['entity_name'].lower().replace('-', ' ').replace(':', ' ').split()
            expected_terms.extend([w for w in words if len(w) > 2])

        # 2) Direct SlotValue values for the candidate entity (PERFORMANCE FIX)
        try:
            entity_id = candidate.get('entity_id') or candidate.get('anchor_id')
            
            if slot_values_batch and entity_id:
                # Use pre-fetched slot values to avoid N+1 query problem
                slot_values = slot_values_batch.get(entity_id, [])
            elif entity_id:
                # Fallback to individual query if batch not available
                slot_values = self.db_manager.get_slot_values_for_entity(entity_id)
            else:
                slot_values = []

            for slot_value in slot_values:
                if slot_value['value']:
                    slot_words = str(slot_value['value']).lower().split()
                    expected_terms.extend([w for w in slot_words if len(w) > 2])
                if slot_value['slot']:
                    expected_terms.append(str(slot_value['slot']).lower())
        
        except Exception as e:
            logger.error(f"Failed to get slot values for {entity_id}: {e}")
        
        # 3) Sample of neighbor names
        for name in (candidate.get('connected_names') or [])[:3]:
            if name:
                words = name.lower().replace('-', ' ').replace(':', ' ').split()
                expected_terms.extend([w for w in words if len(w) > 2])

        # Dedupe and limit
        expected_terms = list(dict.fromkeys(expected_terms))[:self.defaults['N_expected']]
        return expected_terms
    
    def _compute_delta_distances(self, observation_u: Dict, candidate: Dict) -> Dict[str, float]:
        """Compute delta distances for semantic, structural, and terms channels"""
        
        # δ_sem: semantic distance (placeholder - would use actual subgraph embedding)
        delta_sem = max(0.0, 1.0 - candidate.get('retrieval_score', 0.0))
        
        # δ_struct: structural distance (simplified)
        delta_struct = 0.3
        
        # δ_terms: terms distance using Jaccard
        u_terms_set = observation_u.get('u_terms_set', set())
        expected_terms_set = set(candidate.get('expected_terms', []))
        
        if len(u_terms_set) >= self.defaults['small_set_threshold'] and len(expected_terms_set) >= self.defaults['small_set_threshold']:
            # Use Jaccard distance
            intersection = len(u_terms_set.intersection(expected_terms_set))
            union = len(u_terms_set.union(expected_terms_set))
            jaccard = intersection / union if union > 0 else 0.0
            delta_terms = 1.0 - jaccard
        else:
            # Use blended approach for small sets
            intersection = len(u_terms_set.intersection(expected_terms_set))
            union = len(u_terms_set.union(expected_terms_set))
            jaccard = intersection / union if union > 0 else 0.0
            
            # Cosine similarity component (placeholder)
            cosine_sim = 0.3
            
            delta_terms = self.defaults['small_set_blend'] * (1.0 - jaccard) + (1.0 - self.defaults['small_set_blend']) * (1.0 - cosine_sim)
        
        return {
            'delta_sem': delta_sem,
            'delta_struct': delta_struct,
            'delta_terms': delta_terms
        }
    
    def _compute_log_likelihood(self, candidate: Dict[str, Any]) -> float:
        """Compute log likelihood for candidate"""
        
        distances = candidate['distances']
        alpha, beta, gamma = self.defaults['alpha'], self.defaults['beta'], self.defaults['gamma']
        sigma_sem_sq = self.defaults['sigma_sem_sq']
        sigma_struct_sq = self.defaults['sigma_struct_sq'] 
        sigma_terms_sq = self.defaults['sigma_terms_sq']
        
        # Log-likelihood (negative because we're computing negative log-likelihood)
        log_likelihood = -(
            alpha * distances['delta_sem'] / sigma_sem_sq +
            beta * distances['delta_struct'] / sigma_struct_sq +
            gamma * distances['delta_terms'] / sigma_terms_sq
        )
        
        return log_likelihood
