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
        """
        Compute observed structural features for a candidate from full subgraph data.
        This implements the structural observation from attemp1TextOnly.py.
        """

        # Get data from full subgraph
        subgraph = candidate.get('subgraph', {})
        nodes = subgraph.get('nodes', [])
        relationships = subgraph.get('relationships', [])
        facts = subgraph.get('facts', [])

        # Count node labels
        label_counts = {}
        for node in nodes:
            labels = node.get('labels', [])
            for label in labels:
                label_counts[f"label_{label}"] = label_counts.get(f"label_{label}", 0) + 1

        # Count relationship types
        edge_counts = {}
        for rel in relationships:
            rel_type = rel.get('relationship_type', '')
            if rel_type:
                edge_counts[f"edge_{rel_type}"] = edge_counts.get(f"edge_{rel_type}", 0) + 1

        # Count fact types (critical for awards!)
        fact_counts = {}
        for fact in facts:
            fact_kind = fact.get('fact_kind', '')
            if fact_kind and isinstance(fact_kind, str):
                fact_counts[f"fact_{fact_kind}"] = fact_counts.get(f"fact_{fact_kind}", 0) + 1

        # Combine all structural features
        struct_features = {
            'total_nodes': len(nodes),
            'total_relationships': len(relationships),
            'total_facts': len(facts),
            **label_counts,
            **edge_counts,
            **fact_counts
        }

        logger.debug(f"Computed structural features for {candidate['id']}: {len(nodes)} nodes, {len(relationships)} relationships, {len(facts)} facts")
        return struct_features
    
    def _generate_expected_terms(self, candidate: Dict[str, Any], slot_values_batch: Dict[str, List[Dict[str, Any]]] = None) -> List[str]:
        """
        Generate expected terms for candidate from full subgraph data.
        This implements the comprehensive term extraction from attemp1TextOnly.py.
        """

        expected_terms: List[str] = []

        # Access the full subgraph data
        subgraph = candidate.get('subgraph', {})
        nodes = subgraph.get('nodes', [])
        relationships = subgraph.get('relationships', [])
        facts = subgraph.get('facts', [])

        # 1) Anchor entity name words
        anchor_name = candidate.get('anchor_name', '')
        if anchor_name:
            words = anchor_name.lower().replace('-', ' ').replace(':', ' ').split()
            expected_terms.extend([w for w in words if len(w) > 2])

        # 2) All node names in subgraph (entities, types, relation types)
        for node in nodes:
            node_name = node.get('name', '')
            if node_name:
                words = node_name.lower().replace('-', ' ').replace(':', ' ').split()
                expected_terms.extend([w for w in words if len(w) > 2])

        # 3) Relationship types (e.g., "ACTED_IN", "WON_AWARD")
        for rel in relationships:
            rel_type = rel.get('relationship_type', '')
            if rel_type:
                words = rel_type.lower().replace('_', ' ').split()
                expected_terms.extend([w for w in words if len(w) > 2])

        # 4) Fact kinds and properties (CRITICAL for awards!)
        for fact in facts:
            fact_kind = fact.get('fact_kind', '')
            if fact_kind:
                words = fact_kind.lower().replace('_', ' ').split()
                expected_terms.extend([w for w in words if len(w) > 2])

            # Include fact properties that might be descriptive
            properties = fact.get('properties', {})
            if isinstance(properties, dict):
                for key, value in properties.items():
                    if isinstance(value, str) and len(value) > 2:
                        words = value.lower().replace('-', ' ').replace(':', ' ').split()
                        expected_terms.extend([w for w in words if len(w) > 2])

        # 5) SlotValue data (maintain backward compatibility)
        try:
            entity_id = candidate.get('entity_id') or candidate.get('anchor_id')

            if slot_values_batch and entity_id:
                slot_values = slot_values_batch.get(entity_id, [])
            elif entity_id:
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

        # 6) Backward compatibility: connected names
        for name in (candidate.get('connected_names') or [])[:3]:
            if name:
                words = name.lower().replace('-', ' ').replace(':', ' ').split()
                expected_terms.extend([w for w in words if len(w) > 2])

        # Dedupe and limit
        expected_terms = list(dict.fromkeys(expected_terms))[:self.defaults['N_expected']]

        logger.debug(f"Generated {len(expected_terms)} expected terms for candidate {candidate['id']}: {expected_terms[:5]}...")
        return expected_terms
    
    def _compute_delta_distances(self, observation_u: Dict, candidate: Dict) -> Dict[str, float]:
        """
        Compute delta distances for semantic, structural, and terms channels.
        This implements the comprehensive distance computation from attemp1TextOnly.py.
        """

        # δ_sem: semantic distance (placeholder - would use actual subgraph embedding)
        delta_sem = max(0.0, 1.0 - candidate.get('retrieval_score', 0.0))

        # δ_struct: structural distance based on checklist expectations
        delta_struct = self._compute_structural_distance(candidate)

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

            # Cosine similarity component (placeholder - would use actual term embeddings)
            cosine_sim = 0.3

            delta_terms = self.defaults['small_set_blend'] * (1.0 - jaccard) + (1.0 - self.defaults['small_set_blend']) * (1.0 - cosine_sim)

        return {
            'delta_sem': delta_sem,
            'delta_struct': delta_struct,
            'delta_terms': delta_terms
        }

    def _compute_structural_distance(self, candidate: Dict[str, Any]) -> float:
        """
        Compute structural distance based on checklist expectations.
        This implements the checklist-driven structural distance from attemp1TextOnly.py.
        """

        # Get observed structural features
        struct_obs = candidate.get('u_struct_obs', {})

        # Get active checklist from retrieval context (this would come from the retrieval context)
        # For now, we'll use a simple heuristic based on subgraph content

        expected_patterns = self._get_expected_structural_patterns(candidate)

        # Compute distance based on expected vs observed patterns
        penalties = []

        for pattern, expected_count in expected_patterns.items():
            observed_count = struct_obs.get(pattern, 0)

            if expected_count > 0 and observed_count == 0:
                # Missing expected pattern - high penalty
                penalties.append(self.defaults.get('lambda_missing', 0.30))
            elif observed_count > expected_count * 2:
                # Too many instances - moderate penalty
                penalties.append(0.1)
            elif observed_count < expected_count:
                # Fewer than expected - small penalty
                ratio = observed_count / expected_count
                penalties.append((1.0 - ratio) * 0.1)

        # Hub node penalty (simplified)
        total_nodes = struct_obs.get('total_nodes', 0)
        if total_nodes > self.defaults.get('d_cap', 40):
            hub_penalty = 0.02 * (total_nodes - 40)  # lambda_hub * softplus
            penalties.append(hub_penalty)

        # Convert penalties to distance (0-1 scale)
        total_penalty = min(1.0, sum(penalties))

        logger.debug(f"Structural distance for {candidate['id']}: {total_penalty:.3f} (penalties: {penalties})")
        return total_penalty

    def _get_expected_structural_patterns(self, candidate: Dict[str, Any]) -> Dict[str, int]:
        """
        Determine expected structural patterns for a candidate.
        This is a simplified version - in full implementation, this would come from checklist specifications.
        """

        subgraph = candidate.get('subgraph', {})
        facts = subgraph.get('facts', [])
        anchor_name = candidate.get('anchor_name', '').lower()

        expected_patterns = {}

        # Award-related queries should expect Fact nodes
        if any(word in anchor_name for word in ['award', 'oscar', 'bafta', 'golden']):
            expected_patterns['fact_WON_AWARD'] = 1
            expected_patterns['label_Award'] = 1

        # Movie-related queries should expect Person nodes (actors)
        if any(word in anchor_name for word in ['movie', 'film', 'bond', 'matrix']):
            expected_patterns['label_Person'] = 1
            expected_patterns['edge_ACTED_IN'] = 1

        # If we have facts, expect at least some relationships
        if facts:
            expected_patterns['total_relationships'] = max(1, len(facts) // 2)

        # Default expectations for any entity
        expected_patterns['label_Entity'] = 1

        return expected_patterns
    
    def _compute_log_likelihood(self, candidate: Dict[str, Any]) -> float:
        """
        Compute log likelihood for candidate with penalties and noise model.
        This implements the comprehensive likelihood computation from attemp1TextOnly.py.
        """

        distances = candidate['distances']
        alpha, beta, gamma = self.defaults['alpha'], self.defaults['beta'], self.defaults['gamma']
        sigma_sem_sq = self.defaults['sigma_sem_sq']
        sigma_struct_sq = self.defaults['sigma_struct_sq']
        sigma_terms_sq = self.defaults['sigma_terms_sq']

        # Base likelihood from channel distances
        base_log_likelihood = -(
            alpha * distances['delta_sem'] / sigma_sem_sq +
            beta * distances['delta_struct'] / sigma_struct_sq +
            gamma * distances['delta_terms'] / sigma_terms_sq
        )

        # Add penalties from attemp1TextOnly.py
        penalties = self._compute_penalties(candidate)

        # Final log-likelihood with penalties
        log_likelihood = base_log_likelihood - penalties

        logger.debug(f"Likelihood for {candidate['id']}: base={base_log_likelihood:.3f}, penalties={penalties:.3f}, final={log_likelihood:.3f}")
        return log_likelihood

    def _compute_penalties(self, candidate: Dict[str, Any]) -> float:
        """
        Compute penalties for missing required elements and hub nodes.
        This implements the penalty system from attemp1TextOnly.py.
        """

        total_penalty = 0.0
        struct_obs = candidate.get('u_struct_obs', {})

        # Missing required slot penalty
        # In a full implementation, this would check against checklist SlotSpec requirements
        # For now, we'll use heuristics based on expected patterns

        expected_patterns = self._get_expected_structural_patterns(candidate)
        lambda_missing = self.defaults.get('lambda_missing', 0.30)

        for pattern, expected_count in expected_patterns.items():
            observed_count = struct_obs.get(pattern, 0)
            if expected_count > 0 and observed_count == 0:
                total_penalty += lambda_missing
                logger.debug(f"Missing required pattern penalty: {pattern} (expected {expected_count}, got 0)")

        # Hub node penalty
        d_cap = self.defaults.get('d_cap', 40)
        lambda_hub = self.defaults.get('lambda_hub', 0.02)

        total_nodes = struct_obs.get('total_nodes', 0)
        if total_nodes > d_cap:
            # Use softplus approximation: λ_hub * ln(1 + exp(d - d_cap))
            # Simplified to linear for now
            hub_penalty = lambda_hub * (total_nodes - d_cap)
            total_penalty += hub_penalty
            logger.debug(f"Hub node penalty: {total_nodes} nodes (cap: {d_cap}) = {hub_penalty:.3f}")

        return total_penalty
