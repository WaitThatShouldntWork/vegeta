"""
Candidate expansion from anchor nodes
"""

import logging
from typing import Dict, Any, List

from ..utils.database import DatabaseManager
from ..core.config import Config
from ..core.exceptions import RetrievalError

logger = logging.getLogger(__name__)

class CandidateExpander:
    """
    Expand anchor nodes into candidate subgraphs
    """
    
    def __init__(self, db_manager: DatabaseManager, config: Config):
        self.db_manager = db_manager
        self.config = config
        self.defaults = config.system_defaults
    
    def expand_subgraphs(self, anchors: List[Dict], hops: int = None) -> List[Dict[str, Any]]:
        """
        Expand anchors into candidate subgraphs using full subgraph retrieval.
        This implements the selective activation approach from attemp1TextOnly.py.
        """

        if hops is None:
            hops = self.defaults['hops']

        candidates = []

        for i, anchor in enumerate(anchors):
            try:
                # Get FULL subgraph data (includes nodes, relationships, facts)
                subgraph_data = self.db_manager.get_full_subgraph(anchor['id'], hops)

                # Only create candidate if we have meaningful content
                if subgraph_data['metadata']['node_count'] > 1:  # More than just the anchor
                    # Create rich subgraph candidate
                    candidate = {
                        'id': f"subgraph_{i}",
                        'anchor_id': anchor['id'],
                        'anchor_name': anchor['name'],
                        'subgraph': subgraph_data,  # Full subgraph data
                        'anchor_score': anchor['s_combined'],
                        'retrieval_score': anchor['s_combined'],  # Initial score from retrieval

                        # Backward compatibility fields
                        'connected_ids': [node['id'] for node in subgraph_data['nodes'] if node.get('id') and node['id'] != anchor['id']],
                        'connected_names': [node['name'] for node in subgraph_data['nodes'] if node.get('id') and node['id'] != anchor['id']],
                        'subgraph_size': subgraph_data['metadata']['node_count'],
                    }
                    candidates.append(candidate)

            except Exception as e:
                logger.error(f"Subgraph expansion failed for anchor {anchor['id']}: {e}")

        logger.info(f"Expanded {len(anchors)} anchors into {len(candidates)} rich subgraph candidates")
        return candidates
    
    def enumerate_target_candidates_from_anchors(self, anchors: List[Dict[str, Any]], 
                                               target_labels: List[str],
                                               hops: int = None, decay: float = 0.8,
                                               limit_per_anchor: int = 50) -> List[Dict[str, Any]]:
        """Enumerate candidate entities of the target labels within k hops of the anchors."""
        
        if hops is None:
            hops = self.defaults['hops']
        
        if not anchors or not target_labels:
            return []

        # Map candidate_id -> best info
        candidate_map: Dict[str, Dict[str, Any]] = {}

        try:
            for anchor in anchors:
                anchor_id = anchor.get('id')
                anchor_name = anchor.get('name')
                anchor_score = float(anchor.get('s_combined', 0.0))
                if not anchor_id or not anchor_name:
                    continue

                # Use variable-length paths and min(length) to avoid shortestPath start=end issues
                query = f"""
                MATCH (start:Entity {{id: $anchor_id}})
                MATCH p = (start)-[*1..{hops}]-(cand)
                WHERE (cand.id IS NULL OR cand.id <> start.id)
                  AND any(lbl IN labels(cand) WHERE lbl IN $target_labels)
                WITH cand, min(length(p)) AS dist
                RETURN cand.id AS id,
                       CASE WHEN cand.name IS NOT NULL THEN cand.name
                            WHEN cand.title IS NOT NULL THEN cand.title
                            ELSE cand.id END AS name,
                       labels(cand) AS labels, dist
                ORDER BY dist ASC
                LIMIT $limit
                """
                
                results = self.db_manager.execute_query(query, {
                    'anchor_id': anchor_id,
                    'target_labels': target_labels,
                    'limit': limit_per_anchor
                })

                # Debug logging
                logger.debug(f"Anchor {anchor_id}: target_labels={target_labels}, results={len(results)}")
                if results:
                    for r in results[:2]:  # Log first 2 results
                        logger.debug(f"  Found: {r.get('name')} (labels: {r.get('labels')})")
                
                for rec in results:
                    cand_id = rec["id"]
                    cand_name = rec["name"]
                    cand_labels = rec["labels"] or []
                    dist = max(1, int(rec["dist"]))
                    # Path-decayed score from this anchor
                    score = anchor_score * (decay ** (dist - 1))

                    existing = candidate_map.get(cand_id)
                    if (existing is None) or (score > existing.get('retrieval_score', 0.0)):
                        candidate_map[cand_id] = {
                            'entity_id': cand_id,
                            'name': cand_name,
                            'labels': cand_labels,
                            'retrieval_score': float(score),
                            'best_anchor': {
                                'id': anchor_id,
                                'name': anchor_name,
                                'dist': dist,
                                'anchor_score': anchor_score
                            }
                        }
                        
        except Exception as e:
            logger.error(f"Failed to enumerate target candidates: {e}")
            return []

        # Convert to list and sort by retrieval_score
        candidates = sorted(candidate_map.values(), key=lambda x: x['retrieval_score'], reverse=True)
        
        # Convert to standard candidate format
        result_candidates = []
        for i, cand in enumerate(candidates):
            # Skip candidates with invalid best_anchor
            if not cand.get('best_anchor') or not cand['best_anchor'].get('id'):
                continue

            neighbor_data = self.db_manager.get_entity_neighbors(cand['entity_id'], 1)

            result_candidates.append({
                'id': f"cand_{i}",
                'anchor_id': cand['best_anchor']['id'],
                'anchor_name': cand['best_anchor']['name'],
                'connected_ids': neighbor_data['connected_ids'],
                'connected_names': neighbor_data['connected_names'],
                'subgraph_size': len(neighbor_data['connected_ids']),
                'anchor_score': cand['best_anchor']['anchor_score'],
                'retrieval_score': cand['retrieval_score'],
                'entity_id': cand['entity_id'],
                'entity_name': cand['name'],
                'entity_labels': cand['labels']
            })
        
        return result_candidates
