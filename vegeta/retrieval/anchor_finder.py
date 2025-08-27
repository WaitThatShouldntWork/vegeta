"""
Anchor node finding using semantic similarity
"""

import logging
import numpy as np
from typing import Dict, Any, List

from ..utils.database import DatabaseManager
from ..core.config import Config
from ..core.exceptions import RetrievalError

logger = logging.getLogger(__name__)

class AnchorFinder:
    """
    Find top-K anchor nodes using semantic similarity
    """
    
    def __init__(self, db_manager: DatabaseManager, config: Config):
        self.db_manager = db_manager
        self.config = config
        self.defaults = config.system_defaults
    
    def find_anchor_nodes(self, u_sem: np.ndarray, k: int = None) -> List[Dict[str, Any]]:
        """Find top-K anchor nodes using semantic similarity"""
        
        if k is None:
            k = self.defaults['k_anchors']
        
        try:
            # Get entities with embeddings
            entities_with_embeddings = self.db_manager.get_entities_with_embeddings(limit=100)
            
            if not entities_with_embeddings:
                logger.warning("No entities with embeddings found, using fallback")
                return self._fallback_anchor_selection(k)
            
            anchors = []
            
            # Calculate semantic similarities
            for node in entities_with_embeddings:
                if node['sem_emb']:
                    try:
                        node_emb = np.array(node['sem_emb'])
                        # L2 normalize
                        node_emb = node_emb / np.linalg.norm(node_emb)
                        
                        # Cosine similarity
                        s_sem = np.dot(u_sem, node_emb)
                        s_graph = s_sem  # For now, use same as semantic (would use graph embeddings if available)
                        
                        anchors.append({
                            'id': node['id'],
                            'name': node['name'],
                            'labels': node['labels'],
                            's_sem': float(s_sem),
                            's_graph': float(s_graph),
                            's_combined': float(0.7 * s_sem + 0.3 * s_graph)
                        })
                    except Exception as e:
                        logger.error(f"Error processing node {node['id']}: {e}")
            
            # Sort by combined score and take top-K
            anchors.sort(key=lambda x: x['s_combined'], reverse=True)
            return anchors[:k]
            
        except Exception as e:
            logger.error(f"Anchor retrieval failed: {e}")
            return self._fallback_anchor_selection(k)
    
    def _fallback_anchor_selection(self, k: int) -> List[Dict[str, Any]]:
        """Fallback when no embeddings available"""
        try:
            # Get some entities without embeddings and use default scores
            query = """
            MATCH (e:Entity)
            RETURN e.id as id, e.name as name, labels(e) as labels
            LIMIT $limit
            """
            nodes = self.db_manager.execute_query(query, {'limit': min(k * 2, 50)})
            
            anchors = []
            for node in nodes[:k]:
                anchors.append({
                    'id': node['id'],
                    'name': node['name'],
                    'labels': node['labels'],
                    's_sem': 0.5,  # Default similarity
                    's_graph': 0.5,
                    's_combined': 0.5
                })
            
            logger.warning(f"Using {len(anchors)} fallback nodes with default scores")
            return anchors
            
        except Exception as e:
            logger.error(f"Fallback anchor selection failed: {e}")
            return []
