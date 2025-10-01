"""
Main graph retrieval coordinator
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from ..utils.database import DatabaseManager
from ..core.config import Config
from ..core.exceptions import RetrievalError
from .anchor_finder import AnchorFinder
from .candidate_expander import CandidateExpander

logger = logging.getLogger(__name__)

class GraphRetriever:
    """
    Coordinates graph retrieval using anchor selection and candidate expansion
    """
    
    def __init__(self, db_manager: DatabaseManager, config: Config):
        self.db_manager = db_manager
        self.config = config
        self.defaults = config.system_defaults
        
        # Initialize sub-components
        self.anchor_finder = AnchorFinder(db_manager, config)
        self.candidate_expander = CandidateExpander(db_manager, config)
    
    def retrieve_candidates(self, observation_u: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main retrieval pipeline: find anchors and expand to candidates
        
        Args:
            observation_u: User observation with embeddings and metadata
            
        Returns:
            Retrieval context with anchors, candidates, and metadata
        """
        try:
            # Extract semantic embedding
            u_sem = observation_u.get('u_sem')
            if u_sem is None:
                raise RetrievalError("No semantic embedding found in observation")
            
            # Find anchor nodes using semantic similarity
            anchors = self.anchor_finder.find_anchor_nodes(
                u_sem, 
                k=self.defaults['k_anchors']
            )
            
            if not anchors:
                logger.warning("No anchor nodes found")
                return self._create_empty_context(observation_u)
            
            # Link extracted entities to graph
            entities = observation_u.get('u_meta', {}).get('extraction', {}).get('entities', [])
            linked_entity_ids = self._link_entities_to_graph(entities)
            
            # Determine active checklist and target labels
            active_checklist_name, target_labels = self._get_active_checklist_and_target_labels(observation_u)
            
            # Generate candidates based on strategy
            if target_labels:
                # Use target-focused candidate enumeration
                candidates = self.candidate_expander.enumerate_target_candidates_from_anchors(
                    anchors, target_labels
                )
            else:
                # Fallback to general subgraph expansion
                candidates = self.candidate_expander.expand_subgraphs(anchors)
            
            # Create retrieval context
            retrieval_context = {
                'anchors': anchors,
                'linked_entity_ids': linked_entity_ids,
                'candidates': candidates,
                'expansion_params': {
                    'hops': self.defaults['hops'], 
                    'k_anchors': self.defaults['k_anchors']
                },
                'utterance': observation_u.get('u_meta', {}).get('utterance', ''),
                'active_checklist': active_checklist_name,
                'target_labels': target_labels
            }
            
            logger.info(f"Retrieved {len(candidates)} candidates from {len(anchors)} anchors")
            return retrieval_context
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RetrievalError(f"Graph retrieval failed: {e}")
    
    def _link_entities_to_graph(self, entities: List[Dict]) -> List[str]:
        """Link extracted entities to graph nodes using full-text search"""
        
        linked_ids = []
        
        for entity in entities:
            try:
                # Use database manager's fulltext search
                matches = self.db_manager.get_entities_by_fulltext(
                    entity.get('normalized', ''), 
                    limit=3
                )
                
                for match in matches:
                    linked_ids.append(match['id'])
                    logger.info(f"Linked '{entity['surface']}' -> {match['name']} ({match['id']})")
                
            except Exception as e:
                logger.error(f"Entity linking failed for {entity}: {e}")
        
        return linked_ids
    
    def _get_active_checklist_and_target_labels(self, observation_u: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        """Determine the active checklist and its primary target labels in a query-aware way."""

        try:
            # First, check if there's an active checklist based on query content
            utterance = observation_u.get('u_meta', {}).get('utterance', '').lower()

            # Query-aware label selection based on keywords
            query_labels = []

            # Movie/film related queries
            if any(word in utterance for word in ['movie', 'film', 'bond', 'casino', 'skyfall', 'heat']):
                query_labels.extend(['Film', 'Person', 'Year'])

            # Music related queries
            if any(word in utterance for word in ['music', 'rights', 'composer', 'territory', 'sync']):
                query_labels.extend(['MusicTrack', 'Territory', 'Person'])

            # Award related queries
            if any(word in utterance for word in ['award', 'bafta', 'oscar', 'won', 'win']):
                query_labels.extend(['Award', 'Film', 'Year'])

            # Person related queries
            if any(word in utterance for word in ['actor', 'actress', 'director', 'person']):
                query_labels.extend(['Person', 'Film'])

            # Conversational queries (name, introduction, etc.)
            if any(word in utterance for word in ['name', 'i am', 'my name', 'hello', 'hi', 'introduce']):
                # For conversational queries, don't target specific entity types
                # Let the system use general retrieval or fallback to None
                logger.debug("Conversational query detected - using general retrieval")
                return None, []

            # If we found query-specific labels, use them
            if query_labels:
                logger.debug(f"Query-aware labels detected: {query_labels}")
                return None, list(set(query_labels))  # Remove duplicates

            # Fallback to checklist-based approach
            # Get all checklist specs
            query = """
            MATCH (ss:SlotSpec)
            RETURN ss.checklist_name AS checklist_name,
                   ss.name AS name,
                   ss.expect_labels AS expect_labels,
                   ss.required AS required,
                   ss.cardinality AS cardinality
            ORDER BY required DESC
            """
            records = self.db_manager.execute_query(query)

            # Filter to those with expect_labels
            candidates = []
            for r in records:
                expect_labels = r.get("expect_labels") or []
                if expect_labels:
                    candidates.append({
                        "checklist_name": r.get("checklist_name"),
                        "name": r.get("name"),
                        "expect_labels": expect_labels,
                        "required": bool(r.get("required")),
                        "cardinality": r.get("cardinality") or "ONE"
                    })

            if not candidates:
                # Ultimate fallback: use all common entity types
                return None, ['Film', 'Person', 'MusicTrack', 'Territory', 'Award', 'Year', 'Document']

            # Prefer required and cardinality ONE
            def sort_key(x: Dict[str, Any]):
                return (
                    1 if x["required"] else 0,
                    1 if str(x["cardinality"]).upper() == "ONE" else 0,
                    -len(x["expect_labels"])  # fewer labels preferred for specificity
                )

            candidates.sort(key=sort_key, reverse=True)
            picked = candidates[0]
            return picked["checklist_name"], picked["expect_labels"]

        except Exception as e:
            logger.error(f"Failed to fetch checklist target labels: {e}")
            return None, ['Film', 'Person', 'MusicTrack', 'Territory', 'Award', 'Year', 'Document']
    
    def _create_empty_context(self, observation_u: Dict[str, Any]) -> Dict[str, Any]:
        """Create empty retrieval context when no anchors found"""
        return {
            'anchors': [],
            'linked_entity_ids': [],
            'candidates': [],
            'expansion_params': {
                'hops': self.defaults['hops'], 
                'k_anchors': self.defaults['k_anchors']
            },
            'utterance': observation_u.get('u_meta', {}).get('utterance', ''),
            'active_checklist': None,
            'target_labels': []
        }
