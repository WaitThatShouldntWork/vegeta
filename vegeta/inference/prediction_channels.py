"""
Prediction Channels for Generative Model

Three specialized channels for generating expected observations:
- SemanticPredictionChannel: generates expected utterance embeddings
- StructuralPredictionChannel: generates expected graph patterns
- TermsPredictionChannel: generates expected key terms

Each channel implements the g(v) function for its specific observation type.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
import requests

from ..core.exceptions import PredictionError
from .types import HiddenStates

logger = logging.getLogger(__name__)


class SemanticPredictionChannel:
    """
    Generates expected semantic observations u'_sem

    Pools node embeddings from the candidate subgraph and fuses with
    structural information to create expected utterance embedding.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ollama_url = f"{config['ollama_base_url']}/api/embeddings"
        self.model_name = config.get('embedding_model', 'nomic-embed-text:latest')
        self.pooling_method = config.get('pooling_method', 'mean')

    def predict(self, v: HiddenStates, context: Dict[str, Any]) -> NDArray[np.float32]:
        """
        Generate expected semantic vector u'_sem

        Args:
            v: Hidden states (contains subgraph ID)
            context: Context with graph access and node embeddings

        Returns:
            Expected semantic vector (768-dim normalized)
        """
        try:
            subgraph_id = v.z_subgraph
            graph = context.get('graph')

            if not graph or not subgraph_id:
                logger.warning("⚠ No graph or subgraph_id for semantic prediction")
                return np.zeros(768, dtype=np.float32)

            # Get node embeddings from subgraph
            node_embeddings = self._get_subgraph_embeddings(subgraph_id, graph)

            if not node_embeddings:
                logger.warning(f"⚠ No embeddings found for subgraph {subgraph_id}")
                return np.zeros(768, dtype=np.float32)

            # Pool node embeddings
            pooled_embedding = self._pool_embeddings(node_embeddings)

            # Get structural sketch
            structural_sketch = self._get_structural_sketch(subgraph_id, graph)

            # Fuse semantic and structural information
            fused_embedding = self._fuse_semantic_structural(pooled_embedding, structural_sketch)

            # Normalize final embedding
            norm = np.linalg.norm(fused_embedding)
            if norm > 0:
                fused_embedding = fused_embedding / norm

            return fused_embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"❌ Semantic prediction failed: {e}")
            return np.zeros(768, dtype=np.float32)

    def _get_subgraph_embeddings(self, subgraph_id: str, graph) -> List[NDArray[np.float32]]:
        """Extract node embeddings from subgraph"""
        # This would query the graph database for nodes in the subgraph
        # and retrieve their sem_emb properties
        # For now, return mock embeddings
        return [np.random.normal(0, 1, 768).astype(np.float32) for _ in range(5)]

    def _pool_embeddings(self, embeddings: List[NDArray[np.float32]]) -> NDArray[np.float32]:
        """Pool multiple node embeddings into single vector"""
        if not embeddings:
            return np.zeros(768, dtype=np.float32)

        embeddings_array = np.stack(embeddings)

        if self.pooling_method == 'mean':
            pooled = np.mean(embeddings_array, axis=0)
        elif self.pooling_method == 'max':
            pooled = np.max(embeddings_array, axis=0)
        elif self.pooling_method == 'attention':
            # Simple attention-like pooling
            attention_weights = np.random.normal(0, 1, len(embeddings))
            attention_weights = np.exp(attention_weights) / np.sum(np.exp(attention_weights))
            pooled = np.sum(embeddings_array * attention_weights[:, np.newaxis], axis=0)
        else:
            pooled = np.mean(embeddings_array, axis=0)

        return pooled

    def _get_structural_sketch(self, subgraph_id: str, graph) -> Dict[str, float]:
        """Get structural patterns from subgraph"""
        # This would analyze the subgraph structure and return
        # normalized counts of node labels and edge types
        return {
            'Entity': 0.4,
            'Film': 0.3,
            'Person': 0.2,
            'INSTANCE_OF': 0.5,
            'ACTED_IN': 0.3
        }

    def _fuse_semantic_structural(self,
                                 semantic_vec: NDArray[np.float32],
                                 structural_sketch: Dict[str, float]) -> NDArray[np.float32]:
        """Fuse semantic embedding with structural information"""
        # Convert structural sketch to vector
        structural_keys = sorted(structural_sketch.keys())
        structural_vec = np.array([structural_sketch[key] for key in structural_keys])

        # Pad or truncate structural vector to match semantic dimension
        target_dim = len(semantic_vec)
        if len(structural_vec) < target_dim:
            structural_vec = np.pad(structural_vec, (0, target_dim - len(structural_vec)))
        elif len(structural_vec) > target_dim:
            structural_vec = structural_vec[:target_dim]

        # Concatenate and fuse
        concat_vec = np.concatenate([semantic_vec, structural_vec])

        # Simple fusion: weighted combination
        semantic_weight = 0.7
        structural_weight = 0.3

        # Split back into semantic and structural parts
        mid_point = len(semantic_vec)
        fused_semantic = semantic_weight * concat_vec[:mid_point]
        fused_structural = structural_weight * concat_vec[mid_point:mid_point*2]

        # Combine
        result = fused_semantic + fused_structural[:len(fused_semantic)]

        return result


class StructuralPredictionChannel:
    """
    Generates expected structural observations u'_struct

    Predicts expected graph patterns based on checklist requirements
    and subgraph structure.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def predict(self, v: HiddenStates, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Generate expected structural patterns u'_struct

        Args:
            v: Hidden states (contains checklist and subgraph info)
            context: Context with graph access

        Returns:
            Dictionary of expected structural counts/patterns
        """
        try:
            checklist_name = v.z_checklist
            subgraph_id = v.z_subgraph

            # Get checklist requirements
            checklist_specs = self._get_checklist_specs(checklist_name, context)

            # Get actual subgraph structure
            subgraph_structure = self._get_subgraph_structure(subgraph_id, context)

            # Generate expected structure based on checklist requirements
            expected_structure = self._generate_expected_structure(checklist_specs, subgraph_structure)

            # Normalize counts
            expected_structure = self._normalize_structure(expected_structure)

            return expected_structure

        except Exception as e:
            logger.error(f"❌ Structural prediction failed: {e}")
            return {}

    def _get_checklist_specs(self, checklist_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get structural requirements from checklist"""
        # This would query the SlotSpec nodes for the checklist
        # to understand what structural patterns are expected
        return {
            'required_slots': ['film', 'music_track', 'composer'],
            'expected_relations': ['INSTANCE_OF', 'COMPOSED_BY'],
            'expected_node_types': ['Film', 'MusicTrack', 'Person']
        }

    def _get_subgraph_structure(self, subgraph_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get actual structure of the subgraph"""
        # This would analyze the actual subgraph and return
        # counts of node types and relations
        return {
            'node_counts': {'Film': 1, 'Person': 2, 'MusicTrack': 1},
            'relation_counts': {'INSTANCE_OF': 4, 'ACTED_IN': 2, 'COMPOSED_BY': 1}
        }

    def _generate_expected_structure(self,
                                   checklist_specs: Dict[str, Any],
                                   subgraph_structure: Dict[str, Any]) -> Dict[str, float]:
        """Generate expected structure based on checklist requirements"""
        expected = {}

        # Add expected node types from checklist
        for node_type in checklist_specs.get('expected_node_types', []):
            expected[f"node_{node_type}"] = 1.0

        # Add expected relations from checklist
        for relation in checklist_specs.get('expected_relations', []):
            expected[f"rel_{relation}"] = 1.0

        # Add actual counts from subgraph (normalized)
        for node_type, count in subgraph_structure.get('node_counts', {}).items():
            expected[f"actual_node_{node_type}"] = float(count)

        for relation, count in subgraph_structure.get('relation_counts', {}).items():
            expected[f"actual_rel_{relation}"] = float(count)

        return expected

    def _normalize_structure(self, structure: Dict[str, float]) -> Dict[str, float]:
        """Normalize structural counts to [0,1] range"""
        if not structure:
            return {}

        # Use log1p to compress heavy tails, then normalize
        normalized = {}
        for key, value in structure.items():
            normalized[key] = np.log1p(value) / 5.0  # Cap at reasonable maximum
            normalized[key] = min(1.0, max(0.0, normalized[key]))

        return normalized


class TermsPredictionChannel:
    """
    Generates expected terms observations u'_terms

    Predicts key terms that should appear in utterances about this subgraph,
    based on node names, checklist lexicon, and domain knowledge.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_terms = config.get('max_terms', 20)

    def predict(self, v: HiddenStates, context: Dict[str, Any]) -> List[str]:
        """
        Generate expected key terms u'_terms

        Args:
            v: Hidden states (contains checklist and subgraph info)
            context: Context with graph access

        Returns:
            List of expected key terms
        """
        try:
            checklist_name = v.z_checklist
            subgraph_id = v.z_subgraph

            terms_sources = []

            # Get terms from subgraph node names and aliases
            subgraph_terms = self._get_subgraph_terms(subgraph_id, context)
            terms_sources.extend(subgraph_terms)

            # Get terms from checklist lexicon
            checklist_terms = self._get_checklist_lexicon(checklist_name, context)
            terms_sources.extend(checklist_terms)

            # Get domain-specific terms
            domain_terms = self._get_domain_terms(checklist_name)
            terms_sources.extend(domain_terms)

            # Process and filter terms
            processed_terms = self._process_terms(terms_sources)

            # Limit to max_terms
            processed_terms = processed_terms[:self.max_terms]

            return processed_terms

        except Exception as e:
            logger.error(f"❌ Terms prediction failed: {e}")
            return []

    def _get_subgraph_terms(self, subgraph_id: str, context: Dict[str, Any]) -> List[str]:
        """Extract terms from subgraph node names and aliases"""
        # This would query nodes in the subgraph and extract their names/aliases
        return ['Skyfall', 'Adele', 'film', 'music', 'composer']

    def _get_checklist_lexicon(self, checklist_name: str, context: Dict[str, Any]) -> List[str]:
        """Get domain-specific terms from checklist"""
        if checklist_name == 'VerifyMusicRights':
            return ['music', 'rights', 'sync', 'clearance', 'territory', 'composer', 'track']
        return []

    def _get_domain_terms(self, checklist_name: str) -> List[str]:
        """Get general domain terms"""
        if 'Music' in checklist_name:
            return ['royalty', 'license', 'copyright', 'distribution']
        return ['verify', 'check', 'validate']

    def _process_terms(self, terms: List[str]) -> List[str]:
        """Process and filter terms"""
        processed = []

        for term in terms:
            # Convert to lowercase
            term = term.lower()

            # Skip very short terms
            if len(term) < 3:
                continue

            # Skip duplicates
            if term not in processed:
                processed.append(term)

        # Sort by some relevance score (for now, just alphabetically)
        processed.sort()

        return processed
