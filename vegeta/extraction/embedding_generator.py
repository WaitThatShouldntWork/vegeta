"""
Semantic embedding generation for queries and terms
"""

import logging
import numpy as np
from typing import List, Optional

from ..utils.llm_client import LLMClient
from ..core.config import Config
from ..core.exceptions import ExtractionError

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    Generate semantic embeddings for utterances and terms
    """
    
    def __init__(self, llm_client: LLMClient, config: Config):
        self.llm_client = llm_client
        self.config = config
    
    def create_u_sem(self, utterance: str) -> np.ndarray:
        """Create semantic embedding u_sem for the utterance"""
        try:
            embedding = self.llm_client.get_embedding(utterance)
            if embedding is None:
                raise ExtractionError(f"Failed to generate embedding for utterance: {utterance}")
            
            # L2 normalize (already done in LLMClient, but ensure it)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create utterance embedding: {e}")
            raise ExtractionError(f"Utterance embedding failed: {e}")
    
    def create_u_terms_vec(self, terms: List[str]) -> Optional[np.ndarray]:
        """Create term vector by averaging individual term embeddings"""
        if not terms:
            return None
        
        try:
            embeddings = []
            for term in terms:
                emb = self.llm_client.get_embedding(term)
                if emb is not None:
                    embeddings.append(emb)
            
            if embeddings:
                # Average and normalize
                avg_embedding = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    return avg_embedding / norm
                return avg_embedding
            
            logger.warning(f"No valid embeddings generated for terms: {terms}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to create terms vector: {e}")
            return None
