"""
Natural language answer generation
"""

import logging
from typing import Dict, Any, List

from ..core.config import Config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Generate natural language answers for ANSWER actions
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate_answer(self, decision: Dict[str, Any], 
                       candidates: List[Dict[str, Any]]) -> str:
        """Generate an answer when confidence is high enough"""
        
        if decision['action'] != 'ANSWER':
            return None
        
        # Get top candidate
        top_candidate = candidates[0] if candidates else {}
        confidence = decision['confidence']
        
        # Build answer from top candidate
        target_name = decision.get('target', 'unknown')
        
        if top_candidate.get('entity_name'):
            answer = f"Based on your query, I believe you're looking for information about {top_candidate['entity_name']}"
        elif top_candidate.get('anchor_name'):
            answer = f"Based on your query, I believe you're looking for information about {top_candidate['anchor_name']}"
        else:
            answer = f"Based on your query, I believe the answer is: {target_name}"
        
        # Add a couple of connected names if available
        connected_names = top_candidate.get('connected_names', [])
        if connected_names:
            answer += f", which is related to {', '.join(connected_names[:2])}"
        
        answer += f". (Confidence: {confidence:.1%})"
        
        return answer
