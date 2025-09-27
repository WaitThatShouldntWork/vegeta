"""
Natural language answer generation
"""

import logging
from typing import Dict, Any, List, Optional

from ..core.config import Config

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Generate natural language answers for ANSWER actions
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate_answer(self, decision: Dict[str, Any],
                       candidates: List[Dict[str, Any]],
                       session_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate an answer when confidence is high enough"""

        if decision['action'] != 'ANSWER':
            return None

        # Get top candidate
        top_candidate = candidates[0] if candidates else {}
        confidence = decision['confidence']

        # Debug logging
        logger.debug(f"Answer generation - decision target: {decision.get('target')}")
        logger.debug(f"Answer generation - candidates count: {len(candidates)}")
        logger.debug(f"Answer generation - has session context: {session_context is not None}")

        if session_context:
            recent_turns = session_context.get('recent_turns', [])
            logger.debug(f"Answer generation - conversation turns available: {len(recent_turns)}")
            if recent_turns:
                last_turn = recent_turns[-1]
                logger.debug(f"Answer generation - last user utterance: {last_turn.get('user_utterance', '')}")
                logger.debug(f"Answer generation - last system response: {last_turn.get('system_response', '')}")

        if candidates:
            logger.debug(f"Answer generation - top candidate keys: {list(top_candidate.keys())}")
            logger.debug(f"Answer generation - top candidate entity_name: {top_candidate.get('entity_name')}")
            logger.debug(f"Answer generation - top candidate anchor_name: {top_candidate.get('anchor_name')}")

        # Build answer from top candidate - using graph-based inference results
        if top_candidate.get('entity_name'):
            answer = f"Based on your query, I believe you're looking for information about {top_candidate['entity_name']}"
        elif top_candidate.get('anchor_name'):
            answer = f"Based on your query, I believe you're looking for information about {top_candidate['anchor_name']}"
        elif decision.get('target') and decision['target'] != 'unknown':
            answer = f"Based on your query, I believe the answer is: {decision['target']}"
        else:
            answer = "Based on your query, I need more information to provide a specific answer."

        # Add a couple of connected names if available
        connected_names = top_candidate.get('connected_names', [])
        if connected_names:
            answer += f", which is related to {', '.join(connected_names[:2])}"

        answer += f". (Confidence: {confidence:.1%})"

        return answer
