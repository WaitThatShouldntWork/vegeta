"""
Conversation context utilities
"""

from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class ConversationContext:
    """
    Conversation context for multi-turn interactions
    """
    session_id: str
    turn_count: int
    recent_turns: List[Dict[str, Any]]
    conversation_summary: str
    success_rate: float
    domain_expertise: Dict[str, float]
    session_duration: float
    adaptation_context: Dict[str, Any]
