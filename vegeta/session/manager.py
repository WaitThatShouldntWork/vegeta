"""
Enhanced multi-turn session manager
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

from ..core.exceptions import SessionError

logger = logging.getLogger(__name__)

@dataclass
class ConversationTurn:
    """Record of a single conversation turn"""
    turn_id: int
    timestamp: float
    user_utterance: str
    system_action: str  # ASK, SEARCH, ANSWER
    system_response: str
    target: Optional[str]  # What we asked about or searched for
    confidence: float
    entropy: float
    reasoning: str
    
    # Bayesian state at this turn
    posteriors: Dict[str, Any]
    priors: Dict[str, Any] 
    top_candidates: List[Dict[str, Any]]
    
    # Performance tracking
    user_feedback: Optional[str] = None  # User's response if ASK
    outcome: Optional[str] = None  # success/failure/partial
    
    def to_dict(self):
        return asdict(self)

@dataclass 
class SessionState:
    """Persistent session state that carries forward between turns"""
    session_id: str
    start_time: float
    last_active: float
    
    # Conversation history
    turns: List[ConversationTurn]
    conversation_summary: str
    
    # Bayesian state carryover
    belief_state: Dict[str, Any]  # q_t(v) from last turn
    adaptation_context: Dict[str, Any]  # For meta-learning
    
    # Performance tracking
    cumulative_confidence: List[float]
    success_rate: float
    domain_expertise: Dict[str, float]
    
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'start_time': self.start_time,
            'last_active': self.last_active,
            'turns': [turn.to_dict() for turn in self.turns],
            'conversation_summary': self.conversation_summary,
            'belief_state': self.belief_state,
            'adaptation_context': self.adaptation_context,
            'cumulative_confidence': self.cumulative_confidence,
            'success_rate': self.success_rate,
            'domain_expertise': self.domain_expertise
        }

class SessionManager:
    """
    Enhanced multi-turn session manager with belief state carryover
    """
    
    def __init__(self, session_config: Dict[str, Any]):
        self.sessions: Dict[str, SessionState] = {}
        self.session_timeout = session_config.get('timeout', 3600)  # 1 hour
        self.max_turns = session_config.get('max_turns', 20)
        self.inertia_rho = session_config.get('inertia_rho', 0.7)  # Belief carryover strength
        
    def start_session(self, user_id: Optional[str] = None) -> str:
        """Start a new conversation session"""
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = SessionState(
            session_id=session_id,
            start_time=time.time(),
            last_active=time.time(),
            turns=[],
            conversation_summary="",
            belief_state={},
            adaptation_context={},
            cumulative_confidence=[],
            success_rate=0.5,  # Neutral starting point
            domain_expertise={}
        )
        
        logger.info(f"Started new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get session state, checking for timeout"""
        if session_id not in self.sessions:
            return None
            
        session = self.sessions[session_id]
        
        # Check timeout
        if time.time() - session.last_active > self.session_timeout:
            logger.info(f"Session {session_id} timed out")
            del self.sessions[session_id]
            return None
            
        return session
    
    def update_session_state(self, session_id: str, turn_result: Dict[str, Any]) -> bool:
        """Update session with results from current turn"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        # Create turn record
        turn = ConversationTurn(
            turn_id=len(session.turns),
            timestamp=time.time(),
            user_utterance=turn_result['user_utterance'],
            system_action=turn_result['action'],
            system_response=turn_result.get('response', ''),
            target=turn_result.get('target'),
            confidence=turn_result['confidence'],
            entropy=turn_result.get('entropy', 0.0),
            reasoning=turn_result['reasoning'],
            posteriors=turn_result['posteriors'],
            priors=turn_result['priors'],
            top_candidates=turn_result.get('top_candidates', [])
        )
        
        session.turns.append(turn)
        session.last_active = time.time()
        
        # Update belief state for next turn
        session.belief_state = turn_result['posteriors']
        
        # Update conversation summary
        session.conversation_summary = self._update_conversation_summary(session)
        
        # Update performance tracking
        session.cumulative_confidence.append(turn_result['confidence'])
        
        # Update adaptation context for meta-learning
        session.adaptation_context = self._update_adaptation_context(session, turn_result)
        
        logger.info(f"Updated session {session_id}, turn {turn.turn_id}")
        return True
    
    def add_user_feedback(self, session_id: str, feedback: str, outcome: str = None) -> bool:
        """Add user feedback to the last turn (for ASK responses)"""
        session = self.get_session(session_id)
        if not session or not session.turns:
            return False
        
        last_turn = session.turns[-1]
        last_turn.user_feedback = feedback
        if outcome:
            last_turn.outcome = outcome
            
        # Update success rate
        if outcome in ['success', 'failure']:
            total_outcomes = sum(1 for turn in session.turns if turn.outcome in ['success', 'failure'])
            successes = sum(1 for turn in session.turns if turn.outcome == 'success')
            session.success_rate = successes / max(total_outcomes, 1)
        
        return True
    
    def get_conversation_context(self, session_id: str, window_size: int = 5) -> Dict[str, Any]:
        """Get recent conversation context for current turn processing"""
        session = self.get_session(session_id)
        if not session:
            return {}
        
        recent_turns = session.turns[-window_size:] if session.turns else []
        
        return {
            'session_id': session_id,
            'turn_count': len(session.turns),
            'recent_turns': [turn.to_dict() for turn in recent_turns],
            'conversation_summary': session.conversation_summary,
            'success_rate': session.success_rate,
            'domain_expertise': session.domain_expertise,
            'session_duration': time.time() - session.start_time,
            'adaptation_context': session.adaptation_context
        }
    
    def _update_conversation_summary(self, session: SessionState) -> str:
        """Update conversation summary with recent developments"""
        if not session.turns:
            return ""
        
        # Simple summary: last few key facts and current focus
        recent_turns = session.turns[-3:]
        
        summary_parts = []
        
        # What are we trying to figure out?
        if recent_turns:
            last_turn = recent_turns[-1]
            if last_turn.system_action == 'ASK' and last_turn.target:
                summary_parts.append(f"Currently asking about: {last_turn.target}")
            elif last_turn.top_candidates:
                top_candidate = last_turn.top_candidates[0]
                entity_name = top_candidate.get('entity_name', top_candidate.get('name', 'unknown'))
                summary_parts.append(f"Focusing on: {entity_name} (confidence: {last_turn.confidence:.2f})")
        
        # Recent confirmations or facts
        for turn in recent_turns:
            if turn.user_feedback and turn.outcome == 'success':
                summary_parts.append(f"Confirmed: {turn.target} = {turn.user_feedback}")
        
        return "; ".join(summary_parts[-3:])  # Keep it concise
    
    def _update_adaptation_context(self, session: SessionState, turn_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update context for meta-learning adaptation"""
        
        context = session.adaptation_context.copy()
        
        # Domain detection
        domain = turn_result.get('domain', 'general')
        context['current_domain'] = domain
        
        # Performance trends
        if 'performance_trend' not in context:
            context['performance_trend'] = []
        
        context['performance_trend'].append({
            'turn': len(session.turns),
            'confidence': turn_result['confidence'],
            'action': turn_result['action'],
            'entropy': turn_result.get('entropy', 0.0)
        })
        
        # Keep only recent history
        context['performance_trend'] = context['performance_trend'][-10:]
        
        # Complexity assessment
        complexity_indicators = {
            'long_conversation': len(session.turns) > 10,
            'low_confidence_streak': len([t for t in session.turns[-3:] if t.confidence < 0.5]) >= 2,
            'multiple_asks': len([t for t in session.turns if t.system_action == 'ASK']) > 3,
            'domain_uncertainty': turn_result.get('novelty', 0.0) > 0.3
        }
        
        context['complexity_score'] = sum(complexity_indicators.values()) / len(complexity_indicators)
        
        return context
