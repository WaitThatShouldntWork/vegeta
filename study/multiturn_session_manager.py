# %% [markdown]
# # Multi-Turn Session Manager
# Implements conversation state management and posterior carryover for 20Q-style games
# Based on the design in attemp1TextOnly.py section "Lifecycle of the posterior across turns"
# 
# ## Enhanced Brain-Inspired Features:
# - Memory consolidation for posterior→prior transfer
# - Adaptive EIG planning horizon
# - Predictive coding for multi-step lookahead
# - Context-dependent attention mechanisms

# %%
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# %%
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

# %%
class MultiTurnSessionManager:
    """Manages multi-turn conversations with belief state carryover"""
    
    def __init__(self, 
                 session_timeout: int = 3600,  # 1 hour timeout
                 max_turns: int = 20,
                 inertia_rho: float = 0.7):
        self.sessions: Dict[str, SessionState] = {}
        self.session_timeout = session_timeout
        self.max_turns = max_turns
        self.inertia_rho = inertia_rho  # Belief carryover strength
        
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
    
    def build_priors_from_history(self, session: SessionState) -> Dict[str, Any]:
        """Build priors for next turn from conversation history and belief state"""
        
        # Base priors (domain defaults)
        base_priors = self._get_base_priors()
        
        if not session.turns:
            return base_priors
        
        # Get previous belief state
        previous_beliefs = session.belief_state
        
        # Apply inertia carryover: p_{t+1}(z) ∝ (q_t(z))^ρ × base_prior × context
        history_priors = {}
        
        # 1. Checklist prior with inertia
        if 'checklist' in previous_beliefs:
            prev_checklist = previous_beliefs['checklist']
            base_checklist = base_priors['checklist']
            
            # Apply inertia: (q_t)^ρ
            inertia_checklist = {k: v**self.inertia_rho for k, v in prev_checklist.items()}
            
            # Combine with base prior and context boost
            context_boost = self._compute_context_boost(session)
            
            combined_checklist = {}
            all_keys = set(inertia_checklist.keys()) | set(base_checklist.keys())
            
            for key in all_keys:
                inertia_val = inertia_checklist.get(key, 0.0)
                base_val = base_checklist.get(key, 0.1)  # Small floor
                context_val = context_boost.get(key, 1.0)
                
                combined_checklist[key] = inertia_val * base_val * context_val
            
            # Normalize
            total = sum(combined_checklist.values())
            if total > 0:
                history_priors['checklist'] = {k: v/total for k, v in combined_checklist.items()}
            else:
                history_priors['checklist'] = base_checklist
        else:
            history_priors['checklist'] = base_priors['checklist']
        
        # 2. Goal prior with conversation context
        history_priors['goal'] = self._build_goal_prior_from_history(session, base_priors['goal'])
        
        # 3. Subgraph prior with recency boost
        history_priors['subgraph'] = self._build_subgraph_prior_from_history(session, base_priors['subgraph'])
        
        # 4. Dialogue act with transition patterns
        history_priors['dialogue_act'] = self._build_dialogue_act_prior_from_history(session, base_priors['dialogue_act'])
        
        # 5. Novelty with adaptation
        history_priors['novelty'] = self._adapt_novelty_prior(session, base_priors['novelty'])
        
        return history_priors
    
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
    
    def _get_base_priors(self) -> Dict[str, Any]:
        """Get domain-neutral base priors"""
        return {
            'checklist': {'None': 0.7, 'IdentifyFilm': 0.2, 'RecommendFilm': 0.1},
            'goal': {'identify': 0.4, 'recommend': 0.2, 'verify': 0.2, 'explore': 0.2},
            'subgraph': {},  # Filled by retrieval
            'dialogue_act': {'clarify': 0.25, 'confirm': 0.25, 'request': 0.25, 'provide': 0.25},
            'novelty': 0.0
        }
    
    def _compute_context_boost(self, session: SessionState) -> Dict[str, float]:
        """Compute context-dependent boosts for checklist priors"""
        boost = {}
        
        if not session.turns:
            return boost
        
        # Recent checklist mentions get boost
        recent_checklists = []
        for turn in session.turns[-3:]:  # Last 3 turns
            if 'checklist' in turn.posteriors:
                recent_checklists.extend(turn.posteriors['checklist'].keys())
        
        # Frequency-based boost
        for checklist in recent_checklists:
            boost[checklist] = boost.get(checklist, 1.0) + 0.2
        
        return boost
    
    def _build_goal_prior_from_history(self, session: SessionState, base_goal: Dict[str, float]) -> Dict[str, float]:
        """Build goal prior incorporating conversation pattern"""
        if not session.turns:
            return base_goal
        
        # Look at recent dialogue acts to infer goal
        recent_acts = []
        for turn in session.turns[-3:]:
            if turn.system_action:
                recent_acts.append(turn.system_action)
        
        goal_prior = base_goal.copy()
        
        # If we've been asking questions, likely in identify mode
        if recent_acts.count('ASK') >= 2:
            goal_prior['identify'] *= 1.5
            goal_prior['recommend'] *= 0.8
        
        # If user gave specific preferences, likely recommendation
        for turn in session.turns[-2:]:
            if 'recommend' in turn.user_utterance.lower() or 'similar' in turn.user_utterance.lower():
                goal_prior['recommend'] *= 1.4
                goal_prior['identify'] *= 0.9
        
        # Normalize
        total = sum(goal_prior.values())
        return {k: v/total for k, v in goal_prior.items()}
    
    def _build_subgraph_prior_from_history(self, session: SessionState, base_subgraph: Dict[str, float]) -> Dict[str, float]:
        """Build subgraph prior with recency boost for mentioned entities"""
        # This will be filled by the retrieval system, but we can provide hints
        # about entities that were recently mentioned or confirmed
        
        subgraph_hints = {}
        
        # Recent candidates that got high confidence
        for turn in session.turns[-5:]:
            if turn.confidence > 0.6 and turn.top_candidates:
                for candidate in turn.top_candidates[:2]:
                    cand_id = candidate.get('id', candidate.get('entity_id'))
                    if cand_id:
                        # Recency decay
                        turns_ago = len(session.turns) - turn.turn_id
                        recency_weight = 0.9 ** turns_ago
                        subgraph_hints[cand_id] = recency_weight * turn.confidence
        
        return subgraph_hints
    
    def _build_dialogue_act_prior_from_history(self, session: SessionState, base_dialogue: Dict[str, float]) -> Dict[str, float]:
        """Build dialogue act prior based on conversation flow patterns"""
        if not session.turns:
            return base_dialogue
        
        dialogue_prior = base_dialogue.copy()
        last_turn = session.turns[-1]
        
        # Dialogue act transitions (simple Markov model)
        if last_turn.system_action == 'ASK':
            # After asking, expect user to provide or clarify
            dialogue_prior['provide'] *= 2.0
            dialogue_prior['clarify'] *= 1.5
            dialogue_prior['request'] *= 0.5
        elif last_turn.system_action == 'ANSWER':
            # After answering, expect confirmation or new request
            dialogue_prior['confirm'] *= 2.0
            dialogue_prior['request'] *= 1.5
            dialogue_prior['provide'] *= 0.5
        
        # Normalize
        total = sum(dialogue_prior.values())
        return {k: v/total for k, v in dialogue_prior.items()}
    
    def _adapt_novelty_prior(self, session: SessionState, base_novelty: float) -> float:
        """Adapt novelty based on recent performance"""
        if not session.turns:
            return base_novelty
        
        # If recent turns had low confidence, increase novelty
        recent_confidences = [turn.confidence for turn in session.turns[-3:]]
        avg_confidence = np.mean(recent_confidences) if recent_confidences else 0.5
        
        # Low confidence suggests we're in unfamiliar territory
        if avg_confidence < 0.4:
            return min(0.8, base_novelty + 0.3)
        elif avg_confidence > 0.7:
            return max(0.0, base_novelty - 0.2)
        else:
            return base_novelty
    
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

# %%
# Example usage and testing
if __name__ == "__main__":
    
    # Test session management
    manager = MultiTurnSessionManager()
    
    # Start session
    session_id = manager.start_session()
    print(f"Started session: {session_id}")
    
    # Simulate multi-turn conversation
    turns = [
        {
            'user_utterance': "I'm thinking of a film. Try to guess it.",
            'action': 'ASK',
            'target': 'specifics',
            'confidence': 0.1,
            'entropy': 3.5,
            'reasoning': 'Initial query too vague',
            'posteriors': {
                'checklist': {'IdentifyFilm': 0.8, 'None': 0.2},
                'goal': {'identify': 0.9, 'recommend': 0.1},
                'subgraph': {}
            },
            'priors': {},
            'response': 'Could you give me more specific details about what you\'re looking for?'
        },
        {
            'user_utterance': "It's a sci-fi movie from the late 90s",
            'action': 'ASK', 
            'target': 'actor',
            'confidence': 0.4,
            'entropy': 2.1,
            'reasoning': 'Narrowed down by genre and time',
            'posteriors': {
                'checklist': {'IdentifyFilm': 0.9, 'None': 0.1},
                'goal': {'identify': 0.95, 'recommend': 0.05},
                'subgraph': {'matrix_1': 0.6, 'blade_runner_1': 0.3, 'contact_1': 0.1}
            },
            'priors': {},
            'response': 'Can you tell me about any of the actors in this film?'
        }
    ]
    
    # Process turns
    for i, turn_data in enumerate(turns):
        success = manager.update_session_state(session_id, turn_data)
        print(f"Turn {i+1} processed: {success}")
        
        # Get context for next turn
        context = manager.get_conversation_context(session_id)
        print(f"Context: {context['conversation_summary']}")
        
        # Build priors for next turn
        priors = manager.build_priors_from_history(manager.get_session(session_id))
        print(f"Updated priors: {priors['checklist']}")
        print()
