"""
Natural language question generation
"""

import logging
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient
from ..core.config import Config
from ..core.exceptions import GenerationError

logger = logging.getLogger(__name__)

class QuestionGenerator:
    """
    Generate natural language questions for ASK actions
    """
    
    def __init__(self, llm_client: LLMClient, config: Config):
        self.llm_client = llm_client
        self.config = config
    
    def generate_question_llm(self, decision: Dict[str, Any],
                             candidates: List[Dict[str, Any]],
                             session_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a natural language question using LLM"""
        
        if decision['action'] != 'ASK':
            return f"Action: {decision['action']} (target: {decision['target']})"
        
        target_slot = decision['target']
        
        # Handle LLM-based clarification (intelligent instant response)
        if 'quick_response' in decision:
            return decision['quick_response']

        # Get top candidate for context
        top_candidate = candidates[0] if candidates else {}

        # Build conversation history for context (Langgraph-style HumanMessage/AIMessage)
        conversation_history = self._format_conversation_history(session_context)

        # Build context for question generation
        context = {
            "task_context": "identify the target",  # Default task context
            "slot": target_slot,
            "top_candidate": {
                "anchor": top_candidate.get('anchor_name', 'unknown'),
                "connected": top_candidate.get('connected_names', [])[:3]
            },
            "confidence": decision['confidence'],
            "reasoning": f"Need to disambiguate {target_slot} to improve identification"
        }
        
        # Determine fallback template based on target
        fallback_templates = {
            'specifics': "Could you provide more specific details about what you're looking for?",
            'preferences': "What preferences do you have that might help me recommend something?",
            'clarification': "Could you clarify what you're looking for?",
            'film': "Could you tell me more about the film you're thinking of?",
            'actor': "Who are some of the actors in this film?",
            'year': "Do you remember approximately when this was released?",
            'genre': "What genre or type of film is this?"
        }
        
        fallback_template = fallback_templates.get(target_slot, f"Could you tell me more about the {target_slot}?")
        
        # Create prompt for question generation with conversation history
        prompt = f"""Generate a natural, conversational question to ask the user.

Context:
- Task: {context['task_context']}
- Need to ask about: {context['slot']}
- Current focus: {context['top_candidate']['anchor']}
- Confidence: {context['confidence']:.1%}
- Why: {context['reasoning']}

{conversation_history}

Generate ONE short, natural question that would help reduce uncertainty. Be conversational and helpful.
Consider the conversation history above - don't ask about things that have already been confirmed or discussed.

Question:"""
        
        try:
            response = self.llm_client.call_ollama_json(f"{prompt}\n\nReturn JSON with format: {{\"question\": \"your question here\"}}")
            if 'question' in response:
                return response['question']
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
        
        # Fallback to context-appropriate template
        return fallback_template

    def test_conversation_context(self, session_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test method to demonstrate conversation context usage

        Returns the formatted conversation history that would be sent to LLM
        """
        formatted_history = self._format_conversation_history(session_context)

        return {
            'conversation_formatted': formatted_history,
            'turns_count': len(session_context.get('recent_turns', [])),
            'has_conversation_context': bool(formatted_history.strip()),
            'llm_prompt_would_include': bool(session_context and session_context.get('recent_turns'))
        }

    def _format_conversation_history(self, session_context: Dict[str, Any]) -> str:
        """Format conversation history in Langgraph-style HumanMessage/AIMessage format"""
        if not session_context:
            return ""

        recent_turns = session_context.get('recent_turns', [])
        if not recent_turns:
            return ""

        formatted_history = "\n\nRecent conversation history:\n"

        for i, turn in enumerate(recent_turns[-5:]):  # Last 5 turns for comprehensive context
            turn_num = len(recent_turns) - 5 + i + 1

            # Format as HumanMessage
            user_msg = (turn.get('user_utterance') or '').strip()
            if user_msg:
                formatted_history += f"Turn {turn_num} - Human: {user_msg}\n"

            # Format as AIMessage
            ai_msg = (turn.get('system_response') or '').strip()
            if ai_msg:
                formatted_history += f"Turn {turn_num} - Assistant: {ai_msg}\n"

            # Include user feedback if available (for ASK responses)
            user_feedback = (turn.get('user_feedback') or '').strip()
            if user_feedback:
                formatted_history += f"Turn {turn_num} - Human (response): {user_feedback}\n"

        return formatted_history
