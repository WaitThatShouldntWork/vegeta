"""
Natural language question generation
"""

import logging
from typing import Dict, Any, List

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
                             candidates: List[Dict[str, Any]]) -> str:
        """Generate a natural language question using LLM"""
        
        if decision['action'] != 'ASK':
            return f"Action: {decision['action']} (target: {decision['target']})"
        
        target_slot = decision['target']
        
        # Handle LLM-based clarification (intelligent instant response)
        if 'quick_response' in decision:
            return decision['quick_response']
        
        # Get top candidate for context
        top_candidate = candidates[0] if candidates else {}
        
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
        
        # Create prompt for question generation
        prompt = f"""Generate a natural, conversational question to ask the user. Context:

- Task: {context['task_context']} 
- Need to ask about: {context['slot']}
- Current focus: {context['top_candidate']['anchor']}
- Confidence: {context['confidence']:.1%}
- Why: {context['reasoning']}

Generate ONE short, natural question that would help reduce uncertainty. Be conversational and helpful.

Question:"""
        
        try:
            response = self.llm_client.call_ollama_json(f"{prompt}\n\nReturn JSON with format: {{\"question\": \"your question here\"}}")
            if 'question' in response:
                return response['question']
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
        
        # Fallback to context-appropriate template
        return fallback_template
