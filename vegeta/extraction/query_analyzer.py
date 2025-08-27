"""
Query analysis and intent classification
"""

import logging
from typing import Dict, Any

from ..utils.llm_client import LLMClient
from ..core.config import Config
from ..core.exceptions import ExtractionError

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """
    Analyze user queries for intent, dialogue acts, and user goals
    """
    
    def __init__(self, llm_client: LLMClient, config: Config):
        self.llm_client = llm_client
        self.config = config
    
    def classify_user_intent_llm(self, utterance: str, dialogue_acts: Dict[str, float], 
                                extraction: Dict) -> Dict[str, float]:
        """Use LLM to understand user's underlying goals and intentions"""
        
        # Extract context for intent analysis
        canonical_terms = extraction.get('canonical_terms', [])
        entities = extraction.get('entities', [])
        primary_dialogue_act = max(dialogue_acts.items(), key=lambda x: x[1])[0]
        
        analysis_prompt = f"""Analyze this user utterance to understand their underlying goals. Return valid JSON only:

{{
    "user_goals": {{
        "identify": 0.0,
        "recommend": 0.0,
        "verify": 0.0,
        "explore": 0.0,
        "act": 0.0
    }},
    "primary_goal": "identify|recommend|verify|explore|act",
    "reasoning": "brief explanation of intent analysis"
}}

Goal definitions:
- identify: User wants to find/name a specific thing ("What movie is this?", "Who directed X?")
- recommend: User wants suggestions/recommendations ("Suggest movies like X", "What should I watch?")
- verify: User wants to confirm/check facts ("Is this true?", "Did X win an award?")
- explore: User wants to learn/understand more ("Tell me about X", "How does Y work?")
- act: User wants something done/executed ("Play this movie", "Add to watchlist")

Context:
- Utterance: "{utterance}"
- Primary dialogue act: {primary_dialogue_act}
- Key terms: {canonical_terms}
- Entities mentioned: {[e.get('surface', '') for e in entities]}

Guidelines:
- Focus on what the user ultimately wants to accomplish
- Consider specific language patterns (e.g., "like", "similar" → recommend)
- Dialogue acts provide hints but goals are deeper intentions
- Return probabilities that sum to 1.0

JSON:"""
        
        try:
            result = self.llm_client.call_ollama_json(analysis_prompt)
            user_goals = result.get('user_goals', {})
            
            # Validate and normalize probabilities
            total = sum(user_goals.values())
            if total > 0:
                user_goals = {k: v/total for k, v in user_goals.items()}
            else:
                # Fallback based on dialogue act patterns
                if primary_dialogue_act == 'request':
                    user_goals = {'identify': 0.4, 'recommend': 0.4, 'verify': 0.1, 'explore': 0.1, 'act': 0.0}
                elif primary_dialogue_act == 'clarify':
                    user_goals = {'identify': 0.3, 'recommend': 0.1, 'verify': 0.2, 'explore': 0.4, 'act': 0.0}
                else:
                    user_goals = {'identify': 0.4, 'recommend': 0.2, 'verify': 0.15, 'explore': 0.15, 'act': 0.1}
            
            return user_goals
            
        except Exception as e:
            logger.error(f"User intent classification failed: {e}")
            # Fallback to dialogue-act based heuristic
            if primary_dialogue_act == 'request':
                return {'identify': 0.4, 'recommend': 0.4, 'verify': 0.1, 'explore': 0.1, 'act': 0.0}
            elif primary_dialogue_act == 'clarify':
                return {'identify': 0.3, 'recommend': 0.1, 'verify': 0.2, 'explore': 0.4, 'act': 0.0}
            else:
                return {'identify': 0.4, 'recommend': 0.2, 'verify': 0.15, 'explore': 0.15, 'act': 0.1}
    
    def classify_dialogue_act_llm(self, utterance: str) -> Dict[str, float]:
        """Classify dialogue act using LLM understanding of conversational intent"""
        
        analysis_prompt = f"""Analyze this utterance and classify the dialogue act. Return valid JSON only:

{{
    "dialogue_acts": {{
        "clarify": 0.0,
        "confirm": 0.0, 
        "request": 0.0,
        "provide": 0.0
    }},
    "primary_act": "clarify|confirm|request|provide",
    "reasoning": "brief explanation of classification"
}}

Dialogue acts defined:
- clarify: Asking for information, seeking explanation (e.g., "What is X?", "How does Y work?")
- confirm: Seeking verification of information (e.g., "Is this correct?", "Are you sure?")  
- request: Asking for something to be done (e.g., "Find me X", "I want Y", "Help me with Z")
- provide: Giving information or stating facts (e.g., "X is Y", "I think Z")

Return probabilities that sum to 1.0 across the four dialogue acts.

Utterance: "{utterance}"

JSON:"""
        
        try:
            result = self.llm_client.call_ollama_json(analysis_prompt)
            dialogue_acts = result.get('dialogue_acts', {})
            
            # Validate and normalize probabilities
            total = sum(dialogue_acts.values())
            if total > 0:
                dialogue_acts = {k: v/total for k, v in dialogue_acts.items()}
            else:
                # Fallback to uniform distribution
                dialogue_acts = {'clarify': 0.25, 'confirm': 0.25, 'request': 0.25, 'provide': 0.25}
                
            return dialogue_acts
            
        except Exception as e:
            logger.error(f"Dialogue act classification failed: {e}")
            # Fallback to uniform distribution
            return {'clarify': 0.25, 'confirm': 0.25, 'request': 0.25, 'provide': 0.25}
