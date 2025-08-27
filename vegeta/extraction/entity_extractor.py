"""
Entity extraction using LLM-based analysis
"""

import logging
from typing import Dict, Any, List

from ..utils.llm_client import LLMClient
from ..core.config import Config
from ..core.exceptions import ExtractionError

logger = logging.getLogger(__name__)

class EntityExtractor:
    """
    LLM-based entity extraction and query analysis
    """
    
    def __init__(self, llm_client: LLMClient, config: Config):
        self.llm_client = llm_client
        self.config = config
        self.max_terms = config.get('system.defaults.N_terms_max', 15)
    
    def extract_entities_llm(self, utterance: str) -> Dict[str, Any]:
        """Extract entities and terms using LLM with JSON schema validation and query analysis"""
        
        extraction_prompt = f"""Extract information from this user utterance. Return valid JSON only with this exact structure:

{{
    "canonical_terms": ["term1", "term2"],
    "entities": [
        {{"surface": "text_as_written", "normalized": "canonical_form", "type": "EntityType"}}
    ],
    "numbers": [1995, 2010],
    "dates": ["1995", "2010s"]
}}

Rules:
- canonical_terms: lowercase, lemmatized key terms (max {self.max_terms})
- entities: extract any named entities mentioned (people, products, places, works, etc.)
- numbers: extract salient numeric values
- dates: extract date expressions

Utterance: "{utterance}"

JSON:"""
        
        try:
            result = self.llm_client.call_ollama_json(extraction_prompt)
            
            # Validate and clean the result
            canonical_terms = result.get('canonical_terms', [])[:self.max_terms]
            entities = result.get('entities', [])
            numbers = result.get('numbers', [])
            dates = result.get('dates', [])
            
            # Dedupe and clean canonical terms
            canonical_terms = list(dict.fromkeys([
                term.lower().strip() for term in canonical_terms if term.strip()
            ]))
            
            # Add query analysis as a separate LLM call for reliability
            query_analysis = self.analyze_query_clarity(utterance)
            
            return {
                'canonical_terms': canonical_terms,
                'entities': entities,
                'numbers': numbers,
                'dates': dates,
                'query_analysis': query_analysis
            }
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            raise ExtractionError(f"Entity extraction failed: {e}")
    
    def analyze_query_clarity(self, utterance: str) -> Dict[str, Any]:
        """Analyze query clarity and specificity using LLM"""
        
        analysis_prompt = f"""Analyze this user query for clarity and specificity. Return valid JSON only:

{{
    "clarity": "clear|moderate|vague|extremely_vague",
    "specificity": "specific|general|abstract", 
    "domain_identifiable": true/false,
    "immediate_clarification_needed": true/false,
    "clarification_type": "task_domain|intent|specifics|none",
    "reasoning": "brief explanation of the assessment"
}}

Rules for analysis:
- clarity: How understandable is the request?
  - clear: Specific, actionable request
  - moderate: Somewhat unclear but processable  
  - vague: Unclear what user wants
  - extremely_vague: No clear meaning (like "do this", "help me", "it")
- immediate_clarification_needed: true if you cannot proceed without asking for more info
- clarification_type: what type of clarification is most needed?

Query: "{utterance}"

JSON:"""
        
        try:
            result = self.llm_client.call_ollama_json(analysis_prompt)
            return result
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Fallback analysis
            return {
                'clarity': 'moderate',
                'specificity': 'general',
                'domain_identifiable': False,
                'immediate_clarification_needed': False,
                'clarification_type': 'none',
                'reasoning': 'Analysis failed, using defaults'
            }
