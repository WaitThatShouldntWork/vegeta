"""
LLM client for Ollama integration
"""

import json
import logging
import requests
import numpy as np
from typing import Dict, Any, Optional, List

from ..core.exceptions import ConnectionError, GenerationError

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with Ollama LLM services
    """
    
    def __init__(self, llm_config: Dict[str, str]):
        self.config = llm_config
        self.base_url = llm_config['base_url']
        self.default_model = llm_config['default_model']
        self.embedding_model = llm_config['embedding_model']
    
    def test_connection(self) -> bool:
        """Test connection to Ollama service"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [model['name'] for model in response.json().get('models', [])]
                logger.info(f"✓ Ollama connected. Available models: {len(models)}")
                
                # Check if required models are available
                missing_models = []
                if self.default_model not in models:
                    missing_models.append(self.default_model)
                if self.embedding_model not in models:
                    missing_models.append(self.embedding_model)
                
                if missing_models:
                    logger.warning(f"Missing models: {missing_models}")
                
                return True
            else:
                logger.error(f"Ollama service returned HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
    
    def call_ollama_json(self, prompt: str, model: str = None) -> Dict[str, Any]:
        """Call Ollama with JSON format enforcement"""
        model = model or self.default_model
        
        payload = {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent extraction
                "top_p": 0.9
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate", 
                json=payload, 
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            # Parse the JSON response
            response_text = result.get('response', '{}')
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {e}")
                logger.debug(f"Raw response: {response_text}")
                return {}
                
        except requests.RequestException as e:
            raise ConnectionError(f"Ollama request failed: {e}")
        except Exception as e:
            raise GenerationError(f"Ollama JSON call failed: {e}")
    
    def call_ollama_text(self, prompt: str, model: str = None) -> str:
        """Call Ollama for text generation"""
        import time
        
        model = model or self.default_model
        start_time = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9
            }
        }
        
        try:
            logger.info(f"🔄 LLM call to {model} starting...")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get('response', '')
            
            call_time = time.time() - start_time
            logger.info(f"✅ LLM call to {model} completed: {call_time:.2f}s ({len(response_text)} chars)")
            return response_text
            
        except requests.RequestException as e:
            call_time = time.time() - start_time
            logger.error(f"❌ LLM call to {model} failed after {call_time:.2f}s: {e}")
            raise ConnectionError(f"Ollama request failed: {e}")
        except Exception as e:
            call_time = time.time() - start_time
            logger.error(f"❌ LLM call to {model} error after {call_time:.2f}s: {e}")
            raise GenerationError(f"Ollama text call failed: {e}")
    
    def get_embedding(self, text: str, model: str = None) -> Optional[np.ndarray]:
        """Generate embedding using Ollama"""
        import time
        
        model = model or self.embedding_model
        start_time = time.time()
        
        payload = {
            "model": model,
            "prompt": text
        }
        
        try:
            logger.info(f"🔄 Embedding call to {model} starting...")
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            
            embedding = np.array(result['embedding'])
            
            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            call_time = time.time() - start_time
            logger.info(f"✅ Embedding call to {model} completed: {call_time:.2f}s (dims: {len(embedding)})")
            return embedding
            
        except requests.RequestException as e:
            call_time = time.time() - start_time
            logger.error(f"❌ Embedding call to {model} failed after {call_time:.2f}s: {e}")
            logger.error(f"Embedding request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return []
