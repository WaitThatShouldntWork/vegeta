"""
Configuration management for VEGETA system
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from .exceptions import ConfigurationError

class Config:
    """Configuration manager with environment-specific settings"""
    
    def __init__(self, config_path: Optional[str] = None, environment: str = "default"):
        self.environment = environment
        self._config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from YAML files"""
        
        # Default configuration
        default_config = {
            'database': {
                'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
                'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
                'password': os.getenv('NEO4J_PASSWORD', 'password'),
                'database': os.getenv('NEO4J_DATABASE', 'neo4j')
            },
            'ollama': {
                'base_url': os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
                'default_model': os.getenv('OLLAMA_DEFAULT_MODEL', 'gemma:4b'),
                'embedding_model': os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')
            },
            'system': {
                'defaults': {
                    'k_anchors': 10,
                    'M_candidates': 20,
                    'hops': 2,
                    'tau_retrieval': 0.7,
                    'tau_posterior': 0.7,
                    'alpha': 1.0,
                    'beta': 0.5,
                    'gamma': 0.3,
                    'sigma_sem_sq': 0.3,
                    'sigma_struct_sq': 0.2,
                    'sigma_terms_sq': 0.2,
                    'N_terms_max': 15,
                    'N_expected': 20,
                    'small_set_threshold': 3,
                    'small_set_blend': 0.5,
                    'lambda_missing': 0.30,
                    'd_cap': 40,
                    'lambda_hub': 0.02
                }
            },
            'session': {
                'timeout': 3600,  # 1 hour
                'max_turns': 20,
                'inertia_rho': 0.7
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
        
        # Try to load from file if provided
        if config_path:
            config_file = Path(config_path)
        else:
            # Look for config files in standard locations
            possible_paths = [
                Path('config') / f'{self.environment}.yaml',
                Path('config') / 'default.yaml',
                Path('../config') / f'{self.environment}.yaml',
                Path('../config') / 'default.yaml'
            ]
            
            config_file = None
            for path in possible_paths:
                if path.exists():
                    config_file = path
                    break
        
        if config_file and config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        default_config = self._deep_merge(default_config, file_config)
            except Exception as e:
                raise ConfigurationError(f"Failed to load config from {config_file}: {e}")
        
        return default_config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get config value using dot notation (e.g., 'database.uri')"""
        keys = path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any) -> None:
        """Set config value using dot notation"""
        keys = path.split('.')
        config = self._config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    @property
    def database(self) -> Dict[str, str]:
        """Database configuration"""
        return self.get('database', {})
    
    @property 
    def ollama(self) -> Dict[str, str]:
        """Ollama configuration"""
        return self.get('ollama', {})
    
    @property
    def system_defaults(self) -> Dict[str, Any]:
        """System default parameters"""
        return self.get('system.defaults', {})
    
    @property
    def session_config(self) -> Dict[str, Any]:
        """Session management configuration"""
        return self.get('session', {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Return full configuration as dictionary"""
        return self._config.copy()
