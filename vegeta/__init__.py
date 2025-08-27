"""
VEGETA - Bayesian Active Inference System
==========================================

A sophisticated system for active inference using Bayesian decision theory,
designed for 20-questions style interactions and graph-based knowledge retrieval.

Core Components:
- Entity extraction and semantic embedding
- Graph-based retrieval with anchor selection
- Bayesian prior/posterior inference
- Multi-turn session management
- Expected Information Gain (EIG) decision making
"""

__version__ = "0.1.0"
__author__ = "VEGETA Project"

from .core.system import VegetaSystem
from .core.config import Config
from .core.exceptions import VegetaError

__all__ = [
    'VegetaSystem',
    'Config', 
    'VegetaError'
]
