"""
Custom exceptions for VEGETA system
"""

class VegetaError(Exception):
    """Base exception for VEGETA system"""
    pass

class ConfigurationError(VegetaError):
    """Configuration-related errors"""
    pass

class ConnectionError(VegetaError):
    """Database or service connection errors"""
    pass

class ExtractionError(VegetaError):
    """Entity extraction and analysis errors"""
    pass

class RetrievalError(VegetaError):
    """Graph retrieval and search errors"""
    pass

class InferenceError(VegetaError):
    """Bayesian inference and decision errors"""
    pass

class SessionError(VegetaError):
    """Multi-turn session management errors"""
    pass

class GenerationError(VegetaError):
    """Question/answer generation errors"""
    pass
