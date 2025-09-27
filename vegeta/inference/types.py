"""
Type definitions and data classes for Bayesian Active Inference

Contains shared types and data structures used across inference components.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class PredictedObservations:
    """Container for predicted observations across all channels"""
    u_sem: NDArray[np.float32]        # Expected semantic vector (768-dim)
    u_struct: Dict[str, float]        # Expected structural counts/patterns
    u_terms: List[str]               # Expected key terms
    confidence: float                # Overall prediction confidence
    metadata: Dict[str, Any]         # Additional prediction metadata


@dataclass
class HiddenStates:
    """Hidden causes/states v that generate predictions"""
    z_checklist: str                 # Active checklist name
    z_subgraph: str                 # Candidate subgraph ID
    z_goal: Optional[str] = None    # User goal/intent
    z_slots: Optional[Dict[str, str]] = None  # Slot values
    z_step: Optional[str] = None    # Current procedure step


@dataclass
class ObservedFeatures:
    """Container for actual observations u from user utterance"""

    u_sem: NDArray[np.float32]       # Utterance embedding (768-dim)
    u_terms: List[str]              # Canonical terms from utterance
    u_struct: Optional[Dict[str, float]] = None  # Observed structural patterns
