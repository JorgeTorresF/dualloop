"""DualLoop: learned, auditable arbitration between heterogeneous decision
engines (LLMs + typed classifiers + rules), closing the loop
decision -> outcome -> recalibration with no manual retraining.

See README.md and docs/architecture.md for the reasoning behind this design.
"""

from .bandit import AdaptiveThreshold, ReliabilityBandit
from .calibration import ConfidenceCalibrator
from .engines import (
    AnthropicLLMEngine,
    BaseEngine,
    JevEngine,
    OpenAICompatibleLLMEngine,
    RuleEngine,
)
from .heuristics import HumanOverrideHeuristic, OutcomeHeuristic
from .loop import DualLoop
from .store import InMemoryStore, SQLiteStore, Store
from .types import Decision, EngineOutput, Outcome, Question, Vote

__version__ = "0.1.0"

__all__ = [
    "DualLoop",
    "Question",
    "EngineOutput",
    "Vote",
    "Decision",
    "Outcome",
    "InMemoryStore",
    "SQLiteStore",
    "Store",
    "BaseEngine",
    "JevEngine",
    "OpenAICompatibleLLMEngine",
    "AnthropicLLMEngine",
    "RuleEngine",
    "OutcomeHeuristic",
    "HumanOverrideHeuristic",
    "ConfidenceCalibrator",
    "ReliabilityBandit",
    "AdaptiveThreshold",
]
