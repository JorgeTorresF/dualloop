"""DualLoop: arbitraje aprendido y auditable entre motores de decision
heterogeneos (LLM + clasificadores tipados + reglas), con cierre del bucle
decision -> resultado -> recalibracion sin reentrenamiento manual.

Ver README.md y docs/architecture.md para el porque de este diseño.
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
