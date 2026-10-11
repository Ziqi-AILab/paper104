"""Training protocol, token progress, and checkpoint state."""

from .progress import CHECKPOINT_PERCENTAGES, CheckpointSchedule, TokenProgress
from .optimization import PretrainingRecipe, learning_rate_candidates
from .state import TrainingState, load_training_checkpoint, save_training_checkpoint

__all__ = [
    "CHECKPOINT_PERCENTAGES",
    "CheckpointSchedule",
    "PretrainingRecipe",
    "PretrainingRunConfig",
    "PretrainingRunResult",
    "TokenProgress",
    "TrainingState",
    "load_training_checkpoint",
    "learning_rate_candidates",
    "save_training_checkpoint",
    "run_pretraining",
]


def __getattr__(name: str):
    if name in {"PretrainingRunConfig", "PretrainingRunResult", "run_pretraining"}:
        from .runner import PretrainingRunConfig, PretrainingRunResult, run_pretraining

        return {
            "PretrainingRunConfig": PretrainingRunConfig,
            "PretrainingRunResult": PretrainingRunResult,
            "run_pretraining": run_pretraining,
        }[name]
    raise AttributeError(name)
