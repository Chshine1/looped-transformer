from dataclasses import dataclass


@dataclass
class TrainingState:
    epoch: int
    global_step: int
    best_accuracy: float
    evaluation_completed: bool
