"""
Training Data Collection System
================================

Components:
- TrajectoryCollector: Capture assessment trajectories
- RewardCalculator: Calculate educational quality rewards
- DataValidator: Validate training data quality
- StorageManager: Persistent trajectory storage
"""

from .trajectory_collector import TrajectoryCollector, AssessmentTrajectory
from .reward_calculator import RewardCalculator, RewardComponents
from .data_validator import DataValidator, ValidationResult
from .storage_manager import StorageManager

__all__ = [
    "TrajectoryCollector",
    "AssessmentTrajectory",
    "RewardCalculator",
    "RewardComponents",
    "DataValidator",
    "ValidationResult",
    "StorageManager"
]
