"""
Phase 2: Neural Reward Model Training System

Complete implementation of ART training pipeline.
"""

from phase2.training.models.neural_reward_model import (
    NeuralRewardModel,
    create_reward_model
)

from phase2.training.pipelines.dataset_preparation import (
    TrajectoryDataset,
    create_data_loaders
)

from phase2.training.pipelines.training_pipeline import (
    RewardModelTrainer,
    train_reward_model
)

from phase2.training.evaluators.model_evaluator import (
    ModelEvaluator,
    evaluate_checkpoint
)

from phase2.training.pipelines.ruler_integration import (
    RULERClient,
    evaluate_with_ruler
)

__all__ = [
    'NeuralRewardModel',
    'create_reward_model',
    'TrajectoryDataset',
    'create_data_loaders',
    'RewardModelTrainer',
    'train_reward_model',
    'ModelEvaluator',
    'evaluate_checkpoint',
    'RULERClient',
    'evaluate_with_ruler'
]

__version__ = '1.0.0'
__phase__ = 'Phase 2 Complete'
