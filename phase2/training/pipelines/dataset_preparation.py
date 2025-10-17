#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Preparation for Neural Reward Model Training

Converts trajectory data into PyTorch datasets for training.

Author: Andreas Antonos  
Created: 2025-10-16
Version: 1.0.0
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from phase2.training.storage_manager import StorageManager

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

class DataConstants:
    """Constants for dataset preparation"""
    
    # Feature dimensions
    STATE_EMBEDDING_DIM: int = 512
    
    # Data splits
    TRAIN_RATIO: float = 0.8
    VAL_RATIO: float = 0.1
    TEST_RATIO: float = 0.1
    
    # Batch settings
    DEFAULT_BATCH_SIZE: int = 16
    DEFAULT_NUM_WORKERS: int = 2
    
    # Quality filters
    MIN_REWARD_THRESHOLD: float = 0.1
    MAX_REWARD_THRESHOLD: float = 1.0
    MIN_EVENTS_COUNT: int = 3

# ============================================================================
# TRAJECTORY DATASET
# ============================================================================

class TrajectoryDataset(Dataset):
    """PyTorch dataset for trajectory-reward pairs"""
    
    def __init__(self, trajectories: List[Dict], transform=None):
        """
        Initialize dataset with trajectories.
        
        Args:
            trajectories: List of trajectory dictionaries
            transform: Optional data transformation
        """
        self.trajectories = trajectories
        self.transform = transform
        
        # Filter and process trajectories
        self.data = self._prepare_data()
        
    def _prepare_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Process trajectories into (state, reward) pairs"""
        data = []
        
        for trajectory in self.trajectories:
            try:
                # Extract state embedding
                state = self._extract_state_embedding(trajectory)
                
                # Extract reward
                reward = float(trajectory.get('reward_score', 0.0))
                
                # Validate
                if self._validate_sample(state, reward):
                    state_tensor = torch.FloatTensor(state)
                    reward_tensor = torch.FloatTensor([reward])
                    data.append((state_tensor, reward_tensor))
                    
            except Exception as e:
                logger.warning(f"Skipping trajectory: {e}")
                continue
        
        logger.info(f"✅ Prepared {len(data)} samples from {len(self.trajectories)} trajectories")
        return data
    
    def _extract_state_embedding(self, trajectory: Dict) -> np.ndarray:
        """Extract state embedding from trajectory"""
        # For now, create random embedding (will be replaced with actual features)
        # In production, this would use CLIP embeddings, text features, etc.
        
        # Simulate extracting features
        np.random.seed(hash(str(trajectory.get('session_id', ''))) % 2**32)
        embedding = np.random.randn(DataConstants.STATE_EMBEDDING_DIM)
        
        # Add some structure based on trajectory data
        if 'metadata' in trajectory:
            metadata = trajectory['metadata']
            # Influence embedding based on metadata
            if metadata.get('image_size'):
                embedding[0] = np.log(metadata['image_size'] + 1) / 20
        
        return embedding
    
    def _validate_sample(self, state: np.ndarray, reward: float) -> bool:
        """Validate a single sample"""
        # Check state dimension
        if len(state) != DataConstants.STATE_EMBEDDING_DIM:
            return False
        
        # Check reward range
        if reward < DataConstants.MIN_REWARD_THRESHOLD:
            return False
        if reward > DataConstants.MAX_REWARD_THRESHOLD:
            return False
        
        # Check for NaN/Inf
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return False
        
        return True
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        state, reward = self.data[idx]
        
        if self.transform:
            state = self.transform(state)
        
        return state, reward

# ============================================================================
# DATA LOADER FACTORY
# ============================================================================

def create_data_loaders(
    storage_path: Optional[str] = None,
    batch_size: int = DataConstants.DEFAULT_BATCH_SIZE,
    num_workers: int = DataConstants.DEFAULT_NUM_WORKERS,
    validation_split: float = DataConstants.VAL_RATIO
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Load trajectories from storage
    storage = StorageManager(storage_path=storage_path or "phase2/training/data")
    trajectories = storage.query_trajectories(limit=1000)
    
    if not trajectories:
        logger.warning("No trajectories found! Creating synthetic data...")
        trajectories = _create_synthetic_trajectories(100)
    
    # Create dataset
    dataset = TrajectoryDataset(trajectories)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(DataConstants.TRAIN_RATIO * total_size)
    val_size = int(DataConstants.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    logger.info(f"✅ Created data loaders:")
    logger.info(f"   Train: {len(train_dataset)} samples")
    logger.info(f"   Val: {len(val_dataset)} samples")
    logger.info(f"   Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

def _create_synthetic_trajectories(num_samples: int) -> List[Dict]:
    """Create synthetic trajectories for testing"""
    trajectories = []
    
    for i in range(num_samples):
        trajectory = {
            'trajectory_id': f'synthetic_{i}',
            'session_id': f'session_{i}',
            'reward_score': np.random.uniform(0.2, 0.95),
            'metadata': {
                'image_size': np.random.randint(100000, 5000000),
                'assessment_type': 'synthetic'
            },
            'events': [
                {'type': 'start', 'timestamp': 0},
                {'type': 'processing', 'timestamp': 100},
                {'type': 'complete', 'timestamp': 200}
            ]
        }
        trajectories.append(trajectory)
    
    return trajectories

# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    # Test dataset creation
    print("Testing Dataset Preparation...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=4)
    
    # Test one batch
    for batch_idx, (states, rewards) in enumerate(train_loader):
        print(f"\n✅ Batch {batch_idx + 1}:")
        print(f"   States shape: {states.shape}")
        print(f"   Rewards shape: {rewards.shape}")
        print(f"   Reward values: {rewards.squeeze().tolist()}")
        
        if batch_idx >= 2:  # Only show first 3 batches
            break
    
    print("\n✅ Dataset preparation test complete!")

# Finish
