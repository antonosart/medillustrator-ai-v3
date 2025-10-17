#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training Pipeline for Neural Reward Model

Main orchestration for training the reward model.

Author: Andreas Antonos
Created: 2025-10-16
Version: 1.0.0
"""

import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from phase2.training.models.neural_reward_model import create_reward_model, ModelConstants
from phase2.training.pipelines.dataset_preparation import create_data_loaders

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================================
# TRAINING CONFIG
# ============================================================================

class TrainingConfig:
    """Training configuration"""
    
    # Training hyperparameters
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    
    # Checkpointing
    CHECKPOINT_DIR: Path = Path("models/checkpoints")
    SAVE_INTERVAL: int = 5
    
    # Early stopping
    PATIENCE: int = 3
    MIN_DELTA: float = 0.001
    
    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# TRAINER CLASS
# ============================================================================

class RewardModelTrainer:
    """Trainer for the neural reward model"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig = TrainingConfig()
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Ensure checkpoint directory exists
        self.config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (states, rewards) in enumerate(progress_bar):
            # Move to device
            states = states.to(self.device)
            rewards = rewards.to(self.device)
            
            # Forward pass
            predictions = self.model(states)
            loss = self.criterion(predictions, rewards)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                ModelConstants.MAX_GRAD_NORM
            )
            
            self.optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        mae = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for states, rewards in val_loader:
                # Move to device
                states = states.to(self.device)
                rewards = rewards.to(self.device)
                
                # Forward pass
                predictions = self.model(states)
                loss = self.criterion(predictions, rewards)
                
                # Calculate MAE
                mae += torch.mean(torch.abs(predictions - rewards)).item()
                
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        avg_mae = mae / num_batches
        
        return avg_val_loss, avg_mae
    
    def train(
        self, 
        train_loader, 
        val_loader,
        num_epochs: Optional[int] = None
    ):
        """Main training loop"""
        num_epochs = num_epochs or self.config.NUM_EPOCHS
        
        print(f"\n🚀 Starting Training on {self.device}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning Rate: {self.config.LEARNING_RATE}")
        print("=" * 60)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_mae = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Print epoch results
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val MAE:    {val_mae:.4f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss - self.config.MIN_DELTA:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(epoch, is_best=True)
                print("  ✅ New best model saved!")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\n⏹️  Early stopping triggered (patience={self.config.PATIENCE})")
                break
            
            # Regular checkpoint
            if epoch % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(epoch)
        
        print("\n✅ Training Complete!")
        print(f"   Best Val Loss: {self.best_val_loss:.4f}")
        
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_name = "best_model.pth" if is_best else f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path = self.config.CHECKPOINT_DIR / checkpoint_name
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'val_loss': self.val_losses[-1] if self.val_losses else 0,
            'best_val_loss': self.best_val_loss,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_reward_model(
    num_epochs: int = 10,
    batch_size: int = 16
) -> Dict:
    """
    Main function to train the reward model.
    
    Returns:
        Dictionary with training results
    """
    # Create data loaders
    print("📊 Preparing data...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=batch_size)
    
    # Create model
    print("🧠 Creating model...")
    model = create_reward_model()
    
    # Create trainer
    config = TrainingConfig()
    config.NUM_EPOCHS = num_epochs
    trainer = RewardModelTrainer(model, config)
    
    # Train
    train_losses, val_losses = trainer.train(train_loader, val_loader)
    
    # Test
    print("\n📈 Final evaluation on test set...")
    test_loss, test_mae = trainer.validate(test_loader)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test MAE:  {test_mae:.4f}")
    
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'test_mae': test_mae,
        'best_val_loss': trainer.best_val_loss,
        'num_epochs_trained': len(train_losses),
    }
    
    return results

# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("      NEURAL REWARD MODEL TRAINING")
    print("=" * 60)
    
    # Quick test with few epochs
    results = train_reward_model(num_epochs=3, batch_size=8)
    
    print("\n📊 Training Summary:")
    print(f"   Epochs trained: {results['num_epochs_trained']}")
    print(f"   Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"   Test Loss: {results['test_loss']:.4f}")
    print(f"   Test MAE: {results['test_mae']:.4f}")
    
    print("\n✅ Training pipeline test complete!")

# Finish
