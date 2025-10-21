#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Evaluator for Neural Reward Model

Comprehensive evaluation metrics and analysis tools.

Author: Andreas Antonos
Created: 2025-10-17
Version: 1.0.0
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from phase2.training.models.neural_reward_model import NeuralRewardModel, create_reward_model
from phase2.training.pipelines.dataset_preparation import create_data_loaders

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

# ============================================================================
# EVALUATION METRICS
# ============================================================================

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    
    # Basic metrics
    mse: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    r2_score: float = 0.0
    
    # Detailed metrics
    correlation: float = 0.0
    max_error: float = 0.0
    min_error: float = 0.0
    std_error: float = 0.0
    
    # Percentile errors
    p25_error: float = 0.0
    p50_error: float = 0.0  # median
    p75_error: float = 0.0
    p90_error: float = 0.0
    p95_error: float = 0.0
    
    # Performance metrics
    inference_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'basic_metrics': {
                'mse': float(self.mse),
                'mae': float(self.mae),
                'rmse': float(self.rmse),
                'r2_score': float(self.r2_score),
            },
            'detailed_metrics': {
                'correlation': float(self.correlation),
                'max_error': float(self.max_error),
                'min_error': float(self.min_error),
                'std_error': float(self.std_error),
            },
            'percentile_errors': {
                'p25': float(self.p25_error),
                'p50_median': float(self.p50_error),
                'p75': float(self.p75_error),
                'p90': float(self.p90_error),
                'p95': float(self.p95_error),
            },
            'performance': {
                'inference_time_ms': float(self.inference_time_ms),
                'throughput_samples_per_sec': float(self.throughput_samples_per_sec),
            }
        }
    
    def print_summary(self):
        """Print formatted summary of metrics"""
        print("\n" + "="*60)
        print("📊 MODEL EVALUATION METRICS")
        print("="*60)
        
        print("\n📈 Basic Metrics:")
        print(f"  MSE:  {self.mse:.6f}")
        print(f"  MAE:  {self.mae:.6f}")
        print(f"  RMSE: {self.rmse:.6f}")
        print(f"  R²:   {self.r2_score:.6f}")
        
        print("\n📉 Error Distribution:")
        print(f"  Min Error:    {self.min_error:.6f}")
        print(f"  25th %ile:    {self.p25_error:.6f}")
        print(f"  Median:       {self.p50_error:.6f}")
        print(f"  75th %ile:    {self.p75_error:.6f}")
        print(f"  90th %ile:    {self.p90_error:.6f}")
        print(f"  95th %ile:    {self.p95_error:.6f}")
        print(f"  Max Error:    {self.max_error:.6f}")
        print(f"  Std Dev:      {self.std_error:.6f}")
        
        print("\n⚡ Performance:")
        print(f"  Inference Time:     {self.inference_time_ms:.2f} ms/sample")
        print(f"  Throughput:         {self.throughput_samples_per_sec:.0f} samples/sec")
        
        print("\n" + "="*60)

# ============================================================================
# MODEL EVALUATOR
# ============================================================================

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to use (cpu/cuda)
        """
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_dataset(self, data_loader) -> Tuple[EvaluationMetrics, np.ndarray, np.ndarray]:
        """
        Evaluate model on entire dataset.
        
        Returns:
            Tuple of (metrics, predictions, ground_truth)
        """
        all_predictions = []
        all_ground_truth = []
        inference_times = []
        
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, (states, rewards) in enumerate(data_loader):
                # Move to device
                states = states.to(self.device)
                rewards = rewards.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                predictions = self.model(states)
                inference_time = (time.time() - start_time) * 1000  # ms
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_ground_truth.extend(rewards.cpu().numpy().flatten())
                inference_times.append(inference_time / states.size(0))  # per sample
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        ground_truth = np.array(all_ground_truth)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, ground_truth, inference_times)
        
        return metrics, predictions, ground_truth
    
    def _calculate_metrics(
        self, 
        predictions: np.ndarray, 
        ground_truth: np.ndarray,
        inference_times: List[float]
    ) -> EvaluationMetrics:
        """Calculate comprehensive metrics"""
        
        metrics = EvaluationMetrics()
        
        # Basic metrics
        metrics.mse = mean_squared_error(ground_truth, predictions)
        metrics.mae = mean_absolute_error(ground_truth, predictions)
        metrics.rmse = np.sqrt(metrics.mse)
        metrics.r2_score = r2_score(ground_truth, predictions)
        
        # Detailed metrics
        errors = np.abs(predictions - ground_truth)
        metrics.correlation = np.corrcoef(predictions, ground_truth)[0, 1]
        metrics.max_error = np.max(errors)
        metrics.min_error = np.min(errors)
        metrics.std_error = np.std(errors)
        
        # Percentile errors
        metrics.p25_error = np.percentile(errors, 25)
        metrics.p50_error = np.percentile(errors, 50)
        metrics.p75_error = np.percentile(errors, 75)
        metrics.p90_error = np.percentile(errors, 90)
        metrics.p95_error = np.percentile(errors, 95)
        
        # Performance metrics
        metrics.inference_time_ms = np.mean(inference_times)
        metrics.throughput_samples_per_sec = 1000 / metrics.inference_time_ms if metrics.inference_time_ms > 0 else 0
        
        return metrics
    
    def compare_models(
        self,
        baseline_model: nn.Module,
        data_loader
    ) -> Dict:
        """
        Compare this model with a baseline.
        
        Args:
            baseline_model: Model to compare against
            data_loader: Test data
            
        Returns:
            Dictionary with comparison results
        """
        # Evaluate both models
        print("Evaluating trained model...")
        trained_metrics, trained_preds, ground_truth = self.evaluate_dataset(data_loader)
        
        print("Evaluating baseline model...")
        baseline_evaluator = ModelEvaluator(baseline_model, self.device)
        baseline_metrics, baseline_preds, _ = baseline_evaluator.evaluate_dataset(data_loader)
        
        # Calculate improvements
        improvements = {
            'mae_improvement': (baseline_metrics.mae - trained_metrics.mae) / baseline_metrics.mae * 100,
            'mse_improvement': (baseline_metrics.mse - trained_metrics.mse) / baseline_metrics.mse * 100,
            'r2_improvement': (trained_metrics.r2_score - baseline_metrics.r2_score) / abs(baseline_metrics.r2_score) * 100 if baseline_metrics.r2_score != 0 else 0,
            'speed_improvement': (trained_metrics.throughput_samples_per_sec - baseline_metrics.throughput_samples_per_sec) / baseline_metrics.throughput_samples_per_sec * 100 if baseline_metrics.throughput_samples_per_sec != 0 else 0,
        }
        
        return {
            'trained_metrics': trained_metrics.to_dict(),
            'baseline_metrics': baseline_metrics.to_dict(),
            'improvements': improvements,
            'predictions': {
                'trained': trained_preds.tolist(),
                'baseline': baseline_preds.tolist(),
                'ground_truth': ground_truth.tolist(),
            }
        }
    
    def save_evaluation_report(self, metrics: EvaluationMetrics, filepath: Path):
        """Save evaluation report to JSON"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics.to_dict(),
            'model_info': {
                'parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.device),
            }
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📝 Evaluation report saved to {filepath}")

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_checkpoint(
    checkpoint_path: Path,
    batch_size: int = 16
) -> EvaluationMetrics:
    """
    Evaluate a model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        batch_size: Batch size for evaluation
        
    Returns:
        Evaluation metrics
    """
    # Load model
    print(f"📂 Loading model from {checkpoint_path}")
    model = create_reward_model()
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print("⚠️ Checkpoint not found, using random weights")
    
    # Create data loader
    print("📊 Loading test data...")
    _, _, test_loader = create_data_loaders(batch_size=batch_size)
    
    # Evaluate
    print("🔍 Evaluating model...")
    evaluator = ModelEvaluator(model)
    metrics, predictions, ground_truth = evaluator.evaluate_dataset(test_loader)
    
    # Print results
    metrics.print_summary()
    
    return metrics

def compare_checkpoints(
    checkpoint1_path: Path,
    checkpoint2_path: Path,
    batch_size: int = 16
) -> Dict:
    """Compare two model checkpoints"""
    
    print("📊 Model Checkpoint Comparison")
    print("="*60)
    
    # Load models
    model1 = create_reward_model()
    model2 = create_reward_model()
    
    if checkpoint1_path.exists():
        checkpoint1 = torch.load(checkpoint1_path, map_location='cpu')
        model1.load_state_dict(checkpoint1['model_state_dict'])
    
    if checkpoint2_path.exists():
        checkpoint2 = torch.load(checkpoint2_path, map_location='cpu')
        model2.load_state_dict(checkpoint2['model_state_dict'])
    
    # Create data loader
    _, _, test_loader = create_data_loaders(batch_size=batch_size)
    
    # Compare
    evaluator = ModelEvaluator(model1)
    comparison = evaluator.compare_models(model2, test_loader)
    
    print("\n📈 Improvements:")
    for metric, improvement in comparison['improvements'].items():
        print(f"  {metric}: {improvement:+.2f}%")
    
    return comparison

# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("      MODEL EVALUATOR TEST")
    print("="*60)
    
    # Test with best model checkpoint
    checkpoint_path = Path("models/checkpoints/best_model.pth")
    
    if checkpoint_path.exists():
        metrics = evaluate_checkpoint(checkpoint_path)
        
        # Save report
        report_path = Path("models/evaluation/evaluation_report.json")
        evaluator = ModelEvaluator(create_reward_model())
        evaluator.save_evaluation_report(metrics, report_path)
        
        print(f"\n✅ Evaluation complete!")
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path}")
        print("Creating random model for testing...")
        
        # Test with random model
        model = create_reward_model()
        _, _, test_loader = create_data_loaders(batch_size=8)
        
        evaluator = ModelEvaluator(model)
        metrics, _, _ = evaluator.evaluate_dataset(test_loader)
        metrics.print_summary()
    
    print("\n✅ Model evaluator test complete!")

# Finish
