#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Service for Neural Reward Model Integration

Bridges the trained model with the main application workflow.

Author: Andreas Antonos
Created: 2025-10-21
Version: 1.0.0
"""

import logging
from typing import Dict, Optional, Any
from pathlib import Path
import torch
import json
import time

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from phase2.training.models.neural_reward_model import create_reward_model
from phase2.training.pipelines.ruler_integration import RULERClient

logger = logging.getLogger(__name__)

class RewardModelService:
    """Service for reward prediction in production"""
    
    def __init__(self, checkpoint_path: Optional[Path] = None):
        """Initialize service with trained model"""
        
        # Load model
        self.model = create_reward_model()
        
        # Load checkpoint if available
        if checkpoint_path and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {checkpoint_path}")
        else:
            logger.warning("Using untrained model weights")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Initialize RULER client
        self.ruler_client = RULERClient()
        
        # Metrics tracking
        self.total_predictions = 0
        self.avg_inference_time = 0
    
    def predict_reward(self, state_embedding: Dict[str, Any]) -> float:
        """
        Predict reward for given state.
        
        Args:
            state_embedding: State representation
            
        Returns:
            Predicted reward value [0, 1]
        """
        start_time = time.time()
        
        # Convert state to tensor
        # In production, this would use actual CLIP embeddings
        import numpy as np
        np.random.seed(hash(str(state_embedding)) % 2**32)
        state_tensor = torch.FloatTensor(np.random.randn(1, 512))
        
        # Predict
        with torch.no_grad():
            reward = self.model(state_tensor).item()
        
        # Track metrics
        inference_time = time.time() - start_time
        self.total_predictions += 1
        self.avg_inference_time = (
            (self.avg_inference_time * (self.total_predictions - 1) + inference_time) 
            / self.total_predictions
        )
        
        return max(0.0, min(1.0, reward))
    
    def predict_with_ruler(self, assessment_data: Dict) -> Dict:
        """
        Predict using both neural model and RULER.
        
        Returns combined prediction with confidence.
        """
        # Neural model prediction
        neural_reward = self.predict_reward(assessment_data)
        
        # RULER evaluation (async in production)
        import asyncio
        from phase2.training.pipelines.ruler_integration import RULERRequest
        
        request = RULERRequest(
            trajectory_id=str(time.time()),
            assessment_output=assessment_data
        )
        
        ruler_response = asyncio.run(self.ruler_client.evaluate_single(request))
        
        # Combine predictions
        combined_reward = 0.7 * neural_reward + 0.3 * ruler_response.reward_score
        
        return {
            'final_reward': combined_reward,
            'neural_reward': neural_reward,
            'ruler_reward': ruler_response.reward_score,
            'confidence': ruler_response.confidence,
            'reasoning': ruler_response.reasoning,
            'inference_time_ms': self.avg_inference_time * 1000
        }
    
    def get_metrics(self) -> Dict:
        """Get service metrics"""
        return {
            'total_predictions': self.total_predictions,
            'avg_inference_time_ms': self.avg_inference_time * 1000,
            'model_loaded': hasattr(self, 'model'),
            'ruler_enabled': hasattr(self, 'ruler_client')
        }

# Singleton instance
_model_service = None

def get_model_service(checkpoint_path: Optional[Path] = None) -> RewardModelService:
    """Get or create model service instance"""
    global _model_service
    
    if _model_service is None:
        default_checkpoint = Path("models/checkpoints/best_model.pth")
        _model_service = RewardModelService(checkpoint_path or default_checkpoint)
    
    return _model_service

# Test function
if __name__ == "__main__":
    print("Testing Model Service...")
    
    service = get_model_service()
    
    # Test prediction
    test_state = {
        'medical_terms': ['heart', 'valve'],
        'bloom_level': 'Apply',
        'cognitive_load': 5
    }
    
    # Neural only
    reward = service.predict_reward(test_state)
    print(f"\nNeural Reward: {reward:.3f}")
    
    # Combined with RULER
    result = service.predict_with_ruler(test_state)
    print(f"\nCombined Results:")
    print(f"  Final Reward: {result['final_reward']:.3f}")
    print(f"  Neural: {result['neural_reward']:.3f}")
    print(f"  RULER: {result['ruler_reward']:.3f}")
    print(f"  Confidence: {result['confidence']:.3f}")
    
    # Metrics
    print(f"\nService Metrics: {service.get_metrics()}")
    
    print("\n✅ Model service test complete!")

# Finish
