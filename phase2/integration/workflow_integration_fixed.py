#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration with Main LangGraph Workflow - Fixed Version

Author: Andreas Antonos
Created: 2025-10-21
Version: 1.0.1
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import time
import json

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from phase2.integration.model_service import get_model_service
from phase2.training.trajectory_collector import TrajectoryCollector
from phase2.training.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class ARTWorkflowIntegration:
    """Integrates ART components with main workflow"""
    
    def __init__(self):
        self.model_service = get_model_service()
        self.trajectory_collector = TrajectoryCollector()
        self.storage_manager = StorageManager()
        self.assessments_processed = 0
        self.total_reward = 0.0
        logger.info("ART Workflow Integration initialized")
    
    def process_assessment(
        self,
        session_id: str,
        image_data: Dict,
        assessment_results: Dict
    ) -> Dict[str, Any]:
        """Process assessment with ART components"""
        start_time = time.time()
        
        # Start trajectory - using session_id and initial state
        trajectory_id = self.trajectory_collector.start_trajectory(
            session_id,
            initial_state=image_data
        )
        
        # Record assessment state
        self.trajectory_collector.record_state(
            trajectory_id,
            state=assessment_results,
            metadata={'type': 'assessment_results'}
        )
        
        # Get reward prediction
        reward_result = self.model_service.predict_with_ruler(assessment_results)
        
        # Finalize trajectory with reward
        trajectory = self.trajectory_collector.finalize_trajectory(
            trajectory_id,
            final_state={
                **assessment_results,
                'reward_score': reward_result['final_reward']
            }
        )
        
        # Save if we got a trajectory back
        if trajectory:
            self.storage_manager.save_trajectory(trajectory)
        
        # Update stats
        self.assessments_processed += 1
        self.total_reward += reward_result['final_reward']
        
        # Prepare enhanced results
        enhanced_results = {
            **assessment_results,
            'art_enhancement': {
                'trajectory_id': trajectory_id,
                'reward_score': reward_result['final_reward'],
                'neural_reward': reward_result['neural_reward'],
                'ruler_reward': reward_result['ruler_reward'],
                'confidence': reward_result['confidence'],
                'reasoning': reward_result['reasoning'],
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        }
        
        logger.info(f"Assessment {session_id} processed with reward: {reward_result['final_reward']:.3f}")
        
        return enhanced_results
    
    def get_integration_stats(self) -> Dict:
        avg_reward = self.total_reward / max(1, self.assessments_processed)
        
        return {
            'assessments_processed': self.assessments_processed,
            'average_reward': avg_reward,
            'model_metrics': self.model_service.get_metrics(),
            'trajectory_count': len(self.storage_manager.query_trajectories(limit=1000)),
            'storage_backend': self.storage_manager.backend
        }

def enhance_assessment_with_art(
    session_id: str,
    image_data: Dict,
    assessment_results: Dict
) -> Dict:
    """Main integration function for app_v3_langgraph.py"""
    
    if not hasattr(enhance_assessment_with_art, 'integration'):
        enhance_assessment_with_art.integration = ARTWorkflowIntegration()
    
    integration = enhance_assessment_with_art.integration
    
    enhanced_results = integration.process_assessment(
        session_id=session_id,
        image_data=image_data,
        assessment_results=assessment_results
    )
    
    return enhanced_results

if __name__ == "__main__":
    print("Testing Workflow Integration (Fixed)...")
    
    test_session = f"test_session_{int(time.time())}"
    test_image = {'size': 1024*1024, 'format': 'png'}
    test_results = {
        'medical_terms': {
            'terms': ['heart', 'ventricle'],
            'confidence': 0.85
        },
        'bloom_taxonomy': {
            'level': 'Apply',
            'score': 0.75
        }
    }
    
    enhanced = enhance_assessment_with_art(
        session_id=test_session,
        image_data=test_image,
        assessment_results=test_results
    )
    
    print(f"\n✅ Enhanced Assessment:")
    print(f"  Trajectory ID: {enhanced['art_enhancement']['trajectory_id']}")
    print(f"  Final Reward: {enhanced['art_enhancement']['reward_score']:.3f}")
    print(f"  Confidence: {enhanced['art_enhancement']['confidence']:.3f}")
    
    print("\n✅ Workflow integration test complete!")

# Finish
