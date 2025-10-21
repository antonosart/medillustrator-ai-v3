#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration with Main LangGraph Workflow - Final Fixed Version
"""

import logging
from typing import Dict, Any
from pathlib import Path
import time

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from phase2.integration.model_service import get_model_service
from phase2.training.trajectory_collector import TrajectoryCollector
from phase2.training.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class ARTWorkflowIntegration:
    def __init__(self):
        self.model_service = get_model_service()
        self.trajectory_collector = TrajectoryCollector()
        self.storage_manager = StorageManager()
        self.assessments_processed = 0
        self.total_reward = 0.0
    
    def process_assessment(self, session_id: str, image_data: Dict, assessment_results: Dict) -> Dict:
        start_time = time.time()
        
        # Start trajectory with correct parameters
        trajectory = self.trajectory_collector.start_trajectory(
            session_id=session_id,
            image_data=image_data,
            metadata={'timestamp': time.time()}
        )
        
        # Get reward
        reward_result = self.model_service.predict_with_ruler(assessment_results)
        
        # Finalize trajectory
        final_trajectory = self.trajectory_collector.finalize_trajectory(
            trajectory_id=trajectory.trajectory_id,
            final_state={
                **assessment_results,
                'reward_score': reward_result['final_reward']
            }
        )
        
        # Save
        if final_trajectory:
            self.storage_manager.store_trajectory(final_trajectory)
        
        self.assessments_processed += 1
        self.total_reward += reward_result['final_reward']
        
        return {
            **assessment_results,
            'art_enhancement': {
                'trajectory_id': trajectory.trajectory_id,
                'reward_score': reward_result['final_reward'],
                'confidence': reward_result['confidence'],
                'processing_time_ms': (time.time() - start_time) * 1000
            }
        }

# Test
if __name__ == "__main__":
    integration = ARTWorkflowIntegration()
    result = integration.process_assessment(
        session_id=f"test_{int(time.time())}",
        image_data={'size': 1024},
        assessment_results={'medical_terms': ['heart']}
    )
    print(f"✅ Integration working! Reward: {result['art_enhancement']['reward_score']:.3f}")
