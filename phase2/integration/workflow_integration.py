#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration with Main LangGraph Workflow

Connects Neural Reward Model to the assessment pipeline.

Author: Andreas Antonos
Created: 2025-10-21
Version: 1.0.0
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import time
import json

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from phase2.integration.model_service import get_model_service
from phase2.training.trajectory_collector import TrajectoryCollector
from phase2.training.storage_manager import StorageManager

logger = logging.getLogger(__name__)

class ARTWorkflowIntegration:
    """Integrates ART components with main workflow"""
    
    def __init__(self):
        """Initialize integration components"""
        
        # Get model service
        self.model_service = get_model_service()
        
        # Initialize trajectory collector
        self.trajectory_collector = TrajectoryCollector()
        
        # Storage manager for saving trajectories
        self.storage_manager = StorageManager()
        
        # Stats tracking
        self.assessments_processed = 0
        self.total_reward = 0.0
        
        logger.info("ART Workflow Integration initialized")
    
    def process_assessment(
        self,
        session_id: str,
        image_data: Dict,
        assessment_results: Dict
    ) -> Dict[str, Any]:
        """
        Process assessment with ART components.
        
        Args:
            session_id: Unique session identifier
            image_data: Input image information
            assessment_results: Results from agents
            
        Returns:
            Enhanced results with reward and trajectory
        """
        start_time = time.time()
        
        # Start trajectory capture
        self.trajectory_collector.start_trajectory(
            session_id=session_id,
            initial_state=image_data
        )
        
        # Add assessment events
        for agent_name, agent_result in assessment_results.items():
            self.trajectory_collector.add_event(
                session_id=session_id,
                event_type=f"agent_{agent_name}",
                data=agent_result
            )
        
        # Get reward prediction
        reward_result = self.model_service.predict_with_ruler(assessment_results)
        
        # Complete trajectory
        trajectory = self.trajectory_collector.finalize_trajectory(
            session_id=session_id,
            final_state=assessment_results,
            reward_score=reward_result['final_reward']
        )
        
        # Save trajectory for future training
        self.storage_manager.save_trajectory(trajectory)
        
        # Update stats
        self.assessments_processed += 1
        self.total_reward += reward_result['final_reward']
        
        # Prepare enhanced results
        enhanced_results = {
            **assessment_results,
            'art_enhancement': {
                'trajectory_id': trajectory['trajectory_id'],
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
    
    def should_trigger_training(self) -> bool:
        """Check if training should be triggered"""
        
        # Get trajectory count
        trajectories = self.storage_manager.query_trajectories(limit=1000)
        trajectory_count = len(trajectories)
        
        # Training criteria
        MIN_TRAJECTORIES = 100
        MIN_AVG_REWARD_DROP = 0.6
        
        if trajectory_count < MIN_TRAJECTORIES:
            return False
        
        # Check if average reward is dropping
        avg_reward = self.total_reward / max(1, self.assessments_processed)
        
        if avg_reward < MIN_AVG_REWARD_DROP:
            logger.info(f"Training triggered: Low avg reward ({avg_reward:.3f})")
            return True
        
        # Check every N assessments
        if self.assessments_processed > 0 and self.assessments_processed % 500 == 0:
            logger.info(f"Training triggered: Periodic update ({self.assessments_processed} assessments)")
            return True
        
        return False
    
    def get_integration_stats(self) -> Dict:
        """Get integration statistics"""
        
        avg_reward = self.total_reward / max(1, self.assessments_processed)
        
        return {
            'assessments_processed': self.assessments_processed,
            'average_reward': avg_reward,
            'model_metrics': self.model_service.get_metrics(),
            'trajectory_count': len(self.storage_manager.query_trajectories(limit=1000)),
            'storage_backend': self.storage_manager.backend
        }

# Integration hook for main app
def enhance_assessment_with_art(
    session_id: str,
    image_data: Dict,
    assessment_results: Dict
) -> Dict:
    """
    Main integration function for app_v3_langgraph.py
    
    This function can be called from the main app to enhance
    assessments with ART rewards and trajectory capture.
    """
    
    # Get or create integration instance
    if not hasattr(enhance_assessment_with_art, 'integration'):
        enhance_assessment_with_art.integration = ARTWorkflowIntegration()
    
    integration = enhance_assessment_with_art.integration
    
    # Process assessment
    enhanced_results = integration.process_assessment(
        session_id=session_id,
        initial_state=image_data,
        assessment_results=assessment_results
    )
    
    # Check if training should be triggered
    if integration.should_trigger_training():
        logger.info("Training criteria met - would trigger training in production")
        # In production, this would trigger async training job
    
    return enhanced_results

# Test function
if __name__ == "__main__":
    print("Testing Workflow Integration...")
    
    # Create test data
    test_session = f"test_session_{int(time.time())}"
    test_image = {'size': 1024*1024, 'format': 'png'}
    test_results = {
        'medical_terms': {
            'terms': ['heart', 'ventricle', 'atrium'],
            'confidence': 0.85
        },
        'bloom_taxonomy': {
            'level': 'Apply',
            'score': 0.75
        },
        'cognitive_load': {
            'score': 5.5,
            'optimal': True
        },
        'accessibility': {
            'wcag_compliance': 0.8,
            'issues': []
        }
    }
    
    # Test integration
    enhanced = enhance_assessment_with_art(
        session_id=test_session,
        image_data=test_image,
        assessment_results=test_results
    )
    
    print(f"\n✅ Enhanced Assessment:")
    print(f"  Trajectory ID: {enhanced['art_enhancement']['trajectory_id']}")
    print(f"  Final Reward: {enhanced['art_enhancement']['reward_score']:.3f}")
    print(f"  Confidence: {enhanced['art_enhancement']['confidence']:.3f}")
    print(f"  Processing Time: {enhanced['art_enhancement']['processing_time_ms']:.1f}ms")
    
    # Get stats
    integration = enhance_assessment_with_art.integration
    stats = integration.get_integration_stats()
    
    print(f"\n📊 Integration Stats:")
    print(f"  Assessments: {stats['assessments_processed']}")
    print(f"  Avg Reward: {stats['average_reward']:.3f}")
    print(f"  Trajectories: {stats['trajectory_count']}")
    
    print("\n✅ Workflow integration test complete!")

# Finish
