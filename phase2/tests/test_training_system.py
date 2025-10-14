"""
Integration Tests for Training Data Collection System
=====================================================

Tests the complete training data collection workflow.
"""

import pytest
from datetime import datetime
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phase2.training.trajectory_collector import TrajectoryCollector, AssessmentTrajectory
from phase2.training.reward_calculator import RewardCalculator
from phase2.training.data_validator import DataValidator
from phase2.training.storage_manager import StorageManager


class TestTrainingSystemIntegration:
    """Integration tests for training system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_storage_path = "phase2/tests/test_data"
        Path(self.test_storage_path).mkdir(parents=True, exist_ok=True)
        
        self.collector = TrajectoryCollector(storage_path=self.test_storage_path)
        self.calculator = RewardCalculator()
        self.validator = DataValidator(min_quality_score=0.5)
        self.storage = StorageManager(
            backend="local_json",
            storage_path=self.test_storage_path
        )
    
    def test_complete_workflow(self):
        """Test complete training data collection workflow."""
        # 1. Start trajectory
        session_id = "test_session_001"
        image_data = {"filename": "test_heart.jpg", "size": 1024}
        
        trajectory = self.collector.start_trajectory(
            session_id=session_id,
            image_data=image_data,
            metadata={"test": True}
        )
        
        assert trajectory.trajectory_id is not None
        assert trajectory.session_id == session_id
        
        # 2. Record intermediate states
        for i in range(3):
            state_data = {
                "step": i,
                "status": "processing",
                "data": f"state_{i}"
            }
            self.collector.record_state(
                trajectory.trajectory_id,
                state_data,
                agent_name=f"agent_{i}"
            )
        
        assert len(trajectory.intermediate_states) == 3
        
        # 3. Finalize trajectory
        final_state = {
            "medical_terms_analysis": {
                "detected_terms": [{"term": f"term_{i}"} for i in range(10)],
                "confidence_score": 0.85
            },
            "blooms_analysis": {
                "cognitive_level": 3,
                "confidence_score": 0.80
            },
            "cognitive_load_analysis": {
                "intrinsic_load": 5,
                "extraneous_load": 2,
                "germane_load": 6
            },
            "visual_features": {
                "features_extracted": True,
                "quality_metrics": {
                    "clarity": 0.8,
                    "contrast": 0.7,
                    "sharpness": 0.75
                },
                "accessibility_score": 0.8
            }
        }
        
        completed = self.collector.finalize_trajectory(
            trajectory.trajectory_id,
            final_state
        )
        
        assert completed.final_state["status"] == "completed"
        
        # 4. Calculate reward
        rewards = self.calculator.calculate_reward(completed)
        
        assert 0.0 <= rewards.overall_score <= 1.0
        assert rewards.medical_terms_quality > 0
        assert rewards.blooms_appropriateness > 0
        
        # 5. Validate trajectory
        validation = self.validator.validate_trajectory(completed)
        
        assert validation.score > 0.0
        assert isinstance(validation.is_valid, bool)
        
        # 6. Store trajectory
        success = self.storage.store_trajectory(completed)
        
        assert success is True
        
        # 7. Load trajectory
        loaded = self.storage.load_trajectory(trajectory.trajectory_id)
        
        assert loaded is not None
        assert loaded["trajectory_id"] == trajectory.trajectory_id
        
        print("\n✅ Complete workflow test passed!")
        print(f"   - Trajectory ID: {trajectory.trajectory_id}")
        print(f"   - Reward Score: {rewards.overall_score:.3f}")
        print(f"   - Validation Score: {validation.score:.3f}")
        print(f"   - Storage: Success")
    
    def test_statistics(self):
        """Test statistics collection."""
        # Collector stats
        collector_stats = self.collector.get_statistics()
        assert "active_trajectories" in collector_stats
        
        # Storage stats
        storage_stats = self.storage.get_statistics()
        assert "total_trajectories" in storage_stats
        
        print("\n📊 Statistics test passed!")
        print(f"   - Collector: {json.dumps(collector_stats, indent=2)}")
        print(f"   - Storage: {json.dumps(storage_stats, indent=2)}")
    
    def teardown_method(self):
        """Cleanup test data."""
        # Clean up test files
        import shutil
        if Path(self.test_storage_path).exists():
            shutil.rmtree(self.test_storage_path)


if __name__ == "__main__":
    # Run tests
    test = TestTrainingSystemIntegration()
    test.setup_method()
    
    try:
        test.test_complete_workflow()
        test.test_statistics()
        print("\n🎉 All integration tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        test.teardown_method()

# Finish
