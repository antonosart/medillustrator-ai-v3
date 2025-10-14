"""
Training Data Collection System - Trajectory Collector
======================================================

Captures assessment trajectories in real-time for ART training.

Author: MedIllustrator-AI Team
Version: 3.2.0
Date: 2025-10-14
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
import hashlib
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AssessmentTrajectory:
    """
    Represents a complete assessment trajectory.
    
    Attributes:
        trajectory_id: Unique identifier for trajectory
        session_id: Session identifier
        timestamp: Creation timestamp
        image_hash: Hash of assessed image
        initial_state: Initial assessment state
        intermediate_states: List of intermediate states
        final_state: Final assessment state
        agent_executions: Agent execution records
        metadata: Additional metadata
    """
    trajectory_id: str
    session_id: str
    timestamp: datetime
    image_hash: str
    initial_state: Dict[str, Any]
    intermediate_states: List[Dict[str, Any]] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    agent_executions: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trajectory to dictionary format."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Serialize trajectory to JSON."""
        return json.dumps(self.to_dict(), indent=2)


class TrajectoryCollector:
    """
    Collects assessment trajectories for training data.
    
    Features:
    - Real-time trajectory capture
    - State serialization
    - Metadata collection
    - Quality validation
    - Batch optimization
    
    Example:
        >>> collector = TrajectoryCollector(storage_path="data/trajectories")
        >>> trajectory = collector.start_trajectory(session_id="abc123")
        >>> collector.record_state(trajectory.trajectory_id, state_data)
        >>> collector.finalize_trajectory(trajectory.trajectory_id, final_state)
    """
    
    def __init__(
        self,
        storage_path: str = "phase2/training/data",
        auto_save: bool = True,
        batch_size: int = 10
    ):
        """
        Initialize trajectory collector.
        
        Args:
            storage_path: Path for storing trajectories
            auto_save: Enable automatic saving
            batch_size: Number of trajectories before batch save
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.auto_save = auto_save
        self.batch_size = batch_size
        
        # Active trajectories (in-memory)
        self.active_trajectories: Dict[str, AssessmentTrajectory] = {}
        
        # Completed trajectories buffer
        self.completed_buffer: List[AssessmentTrajectory] = []
        
        logger.info(
            f"TrajectoryCollector initialized: "
            f"storage={storage_path}, auto_save={auto_save}"
        )
    
    def start_trajectory(
        self,
        session_id: str,
        image_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AssessmentTrajectory:
        """
        Start a new assessment trajectory.
        
        Args:
            session_id: Session identifier
            image_data: Image data and metadata
            metadata: Additional metadata
            
        Returns:
            New AssessmentTrajectory instance
        """
        # Generate unique trajectory ID
        trajectory_id = self._generate_trajectory_id(session_id)
        
        # Calculate image hash
        image_hash = self._hash_image_data(image_data)
        
        # Create initial state
        initial_state = {
            "session_id": session_id,
            "image_data": image_data,
            "timestamp": datetime.now().isoformat(),
            "status": "initialized"
        }
        
        # Create trajectory
        trajectory = AssessmentTrajectory(
            trajectory_id=trajectory_id,
            session_id=session_id,
            timestamp=datetime.now(),
            image_hash=image_hash,
            initial_state=initial_state,
            metadata=metadata or {}
        )
        
        # Store in active trajectories
        self.active_trajectories[trajectory_id] = trajectory
        
        logger.info(f"Started trajectory: {trajectory_id}")
        
        return trajectory
    
    def record_state(
        self,
        trajectory_id: str,
        state_data: Dict[str, Any],
        agent_name: Optional[str] = None
    ) -> None:
        """
        Record intermediate state in trajectory.
        
        Args:
            trajectory_id: Trajectory identifier
            state_data: State data to record
            agent_name: Name of agent that produced this state
        """
        if trajectory_id not in self.active_trajectories:
            logger.warning(f"Trajectory not found: {trajectory_id}")
            return
        
        trajectory = self.active_trajectories[trajectory_id]
        
        # Add timestamp to state
        timestamped_state = {
            **state_data,
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name
        }
        
        # Append to intermediate states
        trajectory.intermediate_states.append(timestamped_state)
        
        # Record agent execution if provided
        if agent_name:
            execution_record = {
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
                "state_snapshot": timestamped_state
            }
            trajectory.agent_executions.append(execution_record)
        
        logger.debug(f"Recorded state for trajectory: {trajectory_id}")
    
    def finalize_trajectory(
        self,
        trajectory_id: str,
        final_state: Dict[str, Any]
    ) -> AssessmentTrajectory:
        """
        Finalize trajectory and prepare for storage.
        
        Args:
            trajectory_id: Trajectory identifier
            final_state: Final assessment state
            
        Returns:
            Completed AssessmentTrajectory
        """
        if trajectory_id not in self.active_trajectories:
            raise ValueError(f"Trajectory not found: {trajectory_id}")
        
        trajectory = self.active_trajectories[trajectory_id]
        
        # Set final state
        trajectory.final_state = {
            **final_state,
            "timestamp": datetime.now().isoformat(),
            "status": "completed"
        }
        
        # Calculate trajectory duration
        start_time = trajectory.timestamp
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        trajectory.metadata["duration_seconds"] = duration
        trajectory.metadata["num_states"] = len(trajectory.intermediate_states)
        trajectory.metadata["num_agent_executions"] = len(trajectory.agent_executions)
        
        # Move to completed buffer
        self.completed_buffer.append(trajectory)
        del self.active_trajectories[trajectory_id]
        
        logger.info(
            f"Finalized trajectory: {trajectory_id} "
            f"(duration={duration:.2f}s, states={len(trajectory.intermediate_states)})"
        )
        
        # Auto-save if enabled and batch size reached
        if self.auto_save and len(self.completed_buffer) >= self.batch_size:
            self.save_batch()
        
        return trajectory
    
    def save_batch(self) -> int:
        """
        Save completed trajectories to storage.
        
        Returns:
            Number of trajectories saved
        """
        if not self.completed_buffer:
            logger.debug("No trajectories to save")
            return 0
        
        saved_count = 0
        
        for trajectory in self.completed_buffer:
            try:
                # Create filename with timestamp
                filename = (
                    f"trajectory_{trajectory.trajectory_id}_"
                    f"{trajectory.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                )
                filepath = self.storage_path / filename
                
                # Save to file
                with open(filepath, 'w') as f:
                    f.write(trajectory.to_json())
                
                saved_count += 1
                logger.debug(f"Saved trajectory: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to save trajectory {trajectory.trajectory_id}: {e}")
        
        # Clear buffer
        self.completed_buffer.clear()
        
        logger.info(f"Saved batch: {saved_count} trajectories")
        
        return saved_count
    
    def get_trajectory(self, trajectory_id: str) -> Optional[AssessmentTrajectory]:
        """
        Get trajectory by ID.
        
        Args:
            trajectory_id: Trajectory identifier
            
        Returns:
            AssessmentTrajectory if found, None otherwise
        """
        # Check active trajectories
        if trajectory_id in self.active_trajectories:
            return self.active_trajectories[trajectory_id]
        
        # Check completed buffer
        for trajectory in self.completed_buffer:
            if trajectory.trajectory_id == trajectory_id:
                return trajectory
        
        # Try loading from disk
        return self._load_trajectory_from_disk(trajectory_id)
    
    def _generate_trajectory_id(self, session_id: str) -> str:
        """Generate unique trajectory ID."""
        timestamp = datetime.now().isoformat()
        unique_string = f"{session_id}_{timestamp}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:16]
    
    def _hash_image_data(self, image_data: Dict[str, Any]) -> str:
        """Calculate hash of image data."""
        # Use image content or filename for hashing
        image_str = json.dumps(image_data, sort_keys=True)
        return hashlib.sha256(image_str.encode()).hexdigest()[:32]
    
    def _load_trajectory_from_disk(
        self,
        trajectory_id: str
    ) -> Optional[AssessmentTrajectory]:
        """Load trajectory from disk storage."""
        # Search for trajectory file
        for filepath in self.storage_path.glob(f"trajectory_{trajectory_id}_*.json"):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct trajectory object
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                return AssessmentTrajectory(**data)
                
            except Exception as e:
                logger.error(f"Failed to load trajectory from {filepath}: {e}")
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get collector statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "active_trajectories": len(self.active_trajectories),
            "completed_buffer_size": len(self.completed_buffer),
            "storage_path": str(self.storage_path),
            "auto_save": self.auto_save,
            "batch_size": self.batch_size,
            "total_stored": len(list(self.storage_path.glob("trajectory_*.json")))
        }


# Finish
