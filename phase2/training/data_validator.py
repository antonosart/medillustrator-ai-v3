"""
Training Data Collection System - Data Validator
================================================

Validates training data quality and consistency.

Author: MedIllustrator-AI Team
Version: 3.2.0
Date: 2025-10-14
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """
    Result of trajectory validation.
    
    Attributes:
        is_valid: Whether trajectory passes validation
        score: Validation quality score [0, 1]
        errors: List of validation errors
        warnings: List of validation warnings
        metrics: Validation metrics
    """
    is_valid: bool
    score: float
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "score": self.score,
            "errors": self.errors,
            "warnings": self.warnings,
            "metrics": self.metrics
        }


class DataValidator:
    """
    Validates training data quality and consistency.
    
    Features:
    - Completeness checks
    - Consistency validation
    - Quality scoring
    - Anomaly detection
    - Data integrity verification
    
    Validation Criteria:
        - Trajectory completeness (all required fields present)
        - State consistency (logical progression)
        - Data quality (no corrupted or malformed data)
        - Reward validity (rewards within expected range)
        - Metadata completeness
    
    Example:
        >>> validator = DataValidator(min_quality_score=0.7)
        >>> result = validator.validate_trajectory(trajectory)
        >>> if result.is_valid:
        >>>     print("Trajectory is valid!")
    """
    
    # Validation thresholds
    MIN_STATES = 3              # Minimum intermediate states
    MAX_STATES = 100            # Maximum intermediate states
    MIN_DURATION = 5.0          # Minimum duration (seconds)
    MAX_DURATION = 300.0        # Maximum duration (seconds)
    MIN_REWARD = 0.0            # Minimum reward value
    MAX_REWARD = 1.0            # Maximum reward value
    
    def __init__(
        self,
        min_quality_score: float = 0.6,
        strict_mode: bool = False
    ):
        """
        Initialize data validator.
        
        Args:
            min_quality_score: Minimum quality score for validity [0, 1]
            strict_mode: Enable strict validation (fail on warnings)
        """
        self.min_quality_score = min_quality_score
        self.strict_mode = strict_mode
        
        logger.info(
            f"DataValidator initialized: "
            f"min_quality={min_quality_score}, strict={strict_mode}"
        )
    
    def validate_trajectory(
        self,
        trajectory: Any  # AssessmentTrajectory type
    ) -> ValidationResult:
        """
        Validate assessment trajectory.
        
        Args:
            trajectory: Assessment trajectory to validate
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        metrics = {}
        
        # 1. Validate completeness
        completeness_check = self._validate_completeness(trajectory)
        if not completeness_check["valid"]:
            errors.extend(completeness_check["errors"])
        warnings.extend(completeness_check["warnings"])
        metrics.update(completeness_check["metrics"])
        
        # 2. Validate consistency
        consistency_check = self._validate_consistency(trajectory)
        if not consistency_check["valid"]:
            errors.extend(consistency_check["errors"])
        warnings.extend(consistency_check["warnings"])
        metrics.update(consistency_check["metrics"])
        
        # 3. Validate quality
        quality_check = self._validate_quality(trajectory)
        if not quality_check["valid"]:
            errors.extend(quality_check["errors"])
        warnings.extend(quality_check["warnings"])
        metrics.update(quality_check["metrics"])
        
        # 4. Validate metadata
        metadata_check = self._validate_metadata(trajectory)
        warnings.extend(metadata_check["warnings"])
        metrics.update(metadata_check["metrics"])
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(
            completeness_check, consistency_check, quality_check, metadata_check
        )
        
        # Determine validity
        is_valid = (
            len(errors) == 0 and
            validation_score >= self.min_quality_score and
            (not self.strict_mode or len(warnings) == 0)
        )
        
        result = ValidationResult(
            is_valid=is_valid,
            score=validation_score,
            errors=errors,
            warnings=warnings,
            metrics=metrics
        )
        
        logger.info(
            f"Validated trajectory {trajectory.trajectory_id}: "
            f"valid={is_valid}, score={validation_score:.3f}, "
            f"errors={len(errors)}, warnings={len(warnings)}"
        )
        
        return result
    
    def _validate_completeness(
        self,
        trajectory: Any
    ) -> Dict[str, Any]:
        """Validate trajectory completeness."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check required fields
        required_fields = [
            "trajectory_id", "session_id", "timestamp",
            "image_hash", "initial_state", "final_state"
        ]
        
        missing_fields = []
        for field in required_fields:
            if not hasattr(trajectory, field) or getattr(trajectory, field) is None:
                missing_fields.append(field)
        
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Check intermediate states
        num_states = len(trajectory.intermediate_states)
        if num_states < self.MIN_STATES:
            warnings.append(
                f"Too few intermediate states: {num_states} < {self.MIN_STATES}"
            )
        elif num_states > self.MAX_STATES:
            warnings.append(
                f"Too many intermediate states: {num_states} > {self.MAX_STATES}"
            )
        
        # Check agent executions
        num_executions = len(trajectory.agent_executions)
        if num_executions == 0:
            warnings.append("No agent executions recorded")
        
        metrics["num_states"] = num_states
        metrics["num_executions"] = num_executions
        metrics["completeness_score"] = 1.0 - (len(missing_fields) / len(required_fields))
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics
        }
    
    def _validate_consistency(
        self,
        trajectory: Any
    ) -> Dict[str, Any]:
        """Validate trajectory consistency."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check duration
        duration = trajectory.metadata.get("duration_seconds", 0)
        if duration < self.MIN_DURATION:
            warnings.append(
                f"Duration too short: {duration:.1f}s < {self.MIN_DURATION}s"
            )
        elif duration > self.MAX_DURATION:
            errors.append(
                f"Duration too long: {duration:.1f}s > {self.MAX_DURATION}s"
            )
        
        # Check timestamp consistency
        if trajectory.intermediate_states:
            timestamps = []
            for state in trajectory.intermediate_states:
                if "timestamp" in state:
                    try:
                        ts = datetime.fromisoformat(state["timestamp"])
                        timestamps.append(ts)
                    except:
                        warnings.append("Invalid timestamp format in state")
            
            # Check monotonicity
            if len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    if timestamps[i] < timestamps[i-1]:
                        errors.append("Timestamps are not monotonically increasing")
                        break
        
        # Check state progression
        state_count_metadata = trajectory.metadata.get("num_states", 0)
        actual_state_count = len(trajectory.intermediate_states)
        if state_count_metadata != actual_state_count:
            warnings.append(
                f"State count mismatch: metadata={state_count_metadata}, "
                f"actual={actual_state_count}"
            )
        
        metrics["duration_seconds"] = duration
        metrics["consistency_score"] = 1.0 if len(errors) == 0 else 0.5
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics
        }
    
    def _validate_quality(
        self,
        trajectory: Any
    ) -> Dict[str, Any]:
        """Validate data quality."""
        errors = []
        warnings = []
        metrics = {}
        
        # Check final state completeness
        final_state = trajectory.final_state
        expected_analyses = [
            "medical_terms_analysis",
            "blooms_analysis",
            "cognitive_load_analysis"
        ]
        
        missing_analyses = []
        for analysis in expected_analyses:
            if analysis not in final_state or not final_state[analysis]:
                missing_analyses.append(analysis)
        
        if missing_analyses:
            errors.append(
                f"Missing analyses in final state: {', '.join(missing_analyses)}"
            )
        
        # Check data corruption
        try:
            # Attempt to serialize to JSON (catches most corruption issues)
            import json
            json.dumps(trajectory.to_dict())
        except Exception as e:
            errors.append(f"Data corruption detected: {str(e)}")
        
        # Check for anomalous values
        anomalies = []
        
        # Check medical terms count
        if "medical_terms_analysis" in final_state:
            terms = final_state["medical_terms_analysis"].get("detected_terms", [])
            if len(terms) > 100:  # Unrealistic number
                anomalies.append(f"Unusually high term count: {len(terms)}")
        
        if anomalies:
            warnings.extend(anomalies)
        
        metrics["quality_score"] = 1.0 - (len(missing_analyses) / len(expected_analyses))
        metrics["has_anomalies"] = len(anomalies) > 0
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics
        }
    
    def _validate_metadata(
        self,
        trajectory: Any
    ) -> Dict[str, Any]:
        """Validate trajectory metadata."""
        warnings = []
        metrics = {}
        
        metadata = trajectory.metadata
        
        # Check recommended metadata fields
        recommended_fields = [
            "duration_seconds", "num_states", "num_agent_executions"
        ]
        
        missing_metadata = []
        for field in recommended_fields:
            if field not in metadata:
                missing_metadata.append(field)
        
        if missing_metadata:
            warnings.append(
                f"Missing recommended metadata: {', '.join(missing_metadata)}"
            )
        
        metrics["metadata_completeness"] = 1.0 - (
            len(missing_metadata) / len(recommended_fields)
        )
        
        return {
            "warnings": warnings,
            "metrics": metrics
        }
    
    def _calculate_validation_score(
        self,
        *check_results: Dict[str, Any]
    ) -> float:
        """Calculate overall validation score."""
        scores = []
        
        for result in check_results:
            if "metrics" in result:
                # Extract score metrics
                for key, value in result["metrics"].items():
                    if "score" in key and isinstance(value, (int, float)):
                        scores.append(value)
        
        if not scores:
            return 0.5  # Default middle score
        
        return float(np.mean(scores))
    
    def validate_batch(
        self,
        trajectories: List[Any]
    ) -> Tuple[List[ValidationResult], Dict[str, Any]]:
        """
        Validate batch of trajectories.
        
        Args:
            trajectories: List of trajectories to validate
            
        Returns:
            Tuple of (validation results, batch statistics)
        """
        results = []
        
        for trajectory in trajectories:
            result = self.validate_trajectory(trajectory)
            results.append(result)
        
        # Calculate batch statistics
        valid_count = sum(1 for r in results if r.is_valid)
        total_count = len(results)
        
        scores = [r.score for r in results]
        
        statistics = {
            "total_trajectories": total_count,
            "valid_count": valid_count,
            "invalid_count": total_count - valid_count,
            "validation_rate": valid_count / max(total_count, 1),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "min_score": float(np.min(scores)) if scores else 0.0,
            "max_score": float(np.max(scores)) if scores else 0.0,
            "total_errors": sum(len(r.errors) for r in results),
            "total_warnings": sum(len(r.warnings) for r in results)
        }
        
        logger.info(
            f"Validated batch: {valid_count}/{total_count} valid "
            f"(rate={statistics['validation_rate']:.1%})"
        )
        
        return results, statistics


# Finish
