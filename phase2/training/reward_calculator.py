"""
Training Data Collection System - Reward Calculator
===================================================

Calculates educational quality rewards for assessment trajectories.

Author: MedIllustrator-AI Team
Version: 3.2.0
Date: 2025-10-14
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """
    Individual components of the total reward.
    
    Attributes:
        medical_terms_quality: Quality of medical terminology detection
        blooms_appropriateness: Appropriateness of Bloom's level
        cognitive_load_balance: Balance of cognitive load
        visual_quality: Visual analysis quality
        overall_score: Weighted overall score
    """
    medical_terms_quality: float
    blooms_appropriateness: float
    cognitive_load_balance: float
    visual_quality: float
    overall_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "medical_terms_quality": self.medical_terms_quality,
            "blooms_appropriateness": self.blooms_appropriateness,
            "cognitive_load_balance": self.cognitive_load_balance,
            "visual_quality": self.visual_quality,
            "overall_score": self.overall_score
        }


class RewardCalculator:
    """
    Calculates educational quality rewards from assessment trajectories.
    
    Features:
    - Multi-dimensional reward calculation
    - Educational rubric integration
    - Weighted scoring system
    - Normalization and scaling
    - Explainable rewards
    
    Reward Weights:
        - Medical Terms Quality: 30%
        - Bloom's Appropriateness: 25%
        - Cognitive Load Balance: 25%
        - Visual Quality: 20%
    
    Example:
        >>> calculator = RewardCalculator()
        >>> trajectory = collector.get_trajectory(trajectory_id)
        >>> reward = calculator.calculate_reward(trajectory)
        >>> print(f"Overall reward: {reward.overall_score:.2f}")
    """
    
    # Reward component weights
    DEFAULT_WEIGHTS = {
        "medical_terms_quality": 0.30,      # 30% weight
        "blooms_appropriateness": 0.25,     # 25% weight
        "cognitive_load_balance": 0.25,     # 25% weight
        "visual_quality": 0.20              # 20% weight
    }
    
    # Target thresholds for quality
    QUALITY_TARGETS = {
        "medical_terms_count": 15,          # Target: 15+ terms
        "quality_score_percent": 70,        # Target: 70%+
        "blooms_level_appropriate": True,
        "cognitive_load_balanced": True,
        "visual_features_extracted": True
    }
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        rubric_path: Optional[str] = None
    ):
        """
        Initialize reward calculator.
        
        Args:
            weights: Custom reward weights (default: DEFAULT_WEIGHTS)
            rubric_path: Path to educational rubric JSON
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Validate weights sum to 1.0
        weight_sum = sum(self.weights.values())
        if not np.isclose(weight_sum, 1.0):
            logger.warning(
                f"Weights sum to {weight_sum:.3f}, normalizing to 1.0"
            )
            self.weights = {
                k: v / weight_sum for k, v in self.weights.items()
            }
        
        # Load educational rubric
        self.rubric = self._load_rubric(rubric_path)
        
        logger.info(
            f"RewardCalculator initialized with weights: "
            f"{json.dumps(self.weights, indent=2)}"
        )
    
    def calculate_reward(
        self,
        trajectory: Any  # AssessmentTrajectory type
    ) -> RewardComponents:
        """
        Calculate overall reward from trajectory.
        
        Args:
            trajectory: Assessment trajectory
            
        Returns:
            RewardComponents with individual and overall scores
        """
        final_state = trajectory.final_state
        
        # Calculate individual reward components
        medical_terms_reward = self._calculate_medical_terms_reward(final_state)
        blooms_reward = self._calculate_blooms_reward(final_state)
        cognitive_load_reward = self._calculate_cognitive_load_reward(final_state)
        visual_reward = self._calculate_visual_quality_reward(final_state)
        
        # Calculate weighted overall score
        overall_score = (
            self.weights["medical_terms_quality"] * medical_terms_reward +
            self.weights["blooms_appropriateness"] * blooms_reward +
            self.weights["cognitive_load_balance"] * cognitive_load_reward +
            self.weights["visual_quality"] * visual_reward
        )
        
        # Create reward components
        rewards = RewardComponents(
            medical_terms_quality=medical_terms_reward,
            blooms_appropriateness=blooms_reward,
            cognitive_load_balance=cognitive_load_reward,
            visual_quality=visual_reward,
            overall_score=overall_score
        )
        
        logger.info(
            f"Calculated reward for trajectory {trajectory.trajectory_id}: "
            f"overall={overall_score:.3f}"
        )
        
        return rewards
    
    def _calculate_medical_terms_reward(
        self,
        state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for medical terms detection quality.
        
        Criteria:
        - Number of terms detected (target: 15+)
        - Term accuracy and relevance
        - Greek translation completeness
        - Category coverage
        
        Returns:
            Normalized reward [0, 1]
        """
        medical_analysis = state.get("medical_terms_analysis", {})
        
        if not medical_analysis:
            logger.warning("No medical terms analysis found")
            return 0.0
        
        # Extract metrics
        detected_terms = medical_analysis.get("detected_terms", [])
        num_terms = len(detected_terms)
        
        # Confidence score (if available)
        confidence = medical_analysis.get("confidence_score", 0.8)
        
        # Category coverage
        categories = medical_analysis.get("term_categories", {})
        category_coverage = len(categories) / max(5, 1)  # Assume 5 categories max
        
        # Greek translations completeness
        terms_with_greek = sum(
            1 for term in detected_terms 
            if term.get("greek_term")
        )
        greek_completeness = terms_with_greek / max(num_terms, 1)
        
        # Calculate score components
        # 1. Term count score (sigmoid-like curve)
        target_count = self.QUALITY_TARGETS["medical_terms_count"]
        count_score = min(1.0, num_terms / target_count)
        
        # 2. Quality score (confidence)
        quality_score = confidence
        
        # 3. Coverage score
        coverage_score = min(1.0, category_coverage)
        
        # 4. Completeness score
        completeness_score = greek_completeness
        
        # Weighted combination
        reward = (
            0.40 * count_score +           # 40% - term count
            0.30 * quality_score +         # 30% - confidence
            0.20 * coverage_score +        # 20% - category coverage
            0.10 * completeness_score      # 10% - greek translations
        )
        
        logger.debug(
            f"Medical terms reward: {reward:.3f} "
            f"(count={num_terms}, confidence={confidence:.2f})"
        )
        
        return float(np.clip(reward, 0.0, 1.0))
    
    def _calculate_blooms_reward(
        self,
        state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for Bloom's Taxonomy appropriateness.
        
        Criteria:
        - Appropriate cognitive level assignment
        - Justification quality
        - Alignment with image content
        - Recommendation relevance
        
        Returns:
            Normalized reward [0, 1]
        """
        blooms_analysis = state.get("blooms_analysis", {})
        
        if not blooms_analysis:
            logger.warning("No Bloom's analysis found")
            return 0.0
        
        # Extract metrics
        assigned_level = blooms_analysis.get("cognitive_level", 0)
        confidence = blooms_analysis.get("confidence_score", 0.7)
        has_justification = bool(blooms_analysis.get("justification"))
        has_recommendations = bool(blooms_analysis.get("recommendations"))
        
        # Level appropriateness (assuming levels 2-4 are most appropriate)
        level_appropriateness = 1.0 if 2 <= assigned_level <= 4 else 0.6
        
        # Quality indicators
        quality_score = confidence
        justification_score = 1.0 if has_justification else 0.5
        recommendation_score = 1.0 if has_recommendations else 0.7
        
        # Weighted combination
        reward = (
            0.40 * level_appropriateness +    # 40% - level appropriateness
            0.30 * quality_score +             # 30% - confidence
            0.20 * justification_score +       # 20% - justification present
            0.10 * recommendation_score        # 10% - recommendations present
        )
        
        logger.debug(
            f"Bloom's reward: {reward:.3f} "
            f"(level={assigned_level}, confidence={confidence:.2f})"
        )
        
        return float(np.clip(reward, 0.0, 1.0))
    
    def _calculate_cognitive_load_reward(
        self,
        state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for cognitive load balance.
        
        Criteria:
        - Intrinsic load appropriateness
        - Extraneous load minimization
        - Germane load optimization
        - Overall balance
        
        Returns:
            Normalized reward [0, 1]
        """
        cognitive_analysis = state.get("cognitive_load_analysis", {})
        
        if not cognitive_analysis:
            logger.warning("No cognitive load analysis found")
            return 0.0
        
        # Extract load scores (normalized 0-1)
        intrinsic = cognitive_analysis.get("intrinsic_load", 0.5) / 10.0
        extraneous = cognitive_analysis.get("extraneous_load", 0.5) / 10.0
        germane = cognitive_analysis.get("germane_load", 0.5) / 10.0
        
        # Ideal ranges:
        # - Intrinsic: 0.4-0.7 (moderate)
        # - Extraneous: 0.0-0.3 (low)
        # - Germane: 0.5-0.8 (high)
        
        # Score intrinsic load (prefer moderate)
        intrinsic_score = 1.0 - abs(intrinsic - 0.55) / 0.55
        
        # Score extraneous load (prefer low)
        extraneous_score = 1.0 - (extraneous / 0.3) if extraneous <= 0.3 else 0.5
        
        # Score germane load (prefer high)
        germane_score = germane / 0.65 if germane <= 0.65 else 1.0
        
        # Balance score (variance penalty)
        loads = [intrinsic, extraneous, germane]
        balance_score = 1.0 - min(1.0, np.std(loads))
        
        # Weighted combination
        reward = (
            0.25 * intrinsic_score +      # 25% - intrinsic appropriateness
            0.35 * extraneous_score +     # 35% - extraneous minimization
            0.30 * germane_score +        # 30% - germane optimization
            0.10 * balance_score          # 10% - overall balance
        )
        
        logger.debug(
            f"Cognitive load reward: {reward:.3f} "
            f"(I={intrinsic:.2f}, E={extraneous:.2f}, G={germane:.2f})"
        )
        
        return float(np.clip(reward, 0.0, 1.0))
    
    def _calculate_visual_quality_reward(
        self,
        state: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for visual analysis quality.
        
        Criteria:
        - Feature extraction completeness
        - Image quality metrics
        - Accessibility compliance
        - Visual clarity
        
        Returns:
            Normalized reward [0, 1]
        """
        visual_analysis = state.get("visual_features", {})
        
        if not visual_analysis:
            logger.warning("No visual analysis found")
            return 0.0
        
        # Extract metrics
        features_extracted = visual_analysis.get("features_extracted", False)
        quality_metrics = visual_analysis.get("quality_metrics", {})
        accessibility_score = visual_analysis.get("accessibility_score", 0.7)
        
        # Feature completeness
        feature_score = 1.0 if features_extracted else 0.3
        
        # Quality metrics score
        if quality_metrics:
            quality_score = np.mean([
                quality_metrics.get("clarity", 0.7),
                quality_metrics.get("contrast", 0.7),
                quality_metrics.get("sharpness", 0.7)
            ])
        else:
            quality_score = 0.5
        
        # Accessibility score
        accessibility_reward = accessibility_score
        
        # Weighted combination
        reward = (
            0.40 * feature_score +            # 40% - feature extraction
            0.35 * quality_score +            # 35% - quality metrics
            0.25 * accessibility_reward       # 25% - accessibility
        )
        
        logger.debug(
            f"Visual quality reward: {reward:.3f} "
            f"(features={features_extracted}, quality={quality_score:.2f})"
        )
        
        return float(np.clip(reward, 0.0, 1.0))
    
    def _load_rubric(
        self,
        rubric_path: Optional[str]
    ) -> Dict[str, Any]:
        """
        Load educational rubric from file.
        
        Args:
            rubric_path: Path to rubric JSON file
            
        Returns:
            Rubric dictionary
        """
        if not rubric_path:
            # Return default rubric
            return {
                "medical_terms": {
                    "excellent": {"min_terms": 15, "min_confidence": 0.8},
                    "good": {"min_terms": 10, "min_confidence": 0.7},
                    "fair": {"min_terms": 5, "min_confidence": 0.6}
                },
                "blooms_taxonomy": {
                    "appropriate_levels": [2, 3, 4],
                    "min_confidence": 0.7
                },
                "cognitive_load": {
                    "intrinsic_range": [0.4, 0.7],
                    "extraneous_max": 0.3,
                    "germane_min": 0.5
                },
                "visual_quality": {
                    "min_accessibility": 0.7,
                    "min_quality_score": 0.6
                }
            }
        
        try:
            with open(rubric_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load rubric from {rubric_path}: {e}")
            return {}
    
    def explain_reward(
        self,
        rewards: RewardComponents
    ) -> str:
        """
        Generate human-readable explanation of reward.
        
        Args:
            rewards: RewardComponents to explain
            
        Returns:
            Formatted explanation string
        """
        explanation = [
            "=== Reward Breakdown ===",
            f"Overall Score: {rewards.overall_score:.3f}",
            "",
            "Component Scores:",
            f"  Medical Terms Quality:      {rewards.medical_terms_quality:.3f} (weight: {self.weights['medical_terms_quality']:.0%})",
            f"  Bloom's Appropriateness:    {rewards.blooms_appropriateness:.3f} (weight: {self.weights['blooms_appropriateness']:.0%})",
            f"  Cognitive Load Balance:     {rewards.cognitive_load_balance:.3f} (weight: {self.weights['cognitive_load_balance']:.0%})",
            f"  Visual Quality:             {rewards.visual_quality:.3f} (weight: {self.weights['visual_quality']:.0%})",
            "",
            "Interpretation:",
            f"  Overall: {'Excellent' if rewards.overall_score >= 0.8 else 'Good' if rewards.overall_score >= 0.6 else 'Fair' if rewards.overall_score >= 0.4 else 'Needs Improvement'}",
            "======================="
        ]
        
        return "\n".join(explanation)
    
    def get_reward_statistics(
        self,
        reward_history: List[RewardComponents]
    ) -> Dict[str, Any]:
        """
        Calculate statistics from reward history.
        
        Args:
            reward_history: List of RewardComponents
            
        Returns:
            Statistics dictionary
        """
        if not reward_history:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0
            }
        
        overall_scores = [r.overall_score for r in reward_history]
        
        return {
            "count": len(reward_history),
            "mean": float(np.mean(overall_scores)),
            "std": float(np.std(overall_scores)),
            "min": float(np.min(overall_scores)),
            "max": float(np.max(overall_scores)),
            "median": float(np.median(overall_scores)),
            "percentile_25": float(np.percentile(overall_scores, 25)),
            "percentile_75": float(np.percentile(overall_scores, 75))
        }


# Finish
