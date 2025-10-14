"""
config/art_settings.py - Expert-Level ART Integration Configuration
COMPLETE PRODUCTION-READY ART CONFIGURATION με RULER integration
Author: Andreas Antonos (25 years Python experience)
Date: 2025-10-14
Quality Level: 9.5/10 Expert-Level
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Literal
from pathlib import Path
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)


# ============================================================================
# EXPERT IMPROVEMENT 1: COMPREHENSIVE CONSTANTS CLASS
# ============================================================================


class ARTConstants:
    """Centralized constants για ART integration - ELIMINATES MAGIC NUMBERS"""
    
    # Version Control
    VERSION = "3.1.0"
    ART_VERSION = "0.2.5"
    
    # Default Ports
    DEFAULT_ART_SERVER_PORT = 5000
    DEFAULT_RULER_PORT = 5001
    
    # Timeout Configuration (seconds)
    TRAINING_TIMEOUT = 3600  # 1 hour
    INFERENCE_TIMEOUT = 300  # 5 minutes
    RULER_EVALUATION_TIMEOUT = 600  # 10 minutes
    
    # Model Configuration
    DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
    MODEL_MAX_LENGTH = 4096
    
    # Training Hyperparameters
    DEFAULT_BATCH_SIZE = 4
    DEFAULT_LEARNING_RATE = 1e-5
    DEFAULT_NUM_ITERATIONS = 50
    DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
    
    # RULER Configuration
    DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4-20250514"
    DEFAULT_GROUP_SIZE = 6
    MIN_CONFIDENCE_THRESHOLD = 0.7
    
    # Performance Thresholds
    QUALITY_IMPROVEMENT_THRESHOLD = 0.15  # 15% improvement target
    BASELINE_ACCURACY_THRESHOLD = 0.85
    
    # Resource Limits
    MAX_GPU_MEMORY_GB = 16
    MAX_PARALLEL_TRAINING_JOBS = 2
    CHECKPOINT_INTERVAL = 10  # Save every 10 iterations
    
    # Cache Configuration
    TRAJECTORY_CACHE_SIZE_MB = 1024  # 1GB
    MODEL_CACHE_SIZE_GB = 50
    
    # Paths
    DEFAULT_MODELS_DIR = "models"
    DEFAULT_TRAINING_DATA_DIR = "data/training_images"
    DEFAULT_CACHE_DIR = "cache/art_trajectories"


# ============================================================================
# EXPERT IMPROVEMENT 2: ENUMS FOR TYPE SAFETY
# ============================================================================


class ModelVersion(str, Enum):
    """Model version enumeration για type safety"""
    BASELINE = "baseline"
    TRAINED_V1 = "trained-v1"
    TRAINED_V2 = "trained-v2"
    EXPERIMENTAL = "experimental"


class TrainingMode(str, Enum):
    """Training mode enumeration"""
    OFFLINE = "offline"  # Batch training
    ONLINE = "online"  # Continuous learning
    EVALUATION = "evaluation"  # Evaluation only
    

class ResourceTier(str, Enum):
    """Resource allocation tiers"""
    MINIMAL = "minimal"  # CPU only
    STANDARD = "standard"  # Single GPU
    ENHANCED = "enhanced"  # Multi-GPU
    MAXIMUM = "maximum"  # Full cluster


# ============================================================================
# EXPERT IMPROVEMENT 3: STRUCTURED CONFIGURATION CLASSES
# ============================================================================


@dataclass
class ARTServerConfig:
    """ART server configuration με validation"""
    
    enabled: bool = False
    server_url: str = f"http://localhost:{ARTConstants.DEFAULT_ART_SERVER_PORT}"
    api_timeout: int = ARTConstants.INFERENCE_TIMEOUT
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    
    # GPU Configuration
    use_gpu: bool = True
    gpu_device_id: int = 0
    max_gpu_memory_gb: int = ARTConstants.MAX_GPU_MEMORY_GB
    
    # Performance Settings
    batch_size: int = ARTConstants.DEFAULT_BATCH_SIZE
    num_workers: int = 2
    prefetch_factor: int = 2
    
    def validate(self) -> bool:
        """Validate server configuration"""
        if self.enabled:
            if not self.server_url:
                raise ValueError("ART server URL required when enabled")
            if self.api_timeout <= 0:
                raise ValueError("API timeout must be positive")
        return True


@dataclass
class ModelConfiguration:
    """Model configuration με version management"""
    
    # Model Selection
    model_version: ModelVersion = ModelVersion.BASELINE
    base_model_name: str = ARTConstants.DEFAULT_BASE_MODEL
    
    # Model Paths
    models_dir: Path = field(default_factory=lambda: Path(ARTConstants.DEFAULT_MODELS_DIR))
    baseline_path: Optional[Path] = None
    trained_path: Optional[Path] = None
    
    # Model Parameters
    max_length: int = ARTConstants.MODEL_MAX_LENGTH
    temperature: float = 0.7
    top_p: float = 0.9
    
    # Quantization (για resource efficiency)
    use_quantization: bool = False
    quantization_bits: int = 8
    
    def __post_init__(self):
        """Initialize paths"""
        self.models_dir = Path(self.models_dir)
        if self.baseline_path is None:
            self.baseline_path = self.models_dir / "baseline"
        if self.trained_path is None:
            self.trained_path = self.models_dir / "production"
    
    def get_active_model_path(self) -> Path:
        """Get path for currently active model"""
        if self.model_version == ModelVersion.BASELINE:
            return self.baseline_path
        else:
            return self.trained_path / f"{self.model_version.value}.pth"
    
    def validate(self) -> bool:
        """Validate model configuration"""
        if not self.models_dir.exists():
            logger.warning(f"Models directory not found: {self.models_dir}")
            self.models_dir.mkdir(parents=True, exist_ok=True)
        return True


@dataclass
class TrainingConfiguration:
    """Training pipeline configuration"""
    
    # Training Mode
    training_mode: TrainingMode = TrainingMode.OFFLINE
    training_enabled: bool = False
    
    # Hyperparameters
    learning_rate: float = ARTConstants.DEFAULT_LEARNING_RATE
    batch_size: int = ARTConstants.DEFAULT_BATCH_SIZE
    num_iterations: int = ARTConstants.DEFAULT_NUM_ITERATIONS
    gradient_accumulation_steps: int = ARTConstants.DEFAULT_GRADIENT_ACCUMULATION_STEPS
    
    # Advanced Training Settings
    warmup_steps: int = 10
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # LoRA Configuration (για efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Data Configuration
    training_data_dir: Path = field(
        default_factory=lambda: Path(ARTConstants.DEFAULT_TRAINING_DATA_DIR)
    )
    validation_split: float = 0.2
    shuffle_data: bool = True
    
    # Checkpointing
    checkpoint_interval: int = ARTConstants.CHECKPOINT_INTERVAL
    keep_last_n_checkpoints: int = 3
    
    # Early Stopping
    enable_early_stopping: bool = True
    patience: int = 5
    min_delta: float = 0.001
    
    def __post_init__(self):
        """Initialize and validate paths"""
        self.training_data_dir = Path(self.training_data_dir)
        if not self.training_data_dir.exists() and self.training_enabled:
            logger.warning(f"Training data directory not found: {self.training_data_dir}")
            self.training_data_dir.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> bool:
        """Validate training configuration"""
        if self.training_enabled:
            if self.learning_rate <= 0 or self.learning_rate > 1:
                raise ValueError(f"Invalid learning rate: {self.learning_rate}")
            if self.batch_size <= 0:
                raise ValueError(f"Invalid batch size: {self.batch_size}")
            if self.validation_split < 0 or self.validation_split >= 1:
                raise ValueError(f"Invalid validation split: {self.validation_split}")
        return True


@dataclass
class RULERConfiguration:
    """RULER evaluation framework configuration"""
    
    # RULER Settings
    enabled: bool = False
    judge_model: str = ARTConstants.DEFAULT_JUDGE_MODEL
    group_size: int = ARTConstants.DEFAULT_GROUP_SIZE
    
    # Educational Rubric
    use_educational_rubric: bool = True
    rubric_path: Optional[Path] = None
    
    # Evaluation Criteria
    evaluate_medical_accuracy: bool = True
    evaluate_pedagogical_effectiveness: bool = True
    evaluate_cognitive_load: bool = True
    evaluate_accessibility: bool = True
    
    # Scoring Configuration
    min_confidence_threshold: float = ARTConstants.MIN_CONFIDENCE_THRESHOLD
    aggregate_method: Literal["mean", "weighted", "median"] = "weighted"
    
    # Cache Configuration
    cache_evaluations: bool = True
    cache_dir: Path = field(
        default_factory=lambda: Path(ARTConstants.DEFAULT_CACHE_DIR) / "ruler"
    )
    
    # Performance
    parallel_evaluations: bool = True
    max_parallel_requests: int = 5
    evaluation_timeout: int = ARTConstants.RULER_EVALUATION_TIMEOUT
    
    def __post_init__(self):
        """Initialize paths"""
        self.cache_dir = Path(self.cache_dir)
        if self.rubric_path is None:
            self.rubric_path = Path("config/educational_rubric.yaml")
    
    def validate(self) -> bool:
        """Validate RULER configuration"""
        if self.enabled:
            if not self.judge_model:
                raise ValueError("RULER judge model required when enabled")
            if self.group_size <= 0:
                raise ValueError("Invalid RULER group size")
            if self.cache_evaluations and not self.cache_dir.exists():
                self.cache_dir.mkdir(parents=True, exist_ok=True)
        return True


# ============================================================================
# EXPERT IMPROVEMENT 4: MAIN ART CONFIGURATION CLASS
# ============================================================================


class ARTConfig:
    """
    Complete ART integration configuration με expert-level structure
    """
    
    def __init__(self):
        """Initialize ART configuration με environment-aware settings"""
        
        # Core Settings
        self.version = ARTConstants.VERSION
        self.art_version = ARTConstants.ART_VERSION
        
        # Component Configurations
        self.server = self._initialize_server_config()
        self.model = self._initialize_model_config()
        self.training = self._initialize_training_config()
        self.ruler = self._initialize_ruler_config()
        
        # Validate all configurations
        self._validate_all()
        
        logger.info("✅ ART Configuration initialized successfully")
    
    def _initialize_server_config(self) -> ARTServerConfig:
        """Initialize server configuration από environment"""
        return ARTServerConfig(
            enabled=os.getenv("ART_ENABLED", "false").lower() == "true",
            server_url=os.getenv(
                "ART_SERVER_URL",
                f"http://localhost:{ARTConstants.DEFAULT_ART_SERVER_PORT}"
            ),
            use_gpu=os.getenv("ART_USE_GPU", "true").lower() == "true",
        )
    
    def _initialize_model_config(self) -> ModelConfiguration:
        """Initialize model configuration"""
        version_str = os.getenv("MODEL_VERSION", "baseline")
        try:
            model_version = ModelVersion(version_str)
        except ValueError:
            logger.warning(f"Invalid model version: {version_str}, using baseline")
            model_version = ModelVersion.BASELINE
        
        return ModelConfiguration(
            model_version=model_version,
            base_model_name=os.getenv(
                "TRAINING_BASE_MODEL",
                ARTConstants.DEFAULT_BASE_MODEL
            ),
        )
    
    def _initialize_training_config(self) -> TrainingConfiguration:
        """Initialize training configuration"""
        training_enabled = os.getenv("ART_TRAINING_MODE", "false").lower() == "true"
        
        return TrainingConfiguration(
            training_enabled=training_enabled,
            learning_rate=float(os.getenv(
                "TRAINING_LEARNING_RATE",
                str(ARTConstants.DEFAULT_LEARNING_RATE)
            )),
            batch_size=int(os.getenv(
                "TRAINING_BATCH_SIZE",
                str(ARTConstants.DEFAULT_BATCH_SIZE)
            )),
            num_iterations=int(os.getenv(
                "TRAINING_NUM_ITERATIONS",
                str(ARTConstants.DEFAULT_NUM_ITERATIONS)
            )),
        )
    
    def _initialize_ruler_config(self) -> RULERConfiguration:
        """Initialize RULER configuration"""
        return RULERConfiguration(
            enabled=os.getenv("RULER_ENABLED", "false").lower() == "true",
            judge_model=os.getenv(
                "RULER_JUDGE_MODEL",
                ARTConstants.DEFAULT_JUDGE_MODEL
            ),
            group_size=int(os.getenv(
                "RULER_GROUP_SIZE",
                str(ARTConstants.DEFAULT_GROUP_SIZE)
            )),
        )
    
    def _validate_all(self) -> None:
        """Validate all configuration components"""
        try:
            self.server.validate()
            self.model.validate()
            self.training.validate()
            self.ruler.validate()
            logger.info("✅ All ART configurations validated successfully")
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    # Convenience Methods
    
    def is_enabled(self) -> bool:
        """Check if ART integration is enabled"""
        return self.server.enabled
    
    def is_training_enabled(self) -> bool:
        """Check if training is enabled"""
        return self.training.training_enabled
    
    def is_ruler_enabled(self) -> bool:
        """Check if RULER evaluation is enabled"""
        return self.ruler.enabled
    
    def use_trained_model(self) -> bool:
        """Check if using trained model (not baseline)"""
        return self.model.model_version != ModelVersion.BASELINE
    
    def get_current_model_path(self) -> Path:
        """Get path to currently active model"""
        return self.model.get_active_model_path()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary για serialization"""
        return {
            "version": self.version,
            "art_version": self.art_version,
            "server": {
                "enabled": self.server.enabled,
                "url": self.server.server_url,
                "use_gpu": self.server.use_gpu,
            },
            "model": {
                "version": self.model.model_version.value,
                "base_model": self.model.base_model_name,
                "active_path": str(self.model.get_active_model_path()),
            },
            "training": {
                "enabled": self.training.training_enabled,
                "mode": self.training.training_mode.value,
            },
            "ruler": {
                "enabled": self.ruler.enabled,
                "judge_model": self.ruler.judge_model,
            },
        }


# ============================================================================
# EXPERT IMPROVEMENT 5: GLOBAL CONFIGURATION INSTANCE
# ============================================================================


# Global configuration instance
art_config = ARTConfig()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_art_config() -> ARTConfig:
    """Get global ART configuration instance"""
    return art_config


__all__ = [
    "ARTConstants",
    "ModelVersion",
    "TrainingMode",
    "ResourceTier",
    "ARTServerConfig",
    "ModelConfiguration",
    "TrainingConfiguration",
    "RULERConfiguration",
    "ARTConfig",
    "art_config",
    "get_art_config",
]

__version__ = ARTConstants.VERSION
__author__ = "Andreas Antonos"

logger.info("✅ config/art_settings.py loaded successfully")

# Finish
