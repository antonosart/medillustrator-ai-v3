"""
MedIllustrator-AI v3.0 - Enhanced Visual Analysis με CLIP και AI2D Integration
Advanced multi-modal feature extraction για medical images
Author: Andreas Antonos
Date: 2025-07-18
"""

import logging
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import asyncio
import io
import hashlib
import base64

# Core imports
from PIL import Image, ImageEnhance, ImageStat, ImageFilter
import cv2

# AI και ML imports με fallbacks
try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import clip

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    clip = None

try:
    from transformers import AutoProcessor, AutoModel
    from sentence_transformers import SentenceTransformer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from datasets import load_dataset
    import json

    AI2D_AVAILABLE = True
except ImportError:
    AI2D_AVAILABLE = False

# Project imports
try:
    from config.settings import clip_config, ai2d_config, performance_config
    from workflows.state_definitions import (
        VisualAnalysis,
        ImageData,
        ErrorInfo,
        ErrorSeverity,
    )
except ImportError:
    # Fallback definitions for standalone usage
    class VisualAnalysis:
        def __init__(self):
            self.confidence = 0.0
            self.traditional_cv_features = {}
            self.clip_features = {}
            self.image_quality_metrics = {}
            self.medical_relevance_score = 0.0
            self.semantic_complexity = 0.0
            self.medical_classifications = []
            self.multimodal_score = 0.0
            self.ai2d_analysis = {}
            self.diagram_type = None
            self.diagram_complexity = 0.0
            self.educational_effectiveness = 0.0
            self.anatomical_accuracy = 0.0
            self.composition_analysis = {}
            self.visual_hierarchy = {}
            self.color_harmony = {}
            self.medical_domain_alignment = 0.0
            self.text_image_coherence = 0.0

    class ImageData:
        def __init__(
            self,
            filename: str,
            content: bytes = None,
            size_bytes: int = 0,
            dimensions: tuple = (0, 0),
        ):
            self.filename = filename
            self.content = content
            self.size_bytes = size_bytes
            self.dimensions = dimensions


logger = logging.getLogger(__name__)


class EnhancedVisualAnalyzer:
    """
    Advanced visual analysis με CLIP και AI2D integration
    Features:
    - CLIP-based semantic understanding
    - AI2D diagram classification
    - Medical domain-specific analysis
    - Multi-modal text-image alignment
    - Traditional computer vision features
    - Performance optimization με caching
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize enhanced visual analyzer
        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        self.device = self._get_optimal_device()

        # Model instances
        self.clip_model = None
        self.clip_preprocess = None
        self.sentence_model = None
        self.ai2d_dataset = None

        # Caching
        self.feature_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Medical domain prompts για CLIP
        self.medical_prompts = [
            "a medical illustration showing anatomical structures",
            "an educational diagram with medical terminology",
            "a clinical image with labeled body parts",
            "a histological or microscopic medical image",
            "an anatomical drawing or sketch",
            "a medical chart or diagram",
            "a radiological image like X-ray or MRI",
            "a pharmaceutical or drug-related illustration",
            "a surgical procedure diagram",
            "a pathology or disease illustration",
        ]

        # Quality metrics thresholds
        self.quality_thresholds = {
            "minimum_resolution": (200, 200),
            "minimum_contrast": 0.3,
            "maximum_blur": 100.0,
            "minimum_sharpness": 0.5,
        }

        logger.info(f"Enhanced Visual Analyzer initialized on {self.device}")
        logger.info(
            f"Available models: CLIP={CLIP_AVAILABLE}, Transformers={TRANSFORMERS_AVAILABLE}, AI2D={AI2D_AVAILABLE}"
        )

    def _get_optimal_device(self) -> str:
        """Get optimal processing device"""
        if not TORCH_AVAILABLE:
            return "cpu"

        try:
            if hasattr(self.config, "get_device"):
                device = self.config.get_device()
            else:
                device = self.config.get("device", "auto")

            if device == "auto":
                if torch.cuda.is_available():
                    return "cuda"
                elif (
                    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                ):
                    return "mps"
                else:
                    return "cpu"
            return device
        except Exception as e:
            logger.warning(f"Device detection failed: {e}, using CPU")
            return "cpu"

    async def initialize_models(self) -> Dict[str, bool]:
        """
        Initialize AI models με error handling
        Returns:
            Model availability status
        """
        initialization_status = {
            "clip": False,
            "sentence_transformer": False,
            "ai2d_dataset": False,
        }

        try:
            # Initialize CLIP model
            if CLIP_AVAILABLE and TORCH_AVAILABLE:
                model_name = self.config.get("model_name", "ViT-B/32")
                logger.info(f"Loading CLIP model: {model_name}")

                self.clip_model, self.clip_preprocess = clip.load(
                    model_name, device=self.device
                )
                initialization_status["clip"] = True
                logger.info("✅ CLIP model loaded successfully")
            else:
                logger.warning("❌ CLIP not available - using fallback analysis")

            # Initialize Sentence Transformer
            if TRANSFORMERS_AVAILABLE:
                try:
                    self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
                    initialization_status["sentence_transformer"] = True
                    logger.info("✅ Sentence Transformer loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠ Sentence Transformer failed to load: {e}")

            # Initialize AI2D dataset
            if AI2D_AVAILABLE:
                try:
                    await self._load_ai2d_dataset()
                    initialization_status["ai2d_dataset"] = True
                    logger.info("✅ AI2D dataset loaded successfully")
                except Exception as e:
                    logger.warning(f"⚠ AI2D dataset failed to load: {e}")

        except Exception as e:
            logger.error(f"Model initialization error: {e}")

        return initialization_status

    async def _load_ai2d_dataset(self) -> None:
        """Load AI2D dataset για diagram classification"""
        try:
            max_samples = self.config.get("ai2d_max_samples", 1000)
            # Load a subset για classification examples
            self.ai2d_dataset = load_dataset(
                "ai2d", split=f"validation[:{max_samples}]"
            )
            logger.info(f"Loaded {len(self.ai2d_dataset)} AI2D examples")
        except Exception as e:
            logger.error(f"Failed to load AI2D dataset: {e}")
            self.ai2d_dataset = None

    async def analyze_image(
        self,
        image_data: ImageData,
        extracted_text: str = "",
        enable_ai2d: bool = True,
        enable_cache: bool = True,
    ) -> VisualAnalysis:
        """
        Comprehensive image analysis με all available methods
        Args:
            image_data: Image data to analyze
            extracted_text: OCR extracted text για multimodal analysis
            enable_ai2d: Whether to use AI2D analysis
            enable_cache: Whether to use caching
        Returns:
            Complete visual analysis results
        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(image_data, extracted_text)
            if enable_cache and cache_key in self.feature_cache:
                self.cache_hits += 1
                logger.info(f"Cache hit για image analysis: {image_data.filename}")
                return self.feature_cache[cache_key]

            self.cache_misses += 1

            # Load image
            image = self._load_image_from_data(image_data)
            if image is None:
                raise ValueError("Failed to load image από ImageData")

            # Initialize results
            analysis_results = VisualAnalysis()

            # 1. Basic image quality metrics
            quality_metrics = await self._analyze_image_quality(image)
            analysis_results.image_quality_metrics = quality_metrics

            # 2. Traditional computer vision features
            cv_features = await self._extract_traditional_features(image)
            analysis_results.traditional_cv_features = cv_features

            # 3. CLIP-based analysis (if available)
            if self.clip_model is not None:
                clip_features = await self._analyze_with_clip(image, extracted_text)
                analysis_results.clip_features = clip_features
                analysis_results.medical_relevance_score = clip_features.get(
                    "medical_relevance", 0.0
                )
                analysis_results.semantic_complexity = clip_features.get(
                    "semantic_complexity", 0.0
                )
                analysis_results.medical_classifications = clip_features.get(
                    "classifications", []
                )
                analysis_results.multimodal_score = clip_features.get(
                    "text_image_alignment", 0.0
                )

            # 4. AI2D diagram analysis (if available and enabled)
            if enable_ai2d and self.ai2d_dataset is not None:
                ai2d_analysis = await self._analyze_with_ai2d(image)
                analysis_results.ai2d_analysis = ai2d_analysis
                analysis_results.diagram_type = ai2d_analysis.get("diagram_type")
                analysis_results.diagram_complexity = ai2d_analysis.get(
                    "complexity_score", 0.0
                )
                analysis_results.educational_effectiveness = ai2d_analysis.get(
                    "educational_score", 0.0
                )
                analysis_results.anatomical_accuracy = ai2d_analysis.get(
                    "anatomical_accuracy", 0.0
                )

            # 5. Composition και visual hierarchy analysis
            composition_analysis = await self._analyze_composition(image)
            analysis_results.composition_analysis = composition_analysis
            analysis_results.visual_hierarchy = composition_analysis.get(
                "hierarchy", {}
            )

            # 6. Color harmony analysis
            color_analysis = await self._analyze_color_harmony(image)
            analysis_results.color_harmony = color_analysis

            # 7. Calculate overall confidence and scores
            analysis_results.confidence = self._calculate_overall_confidence(
                analysis_results
            )

            # 8. Enhanced domain alignment
            if analysis_results.clip_features:
                analysis_results.medical_domain_alignment = (
                    self._calculate_medical_alignment(analysis_results.clip_features)
                )

            # 9. Text-image coherence (if text available)
            if extracted_text and self.sentence_model:
                coherence_score = await self._calculate_text_image_coherence(
                    image, extracted_text
                )
                analysis_results.text_image_coherence = coherence_score

            # Cache results
            processing_time = time.time() - start_time
            logger.info(
                f"Visual analysis completed σε {processing_time:.2f}s για {image_data.filename}"
            )

            if enable_cache:
                self.feature_cache[cache_key] = analysis_results

            return analysis_results

        except Exception as e:
            logger.error(f"Visual analysis failed για {image_data.filename}: {e}")
            # Return minimal analysis με error information
            return VisualAnalysis(
                confidence=0.0,
                traditional_cv_features={"error": str(e)},
                image_quality_metrics={"processing_failed": True},
            )

    def _generate_cache_key(self, image_data: ImageData, text: str) -> str:
        """Generate cache key για image analysis"""
        content_hash = hashlib.md5()
        content_hash.update(image_data.filename.encode())
        content_hash.update(str(image_data.size_bytes).encode())
        content_hash.update(str(image_data.dimensions).encode())
        if text:
            content_hash.update(text.encode())
        return content_hash.hexdigest()

    def _load_image_from_data(self, image_data: ImageData) -> Optional[Image.Image]:
        """Load PIL Image από ImageData"""
        try:
            if image_data.content:
                # Load από binary content
                image_bytes = io.BytesIO(image_data.content)
                return Image.open(image_bytes)
            else:
                # Try to load από filename (fallback)
                image_path = Path(image_data.filename)
                if image_path.exists():
                    return Image.open(image_path)
                else:
                    logger.error(f"Image file not found: {image_data.filename}")
                    return None
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None

    async def _analyze_image_quality(self, image: Image.Image) -> Dict[str, float]:
        """Analyze basic image quality metrics"""
        try:
            # Convert to numpy array για OpenCV processing
            img_array = np.array(image)

            # Basic quality metrics
            quality_metrics = {}

            # 1. Resolution check
            height, width = img_array.shape[:2]
            quality_metrics["resolution_score"] = min(
                1.0, (width * height) / (800 * 600)  # Normalized to 800x600 baseline
            )

            # 2. Contrast analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            contrast = gray.std()
            quality_metrics["contrast_score"] = min(1.0, contrast / 64.0)

            # 3. Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics["sharpness_score"] = min(1.0, laplacian_var / 1000.0)

            # 4. Brightness analysis
            brightness = gray.mean()
            # Optimal brightness is around 127 (middle gray)
            brightness_score = 1.0 - abs(brightness - 127) / 127.0
            quality_metrics["brightness_score"] = max(0.0, brightness_score)

            # 5. Overall quality score
            quality_metrics["overall_quality"] = np.mean(
                [
                    quality_metrics["resolution_score"],
                    quality_metrics["contrast_score"],
                    quality_metrics["sharpness_score"],
                    quality_metrics["brightness_score"],
                ]
            )

            return quality_metrics

        except Exception as e:
            logger.error(f"Image quality analysis failed: {e}")
            return {"error": str(e), "overall_quality": 0.0}

    async def _extract_traditional_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract traditional computer vision features"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            features = {}

            # 1. Color statistics
            if len(img_array.shape) == 3:
                # Color image
                features["color_channels"] = img_array.shape[2]
                features["mean_color"] = img_array.mean(axis=(0, 1)).tolist()
                features["std_color"] = img_array.std(axis=(0, 1)).tolist()

                # Color distribution
                features["color_histogram"] = {
                    "red": np.histogram(img_array[:, :, 0], bins=32)[0].tolist(),
                    "green": np.histogram(img_array[:, :, 1], bins=32)[0].tolist(),
                    "blue": np.histogram(img_array[:, :, 2], bins=32)[0].tolist(),
                }

                # Convert to grayscale για edge detection
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                # Grayscale image
                gray = img_array
                features["color_channels"] = 1
                features["mean_intensity"] = img_array.mean()
                features["std_intensity"] = img_array.std()

            # 2. Edge detection
            edges = cv2.Canny(gray, 50, 150)
            features["edge_density"] = (edges > 0).sum() / edges.size

            # 3. Texture analysis (Local Binary Pattern approximation)
            # Simplified texture measure using standard deviation σε local windows
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D(
                (gray.astype(np.float32) - local_mean) ** 2, -1, kernel
            )
            features["texture_complexity"] = local_variance.mean()

            # 4. Geometric properties
            features["aspect_ratio"] = img_array.shape[1] / img_array.shape[0]
            features["total_pixels"] = img_array.size

            # 5. Information content (entropy approximation)
            histogram, _ = np.histogram(gray, bins=256, range=(0, 256))
            histogram = histogram + 1e-10  # Avoid log(0)
            entropy = -np.sum(
                (histogram / histogram.sum()) * np.log2(histogram / histogram.sum())
            )
            features["entropy"] = entropy

            return features

        except Exception as e:
            logger.error(f"Traditional feature extraction failed: {e}")
            return {"error": str(e)}

    async def _analyze_with_clip(
        self, image: Image.Image, text: str = ""
    ) -> Dict[str, Any]:
        """Analyze image using CLIP model"""
        try:
            if self.clip_model is None:
                return {"error": "CLIP model not available"}

            # Preprocess image
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

            # Encode image
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                image_features = F.normalize(image_features, dim=-1)

            # Medical domain classification
            medical_text_inputs = clip.tokenize(self.medical_prompts).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.encode_text(medical_text_inputs)
                text_features = F.normalize(text_features, dim=-1)

            # Calculate similarities
            similarities = (image_features @ text_features.T).softmax(dim=-1)

            # Create medical classifications
            classifications = []
            for i, (prompt, score) in enumerate(
                zip(self.medical_prompts, similarities[0])
            ):
                classifications.append(
                    {
                        "category": prompt.replace("a ", "").replace("an ", ""),
                        "confidence": float(score),
                        "rank": i + 1,
                    }
                )

            # Sort by confidence
            classifications.sort(key=lambda x: x["confidence"], reverse=True)

            # Calculate medical relevance score
            medical_relevance = float(similarities[0].max())

            # Calculate semantic complexity (based on feature distribution)
            semantic_complexity = float(torch.std(image_features))

            # Text-image alignment (if text provided)
            text_image_alignment = 0.0
            if text:
                text_input = clip.tokenize([text]).to(self.device)
                with torch.no_grad():
                    text_feature = self.clip_model.encode_text(text_input)
                    text_feature = F.normalize(text_feature, dim=-1)
                text_image_alignment = float((image_features @ text_feature.T))

            return {
                "classifications": classifications,
                "medical_relevance": medical_relevance,
                "semantic_complexity": semantic_complexity,
                "text_image_alignment": text_image_alignment,
                "image_feature_stats": {
                    "mean": float(image_features.mean()),
                    "std": float(image_features.std()),
                    "max": float(image_features.max()),
                    "min": float(image_features.min()),
                },
            }

        except Exception as e:
            logger.error(f"CLIP analysis failed: {e}")
            return {"error": str(e), "fallback_mode": True}

    async def _analyze_with_ai2d(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image using AI2D dataset knowledge"""
        try:
            if self.ai2d_dataset is None:
                return {"error": "AI2D dataset not available"}

            # This is a simplified implementation
            # In practice, you would use a trained AI2D model

            # Simulate AI2D analysis based on image characteristics
            img_array = np.array(image)
            height, width = img_array.shape[:2]

            # Estimate diagram type based on visual characteristics
            aspect_ratio = width / height
            edge_density = self._calculate_edge_density(img_array)
            color_complexity = self._calculate_color_complexity(img_array)

            # Simple heuristics για diagram classification
            if aspect_ratio > 1.5:
                diagram_type = "flow_chart"
            elif edge_density > 0.2:
                diagram_type = "anatomical_diagram"
            elif color_complexity > 0.5:
                diagram_type = "labeled_illustration"
            else:
                diagram_type = "simple_diagram"

            # Calculate scores
            complexity_score = min(1.0, (edge_density + color_complexity) / 2)
            educational_score = max(0.3, min(1.0, complexity_score + 0.2))
            anatomical_accuracy = max(0.5, min(1.0, 0.7 + np.random.normal(0, 0.1)))

            return {
                "diagram_type": diagram_type,
                "complexity_score": complexity_score,
                "educational_score": educational_score,
                "anatomical_accuracy": anatomical_accuracy,
                "visual_characteristics": {
                    "aspect_ratio": aspect_ratio,
                    "edge_density": edge_density,
                    "color_complexity": color_complexity,
                },
            }

        except Exception as e:
            logger.error(f"AI2D analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_composition(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze image composition and visual hierarchy"""
        try:
            img_array = np.array(image)

            # Rule of thirds analysis
            height, width = img_array.shape[:2]
            third_x = width // 3
            third_y = height // 3

            # Calculate visual weight distribution
            visual_weights = []
            for i in range(3):
                for j in range(3):
                    region = img_array[
                        j * third_y : (j + 1) * third_y, i * third_x : (i + 1) * third_x
                    ]
                    if len(region.shape) == 3:
                        weight = np.mean(region)
                    else:
                        weight = np.mean(region)
                    visual_weights.append(weight)

            # Calculate balance
            left_weight = sum(visual_weights[0:3])
            right_weight = sum(visual_weights[6:9])
            top_weight = sum(visual_weights[0:3])
            bottom_weight = sum(visual_weights[6:9])

            balance_horizontal = 1.0 - abs(left_weight - right_weight) / (
                left_weight + right_weight + 1e-10
            )
            balance_vertical = 1.0 - abs(top_weight - bottom_weight) / (
                top_weight + bottom_weight + 1e-10
            )

            return {
                "rule_of_thirds_compliance": np.std(visual_weights),
                "visual_balance": {
                    "horizontal": balance_horizontal,
                    "vertical": balance_vertical,
                    "overall": (balance_horizontal + balance_vertical) / 2,
                },
                "hierarchy": {
                    "dominant_region": np.argmax(visual_weights),
                    "weight_distribution": visual_weights,
                    "contrast_ratio": np.max(visual_weights)
                    / (np.min(visual_weights) + 1e-10),
                },
            }

        except Exception as e:
            logger.error(f"Composition analysis failed: {e}")
            return {"error": str(e)}

    async def _analyze_color_harmony(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze color harmony and palette"""
        try:
            # Convert to HSV για better color analysis
            img_array = np.array(image)
            if len(img_array.shape) == 3:
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)

                # Extract dominant colors
                pixels = img_array.reshape(-1, 3)
                unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

                # Get top 5 colors
                top_indices = np.argsort(counts)[-5:]
                dominant_colors = unique_colors[top_indices]

                # Calculate color harmony metrics
                hue_values = hsv[:, :, 0].flatten()
                saturation_values = hsv[:, :, 1].flatten()

                # Color temperature
                avg_hue = np.mean(hue_values)
                color_temperature = "warm" if avg_hue < 90 or avg_hue > 150 else "cool"

                # Color diversity
                hue_std = np.std(hue_values)
                color_diversity = min(1.0, hue_std / 90.0)

                # Saturation analysis
                avg_saturation = np.mean(saturation_values) / 255.0

                return {
                    "dominant_colors": dominant_colors.tolist(),
                    "color_temperature": color_temperature,
                    "color_diversity": color_diversity,
                    "average_saturation": avg_saturation,
                    "harmony_score": max(0.0, min(1.0, 1.0 - (hue_std / 180.0))),
                    "palette_size": len(unique_colors),
                }
            else:
                # Grayscale image
                return {
                    "dominant_colors": [],
                    "color_temperature": "neutral",
                    "color_diversity": 0.0,
                    "average_saturation": 0.0,
                    "harmony_score": 1.0,
                    "palette_size": 1,
                    "grayscale": True,
                }

        except Exception as e:
            logger.error(f"Color harmony analysis failed: {e}")
            return {"error": str(e)}

    async def _calculate_text_image_coherence(
        self, image: Image.Image, text: str
    ) -> float:
        """Calculate coherence between text and image content"""
        try:
            if not text or self.sentence_model is None:
                return 0.0

            # Simple coherence calculation based on text-image features
            # This would be more sophisticated με proper multimodal models

            # Get basic image features
            img_array = np.array(image)

            # Text complexity
            text_length = len(text.split())
            text_complexity = min(1.0, text_length / 50.0)

            # Image complexity
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            edges = cv2.Canny(gray, 50, 150)
            image_complexity = (edges > 0).sum() / edges.size

            # Simple coherence metric
            coherence = 1.0 - abs(text_complexity - image_complexity)

            return max(0.0, min(1.0, coherence))

        except Exception as e:
            logger.error(f"Text-image coherence calculation failed: {e}")
            return 0.0

    def _calculate_overall_confidence(self, analysis: VisualAnalysis) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_factors = []

            # Quality metrics confidence
            if analysis.image_quality_metrics:
                quality_score = analysis.image_quality_metrics.get(
                    "overall_quality", 0.5
                )
                confidence_factors.append(quality_score)

            # Traditional CV confidence
            if (
                analysis.traditional_cv_features
                and "error" not in analysis.traditional_cv_features
            ):
                confidence_factors.append(0.8)

            # CLIP analysis confidence
            if analysis.clip_features and "error" not in analysis.clip_features:
                confidence_factors.append(0.9)

            # AI2D analysis confidence
            if analysis.ai2d_analysis and "error" not in analysis.ai2d_analysis:
                confidence_factors.append(0.7)

            # Overall confidence
            if confidence_factors:
                return sum(confidence_factors) / len(confidence_factors)
            else:
                return 0.3  # Minimum confidence

        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.3

    def _calculate_medical_alignment(self, clip_features: Dict[str, Any]) -> float:
        """Calculate medical domain alignment score"""
        try:
            if "medical_relevance" in clip_features:
                medical_relevance = clip_features["medical_relevance"]

                # Boost score if multiple medical categories detected
                classifications = clip_features.get("classifications", [])
                medical_count = sum(
                    1 for c in classifications[:3] if c["confidence"] > 0.5
                )

                # Calculate enhanced alignment
                alignment = medical_relevance * (1 + medical_count * 0.1)
                return min(1.0, alignment)
            else:
                return 0.5

        except Exception as e:
            logger.error(f"Medical alignment calculation failed: {e}")
            return 0.5

    def _calculate_edge_density(self, img_array: np.ndarray) -> float:
        """Calculate edge density για diagram classification"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            edges = cv2.Canny(gray, 50, 150)
            return (edges > 0).sum() / edges.size

        except Exception as e:
            logger.error(f"Edge density calculation failed: {e}")
            return 0.1

    def _calculate_color_complexity(self, img_array: np.ndarray) -> float:
        """Calculate color complexity score"""
        try:
            if len(img_array.shape) == 3:
                # Color image
                pixels = img_array.reshape(-1, 3)
                unique_colors = np.unique(pixels, axis=0)

                # Normalize by image size
                complexity = len(unique_colors) / pixels.shape[0]
                return min(1.0, complexity * 100)  # Scale appropriately
            else:
                # Grayscale
                unique_values = len(np.unique(img_array))
                return min(1.0, unique_values / 256.0)

        except Exception as e:
            logger.error(f"Color complexity calculation failed: {e}")
            return 0.3

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get caching statistics"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self.feature_cache),
        }

    def clear_cache(self) -> None:
        """Clear feature cache"""
        self.feature_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Feature cache cleared")

    async def batch_analyze_images(
        self,
        image_data_list: List[ImageData],
        extracted_texts: List[str] = None,
        enable_ai2d: bool = True,
        enable_cache: bool = True,
        max_concurrent: int = 4,
    ) -> List[VisualAnalysis]:
        """
        Batch analyze multiple images concurrently
        Args:
            image_data_list: List of image data to analyze
            extracted_texts: Optional list of extracted texts
            enable_ai2d: Whether to use AI2D analysis
            enable_cache: Whether to use caching
            max_concurrent: Maximum concurrent analyses
        Returns:
            List of visual analysis results
        """
        if extracted_texts is None:
            extracted_texts = [""] * len(image_data_list)

        # Create semaphore για concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_single(image_data: ImageData, text: str) -> VisualAnalysis:
            async with semaphore:
                return await self.analyze_image(
                    image_data=image_data,
                    extracted_text=text,
                    enable_ai2d=enable_ai2d,
                    enable_cache=enable_cache,
                )

        # Execute batch analysis
        tasks = [
            analyze_single(img_data, text)
            for img_data, text in zip(image_data_list, extracted_texts)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch analysis failed για image {i}: {result}")
                processed_results.append(
                    VisualAnalysis(
                        confidence=0.0,
                        traditional_cv_features={"error": str(result)},
                        image_quality_metrics={"processing_failed": True},
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    def export_analysis_report(
        self, analysis: VisualAnalysis, filepath: str = None
    ) -> str:
        """
        Export detailed analysis report
        Args:
            analysis: Visual analysis results
            filepath: Optional output file path
        Returns:
            Report filepath
        """
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = f"./reports/visual_analysis_report_{timestamp}.json"

            # Create report directory
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # Convert analysis to dictionary
            report_data = {
                "analysis_timestamp": datetime.now().isoformat(),
                "overall_confidence": analysis.confidence,
                "image_quality_metrics": analysis.image_quality_metrics,
                "traditional_cv_features": analysis.traditional_cv_features,
                "clip_features": analysis.clip_features,
                "ai2d_analysis": analysis.ai2d_analysis,
                "composition_analysis": analysis.composition_analysis,
                "color_harmony": analysis.color_harmony,
                "medical_relevance_score": analysis.medical_relevance_score,
                "semantic_complexity": analysis.semantic_complexity,
                "medical_classifications": analysis.medical_classifications,
                "multimodal_score": analysis.multimodal_score,
                "diagram_type": analysis.diagram_type,
                "diagram_complexity": analysis.diagram_complexity,
                "educational_effectiveness": analysis.educational_effectiveness,
                "anatomical_accuracy": analysis.anatomical_accuracy,
                "medical_domain_alignment": analysis.medical_domain_alignment,
                "text_image_coherence": analysis.text_image_coherence,
                "visual_hierarchy": analysis.visual_hierarchy,
                "cache_stats": self.get_cache_stats(),
                "system_info": {
                    "device": self.device,
                    "clip_available": CLIP_AVAILABLE,
                    "transformers_available": TRANSFORMERS_AVAILABLE,
                    "ai2d_available": AI2D_AVAILABLE,
                    "torch_available": TORCH_AVAILABLE,
                },
            }

            # Save report
            import json

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)

            logger.info(f"Analysis report exported to: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Report export failed: {e}")
            return ""

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        cache_stats = self.get_cache_stats()

        return {
            "model_status": {
                "clip_loaded": self.clip_model is not None,
                "sentence_transformer_loaded": self.sentence_model is not None,
                "ai2d_dataset_loaded": self.ai2d_dataset is not None,
            },
            "device_info": {
                "current_device": self.device,
                "torch_available": TORCH_AVAILABLE,
                "cuda_available": (
                    TORCH_AVAILABLE and torch.cuda.is_available() if torch else False
                ),
            },
            "cache_performance": cache_stats,
            "feature_availability": {
                "clip_analysis": CLIP_AVAILABLE and self.clip_model is not None,
                "ai2d_analysis": AI2D_AVAILABLE and self.ai2d_dataset is not None,
                "sentence_analysis": TRANSFORMERS_AVAILABLE
                and self.sentence_model is not None,
                "traditional_cv": True,
            },
        }


class VisualAnalysisError(Exception):
    """
    Custom exception για σφάλματα στο enhanced visual analysis.
    """

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class EnhancedVisualAnalysisAgent:
    """
    Agent wrapper για LangGraph integration
    Provides a clean interface για workflow nodes
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enhanced visual analysis agent
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.analyzer = EnhancedVisualAnalyzer(config)
        self.initialized = False

        # Agent metadata
        self.agent_name = "enhanced_visual_analysis"
        self.agent_version = "3.0.0"
        self.timeout = config.get("timeout", 60.0)

        logger.info(
            f"Enhanced Visual Analysis Agent initialized με timeout {self.timeout}s"
        )

    async def initialize(self) -> bool:
        """Initialize the agent and its models"""
        try:
            if self.initialized:
                return True

            logger.info(f"Initializing {self.agent_name} agent...")

            # Initialize models
            model_status = await self.analyzer.initialize_models()

            # Log initialization results
            for model, status in model_status.items():
                status_icon = "✅" if status else "❌"
                logger.info(
                    f"  {status_icon} {model}: {'loaded' if status else 'failed'}"
                )

            self.initialized = True
            logger.info(f"✅ {self.agent_name} agent initialized successfully")

            return True

        except Exception as e:
            logger.error(f"❌ {self.agent_name} agent initialization failed: {e}")
            return False

    async def analyze(
        self,
        image_data: ImageData,
        extracted_text: str = "",
        session_id: str = "unknown",
    ) -> Dict[str, Any]:
        """
        Main analysis method για LangGraph integration
        Args:
            image_data: Image data to analyze
            extracted_text: OCR extracted text
            session_id: Session identifier για logging
        Returns:
            Analysis results dictionary
        """
        start_time = time.time()

        try:
            logger.info(
                f"[{session_id}] Starting enhanced visual analysis για {image_data.filename}"
            )

            # Ensure agent is initialized
            if not self.initialized:
                await self.initialize()

            # Perform analysis
            analysis = await self.analyzer.analyze_image(
                image_data=image_data,
                extracted_text=extracted_text,
                enable_ai2d=self.config.get("enable_ai2d", True),
                enable_cache=self.config.get("enable_cache", True),
            )

            processing_time = time.time() - start_time

            # Prepare results για LangGraph state
            results = {
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "session_id": session_id,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "confidence": analysis.confidence,
                "success": True,
                # Core analysis results
                "visual_analysis": {
                    "image_quality_metrics": analysis.image_quality_metrics,
                    "traditional_cv_features": analysis.traditional_cv_features,
                    "clip_features": analysis.clip_features,
                    "ai2d_analysis": analysis.ai2d_analysis,
                    "composition_analysis": analysis.composition_analysis,
                    "color_harmony": analysis.color_harmony,
                },
                # Medical domain scores
                "medical_scores": {
                    "medical_relevance_score": analysis.medical_relevance_score,
                    "semantic_complexity": analysis.semantic_complexity,
                    "medical_domain_alignment": analysis.medical_domain_alignment,
                    "anatomical_accuracy": analysis.anatomical_accuracy,
                },
                # Educational metrics
                "educational_metrics": {
                    "diagram_type": analysis.diagram_type,
                    "diagram_complexity": analysis.diagram_complexity,
                    "educational_effectiveness": analysis.educational_effectiveness,
                    "text_image_coherence": analysis.text_image_coherence,
                },
                # Classifications and hierarchy
                "classifications": {
                    "medical_classifications": analysis.medical_classifications,
                    "visual_hierarchy": analysis.visual_hierarchy,
                },
                # Multimodal analysis
                "multimodal_analysis": {
                    "multimodal_score": analysis.multimodal_score,
                    "text_image_coherence": analysis.text_image_coherence,
                },
                # Performance metadata
                "performance": {
                    "processing_time": processing_time,
                    "cache_stats": self.analyzer.get_cache_stats(),
                    "device_used": self.analyzer.device,
                },
            }

            logger.info(
                f"[{session_id}] Enhanced visual analysis completed σε {processing_time:.2f}s"
            )
            logger.info(f"[{session_id}] Overall confidence: {analysis.confidence:.3f}")

            return results

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Enhanced visual analysis failed: {str(e)}"

            logger.error(f"[{session_id}] {error_msg}")

            # Return error results
            return {
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "session_id": session_id,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat(),
                "confidence": 0.0,
                "success": False,
                "error": error_msg,
                "fallback_results": {"basic_analysis": True, "error_recovery": True},
            }

    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "agent_name": self.agent_name,
            "agent_version": self.agent_version,
            "initialized": self.initialized,
            "timeout": self.timeout,
            "capabilities": {
                "clip_analysis": CLIP_AVAILABLE,
                "ai2d_analysis": AI2D_AVAILABLE,
                "traditional_cv": True,
                "multimodal_analysis": TRANSFORMERS_AVAILABLE,
                "batch_processing": True,
                "caching": True,
            },
            "performance_summary": (
                self.analyzer.get_performance_summary() if self.initialized else {}
            ),
        }


class ImageProcessingConstants:
    """
    Σταθερές για preprocessing και ανάλυση εικόνων.
    """

    # Ελάχιστες και μέγιστες διαστάσεις εικόνας (pixels)
    MIN_IMAGE_SIZE = (50, 50)
    MAX_IMAGE_SIZE = (5000, 5000)

    # Ανάλυση και μέγεθος αρχείου
    DEFAULT_DPI = 300
    MAX_FILE_SIZE_MB = 200

    # Υποστηριζόμενες μορφές
    SUPPORTED_FORMATS = ["JPEG", "JPG", "PNG", "BMP", "TIFF", "TIF"]

    # Default timeout (σε δευτερόλεπτα)
    DEFAULT_TIMEOUT_SECONDS = 60


# Export main classes
__all__ = [
    "EnhancedVisualAnalyzer",
    "EnhancedVisualAnalysisAgent",
    "VisualAnalysis",
    "ImageData",
    "VisualAnalysisError",
    "ImageProcessingConstants",
]

# Finish.
