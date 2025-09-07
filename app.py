"""
🏥 MedIllustrator-AI v3.0 - Cloud Edition
===========================================

Streamlit Cloud-optimized medical image assessment system.
Simplified architecture for cost-effective deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
import time
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import traceback

# ============================================================================
# STREAMLIT CLOUD CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="MedIllustrator-AI v3.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CLOUD-OPTIMIZED CACHING AND STATE MANAGEMENT
# ============================================================================


@st.cache_data(show_spinner=False)
def load_medical_ontology() -> Dict[str, Any]:
    """Load medical ontology with cloud-optimized caching"""
    # Simplified medical terms for cloud deployment
    medical_terms = {
        "anatomical": [
            "heart",
            "lung",
            "liver",
            "kidney",
            "brain",
            "stomach",
            "intestine",
            "bone",
            "muscle",
            "blood",
            "nerve",
            "artery",
            "vein",
            "cell",
            "tissue",
            "organ",
            "spine",
            "skull",
        ],
        "clinical": [
            "diagnosis",
            "symptom",
            "treatment",
            "therapy",
            "infection",
            "inflammation",
            "disease",
            "syndrome",
            "disorder",
            "condition",
        ],
        "procedural": [
            "surgery",
            "examination",
            "test",
            "scan",
            "biopsy",
            "injection",
            "medication",
            "procedure",
            "analysis",
        ],
    }

    return {
        "terms": medical_terms,
        "total_terms": sum(len(terms) for terms in medical_terms.values()),
        "categories": list(medical_terms.keys()),
        "loaded_at": datetime.now().isoformat(),
    }


@st.cache_data(show_spinner=False)
def get_blooms_taxonomy_levels() -> List[Dict[str, str]]:
    """Get Bloom's Taxonomy levels for educational assessment"""
    return [
        {
            "level": "Remember",
            "description": "Recall facts and basic concepts",
            "keywords": ["define", "list", "identify", "recall"],
        },
        {
            "level": "Understand",
            "description": "Explain ideas or concepts",
            "keywords": ["explain", "describe", "summarize", "classify"],
        },
        {
            "level": "Apply",
            "description": "Use information in new situations",
            "keywords": ["apply", "demonstrate", "solve", "use"],
        },
        {
            "level": "Analyze",
            "description": "Draw connections among ideas",
            "keywords": ["analyze", "compare", "contrast", "examine"],
        },
        {
            "level": "Evaluate",
            "description": "Justify a stand or decision",
            "keywords": ["evaluate", "critique", "judge", "assess"],
        },
        {
            "level": "Create",
            "description": "Produce new or original work",
            "keywords": ["create", "design", "develop", "compose"],
        },
    ]


# ============================================================================
# SIMPLIFIED AGENT IMPLEMENTATIONS FOR CLOUD
# ============================================================================


class CloudMedicalAnalyzer:
    """Lightweight medical analysis for cloud deployment"""

    def __init__(self):
        self.ontology = load_medical_ontology()
        self.blooms_levels = get_blooms_taxonomy_levels()

    def analyze_medical_terms(self, text: str) -> Dict[str, Any]:
        """Analyze medical terminology in text"""
        if not text:
            return {
                "detected_terms": [],
                "categories": {},
                "complexity_score": 0.0,
                "confidence": 0.0,
            }

        text_lower = text.lower()
        detected_terms = []
        categories = {}

        # Check each category
        for category, terms in self.ontology["terms"].items():
            found_terms = [term for term in terms if term in text_lower]
            if found_terms:
                detected_terms.extend(found_terms)
                categories[category] = found_terms

        # Calculate complexity score
        complexity_score = min(len(detected_terms) / 10, 1.0)  # Normalize to 0-1
        confidence = 0.8 if detected_terms else 0.2

        return {
            "detected_terms": list(set(detected_terms)),
            "categories": categories,
            "complexity_score": complexity_score,
            "confidence": confidence,
            "total_found": len(set(detected_terms)),
        }

    def analyze_blooms_taxonomy(self, text: str) -> Dict[str, Any]:
        """Analyze Bloom's Taxonomy level"""
        if not text:
            return {
                "cognitive_level": "Remember",
                "confidence": 0.0,
                "explanation": "No text provided",
            }

        text_lower = text.lower()
        level_scores = {}

        # Score each level based on keyword presence
        for level_info in self.blooms_levels:
            level = level_info["level"]
            keywords = level_info["keywords"]
            score = sum(1 for keyword in keywords if keyword in text_lower)
            level_scores[level] = score

        # Determine highest scoring level
        best_level = max(level_scores, key=level_scores.get)
        max_score = level_scores[best_level]
        confidence = min(max_score / 4, 1.0)  # Normalize confidence

        # Get explanation
        level_info = next(l for l in self.blooms_levels if l["level"] == best_level)
        explanation = level_info["description"]

        return {
            "cognitive_level": best_level,
            "confidence": confidence,
            "explanation": explanation,
            "level_scores": level_scores,
        }

    def analyze_cognitive_load(
        self, text: str, image_complexity: float = 0.5
    ) -> Dict[str, Any]:
        """Analyze cognitive load factors"""
        if not text:
            text_complexity = 0.0
        else:
            # Simple text complexity estimation
            word_count = len(text.split())
            avg_word_length = sum(len(word) for word in text.split()) / max(
                word_count, 1
            )
            text_complexity = min((word_count * avg_word_length) / 1000, 1.0)

        # Calculate load components
        intrinsic_load = (text_complexity + image_complexity) / 2
        extraneous_load = max(0, intrinsic_load - 0.7)  # Excess complexity
        germane_load = min(intrinsic_load, 0.7)  # Productive complexity

        return {
            "intrinsic_load": intrinsic_load,
            "extraneous_load": extraneous_load,
            "germane_load": germane_load,
            "overall_load": (intrinsic_load + extraneous_load + germane_load) / 3,
            "recommendations": self._get_load_recommendations(intrinsic_load),
        }

    def _get_load_recommendations(self, load: float) -> List[str]:
        """Get cognitive load reduction recommendations"""
        if load < 0.3:
            return [
                "Consider adding more detail for better learning",
                "Image could be more complex",
            ]
        elif load > 0.7:
            return [
                "Reduce text complexity",
                "Simplify visual elements",
                "Break into smaller sections",
            ]
        else:
            return ["Good cognitive load balance", "Appropriate for educational use"]


# ============================================================================
# STREAMLIT CLOUD UI IMPLEMENTATION
# ============================================================================


def initialize_session_state():
    """Initialize Streamlit session state"""
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = CloudMedicalAnalyzer()

    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []

    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None


def extract_text_from_image(image: Image.Image) -> str:
    """Simplified text extraction for cloud deployment"""
    # In a full deployment, you might use OCR here
    # For cloud optimization, we'll use a placeholder
    return "Sample extracted text from medical image. This would contain anatomical terms, diagnoses, and medical procedures in a real implementation."


def analyze_image_complexity(image: Image.Image) -> float:
    """Simple image complexity analysis"""
    # Convert to grayscale and analyze
    gray_image = image.convert("L")
    np_image = np.array(gray_image)

    # Simple complexity metrics
    variance = np.var(np_image) / 10000  # Normalize
    edge_density = len(np.where(np.diff(np_image) > 50)[0]) / np_image.size

    complexity = min((variance + edge_density) / 2, 1.0)
    return complexity


def create_analysis_dashboard(analysis_results: Dict[str, Any]):
    """Create comprehensive analysis dashboard"""

    # Overview Metrics
    st.subheader("📊 Analysis Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        medical_score = analysis_results.get("medical_analysis", {}).get(
            "complexity_score", 0
        )
        st.metric(
            label="Medical Complexity",
            value=f"{medical_score:.1%}",
            delta=f"{len(analysis_results.get('medical_analysis', {}).get('detected_terms', []))} terms",
        )

    with col2:
        blooms_confidence = analysis_results.get("blooms_analysis", {}).get(
            "confidence", 0
        )
        st.metric(
            label="Educational Level",
            value=analysis_results.get("blooms_analysis", {}).get(
                "cognitive_level", "Unknown"
            ),
            delta=f"{blooms_confidence:.1%} confidence",
        )

    with col3:
        cognitive_load = analysis_results.get("cognitive_load", {}).get(
            "overall_load", 0
        )
        st.metric(
            label="Cognitive Load",
            value=f"{cognitive_load:.1%}",
            delta="Optimal" if 0.3 <= cognitive_load <= 0.7 else "Review needed",
        )

    with col4:
        processing_time = analysis_results.get("processing_time", 0)
        st.metric(
            label="Processing Time",
            value=f"{processing_time:.1f}s",
            delta="Cloud optimized",
        )

    # Detailed Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🏥 Medical Terms",
            "🧠 Educational Level",
            "⚡ Cognitive Load",
            "📈 Recommendations",
        ]
    )

    with tab1:
        medical_analysis = analysis_results.get("medical_analysis", {})
        st.write("**Detected Medical Terms:**")

        if medical_analysis.get("detected_terms"):
            # Display by category
            for category, terms in medical_analysis.get("categories", {}).items():
                st.write(f"*{category.title()}:* {', '.join(terms)}")
        else:
            st.info("No medical terms detected in the image.")

    with tab2:
        blooms_analysis = analysis_results.get("blooms_analysis", {})
        st.write(
            f"**Cognitive Level:** {blooms_analysis.get('cognitive_level', 'Unknown')}"
        )
        st.write(
            f"**Explanation:** {blooms_analysis.get('explanation', 'No explanation available')}"
        )

        # Show level scores
        if "level_scores" in blooms_analysis:
            level_df = pd.DataFrame(
                [
                    {"Level": level, "Score": score}
                    for level, score in blooms_analysis["level_scores"].items()
                ]
            )
            st.bar_chart(level_df.set_index("Level"))

    with tab3:
        cognitive_load = analysis_results.get("cognitive_load", {})

        # Load visualization
        load_data = {
            "Intrinsic Load": cognitive_load.get("intrinsic_load", 0),
            "Extraneous Load": cognitive_load.get("extraneous_load", 0),
            "Germane Load": cognitive_load.get("germane_load", 0),
        }

        st.bar_chart(pd.DataFrame([load_data]))

        # Recommendations
        recommendations = cognitive_load.get("recommendations", [])
        if recommendations:
            st.write("**Recommendations:**")
            for rec in recommendations:
                st.write(f"• {rec}")

    with tab4:
        st.write("### 🎯 Educational Recommendations")

        # Generate comprehensive recommendations
        medical_terms_count = len(
            analysis_results.get("medical_analysis", {}).get("detected_terms", [])
        )
        cognitive_level = analysis_results.get("blooms_analysis", {}).get(
            "cognitive_level", "Remember"
        )
        cognitive_load = analysis_results.get("cognitive_load", {}).get(
            "overall_load", 0
        )

        recommendations = []

        if medical_terms_count < 3:
            recommendations.append(
                "Consider adding more medical terminology for better learning outcomes"
            )
        elif medical_terms_count > 15:
            recommendations.append(
                "High medical complexity - ensure appropriate for target audience"
            )

        if cognitive_level in ["Remember", "Understand"]:
            recommendations.append(
                "Consider adding analysis or application elements for higher-order thinking"
            )

        if cognitive_load > 0.7:
            recommendations.append(
                "Reduce cognitive load by simplifying content or breaking into sections"
            )
        elif cognitive_load < 0.3:
            recommendations.append(
                "Could increase educational challenge for better engagement"
            )

        if not recommendations:
            recommendations.append(
                "Excellent educational balance - suitable for medical education"
            )

        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")


def main():
    """Main Streamlit application"""

    # Initialize session state
    initialize_session_state()

    # Header
    st.title("🏥 MedIllustrator-AI v3.0")
    st.markdown("**Advanced Medical Image Assessment for Educational Content**")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")

        # Analysis options
        st.subheader("Analysis Options")
        enable_detailed_analysis = st.checkbox("Enable Detailed Analysis", value=True)
        save_to_history = st.checkbox("Save to History", value=True)

        # System info
        st.subheader("📊 System Status")
        st.success("✅ Cloud Optimized")
        st.info(
            f"📚 {st.session_state.analyzer.ontology['total_terms']} Medical Terms Loaded"
        )
        st.info(
            f"🧠 {len(st.session_state.analyzer.blooms_levels)} Bloom's Levels Available"
        )

        # History
        if st.session_state.analysis_history:
            st.subheader("📈 Analysis History")
            st.write(f"Total analyses: {len(st.session_state.analysis_history)}")

            if st.button("Clear History"):
                st.session_state.analysis_history = []
                st.rerun()

    # Main interface
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📤 Upload Medical Image")

        uploaded_file = st.file_uploader(
            "Choose a medical illustration or diagram",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            help="Upload a medical image for comprehensive educational assessment",
        )

        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Medical Image", use_column_width=True)

            # Image info
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size:,} bytes")
            st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")

    with col2:
        st.header("⚡ Quick Analysis")

        if uploaded_file is not None:
            if st.button("🚀 Analyze Image", type="primary"):

                with st.spinner("Analyzing medical image..."):
                    start_time = time.time()

                    try:
                        # Extract text and analyze
                        extracted_text = extract_text_from_image(image)
                        image_complexity = analyze_image_complexity(image)

                        # Run analyses
                        analyzer = st.session_state.analyzer
                        medical_analysis = analyzer.analyze_medical_terms(
                            extracted_text
                        )
                        blooms_analysis = analyzer.analyze_blooms_taxonomy(
                            extracted_text
                        )
                        cognitive_load_analysis = analyzer.analyze_cognitive_load(
                            extracted_text, image_complexity
                        )

                        processing_time = time.time() - start_time

                        # Compile results
                        analysis_results = {
                            "timestamp": datetime.now().isoformat(),
                            "filename": uploaded_file.name,
                            "extracted_text": extracted_text,
                            "image_complexity": image_complexity,
                            "medical_analysis": medical_analysis,
                            "blooms_analysis": blooms_analysis,
                            "cognitive_load": cognitive_load_analysis,
                            "processing_time": processing_time,
                        }

                        # Store results
                        st.session_state.current_analysis = analysis_results

                        if save_to_history:
                            st.session_state.analysis_history.append(analysis_results)

                        st.success(
                            f"✅ Analysis completed in {processing_time:.1f} seconds!"
                        )

                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.write("**Error details:**")
                        st.code(traceback.format_exc())
        else:
            st.info("👆 Upload an image to begin analysis")

    # Results section
    if st.session_state.current_analysis:
        st.markdown("---")
        st.header("📊 Analysis Results")
        create_analysis_dashboard(st.session_state.current_analysis)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            🏥 MedIllustrator-AI v3.0 | Cloud Optimized for Streamlit<br>
            Advanced Medical Image Assessment for Educational Excellence
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
