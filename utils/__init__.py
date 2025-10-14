"""
MedIllustrator-AI v3.0 - Utilities Package
"""

from .ontology_loader import (
    ExpertOntologyLoader,
    MedicalTerm,
    get_ontology_loader,
    quick_lookup,
)

__version__ = "3.0.0"
__all__ = [
    "ExpertOntologyLoader",
    "MedicalTerm", 
    "get_ontology_loader",
    "quick_lookup",
]
