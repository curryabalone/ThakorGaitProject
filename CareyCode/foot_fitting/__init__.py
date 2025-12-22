"""
SMPL Foot Mesh Fitting Module

This module provides functionality for fitting SMPL foot mesh vertices
to STL skeletal geometry from MuJoCo biomechanical models.
"""

from .mesh_loader import MeshLoader
from .foot_extractor import FootExtractor
from .alignment import AlignmentModule, AlignmentResult
from .fitting import FittingModule, FittingResult
from .exporter import Exporter
from .visualizer import Visualizer

__all__ = [
    "MeshLoader",
    "FootExtractor",
    "AlignmentModule",
    "AlignmentResult",
    "FittingModule",
    "FittingResult",
    "Exporter",
    "Visualizer",
]
