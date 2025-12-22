"""
FootExtractor component for extracting foot vertices from SMPL mesh.
"""

import numpy as np


class FootExtractor:
    """Handles extraction of foot vertices from SMPL mesh data."""

    # SMPL-X foot vertex indices (to be populated with actual indices)
    LEFT_FOOT_INDICES: list[int] = []
    RIGHT_FOOT_INDICES: list[int] = []

    @staticmethod
    def load_smpl_vertices(csv_path: str) -> np.ndarray:
        """Load SMPL vertices from CSV (N x 3 array).
        
        Args:
            csv_path: Path to CSV file with x, y, z columns.
            
        Returns:
            Numpy array of shape (N, 3) with vertex coordinates.
            
        Raises:
            ValueError: If CSV format is invalid.
        """
        raise NotImplementedError("To be implemented in task 3")

    @staticmethod
    def extract_foot(
        vertices: np.ndarray, side: str
    ) -> tuple[np.ndarray, list[int]]:
        """Extract foot vertices, return (foot_vertices, original_indices).
        
        Args:
            vertices: Full SMPL vertex array of shape (N, 3).
            side: Either 'left' or 'right'.
            
        Returns:
            Tuple of (foot_vertices, original_indices).
            
        Raises:
            ValueError: If side is not 'left' or 'right'.
        """
        raise NotImplementedError("To be implemented in task 3")
