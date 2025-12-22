"""
Visualizer component for displaying fitting results.
"""

import numpy as np


class Visualizer:
    """Handles 3D visualization of fitting results."""

    @staticmethod
    def show(
        fitted_mesh: np.ndarray,
        bone_points: np.ndarray,
        fitted_faces: np.ndarray = None,
    ) -> None:
        """Display interactive 3D visualization.
        
        Args:
            fitted_mesh: Fitted mesh vertices of shape (N, 3).
            bone_points: Bone point cloud of shape (M, 3).
            fitted_faces: Optional face array for mesh rendering.
        """
        raise NotImplementedError("To be implemented in task 9")

    @staticmethod
    def save_screenshot(
        fitted_mesh: np.ndarray,
        bone_points: np.ndarray,
        output_path: str,
    ) -> None:
        """Save visualization to image file.
        
        Args:
            fitted_mesh: Fitted mesh vertices of shape (N, 3).
            bone_points: Bone point cloud of shape (M, 3).
            output_path: Path to save the screenshot.
        """
        raise NotImplementedError("To be implemented in task 9")
