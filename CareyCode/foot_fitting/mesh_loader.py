"""
MeshLoader component for loading STL geometry files.
"""

import os
import numpy as np
import trimesh


class MeshLoader:
    """Handles loading and combining STL mesh files."""

    @staticmethod
    def validate_path(path: str) -> None:
        """Raise FileNotFoundError with descriptive message if invalid.
        
        Args:
            path: Path to validate.
            
        Raises:
            FileNotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"STL file not found: {path}")

    @staticmethod
    def load_stl(path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load STL file, return (vertices, faces).
        
        Args:
            path: Path to the STL file.
            
        Returns:
            Tuple of (vertices, faces) as numpy arrays.
            vertices: float64 array of shape (N, 3)
            faces: int32 array of shape (M, 3)
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid.
        """
        MeshLoader.validate_path(path)
        
        try:
            mesh = trimesh.load(path, file_type='stl')
        except Exception as e:
            raise ValueError(f"Invalid STL format in file: {path}") from e
        
        # Ensure correct dtypes as per design spec
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        
        return vertices, faces

    @staticmethod
    def load_multiple_stls(paths: list[str]) -> np.ndarray:
        """Load multiple STLs and combine into single point cloud.
        
        Args:
            paths: List of paths to STL files.
            
        Returns:
            Combined vertices as numpy array of shape (N, 3) with float64 dtype.
        """
        all_vertices = []
        
        for path in paths:
            vertices, _ = MeshLoader.load_stl(path)
            all_vertices.append(vertices)
        
        if not all_vertices:
            return np.array([], dtype=np.float64).reshape(0, 3)
        
        return np.vstack(all_vertices)
