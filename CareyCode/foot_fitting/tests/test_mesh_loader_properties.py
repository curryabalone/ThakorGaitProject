"""
Property-based tests for MeshLoader component.

Uses Hypothesis for property-based testing as specified in the design document.
"""

import tempfile
import os
import numpy as np
import trimesh
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from foot_fitting.mesh_loader import MeshLoader


@st.composite
def valid_mesh_strategy(draw):
    """Generate valid mesh with vertices and faces.
    
    Creates a simple triangulated mesh that can be saved as STL.
    STL format uses single precision (float32) internally, so we generate
    float32 values to ensure round-trip compatibility.
    """
    n = draw(st.integers(min_value=3, max_value=50))
    
    vertices = draw(arrays(
        dtype=np.float32,
        shape=(n, 3),
        elements=st.floats(
            min_value=-10.0, 
            max_value=10.0, 
            allow_nan=False, 
            allow_infinity=False,
            width=32
        )
    ))
    vertices = vertices.astype(np.float64)
    
    num_faces = draw(st.integers(min_value=1, max_value=min(20, n)))
    faces = []
    for _ in range(num_faces):
        indices = draw(st.lists(
            st.integers(min_value=0, max_value=n-1),
            min_size=3,
            max_size=3,
            unique=True
        ))
        faces.append(indices)
    
    faces = np.array(faces, dtype=np.int32)
    return vertices, faces



# **Feature: smpl-foot-fitting, Property 1: STL Loading Preserves Coordinates**
# **Validates: Requirements 1.1, 1.4**
@given(mesh_data=valid_mesh_strategy())
@settings(max_examples=100, suppress_health_check=[HealthCheck.filter_too_much])
def test_stl_loading_preserves_coordinates(mesh_data):
    """
    Property 1: STL Loading Preserves Coordinates
    
    For any valid STL file, loading the file and accessing vertex coordinates
    SHALL return values identical to those stored in the original file
    (within floating-point precision).
    
    **Feature: smpl-foot-fitting, Property 1: STL Loading Preserves Coordinates**
    **Validates: Requirements 1.1, 1.4**
    """
    vertices, faces = mesh_data
    
    unique_vertices = np.unique(vertices, axis=0)
    if len(unique_vertices) < 3:
        assume(False)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    if len(mesh.faces) == 0:
        assume(False)
    
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        temp_path = f.name
    
    try:
        mesh.export(temp_path, file_type='stl')
        loaded_vertices, loaded_faces = MeshLoader.load_stl(temp_path)
        
        assert loaded_vertices.dtype == np.float64
        assert loaded_faces.dtype == np.int32
        
        original_face_vertices = vertices[faces]
        loaded_face_vertices = loaded_vertices[loaded_faces]
        
        def sort_faces_by_centroid(face_verts):
            centroids = face_verts.mean(axis=1)
            sort_idx = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))
            return face_verts[sort_idx]
        
        original_sorted = sort_faces_by_centroid(original_face_vertices)
        loaded_sorted = sort_faces_by_centroid(loaded_face_vertices)
        
        assert len(original_sorted) == len(loaded_sorted)
        
        np.testing.assert_allclose(
            original_sorted, 
            loaded_sorted, 
            rtol=1e-5, 
            atol=1e-6,
            err_msg="Vertex coordinates not preserved after STL round-trip"
        )
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    test_stl_loading_preserves_coordinates()
    print("Property test passed!")
