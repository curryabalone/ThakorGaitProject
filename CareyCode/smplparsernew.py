# Conda setup (run these in terminal, not inside Python):
# conda create -n smpl_env python=3.10 -y
# conda activate smpl_env
# conda install pytorch torchvision torchaudio -c pytorch -y
# pip install smplx trimesh pyrender chumpy meshio opencv-python matplotlib

import trimesh
import smplx
import torch
import open3d as o3d
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
from trimesh.proximity import signed_distance



MODEL_PATH = '/Users/careycai/Desktop/Thakor Project/models/smplx/SMPLX_NEUTRAL.npz'
VERTICES_CSV = Path("smplx_vertices.csv")
ROTATED_VERTICES_PATH = Path("smplx_y_to_z.csv")  # optional cache, not used below

# ---------- Load SMPL-X ----------
model = smplx.create(
    MODEL_PATH,
    model_type='smplx',
    gender='neutral',
    use_pca=False
)
faces = model.faces

if VERTICES_CSV.exists():
    vertices = np.loadtxt(VERTICES_CSV, delimiter=",")
    print(f"Loaded cached vertices from {VERTICES_CSV}")
else:
    output = model()
    vertices = output.vertices[0].detach().cpu().numpy()
    np.savetxt(VERTICES_CSV, vertices, delimiter=",")
    print(f"Computed vertices and saved cache to {VERTICES_CSV}")

# ---------- Rotate SMPL to your convention ----------
# multiply all coordinates by a transformation matrix to align the y axis with the z axis
vertices = vertices.T

# FIX: make this a numpy array so that @ works
rotation_matrix = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [-1, 0, 0]
])

vertices_post_transformation = rotation_matrix @ vertices
vertices_post_transformation_transposed = vertices_post_transformation.T

mesh = trimesh.Trimesh(vertices_post_transformation_transposed, faces)

mesh_vertices = np.asarray(mesh.vertices)
mesh_faces = np.asarray(mesh.faces)

x_coords = mesh_vertices[:, 0]
y_coords = mesh_vertices[:, 1]
z_coords = mesh_vertices[:, 2]

min_y = y_coords.min()
min_index = np.argmin(y_coords)
min_x, min_z = x_coords[min_index], z_coords[min_index]

print(f"Lowest vertex at y={min_y:.4f} located at x={min_x:.4f}, z={min_z:.4f}")

# ---------- Open3D geometry for SMPL body ----------
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
o3d_mesh.compute_vertex_normals()
o3d_mesh.paint_uniform_color([0.8, 0.8, 0.8])

# ---------- STL foot geometry, apply offset ----------
o3d_foot = o3d.io.read_triangle_mesh(
    '/Users/careycai/Desktop/Thakor Project/myoconverter/models/mjc/Gait10dof18musc/Geometry/l_foot.stl'
)
o3d_foot.compute_vertex_normals()
foot_vertices = np.asarray(o3d_foot.vertices)
foot_vertices[:, 1] -= 1.0
o3d_foot.vertices = o3d.utility.Vector3dVector(foot_vertices)

# ---------- Sample point clouds ----------
body_pcd = o3d_mesh.sample_points_poisson_disk(10000)
body_pts = np.array(body_pcd.points)

# if you want masking later, you can uncomment these
# mask1 = body_pts[:, 1] < -1
# mask2 = body_pts[:, 2] < 0
# filtered_body_pts = body_pts[mask1 & mask2]
filtered_body_pts = body_pts

foot_pcd = o3d_foot.sample_points_poisson_disk(10000)

filtered_body_pcd = o3d.geometry.PointCloud()
filtered_body_pcd.points = o3d.utility.Vector3dVector(filtered_body_pts)

o3d.visualization.draw_geometries([foot, filtered_body_pcd])

print("Finished optimization and visualization.")
