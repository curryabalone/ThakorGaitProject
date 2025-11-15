'''
conda create -n smpl_env python=3.10 -y
conda activate smpl_env

# Install CPU-only PyTorch (macOS uses this by default)
conda install pytorch torchvision torchaudio -c pytorch -y

# Install SMPL-X + mesh tools
pip install smplx trimesh pyrender chumpy
pip install meshio opencv-python
pip install matplotlib
'''

import smplx
import torch
import trimesh
import open3d as o3d
import numpy as np
from pathlib import Path


MODEL_PATH = '/Users/careycai/Desktop/Thakor Project/models/smplx/SMPLX_NEUTRAL.npz'  # where you place downloaded SMPL-X models
VERTICES_CSV = Path("smplx_vertices.csv")
ROTATED_VERTICES_PATH = Path("smplx_y_to_z.csv")

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

print(vertices[0:2, :])
#multiply all coordinates by a transformation matrix to align the y axis with the z axis
vertices = vertices.T
rotation_matrix = [[1, 0, 0 ],
                   [0, 0, -1],
                   [0, 1, 0]]
vertices_post_transformation = rotation_matrix @ vertices
vertices_post_transformation_transposed = vertices_post_transformation.T

mesh = trimesh.Trimesh(vertices_post_transformation_transposed, faces)

mesh_vertices = np.asarray(mesh.vertices)
faces = np.asarray(mesh.faces)

x_coords = mesh_vertices[:, 0]
y_coords = mesh_vertices[:, 1]
z_coords = mesh_vertices[:, 2]

min_y = y_coords.min()
min_index = np.argmin(y_coords)
min_x, min_z = x_coords[min_index], z_coords[min_index]

print(f"Lowest vertex at y={min_y:.4f} located at x={min_x:.4f}, z={min_z:.4f}")

o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
o3d_mesh.compute_vertex_normals()

# optional: a little color so it’s not gray
o3d_mesh.paint_uniform_color([0.8, 0.8, 0.8])

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
axes.paint_uniform_color([0.1, 0.1, 0.1])

o3d.visualization.draw_geometries([o3d_mesh, axes])

print("Exported smplx.obj successfully!")
