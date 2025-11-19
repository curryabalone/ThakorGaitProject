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
import torch.nn.functional as F
import torch.nn as nn



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

# multiply all coordinates by a transformation matrix to align the y axis with the z axis
vertices = vertices.T
rotation_matrix = [[0, 0, 1],
                   [0, 1, 0],
                   [-1, 0, 0]]
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

#compute geometry for the npz body
o3d_mesh = o3d.geometry.TriangleMesh()
o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces)
o3d_mesh.compute_vertex_normals()
o3d_mesh.paint_uniform_color([0.8, 0.8, 0.8])

#compute stl geometry for the foot and apply offset of -1.2
o3d_foot = o3d.io.read_triangle_mesh('/Users/careycai/Desktop/Thakor Project/myoconverter/models/mjc/Gait10dof18musc/Geometry/l_foot.stl')
o3d_foot.compute_vertex_normals()
foot_vertices = np.asarray(o3d_foot.vertices)
foot_vertices[:,1] -= 1.2
o3d_foot.vertices = o3d.utility.Vector3dVector(foot_vertices)

#generate point cloud for npz body
body_pcd = o3d_mesh.sample_points_poisson_disk(100000)
body_pts = np.array(body_pcd.points)
# mask1 = body_pts[:, 1] < -1
# mask2 = body_pts[:, 2] < 0
filtered_body_pts = body_pts
foot_pcd = o3d_foot.sample_points_poisson_disk(10000)
filtered_body_pcd = o3d.geometry.PointCloud()
filtered_body_pcd.points = o3d.utility.Vector3dVector(filtered_body_pts)

#Here we build the SDF using trimesh
RES = 128
mins = mesh.bounds[0]
maxs = mesh.bounds[1]
xs = np.linspace(mins[0], maxs[0], RES)
ys = np.linspace(mins[1], maxs[1], RES)
zs = np.linspace(mins[2], maxs[2], RES)
grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing='ij')
points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
closest, dist, _ = mesh.nearest.on_surface(points)
inside = mesh.contains(points)  # boolean array
sdf_vals = dist
sdf_vals[inside] *= -1
sdf_grid = sdf_vals.reshape(RES, RES, RES)
sdf_torch = torch.tensor(sdf_grid, dtype=torch.float32)
sdf_torch = sdf_torch.unsqueeze(0).unsqueeze(0) 
bbox_min = torch.tensor(mins, dtype=torch.float32)
bbox_max = torch.tensor(maxs, dtype=torch.float32)
def world_to_grid(p, bbox_min, bbox_max):
    return 2 * (p - bbox_min) / (bbox_max - bbox_min) - 1
def sample_sdf(sdf_grid, points, bbox_min, bbox_max):
    # points: [N,3] world coords
    coords = world_to_grid(points, bbox_min, bbox_max)  # [N,3]
    coords = coords.view(1, -1, 1, 1, 3)  # [1, N, 1, 1, 3]
    
    # grid_sample outputs [1,1,N,1,1]
    sdf = F.grid_sample(
        sdf_grid,
        coords,
        align_corners=True,
        mode='bilinear',
        padding_mode='border'
    )
    return sdf.view(-1)
def axis_angle_to_matrix(axis_angle):
    angle = torch.linalg.norm(axis_angle) + 1e-8
    if angle < 1e-6:
        return torch.eye(3)
    axis = axis_angle / angle
    x, y, z = axis
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    R = torch.stack([
        torch.stack([ca + x*x*C, x*y*C - z*sa, x*z*C + y*sa]),
        torch.stack([y*x*C + z*sa, ca + y*y*C, y*z*C - x*sa]),
        torch.stack([z*x*C - y*sa, z*y*C + x*sa, ca + z*z*C])
    ])
    return R
class RigidTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.axis_angle = nn.Parameter(torch.zeros(3))
        self.translation = nn.Parameter(torch.zeros(3))
        self.log_scale = nn.Parameter(torch.zeros(1))  # s = exp(log_scale)

    def forward(self, pts):
        R = axis_angle_to_matrix(self.axis_angle)
        s = torch.exp(self.log_scale)
        t = self.translation
        return pts @ R.T * s + t
def compute_loss(model, bone_pts, sdf_grid, bbox_min, bbox_max, margin=0.002):
    transformed = model(bone_pts)  # [N,3]
    sdf = sample_sdf(sdf_grid, transformed, bbox_min, bbox_max)

    # cost 1: penalize positive SDF (points outside skin)
    outside = torch.clamp(sdf, min=0)
    L_outside = (outside**2).mean()

    # cost 2: push points slightly inside (target = -margin)
    L_surface = ((sdf + margin)**2).mean()

    # cost 3: regularize transform so it doesn't drift too far
    L_reg = (model.axis_angle**2).sum() * 0.001 + (model.log_scale**2) * 0.001

    return L_outside + L_surface + L_reg, sdf
model = RigidTransform()
optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20)

bone_points = torch.tensor(np.asarray(foot_pcd.points), dtype=torch.float32)
for i in range(50):
    def closure():
        optimizer.zero_grad()
        loss, sdf = compute_loss(
            model, bone_points, sdf_torch, bbox_min, bbox_max
        )
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    print(f"Iter {i} Loss = {loss.item():.6f}")
# ---------- Compute transformed bone points ----------
with torch.no_grad():
    transformed_tensor = model(bone_points)               # [N, 3] torch tensor
    bone_transformed = transformed_tensor.detach().cpu().numpy()

pcd_transformed = o3d.geometry.PointCloud()
pcd_transformed.points = o3d.utility.Vector3dVector(bone_transformed)
pcd_transformed.paint_uniform_color([1, 0, 0])  

o3d.visualization.draw_geometries([foot_pcd, pcd_transformed])

print("Exported smplx.obj successfully!")
