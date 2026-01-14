import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from dataclasses import dataclass


@dataclass
class ContactSphereResult:
    """Result from contact sphere fitting."""
    cell_centers: np.ndarray      # (N, 2) array of cell centers in X-Z plane
    cell_size: float              # Side length of each square cell
    sphere_radius: float          # Radius of each contact sphere
    y_plane: float                # Y coordinate of ground plane
    hull_points: np.ndarray       # (M, 2) array of convex hull vertices
    hull_area: float              # Area of convex hull in m²
    mesh: o3d.geometry.TriangleMesh  # Original mesh
    
    @property
    def num_spheres(self):
        return len(self.cell_centers)
    
    @property
    def sphere_centers_3d(self):
        """Return sphere centers as (N, 3) array in 3D coordinates."""
        centers = np.zeros((len(self.cell_centers), 3))
        centers[:, 0] = self.cell_centers[:, 0]  # X
        centers[:, 1] = self.y_plane              # Y
        centers[:, 2] = self.cell_centers[:, 1]  # Z
        return centers


def _project_to_xz_plane(vertices):
    """Project 3D points to X-Z plane by taking X and Z coordinates."""
    return vertices[:, [0, 2]]


def _compute_convex_hull_contour(points_2d):
    """Compute convex hull of 2D points and return hull vertices in order."""
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]
    return hull, hull_points


def _find_ground_plane_y(vertices):
    """Find the Y coordinate of the ground plane (lowest Y value in mesh)."""
    return np.min(vertices[:, 1])


def _create_square_regions(hull_points, num_regions=240):
    """
    Divide the convex hull area into approximately num_regions square cells.
    Returns cell centers, cell size, and area.
    """
    hull = ConvexHull(hull_points)
    # In 2D, hull.volume is the area (hull.area is the perimeter)
    area = hull.volume
    
    # Each square cell has area = total_area / num_regions
    cell_area = area / num_regions
    cell_size = np.sqrt(cell_area)
    
    # Bounding box
    min_x, min_z = hull_points.min(axis=0)
    max_x, max_z = hull_points.max(axis=0)
    
    # Create grid of cell centers
    x_coords = np.arange(min_x + cell_size / 2, max_x, cell_size)
    z_coords = np.arange(min_z + cell_size / 2, max_z, cell_size)
    xx, zz = np.meshgrid(x_coords, z_coords)
    grid_centers = np.column_stack([xx.ravel(), zz.ravel()])
    
    # Filter to cells whose centers are inside the hull
    delaunay = Delaunay(hull_points)
    inside_mask = delaunay.find_simplex(grid_centers) >= 0
    cell_centers = grid_centers[inside_mask]
    
    return cell_centers, cell_size, area


def fit_contact_spheres(stl_path: str, num_regions: int = 240, radius_ratio: float = 0.4) -> ContactSphereResult:
    """
    Fit contact spheres to a foot mesh STL file.
    
    Args:
        stl_path: Path to the STL mesh file
        num_regions: Target number of square regions/spheres (default 240)
        radius_ratio: Sphere radius as fraction of cell size (default 0.4)
    
    Returns:
        ContactSphereResult containing sphere positions, sizes, and metadata
    """
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    
    # Find ground plane
    y_plane = _find_ground_plane_y(vertices)
    
    # Project to X-Z and compute convex hull
    xz_points = _project_to_xz_plane(vertices)
    hull, hull_points = _compute_convex_hull_contour(xz_points)
    
    # Create square regions
    cell_centers, cell_size, area = _create_square_regions(hull_points, num_regions=num_regions)
    
    # Calculate sphere radius
    sphere_radius = cell_size * radius_ratio
    
    return ContactSphereResult(
        cell_centers=cell_centers,
        cell_size=cell_size,
        sphere_radius=sphere_radius,
        y_plane=y_plane,
        hull_points=hull_points,
        hull_area=area,
        mesh=mesh,
    )


def create_square_wireframes(result: ContactSphereResult, color=[0, 0.8, 0.2]):
    """Create Open3D LineSet for square cell boundaries."""
    half = result.cell_size / 2
    all_points = []
    all_lines = []
    
    for i, (cx, cz) in enumerate(result.cell_centers):
        corners = [
            [cx - half, result.y_plane, cz - half],
            [cx + half, result.y_plane, cz - half],
            [cx + half, result.y_plane, cz + half],
            [cx - half, result.y_plane, cz + half],
        ]
        base_idx = i * 4
        all_points.extend(corners)
        all_lines.extend([
            [base_idx, base_idx + 1],
            [base_idx + 1, base_idx + 2],
            [base_idx + 2, base_idx + 3],
            [base_idx + 3, base_idx],
        ])
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(all_points)
    lineset.lines = o3d.utility.Vector2iVector(all_lines)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(all_lines))
    return lineset


def create_contact_spheres(result: ContactSphereResult, color=[0, 0.7, 0.3]):
    """Create Open3D sphere meshes at each cell center."""
    spheres = []
    for cx, cz in result.cell_centers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=result.sphere_radius)
        sphere.translate([cx, result.y_plane, cz])
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    return spheres


def create_contour_lineset(result: ContactSphereResult, color=[1, 0, 0]):
    """Create Open3D LineSet for the convex hull contour."""
    n = len(result.hull_points)
    points_3d = np.zeros((n, 3))
    points_3d[:, 0] = result.hull_points[:, 0]
    points_3d[:, 1] = result.y_plane
    points_3d[:, 2] = result.hull_points[:, 1]
    
    lines = [[i, (i + 1) % n] for i in range(n)]
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points_3d)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return lineset


def visualize(result: ContactSphereResult, show_squares=True, show_spheres=True):
    """Visualize the contact sphere fitting result."""
    geometries = []
    
    # Mesh
    mesh = result.mesh
    mesh.paint_uniform_color([0.7, 0.7, 0.9])
    geometries.append(mesh)
    
    # Contour
    geometries.append(create_contour_lineset(result))
    
    # Squares
    if show_squares:
        geometries.append(create_square_wireframes(result))
    
    # Spheres
    if show_spheres:
        geometries.extend(create_contact_spheres(result))
    
    # Coordinate frame
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Foot with {result.num_spheres} Contact Spheres"
    )




class FootGraph:
    """
    Geometry-aware graph for the plantar surface.
    Decouples spatial connectivity from the smoothing math.
    """
    def __init__(self, points: np.ndarray, d_thresh: float, k_neighbors: int = 6):
        """
        Build adjacency and normalized Laplacian.
        
        Args:
            points: (N, 2) or (N, 3) array of sphere positions.
            d_thresh: Distance threshold for neighborhood.
            k_neighbors: Minimum number of nearest neighbors to connect.
        """
        self.points = points
        self.num_nodes = len(points)
        self.d_thresh = d_thresh
        self.k_neighbors = k_neighbors
        
        # Build Adjacency Matrix W
        # Using a gaussian kernel for weights
        sigma = d_thresh / 2.0
        
        # Vectorized distance computation
        from scipy.spatial.distance import cdist
        dists = cdist(points, points)
        
        # 1. Distance-based mask
        dist_mask = dists < d_thresh
        
        # 2. k-NN mask
        knn_mask = np.zeros_like(dist_mask, dtype=bool)
        if k_neighbors > 0:
            # Find indices of k+1 nearest neighbors (including self)
            knn_indices = np.argsort(dists, axis=1)[:, :k_neighbors + 1]
            rows = np.arange(self.num_nodes)[:, np.newaxis]
            knn_mask[rows, knn_indices] = True
            
        # Hybrid mask
        mask = dist_mask | knn_mask
        
        W = np.zeros_like(dists)
        W[mask] = np.exp(-(dists[mask]**2) / (sigma**2))
        
        # Remove self-loops
        np.fill_diagonal(W, 0)
        
        # Row-normalize W to get a transition-like matrix
        row_sums = W.sum(axis=1)
        row_sums[row_sums == 0] = 1.0
        self.W_norm = W / row_sums[:, np.newaxis]
        
        # Normalized Laplacian L = I - W_norm
        self.L = np.eye(self.num_nodes) - self.W_norm


class SpatialRegularizer:
    """
    Implements spatial coupling via graph Laplacian smoothing:
    p = (I + lambda * L)^-1 * p_raw
    """
    def __init__(self, graph: FootGraph, lambda_spatial: float = 1.0, preserve_total_force: bool = True):
        """
        Precompute the smoothing operator.
        
        Args:
            graph: FootGraph instance.
            lambda_spatial: Scalar or array (N,) for smoothing strength.
            preserve_total_force: If True, rescale output to match input sum.
        """
        self.graph = graph
        self.preserve_total_force = preserve_total_force
        
        # A = I + lambda * L
        if np.isscalar(lambda_spatial):
            A = np.eye(graph.num_nodes) + lambda_spatial * graph.L
        else:
            # lambda_spatial is an array (N,)
            # Broad-casting lambda per row
            A = np.eye(graph.num_nodes) + lambda_spatial[:, np.newaxis] * graph.L
            
        self.A_inv = np.linalg.inv(A)

    def apply(self, pressures_dict: dict) -> dict:
        """
        Apply smoothing to a dictionary of pressures.
        
        Args:
            pressures_dict: {index: value} (can be signed)
            
        Returns:
            {index: smoothed_value} (always non-negative)
        """
        # Convert dict to vector
        p_raw = np.zeros(self.graph.num_nodes)
        for idx, val in pressures_dict.items():
            if idx < self.graph.num_nodes:
                p_raw[idx] = val
                
        # Target sum: only the positive (actual contact) forces
        p_raw_positive = np.maximum(p_raw, 0.0)
        sum_target = np.sum(p_raw_positive)
        
        # Apply smoothing to the signed signal
        p_smooth_signed = self.A_inv @ p_raw
        
        # Clip to non-negative AFTER smoothing
        p_smooth = np.maximum(p_smooth_signed, 0.0)
        
        # Preserve total positive force
        if self.preserve_total_force and sum_target > 1e-6:
            sum_smooth = np.sum(p_smooth)
            if sum_smooth > 1e-9:
                p_smooth *= (sum_target / sum_smooth)
                
        # Convert back to dict
        return {i: float(v) for i, v in enumerate(p_smooth)}


if __name__ == '__main__':
    stl_path = 'GaitDynamics/output/Geometry/r_foot.stl'
    
    result = fit_contact_spheres(stl_path, num_regions=240, radius_ratio=0.4)
    
    print(f"Total mesh vertices: {len(np.asarray(result.mesh.vertices))}")
    print(f"Ground plane Y: {result.y_plane:.6f}")
    print(f"Convex hull: {len(result.hull_points)} vertices, area: {result.hull_area:.6f} m²")
    print(f"\nContact sphere distribution:")
    print(f"  Num spheres: {result.num_spheres}")
    print(f"  Cell size: {result.cell_size * 1000:.2f} mm")
    print(f"  Sphere radius: {result.sphere_radius * 1000:.2f} mm")
    
    visualize(result)
