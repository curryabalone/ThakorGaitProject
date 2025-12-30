import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull, Delaunay


def project_to_xz_plane(vertices):
    """Project 3D points to X-Z plane by taking X and Z coordinates."""
    return vertices[:, [0, 2]]


def compute_convex_hull_contour(points_2d):
    """Compute convex hull of 2D points and return hull vertices in order."""
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]
    return hull, hull_points


def find_ground_plane_y(vertices):
    """Find the Y coordinate of the ground plane (lowest Y value in mesh)."""
    return np.min(vertices[:, 1])


def create_square_regions(hull_points, num_regions=240):
    """
    Divide the convex hull area into approximately num_regions square cells.
    Returns cell centers and cell size.
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


def create_square_wireframes(cell_centers, cell_size, y_plane, color=[0, 0.8, 0.2]):
    """Create Open3D LineSet for square cell boundaries."""
    half = cell_size / 2
    all_points = []
    all_lines = []
    
    for i, (cx, cz) in enumerate(cell_centers):
        # 4 corners of the square
        corners = [
            [cx - half, y_plane, cz - half],
            [cx + half, y_plane, cz - half],
            [cx + half, y_plane, cz + half],
            [cx - half, y_plane, cz + half],
        ]
        base_idx = i * 4
        all_points.extend(corners)
        # 4 edges forming the square
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


def create_contact_spheres(cell_centers, cell_size, y_plane, color=[0, 0.7, 0.3]):
    """Create Open3D sphere meshes at each cell center. Radius = half cell size."""
    radius = cell_size / 2
    spheres = []
    for cx, cz in cell_centers:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate([cx, y_plane, cz])
        sphere.paint_uniform_color(color)
        spheres.append(sphere)
    return spheres, radius


def create_contour_lineset(hull_points_2d, y_plane, color=[1, 0, 0]):
    """Create Open3D LineSet for the convex hull contour."""
    n = len(hull_points_2d)
    points_3d = np.zeros((n, 3))
    points_3d[:, 0] = hull_points_2d[:, 0]
    points_3d[:, 1] = y_plane
    points_3d[:, 2] = hull_points_2d[:, 1]
    
    lines = [[i, (i + 1) % n] for i in range(n)]
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points_3d)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return lineset


if __name__ == '__main__':
    # Load foot mesh
    o3d_foot = o3d.io.read_triangle_mesh('GaitDynamics/output/Geometry/r_foot.stl')
    o3d_foot.compute_vertex_normals()
    
    vertices = np.asarray(o3d_foot.vertices)
    print(f"Total vertices: {len(vertices)}")
    
    y_plane = find_ground_plane_y(vertices)
    print(f"Ground plane Y: {y_plane:.6f}")
    
    # Project and get convex hull
    xz_points = project_to_xz_plane(vertices)
    hull, hull_points = compute_convex_hull_contour(xz_points)
    # In 2D, hull.volume is area, hull.area is perimeter
    print(f"Convex hull: {len(hull_points)} vertices, area: {hull.volume:.6f} m²")
    
    # Create square regions
    NUM_REGIONS = 240
    cell_centers, cell_size, area = create_square_regions(hull_points, num_regions=NUM_REGIONS)
    print(f"\nSquare region distribution:")
    print(f"  Target regions: {NUM_REGIONS}")
    print(f"  Actual regions: {len(cell_centers)}")
    print(f"  Cell size: {cell_size * 1000:.2f} mm")
    print(f"  Cell area: {(cell_size ** 2) * 1e6:.2f} mm²")
    
    # Create visualization elements
    contour = create_contour_lineset(hull_points, y_plane, color=[1, 0, 0])
    squares = create_square_wireframes(cell_centers, cell_size, y_plane, color=[0, 0.8, 0.2])
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    o3d_foot.paint_uniform_color([0.7, 0.7, 0.9])
    
    print("\nOpening 3D visualization...")
    o3d.visualization.draw_geometries(
        [o3d_foot, contour, squares, coord_frame],
        window_name=f"Foot with {len(cell_centers)} Square Regions"
    )
