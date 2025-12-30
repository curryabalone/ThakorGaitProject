import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull

def project_to_xz_plane(vertices):
    """Project 3D points to X-Z plane by taking X and Z coordinates."""
    return vertices[:, [0, 2]]  # columns 0=X, 2=Z

def compute_convex_hull_contour(points_2d):
    """Compute convex hull of 2D points and return hull vertices in order."""
    hull = ConvexHull(points_2d)
    hull_points = points_2d[hull.vertices]
    return hull, hull_points

def create_contour_lineset(hull_points_2d, y_plane=-0.009839, color=[1, 0, 0]):
    """Create Open3D LineSet for the convex hull contour at the given Y plane."""
    n = len(hull_points_2d)
    
    # Convert 2D hull points back to 3D (insert Y coordinate)
    points_3d = np.zeros((n, 3))
    points_3d[:, 0] = hull_points_2d[:, 0]  # X
    points_3d[:, 1] = y_plane               # Y (constant)
    points_3d[:, 2] = hull_points_2d[:, 1]  # Z
    
    # Create line segments connecting consecutive hull vertices (closed loop)
    lines = [[i, (i + 1) % n] for i in range(n)]
    
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points_3d)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    lineset.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return lineset

if __name__ == '__main__': 
    # Load foot mesh
    o3d_foot = o3d.io.read_triangle_mesh(
        'GaitDynamics/output/Geometry/r_foot.stl'
    )
    o3d_foot.compute_vertex_normals()
    
    # Get vertices
    vertices = np.asarray(o3d_foot.vertices)
    print(f"Total vertices: {len(vertices)}")
    
    # Project to X-Z plane and compute convex hull
    y_plane = -0.009839
    xz_points = project_to_xz_plane(vertices)
    hull, hull_points = compute_convex_hull_contour(xz_points)
    print(f"Convex hull: {len(hull_points)} vertices, area: {hull.area:.6f} m²")
    
    # Create 3D contour lineset
    contour = create_contour_lineset(hull_points, y_plane, color=[1, 0, 0])
    
    # Coordinate frame for reference
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    
    # Visualize
    o3d.visualization.draw_geometries(
        [o3d_foot, contour, coord_frame],
        window_name="Foot with Convex Hull Contour"
    )
