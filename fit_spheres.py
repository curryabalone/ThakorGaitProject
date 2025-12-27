import open3d as o3d

if __name__ == '__main__': 
    o3d_foot = o3d.io.read_triangle_mesh(
        'GaitDynamics/output/Geometry/r_tibia.stl'
    )
    o3d_foot.compute_vertex_normals()
    o3d.visualization.draw_geometries([o3d_foot])