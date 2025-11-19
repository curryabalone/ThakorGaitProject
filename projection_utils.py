import alphashape
import shapely.geometry as geom
import numpy as np
import open3d as o3d

def project_to_plane(food_pcd, project_matrix):
    x_y = np.array([
    [1, 0],
    [0, 1],
    [0, 0]
    ])
    proj_x_y = x_y@np.linalg.pinv(x_y)
    projected_foot_points = proj_x_y @ foot_points.T
    projected_foot_points = projected_foot_points.T

    projected_foot_pcd = o3d.geometry.PointCloud()
    projected_foot_pcd.points = o3d.utility.Vector3dVector(projected_foot_points)

    #compute the outline of the foot using concave hull
    alpha = 0.05
    shape = alphashape.alphashape(projected_foot_points[:, :2], alpha)
    boundary = np.array(shape.exterior.coords)
    boundary3d = np.zeros((boundary.shape[0], 3))
    boundary3d[:, :2] = boundary

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(boundary3d),
        lines=o3d.utility.Vector2iVector([[i, i+1] for i in range(len(boundary3d)-1)])
    )


#generates the projection matrix AA+ that's rotated angle counter-clockwise around axis (0 for X, 1 for Y, 2 for Z). 
def generate_projections(angle, axis, A):
    if(axis == 0):
        A[:,0] = np.array([1, 0, 0])
        A[:,1] = np.array([0, np.cos(angle), np.sin(angle)])
    elif(axis == 1):
        A[:,0] = np.array([0, 1, 0])
        A[:,1] = np.array([np.cos(angle), 0, np.sin(angle)])
    else:
        A[:,0] = np.array([0, 0, 1])
        A[:,1] = np.array([np.cos(angle), np.sin(angle), 0])
    

##calculated the net deviation of the current foot from the body
def loss_function(foot_points, body_points):
    A = np.zeros((3, 2))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for axis in range(0, 3):
        for j in range(0, 5):
            generate_projections(2*np.pi/10*j, axis, A)
            proj = A@np.linalg.pinv(A)
            projected_points = proj@(foot_points.T)
            projected_points_pcd = o3d.geometry.PointCloud()
            projected_points_pcd.points = o3d.utility.Vector3dVector(projected_points.T)
            vis.add_geometry(projected_points_pcd)
            vis.poll_events()
            vis.update_renderer()
            projected_points = proj@(body_points.T)
            projected_points_pcd = o3d.geometry.PointCloud()
            projected_points_pcd.points = o3d.utility.Vector3dVector(projected_points.T)
            vis.poll_events()
            vis.update_renderer()
    vis.run()
    


    
    

    

