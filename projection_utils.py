import alphashape
import shapely.geometry as geom
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree
import math

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
    

#calculates the net deviation of the current foot from the body
def loss_function(foot_points, body_points):
    A = np.zeros((3, 2))
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for axis in range(0, 3):
        for j in range(0, 5):
            #generates the convex hulls for the foot
            generate_projections(2*np.pi/10*j, axis, A)
            proj = np.linalg.pinv(A)
            projected_points = proj@(foot_points.T) 
            projected_points = convex_hull(projected_points.T)
            projected_points = A@projected_points.T
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(projected_points.T)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            #generates the convex hulls for the body
            generate_projections(2*np.pi/10*j, axis, A)
            proj = np.linalg.pinv(A)
            projected_points = proj@(body_points.T) 
            projected_points = convex_hull(projected_points.T)
            projected_points = A@projected_points.T
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(projected_points.T)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
    vis.run()

def convex_hull(points):
    #shape of points: (n, 2)
    alpha = 0.01
    shape = alphashape.alphashape(points, alpha)
    points = np.array(shape.exterior.coords) 
    points = densify_polyline(points, 30)
    return points

def densify_polyline(poly, samples_per_edge=10):
    dense = []
    n = len(poly)

    for i in range(n):
        p0 = poly[i]
        p1 = poly[(i+1) % n]     # wrap-around
        t = np.linspace(0, 1, samples_per_edge, endpoint=False)
        seg = (1-t)[:,None]*p0 + t[:,None]*p1
        dense.append(seg)
    return np.vstack(dense)

def rotation_translation(axis, angle, translation):
    