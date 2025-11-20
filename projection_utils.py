import alphashape
import shapely.geometry as geom
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
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
    total_loss = 0
    count = 0
    for axis in range(0, 3):
        for j in range(0, 5):
            #generates the convex hulls for the foot
            generate_projections(2*np.pi/10*j, axis, A)
            proj = np.linalg.pinv(A)
            projected_points_foot = proj@(foot_points.T) 
            projected_points_foot = convex_hull(projected_points_foot.T)
            pcd = o3d.geometry.PointCloud()
            #generates the convex hulls for the body
            generate_projections(2*np.pi/10*j, axis, A)
            projected_points_body = proj@(body_points.T) 
            projected_points_body = convex_hull(projected_points_body.T)
            #loss function
            count += 1
            total_loss += chamfer_2d(projected_points_body, projected_points_foot)
    loss = total_loss/count
    print("Loss: ", loss)
    return loss

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

def chamfer_2d(F, B):

    treeF = cKDTree(F)
    treeB = cKDTree(B)

    d_FB, _ = treeB.query(F)   # dist each F→nearest-B
    d_BF, _ = treeF.query(B)   # dist each B→nearest-F

    return d_FB.mean() + d_BF.mean()