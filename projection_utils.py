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
            projected_points = concave_hull(projected_points.T)
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
            projected_points = concave_hull(projected_points.T)
            projected_points = A@projected_points.T
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(projected_points.T)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
    vis.run()

def concave_hull(points):
    #shape of points: (n, 2)
    alpha = 0.01
    shape = alphashape.alphashape(points, alpha)
    points = np.array(shape.exterior.coords) 
    points = densify_polyline(points, 30)
    return points

def densify_polyline(poly, samples_per_edge=10):
    dense = []
    for i in range(len(poly)-1):
        p0 = poly[i]
        p1 = poly[i+1]
        t = np.linspace(0, 1, samples_per_edge, endpoint=False)
        seg = (1-t)[:,None]*p0 + t[:,None]*p1
        dense.append(seg)
    dense.append(poly[-1:])   # close the loop
    return np.vstack(dense)

def knn_concave_hull(points, k):
    """
    Compute the concave hull of a set of 2D points using the k-NN algorithm.
    points: (N,2) numpy array
    k: number of nearest neighbors (increase for smoother hull)
    """
    if len(points) < 3:
        return points

    # Start at the lowest y-value (tie-break by x)
    p0_index = np.lexsort((points[:,0], points[:,1]))[0]
    hull = [p0_index]

    current_index = p0_index
    prev_angle = 0

    used = set([p0_index])
    tree = KDTree(points)

    while True:
        # Query k nearest neighbors
        dists, idxs = tree.query(points[current_index], k+1)
        neighbors = [i for i in idxs if i != current_index]

        best_angle = None
        best_neighbor = None

        for n in neighbors:
            if n in used and n != p0_index:
                continue

            # Compute angle from previous edge
            v1 = np.array([math.cos(prev_angle), math.sin(prev_angle)])
            v2 = points[n] - points[current_index]
            angle = math.atan2(v2[1], v2[0])

            # Normalize angle difference
            angle_diff = (angle - prev_angle) % (2*math.pi)

            if best_angle is None or angle_diff < best_angle:
                best_angle = angle_diff
                best_neighbor = n

        if best_neighbor is None:
            break

        if best_neighbor == p0_index:
            # closure
            hull.append(best_neighbor)
            break

        hull.append(best_neighbor)
        used.add(best_neighbor)

        # update direction
        v = points[best_neighbor] - points[current_index]
        prev_angle = math.atan2(v[1], v[0])
        current_index = best_neighbor

        if len(hull) > len(points):
            break

    return points[hull]