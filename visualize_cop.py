#!/usr/bin/env python3
"""
Visualize Center of Pressure (CoP) trajectories from gait simulation.

Generates a video with:
- MuJoCo 3D gait animation
- Left Foot CoP trajectory
- Right Foot CoP trajectory
- Global CoP trajectory (Both feet)

Usage:
    python visualize_cop.py
    python visualize_cop.py --output-dir GaitDynamics/output
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from pathlib import Path
import argparse

from fit_spheres import fit_contact_spheres

# Default paths
DEFAULT_MODEL = "GaitDynamics/output/example_opensim_model_cvt1_contact.xml"
DEFAULT_MOTION = "GaitDynamics/example_mot_complete_kinematics.mot"
DEFAULT_OUTPUT_DIR = "GaitDynamics/output"
DEFAULT_GEOMETRY = "GaitDynamics/output/Geometry"


def parse_mot_file(mot_path):
    """Parse OpenSim .mot file."""
    with open(mot_path, 'r') as f:
        lines = f.readlines()
    in_degrees = True
    header_end = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('inDegrees='):
            in_degrees = line.split('=')[1].lower() == 'yes'
        elif line == 'endheader':
            header_end = i + 1
            break
    joint_names = lines[header_end].strip().split('\t')
    data = []
    for line in lines[header_end + 1:]:
        if line.strip():
            data.append([float(v) for v in line.strip().split('\t')])
    data = np.array(data)
    return data[:, 0], joint_names[1:], data[:, 1:], in_degrees


def map_joints(model, mot_joint_names, in_degrees):
    """Map .mot joints to MuJoCo qpos indices."""
    mj_joint_names = [model.joint(i).name for i in range(model.njnt)]
    qpos_indices, mot_indices, scale_factors = [], [], []
    for mot_idx, mot_name in enumerate(mot_joint_names):
        if mot_name in mj_joint_names:
            jnt_idx = mj_joint_names.index(mot_name)
            qpos_idx = model.joint(jnt_idx).qposadr[0]
            qpos_indices.append(qpos_idx)
            mot_indices.append(mot_idx)
            scale_factors.append(np.pi / 180.0 if in_degrees else 1.0)
    return qpos_indices, mot_indices, scale_factors


def get_foot_sphere_indices(model):
    """Get sphere geom IDs indexed by sphere number for each foot."""
    left_spheres, right_spheres = {}, {}
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name.startswith('contact_sphere_r_'):
            right_spheres[int(name.split('_')[-1])] = {'id': i, 'radius': model.geom_size[i, 0]}
        elif name.startswith('contact_sphere_l_'):
            left_spheres[int(name.split('_')[-1])] = {'id': i, 'radius': model.geom_size[i, 0]}
    return left_spheres, right_spheres


def compute_pressures(model, data, spheres, ground_z, stiffness=50000.0):
    """Compute pressure for each sphere."""
    pressures = {}
    for idx, info in spheres.items():
        sphere_z = data.geom_xpos[info['id'], 2]
        penetration = ground_z - (sphere_z - info['radius'])
        # Refinement: Return signed pressure to allow smoothing to see "near misses"
        pressures[idx] = stiffness * penetration 
    return pressures


def compute_cop(pressures, cell_centers):
    """
    Compute Center of Pressure (CoP).
    
    Args:
        pressures: dict of {sphere_idx: pressure_value}
        cell_centers: (N, 2) array of (x, z) positions for spheres
        
    Returns:
        (x, z) CoP tuple or None if total pressure is too low.
    """
    total_p = 0.0
    weighted_x = 0.0
    weighted_z = 0.0
    
    for idx, p in pressures.items():
        if p > 0:
            # cell_centers is (N, 2) where col 0 is X, col 1 is Z
            # Note: fit_spheres uses X-Z plane where Y is up
            pos = cell_centers[idx]
            total_p += p
            weighted_x += p * pos[0]
            weighted_z += p * pos[1]
            
    if total_p < 1.0:  # Threshold to avoid noise
        return None
        
    return (weighted_x / total_p, weighted_z / total_p)


def compute_global_cop(model, data, left_pressures, right_pressures, left_spheres, right_spheres):
    """
    Compute global CoP based on global sphere positions.
    """
    total_p = 0.0
    weighted_x = 0.0
    weighted_y = 0.0 # MuJoCo uses X-Y ground plane usually, but check model alignment
    
    # In MuJoCo, global frame. Let's assume Z is up, X is forward, Y is left (typical, but we verify)
    # The foot fitting was done in a local frame. Here we use the actual global positions of the spheres.
    
    # Left
    for idx, p in left_pressures.items():
        if p > 0:
            geom_id = left_spheres[idx]['id']
            pos = data.geom_xpos[geom_id] # (x, y, z)
            # Assuming ground is X-Y plane (Z up) or X-Z plane (Y up).
            # The fit_spheres logic implies Y is UP (ground plane Y). 
            # So X and Z are horizontal.
            
            total_p += p
            weighted_x += p * pos[0]
            weighted_y += p * pos[2] # Using Z as the other horizontal dims
            
    # Right
    for idx, p in right_pressures.items():
        if p > 0:
            geom_id = right_spheres[idx]['id']
            pos = data.geom_xpos[geom_id]
            
            total_p += p
            weighted_x += p * pos[0]
            weighted_y += p * pos[2] # Using Z
            
    if total_p < 1.0:
        return None
        
    return (weighted_x / total_p, weighted_y / total_p)


def render_mujoco_frame(model, data, renderer, width=640, height=480):
    renderer.update_scene(data, camera='track')
    return renderer.render()


def create_cop_video(model_path, motion_path, geometry_dir, output_dir, fps=30, stiffness=50000.0):
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Setup renderer
    render_width, render_height = 640, 480
    renderer = mujoco.Renderer(model, render_height, render_width)
    
    print(f"Loading motion: {motion_path}")
    times, joint_names, joint_data, in_degrees = parse_mot_file(motion_path)
    
    qpos_indices, mot_indices, scale_factors = map_joints(model, joint_names, in_degrees)
    left_spheres, right_spheres = get_foot_sphere_indices(model)
    
    print("Loading foot contours...")
    geometry_dir = Path(geometry_dir)
    left_fit = fit_contact_spheres(str(geometry_dir / "l_foot.stl"))
    right_fit = fit_contact_spheres(str(geometry_dir / "r_foot.stl"))
    
    # Compute ground height
    for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
        data.qpos[qpos_idx] = joint_data[0, mot_idx] * scale
    mujoco.mj_forward(model, data)
    
    all_spheres = {**left_spheres, **right_spheres}
    min_z = min(data.geom_xpos[info['id'], 2] - info['radius'] for info in all_spheres.values())
    ground_z = min_z + 0.0005
    
    # Sample times
    dt = 1.0 / fps
    sample_times = np.arange(times[0], times[-1], dt)
    n_frames = len(sample_times)
    
    # History buffers for plotting trails
    left_cop_hist_x, left_cop_hist_z = [], []
    right_cop_hist_x, right_cop_hist_z = [], []
    global_cop_hist_x, global_cop_hist_z = [], []
    
    # Setup Plots
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 4, figure=fig)
    
    # Layout: Animation on left half, CoP plots on right half
    ax_mujoco = fig.add_subplot(gs[:, 0:2])
    ax_left = fig.add_subplot(gs[0, 2])
    ax_right = fig.add_subplot(gs[0, 3])
    ax_global = fig.add_subplot(gs[1, 2:])
    
    # Pre-compute contours (closed loops)
    left_hull = np.vstack([left_fit.hull_points, left_fit.hull_points[0]])
    right_hull = np.vstack([right_fit.hull_points, right_fit.hull_points[0]])
    
    print(f"Total frames to render: {n_frames}")
    print(f"Target FPS: {fps}")
    print(f"Expected video duration: {n_frames / fps:.2f} seconds")
    print(f"Rendering {n_frames} frames...")
    
    def animate(frame_idx):
        t = sample_times[frame_idx]
        mot_frame = min(np.searchsorted(times, t), len(times) - 1)
        
        # Update Physics
        for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
            data.qpos[qpos_idx] = joint_data[mot_frame, mot_idx] * scale
        mujoco.mj_forward(model, data)
        
        # Compute Pressures
        left_press = compute_pressures(model, data, left_spheres, ground_z, stiffness)
        right_press = compute_pressures(model, data, right_spheres, ground_z, stiffness)
        
        # Compute CoPs
        l_cop = compute_cop(left_press, left_fit.cell_centers)
        r_cop = compute_cop(right_press, right_fit.cell_centers)
        
        # Global
        g_cop = compute_global_cop(model, data, left_press, right_press, left_spheres, right_spheres)
        
        # Update History
        if l_cop: 
            left_cop_hist_x.append(l_cop[1]) # Z (width)
            left_cop_hist_z.append(l_cop[0]) # X (length)
        else:
            left_cop_hist_x.append(np.nan)
            left_cop_hist_z.append(np.nan)
            
        if r_cop:
            right_cop_hist_x.append(r_cop[1])
            right_cop_hist_z.append(r_cop[0])
        else:
            right_cop_hist_x.append(np.nan)
            right_cop_hist_z.append(np.nan)
            
        if g_cop:
            global_cop_hist_x.append(g_cop[0]) # Global X
            global_cop_hist_z.append(g_cop[1]) # Global Z
        else:
            global_cop_hist_x.append(np.nan)
            global_cop_hist_z.append(np.nan)
            
        # Keep history reasonable
        # Keep history throughout the entire trial
        max_hist = n_frames
        if len(left_cop_hist_x) > max_hist:
            left_cop_hist_x.pop(0); left_cop_hist_z.pop(0)
            right_cop_hist_x.pop(0); right_cop_hist_z.pop(0)
            global_cop_hist_x.pop(0); global_cop_hist_z.pop(0)
        
        # --- Plotting ---
        
        # 1. MuJoCo Animation
        rgb = render_mujoco_frame(model, data, renderer, render_width, render_height)
        ax_mujoco.clear()
        ax_mujoco.imshow(rgb)
        ax_mujoco.axis('off')
        ax_mujoco.set_title(f'Sim Time: {t:.2f}s', fontsize=14)
        
        # 2. Left Foot
        ax_left.clear()
        # Plot contour - coords are (X, Z), we plot as (Z, X) for standard medical orientation
        ax_left.plot(left_hull[:, 1], left_hull[:, 0], 'k-', lw=1, alpha=0.5)
        # Plot CoP History (Local)
        ax_left.plot(left_cop_hist_x, left_cop_hist_z, 'b-', lw=1.5, alpha=0.6)
        # Current Point
        if l_cop:
            ax_left.plot(l_cop[1], l_cop[0], 'bo', markersize=8, label='CoP')
            
        ax_left.set_title('Left Foot CoP')
        ax_left.set_aspect('equal')
        # Left foot stays normal
        
        # 3. Right Foot
        ax_right.clear()
        ax_right.plot(right_hull[:, 1], right_hull[:, 0], 'k-', lw=1, alpha=0.5) 
        ax_right.plot(right_cop_hist_x, right_cop_hist_z, 'r-', lw=1.5, alpha=0.6)
        if r_cop:
            ax_right.plot(r_cop[1], r_cop[0], 'ro', markersize=8)
            
        ax_right.set_title('Right Foot CoP')
        ax_right.set_aspect('equal')
        ax_right.invert_xaxis()  # Mirror for right foot
        
        # 4. Global CoP (Both Feet)
        ax_global.clear()
        ax_global.plot(global_cop_hist_x, global_cop_hist_z, 'g-', lw=2, alpha=0.7)
        if g_cop:
            ax_global.plot(g_cop[0], g_cop[1], 'go', markersize=10, mec='k')
            
        ax_global.set_title('Global Center of Pressure (World Frame)')
        ax_global.set_xlabel('Global X (m)')
        ax_global.set_ylabel('Global Z (m)')
        ax_global.axis('equal')
        ax_global.grid(True, alpha=0.3)
        
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{n_frames}")
            
        return []
        
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, blit=False)
    
    output_path = Path(output_dir) / "cop_visualization.mp4"
    print(f"Saving to {output_path}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize CoP Trajectories')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--motion', default=DEFAULT_MOTION)
    parser.add_argument('--output-dir', '-o', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--geometry', '-g', default=DEFAULT_GEOMETRY)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--stiffness', type=float, default=50000.0)
    args = parser.parse_args()
    
    create_cop_video(args.model, args.motion, args.geometry, args.output_dir, args.fps, args.stiffness)
