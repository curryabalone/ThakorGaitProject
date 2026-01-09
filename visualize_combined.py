#!/usr/bin/env python3
"""
Combined gait visualization with MuJoCo 3D view and plantar pressure maps.

Renders a single video with:
- Left: MuJoCo 3D walking animation
- Right top: Left foot plantar pressure
- Right bottom: Right foot plantar pressure

Usage:
    python visualize_combined.py
    python visualize_combined.py --output combined_gait.mp4
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from pathlib import Path
import argparse

from fit_spheres import fit_contact_spheres

# Default paths
DEFAULT_MODEL = "GaitDynamics/output/example_opensim_model_cvt1_contact.xml"
DEFAULT_MOTION = "GaitDynamics/example_mot_complete_kinematics.mot"
DEFAULT_OUTPUT = "GaitDynamics/output/combined_gait_pressure.mp4"
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
        pressures[idx] = stiffness * penetration if penetration > 0 else 0.0
    return pressures


def render_mujoco_frame(model, data, renderer, width=640, height=480):
    """Render MuJoCo scene to RGB array."""
    renderer.update_scene(data, camera='track')
    return renderer.render()


def create_combined_video(model_path, motion_path, geometry_dir, output_path, fps=30, stiffness=50000.0):
    """Create combined video with 3D view and pressure maps."""
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Setup renderer with smaller size to fit default framebuffer
    render_width, render_height = 640, 480
    renderer = mujoco.Renderer(model, render_height, render_width)
    
    # Add tracking camera if not exists
    if model.ncam == 0:
        print("  Note: No camera in model, using default view")
    
    print(f"Loading motion: {motion_path}")
    times, joint_names, joint_data, in_degrees = parse_mot_file(motion_path)
    duration = times[-1] - times[0]
    print(f"  Duration: {duration:.2f}s")
    
    qpos_indices, mot_indices, scale_factors = map_joints(model, joint_names, in_degrees)
    left_spheres, right_spheres = get_foot_sphere_indices(model)
    
    # Load foot contours
    print("Loading foot contours...")
    geometry_dir = Path(geometry_dir)
    left_fit = fit_contact_spheres(str(geometry_dir / "l_foot.stl"))
    right_fit = fit_contact_spheres(str(geometry_dir / "r_foot.stl"))
    
    # Compute ground height from first frame
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
    print(f"  Rendering {n_frames} frames at {fps} fps...")
    
    # Colormap for pressure
    cmap = LinearSegmentedColormap.from_list('pressure', ['#e0e0e0', '#00aa00', '#ffff00', '#ff0000'])
    
    # Pre-compute pressure bounds
    max_pressure = 1.0
    
    # Setup figure with GridSpec
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1], figure=fig)
    
    ax_mujoco = fig.add_subplot(gs[:, 0])  # Left: MuJoCo (spans both rows)
    ax_left = fig.add_subplot(gs[0, 1])     # Right top: Left foot
    ax_right = fig.add_subplot(gs[1, 1])    # Right bottom: Right foot
    
    # Foot plot bounds
    left_z = left_fit.cell_centers[:, 1]
    left_x = left_fit.cell_centers[:, 0]
    right_z = right_fit.cell_centers[:, 1]
    right_x = right_fit.cell_centers[:, 0]
    
    left_hull = np.vstack([left_fit.hull_points, left_fit.hull_points[0]])
    right_hull = np.vstack([right_fit.hull_points, right_fit.hull_points[0]])
    
    mujoco_img = None
    
    def animate(frame_idx):
        nonlocal max_pressure, mujoco_img
        
        t = sample_times[frame_idx]
        mot_frame = min(np.searchsorted(times, t), len(times) - 1)
        
        # Set joint positions
        for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
            data.qpos[qpos_idx] = joint_data[mot_frame, mot_idx] * scale
        mujoco.mj_forward(model, data)
        
        # Compute pressures
        left_press = compute_pressures(model, data, left_spheres, ground_z, stiffness)
        right_press = compute_pressures(model, data, right_spheres, ground_z, stiffness)
        
        # Update max pressure
        curr_max = max(max(left_press.values(), default=0), max(right_press.values(), default=0))
        if curr_max > max_pressure:
            max_pressure = curr_max * 1.2
        
        # Render MuJoCo
        rgb = render_mujoco_frame(model, data, renderer, render_width, render_height)
        
        # Clear and redraw
        ax_mujoco.clear()
        ax_mujoco.imshow(rgb)
        ax_mujoco.axis('off')
        ax_mujoco.set_title(f'Gait Animation  t={t:.2f}s', fontsize=14)
        
        # Left foot pressure
        ax_left.clear()
        ax_left.plot(left_hull[:, 1], left_hull[:, 0], 'k-', lw=2)
        colors_l = [left_press.get(i, 0) / max_pressure for i in range(len(left_z))]
        ax_left.scatter(left_z, left_x, c=colors_l, cmap=cmap, vmin=0, vmax=1,
                       s=left_fit.cell_size * 30000, alpha=0.85)
        ax_left.set_xlim(left_z.min()-0.01, left_z.max()+0.01)
        ax_left.set_ylim(left_x.min()-0.01, left_x.max()+0.01)
        ax_left.set_aspect('equal')
        ax_left.set_title(f'Left Foot ({sum(left_press.values()):.0f}N)', fontsize=12)
        ax_left.set_xlabel('M-L (m)', fontsize=9)
        ax_left.set_ylabel('A-P (m)', fontsize=9)
        
        # Right foot pressure (flip Z for mirror view)
        ax_right.clear()
        ax_right.plot(right_hull[:, 1], right_hull[:, 0], 'k-', lw=2)
        colors_r = [right_press.get(i, 0) / max_pressure for i in range(len(right_z))]
        ax_right.scatter(right_z, right_x, c=colors_r, cmap=cmap, vmin=0, vmax=1,
                        s=right_fit.cell_size * 30000, alpha=0.85)
        ax_right.set_xlim(right_z.max()+0.01, right_z.min()-0.01)  # Flipped
        ax_right.set_ylim(right_x.min()-0.01, right_x.max()+0.01)
        ax_right.set_aspect('equal')
        ax_right.set_title(f'Right Foot ({sum(right_press.values()):.0f}N)', fontsize=12)
        ax_right.set_xlabel('M-L (m)', fontsize=9)
        ax_right.set_ylabel('A-P (m)', fontsize=9)
        
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{n_frames}")
        
        return []
    
    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=1000/fps, blit=False)
    
    plt.tight_layout()
    
    print(f"Saving to {output_path}...")
    writer = animation.FFMpegWriter(fps=fps, bitrate=5000)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    
    print(f"Done! Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combined gait and pressure visualization')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--motion', default=DEFAULT_MOTION)
    parser.add_argument('--geometry', '-g', default=DEFAULT_GEOMETRY)
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--stiffness', type=float, default=50000.0)
    args = parser.parse_args()
    
    create_combined_video(args.model, args.motion, args.geometry, args.output, args.fps, args.stiffness)
