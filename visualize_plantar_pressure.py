#!/usr/bin/env python3
"""
Generate plantar pressure heatmap videos from gait simulation.

Uses the foot contour from fit_spheres to create proper 2D plantar views.

Usage:
    python visualize_plantar_pressure.py
    python visualize_plantar_pressure.py --output-dir GaitDynamics/output/pressure_videos
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from pathlib import Path
import argparse

from fit_spheres import fit_contact_spheres, ContactSphereResult

# Default paths
DEFAULT_MODEL = "GaitDynamics/output/example_opensim_model_cvt1_contact.xml"
DEFAULT_MOTION = "GaitDynamics/example_mot_complete_kinematics.mot"
DEFAULT_OUTPUT_DIR = "GaitDynamics/output/pressure_videos"
DEFAULT_OUTPUT = "GaitDynamics/output/contact_forces.csv"
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
            values = [float(v) for v in line.strip().split('\t')]
            data.append(values)
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
            idx = int(name.split('_')[-1])
            right_spheres[idx] = {'id': i, 'radius': model.geom_size[i, 0]}
        elif name.startswith('contact_sphere_l_'):
            idx = int(name.split('_')[-1])
            left_spheres[idx] = {'id': i, 'radius': model.geom_size[i, 0]}
    return left_spheres, right_spheres


def compute_pressure_for_spheres(model, data, sphere_dict, ground_z, stiffness=50000.0):
    """Compute pressure for each sphere based on penetration."""
    pressures = {}
    for idx, info in sphere_dict.items():
        geom_id = info['id']
        radius = info['radius']
        sphere_z = data.geom_xpos[geom_id, 2]
        penetration = ground_z - (sphere_z - radius)
        # Refinement: Signed pressure
        pressures[idx] = stiffness * penetration
    return pressures


def simulate_and_collect_pressure(model_path, motion_path, sample_rate=30.0, stiffness=50000.0):
    """Run simulation and collect pressure data."""
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"Loading motion: {motion_path}")
    times, joint_names, joint_data, in_degrees = parse_mot_file(motion_path)
    print(f"  Duration: {times[-1] - times[0]:.2f}s, {len(times)} frames")
    
    qpos_indices, mot_indices, scale_factors = map_joints(model, joint_names, in_degrees)
    left_spheres, right_spheres = get_foot_sphere_indices(model)
    print(f"  Left: {len(left_spheres)} spheres, Right: {len(right_spheres)} spheres")
    
    # Set first frame to compute ground height
    for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
        data.qpos[qpos_idx] = joint_data[0, mot_idx] * scale
    mujoco.mj_forward(model, data)
    
    # Find ground height from lowest sphere
    all_spheres = {**left_spheres, **right_spheres}
    min_z, radius = float('inf'), 0.0
    for idx, info in all_spheres.items():
        z = data.geom_xpos[info['id'], 2]
        if z < min_z:
            min_z, radius = z, info['radius']
    ground_z = min_z - radius + 0.0005
    print(f"  Ground Z: {ground_z:.4f}")
    
    # Sample frames
    dt = 1.0 / sample_rate
    sample_times = np.arange(times[0], times[-1], dt)
    left_pressures, right_pressures = [], []
    
    print(f"\nCollecting pressure data ({len(sample_times)} frames)...")
    for i, t in enumerate(sample_times):
        frame_idx = min(np.searchsorted(times, t), len(times) - 1)
        for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
            data.qpos[qpos_idx] = joint_data[frame_idx, mot_idx] * scale
        mujoco.mj_forward(model, data)
        
        left_pressures.append(compute_pressure_for_spheres(model, data, left_spheres, ground_z, stiffness))
        right_pressures.append(compute_pressure_for_spheres(model, data, right_spheres, ground_z, stiffness))
            
        if (i + 1) % 100 == 0:
            print(f"  Frame {i+1}/{len(sample_times)}")
    
    return sample_times, left_pressures, right_pressures


def create_pressure_video(sample_times, pressure_data, fit_result, output_path, foot_name, fps=30, flip_z=False):
    """Create pressure heatmap video using fit_spheres cell centers and contour."""
    print(f"\nCreating {foot_name} foot video: {output_path}")
    
    cell_centers = fit_result.cell_centers  # (N, 2) as (X, Z)
    hull_points = fit_result.hull_points
    cell_size = fit_result.cell_size
    
    # For plantar view: Z (medial-lateral) on X-axis, X (anterior-posterior) on Y-axis
    z_vals = cell_centers[:, 1]  # width
    x_vals = cell_centers[:, 0]  # length
    
    z_min, z_max = z_vals.min() - 0.01, z_vals.max() + 0.01
    x_min, x_max = x_vals.min() - 0.01, x_vals.max() + 0.01
    
    # Find max pressure for color scaling
    # Refinement: Use a fixed max to better visualize smoothing across time
    # Typical gait GRF for one foot is ~1.2x body weight. Let's assume ~800N total,
    # divided among spheres. A peak of 50-100N per sphere is reasonable.
    max_pressure = 100.0 # Fixed max N for consistent color mapping
    print(f"  Fixed Max pressure for visualization: {max_pressure:.1f} N")
    
    # Colormap
    cmap = LinearSegmentedColormap.from_list('pressure', ['#e0e0e0', '#00aa00', '#ffff00', '#ff0000'])
    
    fig, ax = plt.subplots(figsize=(5, 12))
    
    # Hull contour (closed loop) - swap to (Z, X) for plotting
    hull_closed = np.vstack([hull_points, hull_points[0]])
    
    def animate(frame_idx):
        ax.clear()
        
        # Set limits - Z on X-axis, X on Y-axis
        if flip_z:
            ax.set_xlim(z_max, z_min)
        else:
            ax.set_xlim(z_min, z_max)
        ax.set_ylim(x_min, x_max)
        ax.set_aspect('equal')
        ax.set_xlabel('Medial-Lateral (m)')
        ax.set_ylabel('Anterior-Posterior (m)')
        ax.set_title(f'{foot_name} Foot Plantar Pressure')
        
        # Draw foot contour
        ax.plot(hull_closed[:, 1], hull_closed[:, 0], 'k-', linewidth=2)
        
        # Get pressures
        pressures = pressure_data[frame_idx]
        colors = [pressures.get(idx, 0.0) / max_pressure for idx in range(len(cell_centers))]
        
        # Plot pressure circles - Z on X, X on Y
        ax.scatter(z_vals, x_vals, c=colors, cmap=cmap, vmin=0, vmax=1,
                  s=cell_size * 25000, edgecolors='gray', linewidths=0.3, alpha=0.85)
        
        t = sample_times[frame_idx]
        total = sum(pressures.values())
        ax.text(0.02, 0.98, f't = {t:.2f}s', transform=ax.transAxes, fontsize=14,
                fontweight='bold', va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.text(0.02, 0.94, f'Total: {total:.1f} N', transform=ax.transAxes, fontsize=12,
                va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        return []
    
    anim = animation.FuncAnimation(fig, animate, frames=len(pressure_data), interval=1000/fps, blit=False)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_pressure))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Pressure (N)', shrink=0.6)
    
    print(f"  Rendering {len(pressure_data)} frames...")
    anim.save(output_path, writer=animation.FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main(model_path, motion_path, output_dir, geometry_dir, fps=30.0, stiffness=50000.0):
    """Generate plantar pressure videos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir = Path(geometry_dir)
    
    # Load foot mesh fitting results for contours and cell positions
    print("Loading foot mesh contours...")
    left_fit = fit_contact_spheres(str(geometry_dir / "l_foot.stl"))
    right_fit = fit_contact_spheres(str(geometry_dir / "r_foot.stl"))
    print(f"  Left: {left_fit.num_spheres} cells, Right: {right_fit.num_spheres} cells")
    
    # Run simulation to collect pressure data
    sample_times, left_pressures, right_pressures = \
        simulate_and_collect_pressure(model_path, motion_path, fps, stiffness)
    
    # Create videos
    create_pressure_video(sample_times, left_pressures, left_fit,
                         output_dir / "left_foot_pressure.mp4", "Left", fps, flip_z=False)
    create_pressure_video(sample_times, right_pressures, right_fit,
                         output_dir / "right_foot_pressure.mp4", "Right", fps, flip_z=True)
    
    print(f"\nDone! Videos saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate plantar pressure heatmap videos')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--motion', default=DEFAULT_MOTION)
    parser.add_argument('--output-dir', '-o', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--geometry', '-g', default=DEFAULT_GEOMETRY)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--stiffness', type=float, default=50000.0)
    args = parser.parse_args()
    
    main(args.model, args.motion, args.output_dir, args.geometry, args.fps, args.stiffness)
