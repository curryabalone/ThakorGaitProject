#!/usr/bin/env python3
"""
Simulate gait motion and visualize contact pressure distribution.

Uses kinematic playback from .mot file and computes pressure based on
sphere penetration depth into the ground plane.

Usage:
    python simulate_gait_forces.py
    python simulate_gait_forces.py --speed 0.5
    python simulate_gait_forces.py --output forces.csv
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
import csv
from pathlib import Path

# Default paths
DEFAULT_MODEL = "GaitDynamics/output/example_opensim_model_cvt1_contact.xml"
DEFAULT_MOTION = "GaitDynamics/example_mot_complete_kinematics.mot"
DEFAULT_OUTPUT = "GaitDynamics/output/contact_forces.csv"


def parse_mot_file(mot_path):
    """Parse OpenSim .mot file and return time/joint data."""
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
    times = data[:, 0]
    joint_data = data[:, 1:]
    joint_names = joint_names[1:]
    
    return times, joint_names, joint_data, in_degrees


def map_joints(model, mot_joint_names, in_degrees):
    """Map .mot joints to MuJoCo qpos indices."""
    mj_joint_names = [model.joint(i).name for i in range(model.njnt)]
    
    qpos_indices = []
    mot_indices = []
    scale_factors = []
    
    for mot_idx, mot_name in enumerate(mot_joint_names):
        if mot_name in mj_joint_names:
            jnt_idx = mj_joint_names.index(mot_name)
            qpos_idx = model.joint(jnt_idx).qposadr[0]
            qpos_indices.append(qpos_idx)
            mot_indices.append(mot_idx)
            scale_factors.append(np.pi / 180.0 if in_degrees else 1.0)
    
    return qpos_indices, mot_indices, scale_factors


def get_contact_sphere_info(model):
    """Find all contact sphere geom IDs, names, and radii."""
    spheres = {}
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name.startswith('contact_sphere_'):
            spheres[name] = {
                'id': i,
                'radius': model.geom_size[i, 0],
            }
    return spheres


def get_ground_height(model, data):
    """Get the Z position of the ground plane."""
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ground_plane')
    if ground_id >= 0:
        return data.geom_xpos[ground_id, 2]
    return 0.0


def compute_penetration_pressure(model, data, spheres, ground_z, stiffness=50000.0):
    """
    Compute pressure for each sphere based on penetration depth.
    
    Uses a simple spring model: force = stiffness * penetration_depth
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        spheres: Dict of sphere info from get_contact_sphere_info
        ground_z: Z height of ground plane
        stiffness: Spring stiffness (N/m) for pressure calculation
    
    Returns:
        Dict mapping sphere name -> pressure (force in N)
    """
    pressures = {}
    
    for name, info in spheres.items():
        geom_id = info['id']
        radius = info['radius']
        
        # Get sphere center position
        sphere_z = data.geom_xpos[geom_id, 2]
        sphere_bottom = sphere_z - radius
        
        # Compute penetration (positive = into ground)
        penetration = ground_z - sphere_bottom
        
        if penetration > 0:
            # Simple spring model for contact force
            force = stiffness * penetration
            pressures[name] = force
        else:
            pressures[name] = 0.0
    
    return pressures


def pressure_to_color(pressure, max_pressure):
    """
    Convert pressure to RGBA color (green -> yellow -> red).
    """
    if max_pressure <= 0 or pressure <= 0:
        return np.array([0.0, 0.7, 0.3, 0.5])  # Default green
    
    t = min(pressure / max_pressure, 1.0)
    
    if t < 0.5:
        r = t * 2
        g = 1.0
        b = 0.0
    else:
        r = 1.0
        g = 1.0 - (t - 0.5) * 2
        b = 0.0
    
    a = 0.5 + 0.5 * t
    
    return np.array([r, g, b, a])


def update_sphere_colors(model, spheres, pressures, max_pressure):
    """Update sphere colors based on pressure values."""
    for name, info in spheres.items():
        pressure = pressures.get(name, 0.0)
        color = pressure_to_color(pressure, max_pressure)
        model.geom_rgba[info['id']] = color


def simulate_with_pressure_viz(
    model_path: str,
    motion_path: str,
    output_path: str | None = None,
    speed: float = 1.0,
    max_pressure: float = 50.0,
    auto_scale: bool = True,
    stiffness: float = 50000.0,
):
    """
    Run kinematic playback with pressure visualization.
    
    Args:
        model_path: Path to MuJoCo XML with contact spheres
        motion_path: Path to .mot motion file
        output_path: Optional path for output CSV
        speed: Playback speed multiplier
        max_pressure: Maximum pressure for color scaling (N)
        auto_scale: Auto-adjust max based on observed pressures
        stiffness: Contact stiffness for pressure calculation (N/m)
    """
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"Loading motion: {motion_path}")
    times, joint_names, joint_data, in_degrees = parse_mot_file(motion_path)
    duration = times[-1] - times[0]
    print(f"  {len(times)} frames, duration: {duration:.2f}s")
    
    qpos_indices, mot_indices, scale_factors = map_joints(model, joint_names, in_degrees)
    print(f"  Mapped {len(qpos_indices)}/{len(joint_names)} joints")
    
    # Find contact spheres
    spheres = get_contact_sphere_info(model)
    sphere_names = sorted(spheres.keys())
    print(f"  Found {len(sphere_names)} contact spheres")
    
    if not spheres:
        print("ERROR: No contact spheres found!")
        return
    
    # Apply first frame of motion to get correct pose
    for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
        data.qpos[qpos_idx] = joint_data[0, mot_idx] * scale
    mujoco.mj_forward(model, data)
    
    # Find lowest sphere in this pose to set ground height
    min_sphere_z = float('inf')
    sphere_radius = 0.0
    for name, info in spheres.items():
        z = data.geom_xpos[info['id'], 2]
        if z < min_sphere_z:
            min_sphere_z = z
            sphere_radius = info['radius']
    
    # Set ground just at sphere bottoms (slight penetration for contact)
    ground_z = min_sphere_z - sphere_radius + 0.0005  # 0.5mm penetration
    print(f"  Computed ground Z: {ground_z:.4f} (from motion frame 0)")
    
    # Update ground plane position in model
    ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ground_plane')
    if ground_id >= 0:
        model.geom_pos[ground_id, 2] = ground_z
    
    # Prepare logging
    force_log = []
    header = ['time'] + [f'{name}_force' for name in sphere_names]
    
    observed_max = max_pressure
    
    print(f"\nStarting pressure visualization...")
    print(f"  Speed: {speed}x")
    print(f"  Stiffness: {stiffness} N/m")
    print(f"  Color scale: Green (0) -> Yellow -> Red (max)")
    print(f"\nClose window or Ctrl+C to stop.\n")
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    try:
        frame_idx = 0
        start_real_time = time.time()
        last_print_time = -1.0
        
        while viewer.is_running():
            # Get current motion time (no loop - stop at end)
            elapsed_real = time.time() - start_real_time
            sim_time = times[0] + elapsed_real * speed
            
            # Stop at end of motion
            if sim_time > times[-1]:
                print(f"\nPlayback complete at t={times[-1]:.2f}s")
                break
            
            # Find closest frame
            frame_idx = np.searchsorted(times, sim_time)
            frame_idx = min(frame_idx, len(times) - 1)
            
            # Set joint positions
            for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
                data.qpos[qpos_idx] = joint_data[frame_idx, mot_idx] * scale
            
            # Forward kinematics (no physics step)
            mujoco.mj_forward(model, data)
            
            # Compute penetration-based pressure
            pressures = compute_penetration_pressure(model, data, spheres, ground_z, stiffness)
            
            # Auto-scale
            current_max = max(pressures.values()) if pressures else 0
            if auto_scale and current_max > observed_max:
                observed_max = current_max * 1.2
            
            # Update colors
            update_sphere_colors(model, spheres, pressures, observed_max)
            
            # Log
            if output_path:
                row = [sim_time] + [pressures.get(name, 0.0) for name in sphere_names]
                force_log.append(row)
            
            viewer.sync()
            
            # Progress (every 0.5s of sim time)
            if int(sim_time * 2) != int(last_print_time * 2):
                active = sum(1 for p in pressures.values() if p > 0.1)
                total = sum(pressures.values())
                pct = 100 * (sim_time - times[0]) / duration
                print(f"t={sim_time:.2f}s ({pct:.0f}%) | Active: {active} | "
                      f"Total: {total:.1f}N | Max: {current_max:.1f}N")
                last_print_time = sim_time
            
            # Small sleep to not hog CPU
            time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\nStopped.")
    
    finally:
        viewer.close()
    
    # Save CSV
    if output_path and force_log:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nWriting {len(force_log)} samples to {output_path}")
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(force_log)
    
    print(f"\nDone. Max observed pressure: {observed_max:.1f}N")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize gait pressure distribution'
    )
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--motion', default=DEFAULT_MOTION)
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT, help='Output CSV path')
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--max-pressure', type=float, default=50.0)
    parser.add_argument('--stiffness', type=float, default=50000.0, help='Contact stiffness N/m')
    parser.add_argument('--no-auto-scale', action='store_true')
    args = parser.parse_args()
    
    simulate_with_pressure_viz(
        args.model,
        args.motion,
        args.output,
        args.speed,
        args.max_pressure,
        auto_scale=not args.no_auto_scale,
        stiffness=args.stiffness,
    )
