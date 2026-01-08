#!/usr/bin/env python3
"""
Simulate gait motion and log contact forces on foot spheres.

Runs physics simulation while tracking joint positions from .mot file,
and records contact forces experienced by each contact sphere.

Usage:
    python simulate_gait_forces.py
    python simulate_gait_forces.py --output forces.csv --visualize
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


def get_contact_sphere_geoms(model):
    """Find all contact sphere geom IDs and names."""
    sphere_geoms = {}
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name.startswith('contact_sphere_'):
            sphere_geoms[name] = i
    return sphere_geoms


def get_contact_forces(model, data, sphere_geom_ids):
    """
    Extract contact forces for each sphere from the contact data.
    
    Returns dict mapping geom_name -> force vector (fx, fy, fz)
    """
    forces = {name: np.zeros(3) for name in sphere_geom_ids}
    
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        # Check if either geom is a contact sphere
        for name, geom_id in sphere_geom_ids.items():
            if geom1_id == geom_id or geom2_id == geom_id:
                # Get contact force
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                # force[:3] is normal + tangent forces in contact frame
                # Convert to world frame using contact frame rotation
                frame = contact.frame.reshape(3, 3)
                world_force = frame.T @ force[:3]
                forces[name] += world_force
    
    return forces


def simulate_and_log_forces(
    model_path: str,
    motion_path: str,
    output_path: str,
    visualize: bool = False,
    speed: float = 1.0,
):
    """
    Run simulation and log contact forces to CSV.
    
    Args:
        model_path: Path to MuJoCo XML with contact spheres
        motion_path: Path to .mot motion file
        output_path: Path for output CSV
        visualize: Whether to show viewer
        speed: Playback speed multiplier
    """
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"Loading motion: {motion_path}")
    times, joint_names, joint_data, in_degrees = parse_mot_file(motion_path)
    print(f"  {len(times)} frames, duration: {times[-1] - times[0]:.2f}s")
    
    qpos_indices, mot_indices, scale_factors = map_joints(model, joint_names, in_degrees)
    print(f"  Mapped {len(qpos_indices)}/{len(joint_names)} joints")
    
    # Find contact spheres
    sphere_geoms = get_contact_sphere_geoms(model)
    sphere_names = sorted(sphere_geoms.keys())
    print(f"  Found {len(sphere_names)} contact spheres")
    
    if not sphere_names:
        print("ERROR: No contact spheres found!")
        return
    
    # Prepare output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # CSV header: time, then fx/fy/fz for each sphere
    header = ['time']
    for name in sphere_names:
        header.extend([f'{name}_fx', f'{name}_fy', f'{name}_fz'])
    
    force_log = []
    
    print(f"\nSimulating...")
    
    if visualize:
        viewer = mujoco.viewer.launch_passive(model, data)
    
    try:
        sim_time = times[0]
        start_real_time = time.time()
        frame_count = 0
        
        while sim_time <= times[-1]:
            # Find motion frame
            frame_idx = min(np.searchsorted(times, sim_time), len(times) - 1)
            
            # Set joint positions from motion data
            for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
                data.qpos[qpos_idx] = joint_data[frame_idx, mot_idx] * scale
            
            # Step physics
            mujoco.mj_step(model, data)
            
            # Get contact forces
            forces = get_contact_forces(model, data, sphere_geoms)
            
            # Log forces
            row = [sim_time]
            for name in sphere_names:
                row.extend(forces[name].tolist())
            force_log.append(row)
            
            # Update viewer if visualizing
            if visualize:
                if not viewer.is_running():
                    break
                viewer.sync()
                # Real-time pacing
                elapsed = time.time() - start_real_time
                target_time = (sim_time - times[0]) / speed
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)
            
            sim_time += model.opt.timestep
            frame_count += 1
            
            if frame_count % 500 == 0:
                print(f"  t={sim_time:.3f}s ({100*(sim_time-times[0])/(times[-1]-times[0]):.1f}%)")
    
    finally:
        if visualize:
            viewer.close()
    
    # Write CSV
    print(f"\nWriting {len(force_log)} samples to {output_path}")
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(force_log)
    
    # Summary stats
    force_data = np.array(force_log)
    total_force_magnitude = np.sqrt(
        force_data[:, 1::3]**2 + force_data[:, 2::3]**2 + force_data[:, 3::3]**2
    )
    max_force = total_force_magnitude.max()
    mean_force = total_force_magnitude[total_force_magnitude > 0].mean() if (total_force_magnitude > 0).any() else 0
    
    print(f"\nForce statistics:")
    print(f"  Max force magnitude: {max_force:.2f} N")
    print(f"  Mean non-zero force: {mean_force:.2f} N")
    print(f"  Spheres with contact: {(total_force_magnitude.max(axis=0) > 0).sum()}/{len(sphere_names)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate gait and log contact forces')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='MuJoCo XML model path')
    parser.add_argument('--motion', default=DEFAULT_MOTION, help='.mot motion file path')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT, help='Output CSV path')
    parser.add_argument('--visualize', '-v', action='store_true', help='Show viewer')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed')
    args = parser.parse_args()
    
    simulate_and_log_forces(
        args.model,
        args.motion,
        args.output,
        args.visualize,
        args.speed,
    )
