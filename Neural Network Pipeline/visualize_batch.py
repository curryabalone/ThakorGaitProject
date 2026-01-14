#!/usr/bin/env python3
"""
Visualize motion batch from get_batch() using MuJoCo passive viewer.

This script loads a batch of frames and plays them back in the MuJoCo viewer
to verify the motion data is correct.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
from get_batch import get_batch


def get_joint_names_from_mot(mot_path):
    """Extract joint names from .mot file header."""
    with open(mot_path, 'r') as f:
        lines = f.readlines()
    
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == 'endheader':
            header_end = i + 1
            break
    
    # Column names are on the line after 'endheader'
    joint_names = lines[header_end].strip().split('\t')
    return joint_names[1:]  # Skip 'time' column


def map_mot_to_mujoco(model, mot_joint_names):
    """
    Map .mot joint names to MuJoCo qpos indices.
    
    Returns:
        qpos_indices: list of MuJoCo qpos indices
        mot_indices: list of mot data column indices (excluding time)
    """
    qpos_indices = []
    mot_indices = []
    
    mj_joint_names = [model.joint(i).name for i in range(model.njnt)]
    
    print(f"\nMapping {len(mot_joint_names)} joints...")
    
    for mot_idx, mot_name in enumerate(mot_joint_names):
        if mot_name in mj_joint_names:
            jnt_idx = mj_joint_names.index(mot_name)
            qpos_idx = model.joint(jnt_idx).qposadr[0]
            qpos_indices.append(qpos_idx)
            mot_indices.append(mot_idx)
            print(f"  ✓ {mot_name} -> qpos[{qpos_idx}]")
        else:
            print(f"  ✗ {mot_name} (no match)")
    
    print(f"Mapped {len(qpos_indices)}/{len(mot_joint_names)} joints")
    return qpos_indices, mot_indices


def visualize_batch(mjcf_path, mot_path, batch_size=10, playback_speed=1.0):
    """
    Visualize a random batch of frames from get_batch().
    
    Args:
        mjcf_path: Path to MuJoCo XML model
        mot_path: Path to .mot file
        batch_size: Number of frames to load
        playback_speed: Playback speed multiplier
    """
    # Load model
    print(f"Loading MuJoCo model: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    
    # Get batch
    print(f"\nLoading batch of {batch_size} frames from: {mot_path}")
    batch = get_batch(mot_path, batch_size)
    print(f"Batch shape: {batch.shape}")
    print(f"  Frames: {batch.shape[0]}")
    print(f"  Features: {batch.shape[1]} (time + {batch.shape[1]-1} joints)")
    
    # Extract time and joint data
    times = batch[:, 0]
    joint_data = batch[:, 1:]
    
    print(f"\nTime range: {times[0]:.3f}s to {times[-1]:.3f}s")
    print(f"Frame interval: {np.mean(np.diff(times)):.4f}s")
    
    # Get joint names and map to MuJoCo
    joint_names = get_joint_names_from_mot(mot_path)
    qpos_indices, mot_indices = map_mot_to_mujoco(model, joint_names)
    
    if not qpos_indices:
        print("\nERROR: No joints could be mapped!")
        return
    
    # Check if angles are in degrees (common for OpenSim)
    # Heuristic: if max absolute value > 2π, assume degrees
    in_degrees = np.abs(joint_data).max() > 2 * np.pi
    scale = np.pi / 180.0 if in_degrees else 1.0
    print(f"\nAngles in degrees: {in_degrees}")
    
    # Launch viewer
    print("\nLaunching viewer...")
    print("Controls:")
    print("  Space: Pause/Resume")
    print("  R: Reset to start")
    print("  Arrow keys: Step forward/backward")
    print("  Q/Esc: Quit")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        paused = False
        last_time = time.time()
        
        while viewer.is_running():
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            if not paused:
                # Auto-advance frames based on actual time intervals
                frame_idx += 1
                if frame_idx >= len(times):
                    frame_idx = 0  # Loop
            
            # Apply joint positions
            for qpos_idx, mot_idx in zip(qpos_indices, mot_indices):
                data.qpos[qpos_idx] = joint_data[frame_idx, mot_idx] * scale
            
            # Forward kinematics
            mujoco.mj_forward(model, data)
            
            # Update viewer
            viewer.sync()
            
            # Display frame info
            if frame_idx % 10 == 0 or paused:
                print(f"\rFrame {frame_idx+1}/{len(times)} | Time: {times[frame_idx]:.3f}s", end='')
            
            time.sleep(0.03 * playback_speed)  # ~30 Hz


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize random batch from get_batch() in MuJoCo viewer'
    )
    parser.add_argument('--model', type=str,
                        default='../Session1/OpenSimData/Model/MuJoCo_Output/LaiUhlrich2022_scaled_cvt1.xml',
                        help='Path to MuJoCo XML model')
    parser.add_argument('--motion', type=str,
                        default='../GaitDynamics/example_mot_complete_kinematics.mot',
                        help='Path to .mot motion file')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Number of frames to load')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier')
    
    args = parser.parse_args()
    
    visualize_batch(args.model, args.motion, args.batch_size, args.speed)
