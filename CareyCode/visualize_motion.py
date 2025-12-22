#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize a MuJoCo model with OpenSim motion (.mot) file data.

This script loads a converted MuJoCo model and plays back joint kinematics
from an OpenSim .mot file.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time


def parse_mot_file(mot_path):
    """
    Parse an OpenSim .mot file and return time and joint angle data.
    
    Returns:
        times: numpy array of time values
        joint_names: list of joint/coordinate names
        joint_data: numpy array of shape (n_frames, n_joints)
        in_degrees: bool, whether angles are in degrees
    """
    with open(mot_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    in_degrees = True
    header_end = 0
    n_rows = 0
    n_cols = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('nRows='):
            n_rows = int(line.split('=')[1])
        elif line.startswith('nColumns='):
            n_cols = int(line.split('=')[1])
        elif line.startswith('inDegrees='):
            in_degrees = line.split('=')[1].lower() == 'yes'
        elif line == 'endheader':
            header_end = i + 1
            break
    
    # Parse column names (first line after header)
    joint_names = lines[header_end].strip().split('\t')
    
    # Parse data
    data = []
    for line in lines[header_end + 1:]:
        if line.strip():
            values = [float(v) for v in line.strip().split('\t')]
            data.append(values)
    
    data = np.array(data)
    times = data[:, 0]  # First column is time
    joint_data = data[:, 1:]  # Rest are joint angles
    joint_names = joint_names[1:]  # Remove 'time' from names
    
    return times, joint_names, joint_data, in_degrees


def map_mot_to_mujoco(model, mot_joint_names, mot_data, in_degrees):
    """
    Map .mot joint data to MuJoCo qpos indices.
    
    Returns:
        qpos_indices: list of MuJoCo qpos indices for each mot joint
        mot_indices: list of mot data column indices that have matches
        scale_factors: conversion factors (deg to rad if needed)
    """
    qpos_indices = []
    mot_indices = []
    scale_factors = []
    
    # Get MuJoCo joint names
    mj_joint_names = [model.joint(i).name for i in range(model.njnt)]
    
    print(f"\nMuJoCo model has {model.njnt} joints:")
    for name in mj_joint_names:
        print(f"  - {name}")
    
    print(f"\n.mot file has {len(mot_joint_names)} coordinates:")
    for name in mot_joint_names:
        print(f"  - {name}")
    
    print("\nMapping joints...")
    
    for mot_idx, mot_name in enumerate(mot_joint_names):
        # Try exact match first
        if mot_name in mj_joint_names:
            jnt_idx = mj_joint_names.index(mot_name)
            qpos_idx = model.joint(jnt_idx).qposadr[0]
            qpos_indices.append(qpos_idx)
            mot_indices.append(mot_idx)
            scale_factors.append(np.pi / 180.0 if in_degrees else 1.0)
            print(f"  Matched: {mot_name} -> qpos[{qpos_idx}]")
        else:
            # Try common naming variations
            variations = [
                mot_name.replace('_', ''),
                mot_name.lower(),
                mot_name.replace('_r', '_right').replace('_l', '_left'),
            ]
            matched = False
            for var in variations:
                for mj_idx, mj_name in enumerate(mj_joint_names):
                    if var == mj_name.lower() or var == mj_name.replace('_', ''):
                        qpos_idx = model.joint(mj_idx).qposadr[0]
                        qpos_indices.append(qpos_idx)
                        mot_indices.append(mot_idx)
                        scale_factors.append(np.pi / 180.0 if in_degrees else 1.0)
                        print(f"  Matched: {mot_name} -> {mj_name} (qpos[{qpos_idx}])")
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                print(f"  No match: {mot_name}")
    
    return qpos_indices, mot_indices, scale_factors


def visualize_motion(mjcf_path, mot_path, playback_speed=1.0):
    """
    Load MuJoCo model and visualize with .mot motion data.
    
    Args:
        mjcf_path: Path to MuJoCo XML model file
        mot_path: Path to OpenSim .mot file
        playback_speed: Speed multiplier for playback (1.0 = realtime)
    """
    # Load MuJoCo model
    print(f"Loading MuJoCo model: {mjcf_path}")
    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)
    
    # Parse .mot file
    print(f"Loading motion data: {mot_path}")
    times, joint_names, joint_data, in_degrees = parse_mot_file(mot_path)
    print(f"  {len(times)} frames, {len(joint_names)} joints")
    print(f"  Duration: {times[-1] - times[0]:.2f} seconds")
    print(f"  Angles in degrees: {in_degrees}")
    
    # Map joints
    qpos_indices, mot_indices, scale_factors = map_mot_to_mujoco(
        model, joint_names, joint_data, in_degrees
    )
    
    if not qpos_indices:
        print("\nERROR: No joints could be mapped!")
        return
    
    print(f"\nSuccessfully mapped {len(qpos_indices)} joints")
    
    # Launch viewer
    print("\nLaunching viewer...")
    print("Controls:")
    print("  Space: Pause/Resume")
    print("  R: Reset to start")
    print("  Q/Esc: Quit")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        paused = False
        last_time = time.time()
        sim_time = times[0]
        
        while viewer.is_running():
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            if not paused:
                sim_time += dt * playback_speed
                
                # Loop back to start
                if sim_time > times[-1]:
                    sim_time = times[0]
                
                # Find closest frame
                frame_idx = np.searchsorted(times, sim_time)
                frame_idx = min(frame_idx, len(times) - 1)
                
                # Set joint positions
                for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
                    data.qpos[qpos_idx] = joint_data[frame_idx, mot_idx] * scale
                
                # Forward kinematics
                mujoco.mj_forward(model, data)
            
            viewer.sync()
            time.sleep(0.01)  # ~100 Hz update


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize MuJoCo model with .mot motion data')
    parser.add_argument('--model', type=str, 
                        default='../Session1/OpenSimData/Model/MuJoCo_Output/LaiUhlrich2022_scaled_cvt1.xml',
                        help='Path to MuJoCo XML model')
    parser.add_argument('--motion', type=str,
                        default='../Session1/OpenSimData/Kinematics/0_8_m_s.mot',
                        help='Path to .mot motion file')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier')
    
    args = parser.parse_args()
    
    visualize_motion(args.model, args.motion, args.speed)
