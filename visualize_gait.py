#!/usr/bin/env python3
"""
Visualize GaitDynamics skeleton with motion data.

Usage:
    python visualize_gait.py
    python visualize_gait.py --speed 0.5
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse

# Default paths for GaitDynamics
DEFAULT_MODEL = "GaitDynamics/output/example_opensim_model_cvt1.xml"
DEFAULT_MOTION = "GaitDynamics/example_mot_complete_kinematics.mot"


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


def visualize_motion(model_path, motion_path, speed=1.0):
    """Load model and visualize with motion data."""
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    print(f"Loading motion: {motion_path}")
    times, joint_names, joint_data, in_degrees = parse_mot_file(motion_path)
    print(f"  {len(times)} frames, duration: {times[-1] - times[0]:.2f}s")
    
    qpos_indices, mot_indices, scale_factors = map_joints(model, joint_names, in_degrees)
    print(f"  Mapped {len(qpos_indices)}/{len(joint_names)} joints")
    
    if not qpos_indices:
        print("ERROR: No joints mapped!")
        return
    
    print("\nControls: Space=Pause, R=Reset, Esc=Quit")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_time = times[0]
        last_time = time.time()
        
        while viewer.is_running():
            dt = time.time() - last_time
            last_time = time.time()
            
            sim_time += dt * speed
            if sim_time > times[-1]:
                sim_time = times[0]
            
            frame_idx = min(np.searchsorted(times, sim_time), len(times) - 1)
            
            for qpos_idx, mot_idx, scale in zip(qpos_indices, mot_indices, scale_factors):
                data.qpos[qpos_idx] = joint_data[frame_idx, mot_idx] * scale
            
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize gait motion on MuJoCo skeleton')
    parser.add_argument('--model', default=DEFAULT_MODEL, help='MuJoCo XML model path')
    parser.add_argument('--motion', default=DEFAULT_MOTION, help='.mot motion file path')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed')
    args = parser.parse_args()
    
    visualize_motion(args.model, args.motion, args.speed)
