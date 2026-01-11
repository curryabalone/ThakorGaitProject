"""
Data loading and preprocessing for neural contact force estimation.
"""

import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class MotionData:
    """Container for motion capture data."""
    times: np.ndarray           # (n_frames,) time stamps
    joint_names: List[str]      # joint/coordinate names
    joint_angles: np.ndarray    # (n_frames, n_joints) joint angles in radians
    in_degrees: bool            # original format (for reference)
    

def load_mot_file(mot_path: str) -> MotionData:
    """
    Parse an OpenSim .mot file.
    
    Args:
        mot_path: path to .mot file
        
    Returns:
        MotionData with times, joint names, and angles (converted to radians)
    """
    with open(mot_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    in_degrees = True
    header_end = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('inDegrees='):
            in_degrees = line.split('=')[1].lower() == 'yes'
        elif line == 'endheader':
            header_end = i + 1
            break
    
    # Parse column names
    joint_names = lines[header_end].strip().split('\t')[1:]  # Skip 'time'
    
    # Parse data
    data = []
    for line in lines[header_end + 1:]:
        if line.strip():
            values = [float(v) for v in line.strip().split('\t')]
            data.append(values)
    
    data = np.array(data)
    times = data[:, 0]
    joint_angles = data[:, 1:]
    
    # Convert to radians if needed
    if in_degrees:
        joint_angles = np.deg2rad(joint_angles)
    
    return MotionData(
        times=times,
        joint_names=joint_names,
        joint_angles=joint_angles,
        in_degrees=in_degrees,
    )


def extract_foot_pose(
    motion_data: MotionData,
    foot: str = 'right',
    include_pelvis: bool = True,
) -> np.ndarray:
    """
    Extract foot-relevant pose from full body motion.
    
    Args:
        motion_data: full body motion data
        foot: 'left' or 'right'
        include_pelvis: whether to include pelvis position/orientation
        
    Returns:
        foot_pose: (n_frames, pose_dim) array
    """
    suffix = '_r' if foot == 'right' else '_l'
    
    # Joints relevant to foot pose
    foot_joints = [
        f'hip_flexion{suffix}',
        f'hip_adduction{suffix}',
        f'hip_rotation{suffix}',
        f'knee_angle{suffix}',
        f'ankle_angle{suffix}',
        f'subtalar_angle{suffix}',
        f'mtp_angle{suffix}',
    ]
    
    pelvis_joints = [
        'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
        'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
    ]
    
    # Find matching columns
    selected_joints = pelvis_joints + foot_joints if include_pelvis else foot_joints
    indices = []
    found_names = []
    
    for joint in selected_joints:
        if joint in motion_data.joint_names:
            indices.append(motion_data.joint_names.index(joint))
            found_names.append(joint)
    
    if not indices:
        raise ValueError(f"No matching joints found for {foot} foot")
    
    return motion_data.joint_angles[:, indices], found_names


def create_pose_sequences(
    foot_pose: np.ndarray,
    history_length: int = 10,
    stride: int = 1,
) -> np.ndarray:
    """
    Create overlapping sequences for training.
    
    Args:
        foot_pose: (n_frames, pose_dim) foot pose data
        history_length: number of frames per sequence
        stride: step between sequences
        
    Returns:
        sequences: (n_sequences, history_length, pose_dim)
    """
    n_frames = foot_pose.shape[0]
    sequences = []
    
    for start in range(0, n_frames - history_length + 1, stride):
        seq = foot_pose[start:start + history_length]
        sequences.append(seq)
    
    return np.stack(sequences, axis=0)



class MotionDataset:
    """Dataset for training contact force model."""
    
    def __init__(
        self,
        mot_files: List[str],
        history_length: int = 10,
        stride: int = 1,
        foot: str = 'right',
    ):
        """
        Args:
            mot_files: list of paths to .mot files
            history_length: frames of history for each sample
            stride: step between samples
            foot: 'left' or 'right'
        """
        self.history_length = history_length
        self.stride = stride
        self.foot = foot
        
        # Load and process all motion files
        all_sequences = []
        all_times = []
        
        for mot_path in mot_files:
            motion_data = load_mot_file(mot_path)
            foot_pose, joint_names = extract_foot_pose(motion_data, foot=foot)
            
            sequences = create_pose_sequences(
                foot_pose, 
                history_length=history_length,
                stride=stride,
            )
            all_sequences.append(sequences)
            
            # Store time info for each sequence
            for start in range(0, len(motion_data.times) - history_length + 1, stride):
                all_times.append(motion_data.times[start + history_length - 1])
        
        self.sequences = np.concatenate(all_sequences, axis=0)
        self.times = np.array(all_times)
        self.joint_names = joint_names
        self.pose_dim = self.sequences.shape[-1]
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        return self.sequences[idx]
    
    def get_batch(self, indices: np.ndarray) -> jnp.ndarray:
        """Get a batch of sequences as JAX array."""
        return jnp.array(self.sequences[indices])
    
    def random_batch(self, batch_size: int, rng: np.random.Generator) -> jnp.ndarray:
        """Get a random batch of sequences."""
        indices = rng.choice(len(self), size=batch_size, replace=False)
        return self.get_batch(indices)


def load_sphere_geometry(xml_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load contact sphere positions and radii from MuJoCo XML.
    
    Args:
        xml_path: path to MuJoCo model XML
        
    Returns:
        positions: (num_spheres, 3) sphere center positions
        radii: (num_spheres,) sphere radii
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    positions = []
    radii = []
    
    # Find all sphere geoms that are contact spheres
    # Convention: contact spheres have class="contact" or name contains "contact"
    for geom in root.iter('geom'):
        geom_type = geom.get('type', 'sphere')
        geom_class = geom.get('class', '')
        geom_name = geom.get('name', '')
        
        if geom_type == 'sphere' and ('contact' in geom_class.lower() or 'contact' in geom_name.lower()):
            pos_str = geom.get('pos', '0 0 0')
            pos = [float(x) for x in pos_str.split()]
            positions.append(pos)
            
            size_str = geom.get('size', '0.01')
            radius = float(size_str.split()[0])
            radii.append(radius)
    
    if not positions:
        # Fallback: get all spheres from foot bodies
        foot_bodies = ['calcn_r', 'calcn_l', 'toes_r', 'toes_l', 
                       'talus_r', 'talus_l', 'foot_r', 'foot_l']
        
        for body in root.iter('body'):
            body_name = body.get('name', '').lower()
            if any(fb in body_name for fb in foot_bodies):
                for geom in body.findall('geom'):
                    if geom.get('type', 'sphere') == 'sphere':
                        pos_str = geom.get('pos', '0 0 0')
                        pos = [float(x) for x in pos_str.split()]
                        positions.append(pos)
                        
                        size_str = geom.get('size', '0.01')
                        radius = float(size_str.split()[0])
                        radii.append(radius)
    
    return np.array(positions), np.array(radii)


def estimate_gait_phase(foot_pose: np.ndarray, times: np.ndarray) -> np.ndarray:
    """
    Estimate gait phase (0-1) from foot kinematics.
    
    Uses heel height and velocity to detect heel strike (phase=0).
    
    Args:
        foot_pose: (n_frames, pose_dim) foot pose
        times: (n_frames,) time stamps
        
    Returns:
        phase: (n_frames,) gait phase 0-1
    """
    # Simple heuristic: use pelvis_ty (vertical position) if available
    # or ankle angle to estimate phase
    
    # For now, return linear phase (will be refined with actual gait detection)
    n_frames = len(times)
    
    # Detect approximate gait cycle from ankle angle oscillation
    # Assuming ankle_angle is in the pose
    ankle_idx = 4  # Typical index for ankle_angle after pelvis coords
    
    if foot_pose.shape[1] > ankle_idx:
        ankle = foot_pose[:, ankle_idx]
        
        # Find peaks (heel strikes) using simple peak detection
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(-ankle, distance=20)  # Heel strike = ankle dorsiflexion
        
        if len(peaks) >= 2:
            # Interpolate phase between peaks
            phase = np.zeros(n_frames)
            for i in range(len(peaks) - 1):
                start, end = peaks[i], peaks[i + 1]
                phase[start:end] = np.linspace(0, 1, end - start, endpoint=False)
            
            # Handle edges
            if peaks[0] > 0:
                phase[:peaks[0]] = np.linspace(0, 0, peaks[0])
            if peaks[-1] < n_frames - 1:
                phase[peaks[-1]:] = np.linspace(0, 1, n_frames - peaks[-1])
            
            return phase
    
    # Fallback: linear phase
    return np.linspace(0, 1, n_frames)
