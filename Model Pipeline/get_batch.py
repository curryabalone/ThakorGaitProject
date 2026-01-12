import argparse
import numpy as np
import os

DEFAULT_MOTION = "../GaitDynamics/example_mot_complete_kinematics.mot"

# Global cache for loaded .mot files
# Maps absolute path -> numpy array
_MOT_CACHE = {}

def get_batch(mot_path, batch_size=10):
    """
    Get a random batch of frames from an OpenSim .mot file.
    Uses in-memory caching to avoid re-reading the file.
    """
    global _MOT_CACHE
    
    # Resolve absolute path to ensure cache hits match correctly
    abs_path = os.path.abspath(mot_path)
    
    if abs_path not in _MOT_CACHE:
        # Load and parse if not in cache
        print(f"Loading {mot_path} into cache...") # Optional logging
        with open(abs_path, 'r') as f:
            lines = f.readlines()
            
        header_end = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line == 'endheader':
                header_end = i + 1
                break
        
        data = []
        for line in lines[header_end + 1:]:
            if line.strip():
                data.append([float(v) for v in line.strip().split('\t')])
        
        # Store as numpy array
        _MOT_CACHE[abs_path] = np.array(data)
    
    # Retrieve from cache
    data_array = _MOT_CACHE[abs_path]
    num_frames = len(data_array)
    
    if num_frames <= batch_size:
        # If files are too short, just return what we have or loop? 
        # For now, let's just return what we have if it's too small, 
        # or error out if strict batch size is needed.
        # Returning available frames is safer for now.
        return data_array
    
    # Select random start index
    start_idx = np.random.randint(0, num_frames - batch_size + 1)
    
    return data_array[start_idx : start_idx + batch_size]


def parse_mot_file(mot_path):
    # Keeping this for backward compatibility if needed, 
    # but strictly speaking user asked for get_batch to do the work.
    # We can alias it or leave the old one?
    # The user specifically said "run get_batch to return at random 10 frames".
    # The previous main block called parse_mot_file, so I'll leave a simple runner.
    batch = get_batch(mot_path)
    print(f"Retrieved batch of shape: {batch.shape}")
    print(batch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', default=DEFAULT_MOTION)
    args = parser.parse_args()
    parse_mot_file(args.motion)
