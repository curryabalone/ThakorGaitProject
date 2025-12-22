#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline to convert OpenSim (.osim) files to MuJoCo format using myoconverter.

This script converts the LaiUhlrich2022_scaled.osim model from Session1.
"""

import sys
import os

# Add myoconverter to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'myoconverter'))

from myoconverter.O2MPipeline import O2MPipeline

# Define pipeline configurations
kwargs = {}
kwargs['convert_steps'] = [1]           # Step 1 only: XML/geometry conversion (no muscle optimization)
kwargs['muscle_list'] = None           # No specific muscle selected, optimize all of them
kwargs['osim_data_overwrite'] = True   # Overwrite the Osim model state files
kwargs['conversion'] = True            # Perform conversion process
kwargs['validation'] = True            # Perform validation process
kwargs['speedy'] = False               # Do not reduce checking nodes (more accurate)
kwargs['generate_pdf'] = False         # Skip PDF report (avoids library compatibility issues)
kwargs['add_ground_geom'] = True       # Add ground plane to the model
kwargs['treat_as_normal_path_point'] = True   # Treat complex path points as normal (simpler conversion)

# Input paths
osim_file = '../Session1/OpenSimData/Model/LaiUhlrich2022_scaled.osim'
geometry_folder = '../Session1/OpenSimData/Model/Geometry'

# Output folder for converted MuJoCo model
output_folder = '../Session1/OpenSimData/Model/MuJoCo_Output'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

if __name__ == '__main__':
    print(f"Converting OpenSim model: {osim_file}")
    print(f"Geometry folder: {geometry_folder}")
    print(f"Output folder: {output_folder}")
    print("-" * 50)
    
    O2MPipeline(osim_file, geometry_folder, output_folder, **kwargs)
    
    print("-" * 50)
    print("Conversion complete!")
