#!/usr/bin/env python3
"""
Generate comparison videos: raw vs spatially smoothed plantar pressure and CoP.

Creates 4 videos:
1. cop_visualization_raw.mp4 - Raw CoP (no spatial coupling)
2. cop_visualization_smoothed.mp4 - Smoothed CoP (with spatial coupling)
3. left/right_foot_pressure_raw.mp4 - Raw plantar pressure
4. left/right_foot_pressure_smoothed.mp4 - Smoothed plantar pressure
"""

from pathlib import Path
import sys

# Import the visualization functions
from visualize_cop import create_cop_video
from visualize_plantar_pressure import main as create_pressure_videos

# Default paths
MODEL = "GaitDynamics/output/example_opensim_model_cvt1_contact.xml"
MOTION = "GaitDynamics/example_mot_complete_kinematics.mot"
OUTPUT_DIR = "GaitDynamics/output"
GEOMETRY = "GaitDynamics/output/Geometry"

def main():
    print("Generating comparison videos: Raw vs Smoothed")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # 1. Generate RAW CoP visualization
    print("\n[1/4] Generating RAW CoP visualization...")
    create_cop_video(MODEL, MOTION, GEOMETRY, OUTPUT_DIR, 
                    fps=30, stiffness=50000.0, 
                    enable_spatial=False, lambda_spatial=0.1, 
                    output_suffix="_raw")
    
    # 2. Generate SMOOTHED CoP visualization
    print("\n[2/4] Generating SMOOTHED CoP visualization...")
    create_cop_video(MODEL, MOTION, GEOMETRY, OUTPUT_DIR, 
                    fps=30, stiffness=50000.0, 
                    enable_spatial=True, lambda_spatial=5.0, 
                    output_suffix="_smoothed")
    
    # 3. Generate RAW plantar pressure
    print("\n[3/4] Generating RAW plantar pressure...")
    create_pressure_videos(MODEL, MOTION, OUTPUT_DIR, GEOMETRY, 
                          fps=30, stiffness=50000.0,
                          enable_spatial=False, lambda_spatial=0.1,
                          output_suffix="_raw")
    
    # 4. Generate SMOOTHED plantar pressure
    print("\n[4/4] Generating SMOOTHED plantar pressure...")
    create_pressure_videos(MODEL, MOTION, OUTPUT_DIR, GEOMETRY, 
                          fps=30, stiffness=50000.0,
                          enable_spatial=True, lambda_spatial=5.0,
                          output_suffix="_smoothed")
    
    print("\n" + "="*80)
    print("✓ All videos generated successfully!")
    print("="*80)
    print("\nComparison files created:")
    output_path = Path(OUTPUT_DIR)
    print(f"  CoP (Raw):       {output_path / 'cop_visualization_raw.mp4'}")
    print(f"  CoP (Smoothed):  {output_path / 'cop_visualization_smoothed.mp4'}")
    print(f"  Pressure (Raw):  {output_path / 'left_foot_pressure_raw.mp4'}")
    print(f"                   {output_path / 'right_foot_pressure_raw.mp4'}")
    print(f"  Pressure (Smooth): {output_path / 'left_foot_pressure_smoothed.mp4'}")
    print(f"                     {output_path / 'right_foot_pressure_smoothed.mp4'}")

if __name__ == '__main__':
    main()
