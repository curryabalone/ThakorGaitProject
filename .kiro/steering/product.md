# Thakor Gait Project

Biomechanics research project from Johns Hopkins University Thakor Lab focused on plantar pressure prediction.

## Core Capabilities

- **OpenSim to MuJoCo Conversion**: Convert musculoskeletal models from OpenSim (.osim) format to MuJoCo XML with optimized muscle kinematics and kinetics (via myoconverter)
- **Motion Visualization**: Playback OpenSim motion files (.mot) on converted MuJoCo models
- **Foot Mesh Fitting**: SMPL-X body model integration with foot mesh alignment for biomechanical analysis

## Key Workflows

1. Convert OpenSim models using the 3-step myoconverter pipeline (XML conversion → muscle kinematics optimization → muscle kinetics optimization)
2. Visualize gait motion data on converted models
3. Fit and align foot meshes from biomechanical models to SMPL-X body representations
