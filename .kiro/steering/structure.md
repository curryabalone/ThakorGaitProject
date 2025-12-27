# Project Structure

```
├── CareyCode/                    # Main application code
│   ├── main.py                   # SMPL-X body model + foot mesh alignment
│   ├── visualize_motion.py       # MuJoCo motion playback from .mot files
│   ├── projection_utils.py       # Alignment optimization utilities
│   ├── myo.py                    # MuJoCo utilities
│   └── foot_fitting/             # Modular foot fitting package
│       ├── alignment.py          # Mesh alignment operations
│       ├── fitting.py            # ICP-based mesh fitting
│       ├── mesh_loader.py        # Mesh I/O operations
│       ├── foot_extractor.py     # Foot region extraction
│       ├── exporter.py           # Export utilities
│       ├── visualizer.py         # Visualization helpers
│       └── tests/                # Unit and property tests
│
├── myoconverter/                 # OpenSim → MuJoCo converter (submodule)
│   ├── myoconverter/             # Core conversion library
│   │   ├── O2MPipeline.py        # Main pipeline entry point
│   │   ├── conversion_steps/     # 3-step conversion process
│   │   ├── xml/                  # XML parsing and generation
│   │   │   ├── bodies/           # Body element converters
│   │   │   ├── joints/           # Joint converters
│   │   │   ├── forces/           # Muscle/force converters
│   │   │   └── path_points/      # Muscle path point handling
│   │   └── optimization/         # Muscle kinematics/kinetics optimization
│   ├── models/                   # Example models
│   │   ├── osim/                 # Source OpenSim models
│   │   └── mjc/                  # Converted MuJoCo models
│   └── examples/                 # Conversion example scripts
│
├── Session1/                     # Sample session data
│   ├── MarkerData/               # Motion capture marker data (.trc)
│   ├── OpenSimData/              # OpenSim outputs
│   │   ├── Model/                # Scaled model + MuJoCo conversion
│   │   └── Kinematics/           # Joint angle data (.mot)
│   ├── Videos/                   # Camera data and keypoints
│   └── CalibrationImages/        # Camera calibration
│
└── Gait2354Simbody/              # Gait model conversion workspace
```

## Key Patterns

- **Conversion outputs**: MuJoCo models saved to `<output>/` with `_cvt1.xml`, `_cvt2.xml`, `_cvt3.xml` suffixes for each pipeline step
- **Geometry files**: STL meshes in `Geometry/` subfolder alongside models
- **Session data**: Each session contains MarkerData, OpenSimData, and Videos folders
