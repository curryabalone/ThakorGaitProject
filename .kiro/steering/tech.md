# Tech Stack

## Language & Runtime
- Python 3.10 (required for OpenSim compatibility)

## Environment Management
- Conda/Micromamba with `myoconverter/conda_env.yml`

## Core Libraries

| Library | Purpose |
|---------|---------|
| OpenSim 4.4.1 | Musculoskeletal modeling and simulation |
| MuJoCo 3.1.1 | Physics simulation engine |
| NumPy, SciPy | Numerical computing |
| Trimesh | 3D mesh processing |
| Open3D | Point cloud and mesh operations |
| PyVista | 3D visualization |
| SMPL-X | Human body model |
| Loguru | Logging |
| Matplotlib, Seaborn | Plotting |

## Testing
- pytest
- hypothesis (property-based testing)

## Common Commands

```bash
# Create environment
micromamba create -f myoconverter/conda_env.yml
micromamba activate myoconverter

# Run motion visualizer
cd CareyCode
python visualize_motion.py --model <path.xml> --motion <path.mot>

# Run myoconverter pipeline (from myoconverter/)
python myoconverter/O2MPipeline.py <osim_file> <geometry_folder> <output_folder>

# Run tests
pytest CareyCode/foot_fitting/tests/
```

## Build Notes
- OpenSim requires conda-forge channel installation (not pip)
- MuJoCo installed via pip for ARM compatibility
- Export PYTHONPATH to include myoconverter folder when using as library
