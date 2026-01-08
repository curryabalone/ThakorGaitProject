# Thakor Gait Project
This is the repository to the Johns Hopkins University Thakor Lab Plantar Pressure Prediction Project. 

## Setup

### Prerequisites
- [Micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) (or Conda/Mamba)

### Create Environment

```bash
micromamba create -f myoconverter/conda_env.yml
micromamba activate myoconverter
```

This installs Python 3.10 with all required dependencies including:
- OpenSim 4.4.1
- MuJoCo 3.1.1
- NumPy, SciPy, Matplotlib
- Trimesh, PyVista for mesh processing

## Running the Motion Visualizer

The `visualize_motion.py` script plays back OpenSim motion (`.mot`) files on a MuJoCo model.

### Basic Usage

```bash
cd CareyCode
python visualize_motion.py
```

This runs with default paths pointing to the Session1 data.

### Custom Paths

```bash
python visualize_motion.py \
    --model ../Session1/OpenSimData/Model/MuJoCo_Output/LaiUhlrich2022_scaled_cvt1.xml \
    --motion ../Session1/OpenSimData/Kinematics/0_8_m_s.mot \
    --speed 1.0
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Path to MuJoCo XML model | `../Session1/OpenSimData/Model/MuJoCo_Output/LaiUhlrich2022_scaled_cvt1.xml` |
| `--motion` | Path to OpenSim .mot file | `../Session1/OpenSimData/Kinematics/0_8_m_s.mot` |
| `--speed` | Playback speed multiplier | `1.0` |

### Viewer Controls

- **Space**: Pause/Resume playback
- **R**: Reset to start
- **Q/Esc**: Quit

### Troubleshooting (Windows)

If you encounter a `TypeError` related to `os.PathLike` when importing `mujoco`, you may need to manually set the `CONDA_PREFIX` environment variable before running the script:

```powershell
# Set conda prefix (adjust path to your environment location if different)
$env:CONDA_PREFIX='C:\Users\jaymo\anaconda3\envs\myoconverter'

# Ensure you are in the CareyCode directory
cd CareyCode

# Run the script
python visualize_motion.py
```
