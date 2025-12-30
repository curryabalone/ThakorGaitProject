"""
Simple static viewer for the MuJoCo model with contact spheres.
"""

import mujoco
import mujoco.viewer


def view_model(xml_path: str):
    """Load and view a MuJoCo model in static/paused mode."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Set initial pose from keyframe if available
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # Forward kinematics only (no dynamics)
    mujoco.mj_forward(model, data)
    
    # Print some info about the foot mesh
    print("\nModel loaded successfully")
    
    # Launch viewer paused and keep it open
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Enable visibility for group 3 (contact spheres)
        viewer.opt.geomgroup[3] = True
        viewer.sync()
        
        while viewer.is_running():
            pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="View MuJoCo model with contact spheres")
    parser.add_argument(
        "xml_path",
        nargs="?",
        default="GaitDynamics/output/example_opensim_model_cvt1_contact.xml",
        help="Path to MuJoCo XML file",
    )
    args = parser.parse_args()
    
    view_model(args.xml_path)
