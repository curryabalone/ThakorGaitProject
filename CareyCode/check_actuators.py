import mujoco
import numpy as np

model_path = '../Session1/OpenSimData/Model/MuJoCo_Output/LaiUhlrich2022_scaled_cvt1.xml'

try:
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    
    print(f"\nModel Stats:")
    print(f"  njoints: {model.njnt}")
    print(f"  nq: {model.nq}")
    print(f"  nv: {model.nv}")
    print(f"  nu (actuators): {model.nu}")
    
    print("\nActuators:")
    if model.nu == 0:
        print("  NO ACTUATORS FOUND! PD control via data.ctrl will not work.")
    else:
        for i in range(model.nu):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            joint_id = model.actuator_trnid[i, 0]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            print(f"  Actuator {i}: {name} -> Joint {joint_name} (ID {joint_id})")

except Exception as e:
    print(f"Error: {e}")
