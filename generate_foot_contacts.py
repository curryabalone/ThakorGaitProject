import numpy as np
import fit_spheres

def generate_contact_xml_for_mesh(stl_path, body_name, color="0.8 0 0 1"):
    """
    Fits spheres to the mesh and returns a MuJoCo XML string with geom elements.
    """
    print(f"Processing {stl_path} for body {body_name}...")
    
    # 1. Run the sphere fitting algorithm
    try:
        # Use existing parameters from fit_spheres logic
        result = fit_spheres.fit_contact_spheres(stl_path, num_regions=240, radius_ratio=0.4)
    except Exception as e:
        print(f"Error fitting spheres for {stl_path}: {e}")
        return ""

    print(f"  Generated {result.num_spheres} spheres (r={result.sphere_radius*1000:.1f}mm)")
    
    # 2. Convert to XML geoms
    # Geoms should be attached to the foot body. 
    # Since fit_spheres works in global mesh coordinates, we need to be careful.
    # If the mesh in MuJoCo is applied to a body that moves, we generally want these 
    # spheres to be defined relative to that body.
    #
    # However, fit_spheres loads the STL directly. If the STL is 'pre-transformed' 
    # (i.e. the foot mesh is at (0,0,0)), then the sphere coordinates are local. 
    # If the STL is in global coords, we might have an offset issue.
    #
    # For this specific model (LaiUhlrich2022), the feet meshes are typically 
    # defined in a local frame or the body is positioned to match the mesh.
    # We will assume the spheres should be added as children of the foot body.
    
    xml_lines = []
    xml_lines.append(f'<!-- Contact spheres for {body_name} -->')
    
    for i, (cx, cz) in enumerate(result.cell_centers):
        # Result gives X, Z and a ground plane Y.
        # MuJoCo uses (x, y, z).
        # We need to map:
        #   fit_spheres X -> MuJoCo X
        #   fit_spheres Y (plane) -> MuJoCo Y
        #   fit_spheres Z -> MuJoCo Z
        
        pos_str = f"{cx:.5f} {result.y_plane:.5f} {cz:.5f}"
        
        # Create a geom
        # type="sphere"
        # size="{radius}"
        # pos="{x} {y} {z}"
        # condim="3" (for friction)
        # group="1" (visualize)
        geom = (f'<geom name="contact_{body_name}_{i}" '
                f'type="sphere" size="{result.sphere_radius:.5f}" '
                f'pos="{pos_str}" rgba="{color}" group="1" '
                f'contype="1" conaffinity="1" solimp="0.9 0.95 0.001 0.5 2" solref="0.02 1"/>')
        xml_lines.append(geom)
        
    return "\n".join(xml_lines)

if __name__ == "__main__":
    # Define paths based on project structure
    # Running from ThakorGaitProject root
    
    # Right Foot
    r_foot_stl = "GaitDynamics/output/Geometry/r_foot.stl"
    r_xml = generate_contact_xml_for_mesh(r_foot_stl, "calcn_r", color="0.9 0.2 0.2 0.5")
    
    # Left Foot
    # Assuming similiar path or mirrored
    l_foot_stl = "GaitDynamics/output/Geometry/l_foot.stl"  # Guessed path, need to verify
    l_xml = generate_contact_xml_for_mesh(l_foot_stl, "calcn_l", color="0.2 0.9 0.2 0.5")
    
    # Save to file
    with open("contacts_snippet.xml", "w") as f:
        f.write("<mujoco>\n")
        f.write("<!-- Paste these into the corresponding <body name='...'> -->\n\n")
        
        f.write(f"<!-- RIGHT FOOT ({r_foot_stl}) -->\n")
        f.write(r_xml)
        f.write("\n\n")
        
        f.write(f"<!-- LEFT FOOT ({l_foot_stl}) -->\n")
        f.write(l_xml)
        f.write("\n</mujoco>")
        
    print("\nSaved contact geoms to contacts_snippet.xml")
