"""
Modify MuJoCo XML to add contact spheres to foot bodies.

Uses the same sphere fitting algorithm as fit_spheres.py to generate
contact spheres for the calcn_r and calcn_l bodies.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from fit_spheres import fit_contact_spheres, ContactSphereResult


def add_ground_plane(worldbody: ET.Element, ground_height: float = 0.0) -> str:
    """Add a ground plane geom directly to worldbody. Returns the geom name.
    
    Args:
        worldbody: The worldbody XML element
        ground_height: Height of ground plane in MuJoCo world Z coords
    """
    # Add ground plane directly to worldbody (no body wrapper needed)
    # MuJoCo's default plane has normal +Z which is "up" in world frame
    ground_geom = ET.SubElement(worldbody, "geom")
    ground_geom.set("name", "ground_plane")
    ground_geom.set("type", "plane")
    ground_geom.set("size", "10 10 0.1")
    ground_geom.set("pos", f"0 0 {ground_height:.6f}")
    ground_geom.set("rgba", "0.8 0.8 0.8 1")
    ground_geom.set("contype", "1")
    ground_geom.set("conaffinity", "1")
    
    return "ground_plane"


def compute_ground_height_from_mujoco(xml_path: str, clearance: float = 0.001) -> float:
    """
    Load the model in MuJoCo and find the actual lowest contact sphere position.
    
    This accounts for keyframes and initial joint positions that affect the pose.
    
    Args:
        xml_path: Path to the MuJoCo XML with contact spheres already added
        clearance: Gap between sphere bottoms and ground
    
    Returns:
        Ground height in world Z coordinates
    """
    import mujoco
    
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    # Apply keyframe if exists (this sets the default pose)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    
    mujoco.mj_forward(m, d)
    
    # Find lowest contact sphere
    min_z = float('inf')
    sphere_radius = 0.0
    
    for i in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and 'contact_sphere' in name:
            z = d.geom_xpos[i][2]
            if z < min_z:
                min_z = z
                sphere_radius = m.geom_size[i][0]
    
    if min_z == float('inf'):
        print("Warning: No contact spheres found")
        return 0.0
    
    # Ground should be just below the sphere bottoms
    ground_height = min_z - sphere_radius - clearance
    
    print(f"Lowest sphere center Z: {min_z:.6f}")
    print(f"Sphere radius: {sphere_radius:.6f}")
    print(f"Sphere bottom Z: {min_z - sphere_radius:.6f}")
    print(f"Ground height (with {clearance*1000:.1f}mm clearance): {ground_height:.6f}")
    
    return ground_height


def add_contact_spheres_to_body(
    body: ET.Element,
    result: ContactSphereResult,
    prefix: str,
) -> list[str]:
    """
    Add contact sphere geoms to a body element.
    
    Args:
        body: The XML body element (calcn_r or calcn_l)
        result: ContactSphereResult from fit_contact_spheres
        prefix: Prefix for sphere names (e.g., 'r' or 'l')
    
    Returns:
        List of sphere geom names added
    """
    sphere_names = []
    
    for i, (cx, cz) in enumerate(result.cell_centers):
        geom = ET.SubElement(body, "geom")
        name = f"contact_sphere_{prefix}_{i:03d}"
        geom.set("name", name)
        geom.set("type", "sphere")
        # Position relative to body frame
        # The sphere centers are in the mesh coordinate frame
        geom.set("pos", f"{cx:.6f} {result.y_plane:.6f} {cz:.6f}")
        geom.set("size", f"{result.sphere_radius:.6f}")
        geom.set("rgba", "0 0.7 0.3 0.5")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("group", "3")  # Separate group for contact spheres
        
        sphere_names.append(name)
    
    return sphere_names


def add_contact_pairs(
    contact_elem: ET.Element,
    sphere_names: list[str],
    ground_name: str,
):
    """Add contact pair definitions between spheres and ground."""
    # Clear existing content but keep the element
    for child in list(contact_elem):
        contact_elem.remove(child)
    
    # Add comment
    comment = ET.Comment(" Contact pairs between foot spheres and ground ")
    contact_elem.append(comment)
    
    # Add contact pairs for each sphere
    for sphere_name in sphere_names:
        pair = ET.SubElement(contact_elem, "pair")
        pair.set("geom1", sphere_name)
        pair.set("geom2", ground_name)
        pair.set("condim", "3")
        pair.set("friction", "1 1 0.005")


def find_body_recursive(element: ET.Element, name: str) -> ET.Element | None:
    """Recursively find a body element by name."""
    if element.tag == "body" and element.get("name") == name:
        return element
    for child in element:
        result = find_body_recursive(child, name)
        if result is not None:
            return result
    return None


def get_mesh_scale(root: ET.Element, mesh_name: str) -> np.ndarray:
    """
    Get the scale factors for a mesh from the XML asset section.
    
    Args:
        root: XML root element
        mesh_name: Name pattern to search for (e.g., 'r_foot')
    
    Returns:
        numpy array of [scale_x, scale_y, scale_z], defaults to [1, 1, 1]
    """
    asset = root.find("asset")
    if asset is None:
        return np.array([1.0, 1.0, 1.0])
    
    for mesh in asset.findall("mesh"):
        file_attr = mesh.get("file", "")
        if mesh_name in file_attr:
            scale_str = mesh.get("scale", "1 1 1")
            scale = np.array([float(x) for x in scale_str.split()])
            return scale
    
    return np.array([1.0, 1.0, 1.0])


def add_contact_spheres_to_body_scaled(
    body: ET.Element,
    result: ContactSphereResult,
    prefix: str,
    scale: np.ndarray,
) -> list[str]:
    """
    Add contact sphere geoms to a body element with scaling applied.
    
    Args:
        body: The XML body element (calcn_r or calcn_l)
        result: ContactSphereResult from fit_contact_spheres
        prefix: Prefix for sphere names (e.g., 'r' or 'l')
        scale: Scale factors [scale_x, scale_y, scale_z] from mesh definition
    
    Returns:
        List of sphere geom names added
    """
    sphere_names = []
    
    # Scale the sphere radius (use average of X and Z scales for radius)
    scaled_radius = result.sphere_radius * (scale[0] + scale[2]) / 2
    
    for i, (cx, cz) in enumerate(result.cell_centers):
        geom = ET.SubElement(body, "geom")
        name = f"contact_sphere_{prefix}_{i:03d}"
        geom.set("name", name)
        geom.set("type", "sphere")
        
        # Apply scaling to positions
        # X -> scale[0], Y -> scale[1], Z -> scale[2]
        scaled_x = cx * scale[0]
        scaled_y = result.y_plane * scale[1]
        scaled_z = cz * scale[2]
        
        geom.set("pos", f"{scaled_x:.6f} {scaled_y:.6f} {scaled_z:.6f}")
        geom.set("size", f"{scaled_radius:.6f}")
        geom.set("rgba", "0 0.7 0.3 0.5")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("group", "3")  # Separate group for contact spheres
        
        sphere_names.append(name)
    
    return sphere_names



def modify_xml_with_contact_spheres(
    xml_path: str,
    output_path: str | None = None,
    geometry_folder: str | None = None,
    num_regions: int = 240,
    radius_ratio: float = 0.4,
    ground_clearance: float = 0.001,  # 1mm clearance for pressure sensing
) -> str:
    """
    Modify a MuJoCo XML file to add contact spheres to foot bodies.
    
    Args:
        xml_path: Path to input XML file
        output_path: Path for output XML (defaults to input with '_contact' suffix)
        geometry_folder: Folder containing foot STL files (defaults to Geometry/ next to XML)
        num_regions: Target number of contact spheres per foot
        radius_ratio: Sphere radius as fraction of cell size
        ground_clearance: Gap between sphere bottoms and ground (for pressure sensing)
    
    Returns:
        Path to the output XML file
    """
    xml_path = Path(xml_path)
    
    if output_path is None:
        output_path = xml_path.parent / f"{xml_path.stem}_contact.xml"
    else:
        output_path = Path(output_path)
    
    if geometry_folder is None:
        geometry_folder = xml_path.parent / "Geometry"
    else:
        geometry_folder = Path(geometry_folder)
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Find worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Could not find 'worldbody' element")
    
    all_sphere_names = []
    
    # Process right foot (calcn_r with r_foot.stl)
    r_foot_stl = geometry_folder / "r_foot.stl"
    if r_foot_stl.exists():
        calcn_r = find_body_recursive(worldbody, "calcn_r")
        if calcn_r is not None:
            result_r = fit_contact_spheres(str(r_foot_stl), num_regions, radius_ratio)
            scale_r = get_mesh_scale(root, "r_foot")
            print(f"Right foot scale: X={scale_r[0]:.4f}, Y={scale_r[1]:.4f}, Z={scale_r[2]:.4f}")
            
            sphere_names_r = add_contact_spheres_to_body_scaled(calcn_r, result_r, "r", scale_r)
            all_sphere_names.extend(sphere_names_r)
            print(f"Added {len(sphere_names_r)} contact spheres to calcn_r")
        else:
            print("Warning: Could not find 'calcn_r' body")
    else:
        print(f"Warning: Right foot STL not found: {r_foot_stl}")
    
    # Process left foot (calcn_l with l_foot.stl)
    l_foot_stl = geometry_folder / "l_foot.stl"
    if l_foot_stl.exists():
        calcn_l = find_body_recursive(worldbody, "calcn_l")
        if calcn_l is not None:
            result_l = fit_contact_spheres(str(l_foot_stl), num_regions, radius_ratio)
            scale_l = get_mesh_scale(root, "l_foot")
            print(f"Left foot scale: X={scale_l[0]:.4f}, Y={scale_l[1]:.4f}, Z={scale_l[2]:.4f}")
            
            sphere_names_l = add_contact_spheres_to_body_scaled(calcn_l, result_l, "l", scale_l)
            all_sphere_names.extend(sphere_names_l)
            print(f"Added {len(sphere_names_l)} contact spheres to calcn_l")
        else:
            print("Warning: Could not find 'calcn_l' body")
    else:
        print(f"Warning: Left foot STL not found: {l_foot_stl}")
    
    # Add temporary ground plane at 0 (will update after MuJoCo check)
    ground_name = add_ground_plane(worldbody, 0.0)
    
    # Find or create contact element
    contact_elem = root.find("contact")
    if contact_elem is None:
        contact_elem = ET.SubElement(root, "contact")
    
    # Add contact pairs
    add_contact_pairs(contact_elem, all_sphere_names, ground_name)
    print(f"Added {len(all_sphere_names)} contact pairs")
    
    # Update size element for more contacts
    size_elem = root.find("size")
    if size_elem is not None:
        current_nconmax = int(size_elem.get("nconmax", "400"))
        new_nconmax = max(current_nconmax, len(all_sphere_names) * 2 + 100)
        size_elem.set("nconmax", str(new_nconmax))
        print(f"Updated nconmax: {current_nconmax} -> {new_nconmax}")
    
    # Write temporary output
    tree.write(output_path, encoding="unicode", xml_declaration=False)
    
    # Now use MuJoCo to find the correct ground height (accounts for keyframes)
    print("\nComputing ground height from MuJoCo simulation...")
    ground_height = compute_ground_height_from_mujoco(str(output_path), ground_clearance)
    
    # Update ground plane position in the XML
    ground_geom = worldbody.find("geom[@name='ground_plane']")
    if ground_geom is not None:
        ground_geom.set("pos", f"0 0 {ground_height:.6f}")
    
    # Write final output
    tree.write(output_path, encoding="unicode", xml_declaration=False)
    
    # Pretty print
    with open(output_path, "r") as f:
        content = f.read()
    
    import xml.dom.minidom as minidom
    dom = minidom.parseString(content)
    pretty_xml = dom.toprettyxml(indent="  ")
    
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
    
    with open(output_path, "w") as f:
        f.write('\n'.join(lines))
    
    print(f"\nOutput written to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add contact spheres to MuJoCo XML foot bodies"
    )
    parser.add_argument(
        "xml_path",
        nargs="?",
        default="GaitDynamics/output/example_opensim_model_cvt1.xml",
        help="Path to input XML file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output XML path (default: input_contact.xml)",
    )
    parser.add_argument(
        "-g", "--geometry",
        help="Geometry folder path",
    )
    parser.add_argument(
        "-n", "--num-regions",
        type=int,
        default=240,
        help="Target number of contact spheres per foot (default: 240)",
    )
    parser.add_argument(
        "-r", "--radius-ratio",
        type=float,
        default=0.4,
        help="Sphere radius as fraction of cell size (default: 0.4)",
    )
    
    args = parser.parse_args()
    
    modify_xml_with_contact_spheres(
        args.xml_path,
        args.output,
        args.geometry,
        args.num_regions,
        args.radius_ratio,
    )
