import os
import mujoco

def inject_contacts(original_xml, contacts_xml, output_xml):
    """
    Injects contact geoms into the foot bodies of the MuJoCo model.
    """
    print(f"Reading original model: {original_xml}")
    with open(original_xml, 'r') as f:
        xml_content = f.read()
    
    # We need to read the contacts snippet
    print(f"Reading contacts snippet: {contacts_xml}")
    if not os.path.exists(contacts_xml):
        print("Error: contacts snippet not found! Did generate_foot_contacts.py run?")
        return
        
    # In generate_foot_contacts.py, we created a file with comments identifying right/left foot
    # But for a robust injection we need to parse it or just blindly insert.
    # A robust way is tough with simple string manipulation if the snippet is complex.
    
    # Let's try a smarter find-and-replace strategy.
    # We know the body names are "calcn_r" and "calcn_l" (calcaneus/heel).
    
    # Load the generated snippet
    with open(contacts_xml, 'r') as f:
        snippet_lines = f.readlines()
        
    # Separate Right and Left foot geoms from the snippet
    # The snippet has comments <!-- RIGHT FOOT ... -->
    r_geoms = []
    l_geoms = []
    current_list = None
    
    for line in snippet_lines:
        if "RIGHT FOOT" in line:
            current_list = r_geoms
        elif "LEFT FOOT" in line:
            current_list = l_geoms
        elif "<geom" in line and current_list is not None:
            current_list.append(line.strip())

    print(f"Parsed {len(r_geoms)} right foot geoms and {len(l_geoms)} left foot geoms.")

    # Now inject into main XML
    # Find <body name="calcn_r" ... > ... </body>
    # We will insert before the closing </body> of that body.
    
    new_xml = xml_content
    
    if r_geoms:
        # Find end of calcn_r body
        # This is tricky with regex because bodies can be nested. 
        # But usually foot is a terminal body or close to it.
        # Let's search for the body tag start
        
        # Strategy: Use string replacement on a known unique marker inside the body if possible,
        # or just look for the first closing tag after the opening tag? No, that's risky.
        
        # Better Strategy: The model file is structured. 
        # Let's assume standard formatting. 
        
        # Actually, let's look for the body definition and insert right after the opening tag.
        # <body name="calcn_r" pos="...">
        #   <geom ... >
        #   <site ... >
        #   <-- INSERT HERE -->
        
        # Searching for 'name="calcn_r"'
        idx_r = new_xml.find('name="calcn_r"')
        if idx_r != -1:
            # Find the end of this opening tag ">"
            end_tag_r = new_xml.find('>', idx_r)
            if end_tag_r != -1:
                insertion_point = end_tag_r + 1
                # Insert
                block = "\n      <!-- Generated Contact Spheres -->\n      " + "\n      ".join(r_geoms) + "\n"
                new_xml = new_xml[:insertion_point] + block + new_xml[insertion_point:]
                print("Injected right foot contacts.")
        else:
             print("Warning: Could not find body 'calcn_r'")

    # Refind for left because indices shifted
    if l_geoms:
        idx_l = new_xml.find('name="calcn_l"')
        if idx_l != -1:
            end_tag_l = new_xml.find('>', idx_l)
            if end_tag_l != -1:
                insertion_point = end_tag_l + 1
                block = "\n      <!-- Generated Contact Spheres -->\n      " + "\n      ".join(l_geoms) + "\n"
                new_xml = new_xml[:insertion_point] + block + new_xml[insertion_point:]
                print("Injected left foot contacts.")
        else:
             print("Warning: Could not find body 'calcn_l'")

    print(f"Saving to {output_xml}")
    with open(output_xml, 'w') as f:
        f.write(new_xml)


if __name__ == "__main__":
    original = "Session1/OpenSimData/Model/MuJoCo_Output/LaiUhlrich2022_scaled_cvt1.xml"
    contacts = "contacts_snippet.xml"
    output = "Session1/OpenSimData/Model/MuJoCo_Output/LaiUhlrich2022_contact_sensors.xml"
    
    inject_contacts(original, contacts, output)
