# Requirements Document

## Introduction

This feature enables synthetic plantar pressure data generation by fitting a dense array of Hunt-Crossley contact spheres (~1000) to the foot mesh of a MuJoCo musculoskeletal model. During gait simulation, contact forces from these spheres are recorded and visualized as pressure maps, providing ground-truth data for plantar pressure prediction research.

## Glossary

- **Contact_Sphere_Array**: A collection of approximately 1000 small Hunt-Crossley contact spheres distributed across the plantar (bottom) surface of the foot mesh
- **Hunt_Crossley_Contact**: A nonlinear viscoelastic contact model in MuJoCo that computes contact forces based on penetration depth and velocity
- **Plantar_Surface**: The bottom surface of the foot that contacts the ground during gait
- **Pressure_Map**: A 2D visualization showing the spatial distribution of contact forces across the foot sole
- **Contact_Force_Data**: Time-series data containing force magnitude and location for each contact sphere during simulation
- **Foot_Mesh**: The 3D geometry representing the foot (calcn and toes bodies) in the MuJoCo model
- **Ground_Plane**: The flat surface representing the floor that the foot contacts during gait

## Requirements

### Requirement 1: Contact Sphere Placement

**User Story:** As a biomechanics researcher, I want contact spheres automatically distributed across the foot's plantar surface, so that I can capture spatially-resolved ground reaction forces.

#### Acceptance Criteria

1. WHEN the Contact_Sphere_Array is generated, THE System SHALL place spheres only on vertices of the Foot_Mesh that face downward (plantar surface)
2. WHEN distributing spheres, THE System SHALL achieve approximately uniform spacing across the Plantar_Surface
3. THE System SHALL generate between 800 and 1200 contact spheres per foot
4. WHEN a sphere position is computed, THE System SHALL attach it to the correct parent body (calcn or toes)
5. THE System SHALL configure each sphere with Hunt_Crossley_Contact parameters (stiffness, damping, friction)

### Requirement 2: MuJoCo Model Modification

**User Story:** As a developer, I want the contact spheres added to the existing MuJoCo XML model, so that I can run simulations with contact force sensing.

#### Acceptance Criteria

1. WHEN modifying the model, THE System SHALL preserve all existing model elements (bodies, joints, muscles, tendons)
2. WHEN adding contact spheres, THE System SHALL generate valid MuJoCo XML with proper geom elements
3. THE System SHALL define contact pairs between each sphere and the Ground_Plane
4. WHEN the modified model is loaded, THE MuJoCo_Loader SHALL parse it without errors
5. THE System SHALL output a new XML file without modifying the original model file

### Requirement 3: Contact Force Data Collection

**User Story:** As a researcher, I want to record contact forces from all spheres during simulation, so that I can analyze plantar pressure patterns.

#### Acceptance Criteria

1. WHEN simulation runs, THE System SHALL capture contact force magnitude for each active contact
2. WHEN a contact occurs, THE System SHALL record the sphere identifier, force vector, and contact location
3. THE System SHALL store Contact_Force_Data at each simulation timestep
4. WHEN simulation completes, THE System SHALL export data to a structured format (CSV or NumPy)
5. THE System SHALL synchronize force data timestamps with motion data timestamps

### Requirement 4: Pressure Map Visualization

**User Story:** As a researcher, I want to visualize plantar pressure as a heatmap, so that I can interpret the spatial distribution of ground reaction forces.

#### Acceptance Criteria

1. WHEN Contact_Force_Data is loaded, THE Visualizer SHALL render a 2D Pressure_Map of the foot outline
2. WHEN displaying pressure, THE Visualizer SHALL use a color gradient from low (blue) to high (red) force
3. THE Visualizer SHALL support frame-by-frame playback synchronized with motion data
4. WHEN a frame is displayed, THE Visualizer SHALL show the current timestamp and total vertical force
5. THE Visualizer SHALL allow export of Pressure_Map frames as images

### Requirement 5: Sphere Parameter Configuration

**User Story:** As a researcher, I want to configure contact sphere parameters, so that I can tune the contact model for realistic pressure simulation.

#### Acceptance Criteria

1. THE System SHALL allow configuration of sphere radius (default ~2-3mm for dense coverage)
2. THE System SHALL allow configuration of Hunt-Crossley stiffness coefficient
3. THE System SHALL allow configuration of Hunt-Crossley damping coefficient
4. THE System SHALL allow configuration of friction coefficients (static and dynamic)
5. WHEN parameters are changed, THE System SHALL regenerate the contact sphere array with new values

### Requirement 6: Bilateral Foot Support

**User Story:** As a researcher, I want contact spheres on both feet, so that I can capture complete gait cycle pressure data.

#### Acceptance Criteria

1. THE System SHALL generate Contact_Sphere_Array for both left and right feet
2. WHEN naming spheres, THE System SHALL use distinct prefixes for left (_l) and right (_r) feet
3. THE System SHALL maintain symmetric sphere placement between left and right feet
4. WHEN collecting data, THE System SHALL distinguish forces from left versus right foot contacts
