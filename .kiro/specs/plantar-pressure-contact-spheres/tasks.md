# Implementation Plan: Plantar Pressure Contact Spheres

## Overview

This implementation plan breaks down the plantar pressure contact sphere feature into incremental coding tasks. Each task builds on previous work, with property-based tests validating correctness at each stage.

## Tasks

- [x] 1. Create project structure and configuration
  - Create `CareyCode/plantar_pressure/` package directory
  - Create `__init__.py` with package exports
  - Create `config.py` with `SphereConfig` dataclass
  - Create `types.py` with `ContactSphere` and `ContactRecord` dataclasses
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 2. Implement PlantarExtractor component
  - [x] 2.1 Implement `PlantarExtractor` class with normal-based vertex filtering
    - Load mesh vertices and compute vertex normals from faces
    - Filter vertices where normal Y-component < threshold
    - Return filtered vertices and their original indices
    - _Requirements: 1.1_
  - [x] 2.2 Write unit tests for PlantarExtractor
    - Test normal computation for upward/downward triangles
    - Test plantar vertex filtering
    - Test error handling for empty inputs
    - _Requirements: 1.1_

- [-] 3. Implement SphereGenerator component
  - [x] 3.1 Implement Poisson disk sampling algorithm
    - Create spatial grid for efficient neighbor lookup
    - Implement dart-throwing with rejection sampling
    - Generate candidate points until target count reached
    - _Requirements: 1.2, 1.3_
  - [x] 3.2 Implement `SphereGenerator.generate_spheres()` method
    - Apply Poisson disk sampling to plantar vertices
    - Create `ContactSphere` objects with positions and parameters
    - Assign spheres to correct parent body (calcn vs toes)
    - Generate unique sphere names with side prefix
    - _Requirements: 1.4, 1.5, 6.1, 6.2_
  - [x] 3.3 Implement `SphereGenerator.generate_bilateral_spheres()` method
    - Generate spheres for both left and right feet
    - _Requirements: 6.1, 6.2_
  - [ ]* 3.4 Write property test for plantar surface filtering
    - **Property 1: Plantar Surface Filtering**
    - **Validates: Requirements 1.1**
  - [ ]* 3.5 Write property test for sphere count bounds
    - **Property 2: Sphere Count Bounds**
    - **Validates: Requirements 1.3**
  - [ ]* 3.6 Write property test for uniform spacing
    - **Property 3: Uniform Sphere Spacing**
    - **Validates: Requirements 1.2**
  - [ ]* 3.7 Write property test for body assignment
    - **Property 4: Body Assignment Correctness**
    - **Validates: Requirements 1.4**

- [x] 4. Checkpoint - Ensure sphere generation tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement ModelModifier component
  - [x] 5.1 Create `model_modifier.py` with `ModelModifier` class
    - Parse MuJoCo XML using ElementTree
    - Build body name to element mapping
    - Implement `get_body_transform()` for coordinate conversion
    - _Requirements: 2.1_
  - [x] 5.2 Implement `add_contact_spheres()` method
    - Generate geom XML elements for each sphere
    - Insert geoms into correct body elements
    - Apply contact parameters (solref, solimp, friction)
    - _Requirements: 2.2, 5.1, 5.2, 5.3, 5.4_
  - [x] 5.3 Implement `add_contact_pairs()` method
    - Find or create contact element in XML
    - Add pair elements for each sphere-ground combination
    - _Requirements: 2.3_
  - [x] 5.4 Implement `save()` method with original file protection
    - Write to new file path only
    - Verify original file unchanged
    - _Requirements: 2.5_
  - [ ]* 5.5 Write property test for configuration application
    - **Property 5: Configuration Application**
    - **Validates: Requirements 1.5, 5.1, 5.2, 5.3, 5.4**
  - [ ]* 5.6 Write property test for model element preservation
    - **Property 6: Model Element Preservation**
    - **Validates: Requirements 2.1**
  - [ ]* 5.7 Write property test for valid XML generation
    - **Property 7: Valid XML Generation**
    - **Validates: Requirements 2.2, 2.3, 2.4**
  - [ ]* 5.8 Write property test for original file immutability
    - **Property 8: Original File Immutability**
    - **Validates: Requirements 2.5**

- [x] 6. Checkpoint - Ensure model modification tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement ForceCollector component
  - [x] 7.1 Implement contact force extraction from MuJoCo data
    - Access `data.contact` array for active contacts
    - Map contact geom IDs to sphere names
    - Extract force vectors and contact positions
    - _Requirements: 3.1, 3.2_
  - [x] 7.2 Implement `collect_frame()` method
    - Create `ContactRecord` for each active contact
    - Include timestamp, sphere ID, foot side, forces
    - _Requirements: 3.2, 6.4_
  - [x] 7.3 Implement CSV and NumPy export methods
    - Write CSV with header row and all records
    - Write NumPy .npz with structured arrays
    - _Requirements: 3.4_
  - [ ]* 7.4 Write property test for contact data completeness
    - **Property 9: Contact Data Completeness**
    - **Validates: Requirements 3.1, 3.2**
  - [ ]* 7.5 Write property test for data export round-trip
    - **Property 11: Data Export Round-Trip**
    - **Validates: Requirements 3.4**

- [x] 8. Implement simulation integration
  - [x] 8.1 Create `run_simulation_with_contacts()` function
    - Load modified model and motion data
    - Step through simulation collecting forces each frame
    - Synchronize timestamps with motion data
    - _Requirements: 3.3, 3.5_
  - [ ]* 8.2 Write property test for timestep consistency
    - **Property 10: Timestep Data Consistency**
    - **Validates: Requirements 3.3**
  - [ ]* 8.3 Write property test for timestamp synchronization
    - **Property 12: Timestamp Synchronization**
    - **Validates: Requirements 3.5**

- [x] 9. Checkpoint - Ensure force collection tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Implement PressureVisualizer component
  - [x] 10.1 Implement foot outline and coordinate mapping
    - Create 2D foot outline from sphere positions
    - Map 3D sphere positions to 2D visualization coordinates
    - _Requirements: 4.1_
  - [x] 10.2 Implement `render_frame()` with color gradient
    - Create matplotlib figure with foot outline
    - Apply blue-to-red colormap based on force magnitude
    - Display timestamp and total vertical force
    - _Requirements: 4.2, 4.4_
  - [x] 10.3 Implement frame export and animation creation
    - Export individual frames as PNG images
    - Create video animation using matplotlib animation
    - _Requirements: 4.3, 4.5_
  - [ ]* 10.4 Write property test for displayed values
    - **Property: Displayed values match underlying data**
    - **Validates: Requirements 4.4**

- [x] 11. Implement bilateral foot support
  - [x] 11.1 Extend sphere generation for both feet
    - Process both left and right foot meshes
    - Apply mirroring for symmetric placement
    - _Requirements: 6.1, 6.3_
  - [ ]* 11.2 Write property test for bilateral generation
    - **Property 13: Bilateral Sphere Generation**
    - **Validates: Requirements 6.1, 6.2**
  - [ ]* 11.3 Write property test for bilateral symmetry
    - **Property 14: Bilateral Symmetry**
    - **Validates: Requirements 6.3**
  - [ ]* 11.4 Write property test for foot identification
    - **Property 15: Force Data Foot Identification**
    - **Validates: Requirements 6.4**

- [x] 12. Create CLI and integration script
  - [x] 12.1 Create `add_contact_spheres.py` CLI script
    - Parse command line arguments for model path, config
    - Load model, generate spheres, save modified model
    - _Requirements: 2.5, 5.5_
  - [x] 12.2 Create `collect_pressure_data.py` CLI script
    - Run simulation with contact collection
    - Export force data to specified format
    - _Requirements: 3.4_
  - [x] 12.3 Create `visualize_pressure.py` CLI script
    - Load force data and render pressure maps
    - Support frame export and animation
    - _Requirements: 4.3, 4.5_

- [x] 13. Final checkpoint - End-to-end integration test
  - Run full pipeline: add spheres → simulate → collect → visualize
  - Verify all components work together
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests use Hypothesis library with minimum 100 iterations
- Unit tests validate specific examples and edge cases
