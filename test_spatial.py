import numpy as np
from fit_spheres import FootGraph, SpatialRegularizer

def test_spatial_smoothing():
    print("Testing Spatial Regularization...")
    
    # 1. Create a 5x5 grid of points
    x = np.linspace(0, 0.1, 5)
    z = np.linspace(0, 0.1, 5)
    xx, zz = np.meshgrid(x, z)
    points = np.column_stack([xx.ravel(), zz.ravel()])
    
    # d_thresh should be slightly more than the spacing (0.025)
    d_thresh = 0.03
    graph = FootGraph(points, d_thresh)
    reg = SpatialRegularizer(graph, lambda_spatial=10.0, preserve_total_force=True)
    
    # 2. Add an impulse in the center (index 12)
    p_raw = {12: 100.0}
    p_smooth = reg.apply(p_raw)
    
    # 3. Verify total force preservation
    sum_raw = sum(p_raw.values())
    sum_smooth = sum(p_smooth.values())
    print(f"  Sum Raw: {sum_raw:.2f}, Sum Smooth: {sum_smooth:.2f}")
    assert abs(sum_raw - sum_smooth) < 1e-6, "Force preservation failed!"
    
    # 4. Verify diffusion
    print(f"  Impulse center (12) raw: {p_raw[12]:.2f}, smooth: {p_smooth[12]:.2f}")
    assert p_smooth[12] < 100.0, "Smoothing did not reduce peak value!"
    
    # Check a neighbor (index 13 or 7 or 11 or 17)
    neighbor_val = p_smooth[13]
    print(f"  Neighbor (13) smooth value: {neighbor_val:.2f}")
    assert neighbor_val > 0, "Smoothing did not diffuse to neighbor!"
    
    # Check a far point (index 0)
    far_val = p_smooth[0]
    print(f"  Far point (0) smooth value: {far_val:.2f}")
    assert far_val < neighbor_val, "Far point should have less pressure than neighbor!"

    print("  Tests passed!")

if __name__ == "__main__":
    test_spatial_smoothing()
