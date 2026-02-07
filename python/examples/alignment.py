"""
Point cloud alignment example using trueform

Demonstrates alignment between point clouds:
1. With correspondences (smoothed mesh -> same vertex count)
2. Without correspondences (shuffled points)
3. ICP refinement (Point-to-Point vs Point-to-Plane)
4. Different mesh resolutions

Usage:
    python alignment.py [mesh.stl]

Default mesh: dragon-500k.stl
"""

import sys
import os
import time
import numpy as np
import trueform as tf


def random_transformation_at(centroid: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create random rotation around centroid + translation."""
    # Random rotation matrix using QR decomposition
    A = np.random.randn(3, 3).astype(np.float32)
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    # Build: translate_back @ rotate @ translate_to_origin @ translate_far
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Q
    T[:3, 3] = centroid - Q @ centroid + translation
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply transformation matrix to points."""
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    homogeneous = np.hstack([points, ones])
    transformed = (T @ homogeneous.T).T
    return transformed[:, :3].astype(points.dtype)


def compute_rms_error(A: np.ndarray, B: np.ndarray) -> float:
    """Compute RMS error between corresponding points."""
    diff = A - B
    return np.sqrt(np.mean(np.sum(diff * diff, axis=1)))


def compute_max_error(A: np.ndarray, B: np.ndarray) -> float:
    """Compute max error between corresponding points."""
    diff = A - B
    return np.sqrt(np.max(np.sum(diff * diff, axis=1)))


def main():
    # Default data directory
    data_dir = os.path.join(os.path.dirname(__file__), '../../benchmarks/data/')

    # Parse command line arguments
    if len(sys.argv) >= 2:
        mesh_path = sys.argv[1]
    else:
        mesh_path = os.path.join(data_dir, 'dragon-500k.stl')

    print(f"Loading mesh: {mesh_path}")

    # Read the mesh
    faces, points = tf.read_stl(mesh_path)
    if len(faces) == 0:
        print("Failed to load mesh or mesh is empty")
        return 1

    print(f"Loaded {len(faces)} triangles, {len(points)} vertices")

    # Compute AABB and diagonal
    aabb_min = points.min(axis=0)
    aabb_max = points.max(axis=0)
    diagonal = np.linalg.norm(aabb_max - aabb_min)
    print(f"AABB diagonal: {diagonal:.2f}")

    # Build vertex connectivity and create smoothed source mesh
    print("\nBuilding vertex link...")
    mesh = tf.Mesh(faces, points)
    vlink = mesh.vertex_link

    smooth_iters = 200
    smooth_lambda = 0.9
    print(f"Smoothing mesh ({smooth_iters} iterations, lambda={smooth_lambda})...")

    smoothed_points = tf.laplacian_smoothed(mesh, iterations=smooth_iters, lambda_=smooth_lambda)

    smooth_rms = compute_rms_error(points, smoothed_points)
    print(f"Smoothing RMS displacement: {smooth_rms:.6f} ({100.0 * smooth_rms / diagonal:.2f}% of diagonal)")

    # Build tree on target for OBB disambiguation and ICP
    target_cloud = tf.PointCloud(points)

    # Compute normals for point-to-plane ICP
    print("Computing point normals...")
    target_normals = tf.point_normals(mesh)

    # =========================================================================
    # Part 1: With correspondences (rigid transformation)
    # =========================================================================
    print("\n" + "=" * 60)
    print("=== PART 1: With correspondences ===")
    print("=" * 60)

    # Compute centroid of smoothed mesh
    centroid = smoothed_points.mean(axis=0)

    # Random rotation around centroid + large translation (2.5x diagonal away)
    far_translation = np.array([diagonal * 2.5, diagonal * -1.5, diagonal * 2.0], dtype=np.float32)
    T1 = random_transformation_at(centroid, far_translation)

    print("\nTransforming smoothed mesh (rotation around centroid + translation)")

    # Create transformed copy of smoothed points
    source1_pts = transform_points(smoothed_points, T1)

    initial1 = compute_max_error(points, source1_pts)
    print(f"Initial error: {initial1:.2f}")

    source1 = tf.PointCloud(source1_pts)
    baseline1 = tf.PointCloud(smoothed_points)

    print("\nRigid alignment:")
    T_rigid1 = tf.fit_rigid_alignment(source1, baseline1)
    source1.transformation = T_rigid1
    rigid1_pts = transform_points(source1_pts, T_rigid1)
    rigid1_rms = compute_rms_error(points, rigid1_pts)
    print(f"  RMS error: {rigid1_rms:.6f}")
    source1.transformation = None

    print("\nOBB alignment (no tree):")
    baseline_cloud = tf.PointCloud(points)
    T_obb1_no_tree = tf.fit_obb_alignment(source1, baseline_cloud, sample_size=0)
    obb1_no_tree_pts = transform_points(source1_pts, T_obb1_no_tree)
    obb1_no_tree_rms = compute_rms_error(points, obb1_no_tree_pts)
    print(f"  RMS error: {obb1_no_tree_rms:.6f}")

    print("\nOBB alignment (with tree):")
    T_obb1_tree = tf.fit_obb_alignment(source1, target_cloud)
    obb1_tree_pts = transform_points(source1_pts, T_obb1_tree)
    obb1_tree_rms = compute_rms_error(points, obb1_tree_pts)
    print(f"  RMS error: {obb1_tree_rms:.6f}")

    print("\n--- Summary (Part 1) ---")
    print(f"  Ground truth:    {smooth_rms:.6f}")
    print(f"  Rigid:           {rigid1_rms:.6f}")
    print(f"  OBB (no tree):   {obb1_no_tree_rms:.6f}")
    print(f"  OBB (with tree): {obb1_tree_rms:.6f}")

    # =========================================================================
    # Part 2: Without correspondences (shuffled source)
    # =========================================================================
    print("\n" + "=" * 60)
    print("=== PART 2: Without correspondences (shuffled) ===")
    print("=" * 60)

    # Create shuffled indices
    shuffle_ids = np.random.permutation(len(source1_pts))
    source2_pts = source1_pts[shuffle_ids]
    target_shuffled = points[shuffle_ids]

    source2 = tf.PointCloud(source2_pts)

    print("\nRigid alignment (will fail - no correspondences):")
    T_rigid2 = tf.fit_rigid_alignment(source2, baseline_cloud)
    rigid2_pts = transform_points(source2_pts, T_rigid2)
    rigid2_rms = compute_rms_error(target_shuffled, rigid2_pts)
    print(f"  RMS error: {rigid2_rms:.6f}")

    print("\nOBB alignment (no tree - ambiguous):")
    T_obb2_no_tree = tf.fit_obb_alignment(source2, baseline_cloud, sample_size=0)
    obb2_no_tree_pts = transform_points(source2_pts, T_obb2_no_tree)
    obb2_no_tree_rms = compute_rms_error(target_shuffled, obb2_no_tree_pts)
    print(f"  RMS error: {obb2_no_tree_rms:.6f}")

    print("\nOBB alignment (with tree - disambiguated):")
    T_obb2_tree = tf.fit_obb_alignment(source2, target_cloud)
    obb2_tree_pts = transform_points(source2_pts, T_obb2_tree)
    obb2_tree_rms = compute_rms_error(target_shuffled, obb2_tree_pts)
    print(f"  RMS error: {obb2_tree_rms:.6f}")

    print("\n--- Summary (Part 2) ---")
    print(f"  Ground truth:    {smooth_rms:.6f}")
    print(f"  Rigid:           {rigid2_rms:.6f} (FAILS)")
    print(f"  OBB (no tree):   {obb2_no_tree_rms:.6f} (may be wrong orientation)")
    print(f"  OBB (with tree): {obb2_tree_rms:.6f}")

    # =========================================================================
    # Part 3: ICP refinement - Point-to-Point vs Point-to-Plane
    # =========================================================================
    print("\n" + "=" * 60)
    print("=== PART 3: ICP refinement ===")
    print("=" * 60)

    print(f"Ground truth RMS: {smooth_rms:.6f}")
    print(f"Starting from OBB with tree: RMS = {obb2_tree_rms:.6f}")

    # ICP configuration
    max_iterations = 50
    n_samples = 1000
    k = 1
    min_relative_improvement = 1e-6

    print(f"Subsampling: ~{n_samples} / {len(source2_pts)} points per iteration")

    # Run Point-to-Point ICP (returns delta, compose with initial transform)
    print("\nPoint-to-Point ICP...")
    source2.transformation = T_obb2_tree
    t0 = time.perf_counter()
    T_p2p_delta = tf.fit_icp_alignment(
        source2, target_cloud,
        max_iterations=max_iterations,
        n_samples=n_samples,
        k=k,
        min_relative_improvement=min_relative_improvement
    )
    p2p_time = (time.perf_counter() - t0) * 1000
    T_p2p = T_p2p_delta @ T_obb2_tree  # total = delta @ init
    p2p_pts = transform_points(source2_pts, T_p2p)
    p2p_rms = compute_rms_error(target_shuffled, p2p_pts)
    print(f"  Final RMS: {p2p_rms:.6f}, time: {p2p_time:.1f} ms")

    # Run Point-to-Plane ICP (returns delta, compose with initial transform)
    print("\nPoint-to-Plane ICP...")
    source2.transformation = T_obb2_tree
    t0 = time.perf_counter()
    T_p2l_delta = tf.fit_icp_alignment(
        source2, (target_cloud, target_normals),
        max_iterations=max_iterations,
        n_samples=n_samples,
        k=k,
        min_relative_improvement=min_relative_improvement
    )
    p2l_time = (time.perf_counter() - t0) * 1000
    T_p2l = T_p2l_delta @ T_obb2_tree  # total = delta @ init
    p2l_pts = transform_points(source2_pts, T_p2l)
    p2l_rms = compute_rms_error(target_shuffled, p2l_pts)
    print(f"  Final RMS: {p2l_rms:.6f}, time: {p2l_time:.1f} ms")

    print("\n--- ICP Comparison ---")
    print(f"  Ground truth RMS:    {smooth_rms:.6f}")
    print(f"  Point-to-Point: RMS={p2p_rms:.6f}, time={p2p_time:.1f} ms")
    print(f"  Point-to-Plane: RMS={p2l_rms:.6f}, time={p2l_time:.1f} ms")
    if p2l_time < p2p_time:
        print(f"  Point-to-Plane was {p2p_time / p2l_time:.1f}x faster!")

    # =========================================================================
    # Part 4: Different mesh resolutions (no correspondences possible)
    # =========================================================================
    print("\n" + "=" * 60)
    print("=== PART 4: Different mesh resolutions ===")
    print("=" * 60)

    # Load a lower-resolution version of the mesh
    low_res_path = os.path.join(data_dir, 'dragon-50k.stl')
    print(f"\nLoading low-res mesh: {low_res_path}")

    try:
        faces_low, points_low = tf.read_stl(low_res_path)
        if len(faces_low) == 0:
            print("Failed to load low-res mesh, skipping Part 4")
            return 0
    except Exception:
        print("Failed to load low-res mesh, skipping Part 4")
        return 0

    print(f"High-res: {len(points)} vertices")
    print(f"Low-res:  {len(points_low)} vertices")

    # Build cloud on low-res mesh
    low_res_cloud = tf.PointCloud(points_low)

    # Baseline Chamfer: how different are the meshes due to resolution?
    chamfer_baseline_fwd = tf.chamfer_error(low_res_cloud, target_cloud)
    chamfer_baseline_bwd = tf.chamfer_error(target_cloud, low_res_cloud)
    print("\nBaseline Chamfer (aligned, different resolutions):")
    print(f"  Low->High: {chamfer_baseline_fwd:.6f}")
    print(f"  High->Low: {chamfer_baseline_bwd:.6f}")
    print(f"  Symmetric: {(chamfer_baseline_fwd + chamfer_baseline_bwd) / 2:.6f}")

    # Transform low-res mesh far away
    centroid_low = points_low.mean(axis=0)
    T_low = random_transformation_at(centroid_low, far_translation)

    source_low_pts = transform_points(points_low, T_low)
    source_low = tf.PointCloud(source_low_pts)

    # Initial Chamfer error (meshes far apart)
    chamfer_init_fwd = tf.chamfer_error(source_low, target_cloud)
    chamfer_init_bwd = tf.chamfer_error(target_cloud, source_low)
    print("\nInitial Chamfer error:")
    print(f"  Low->High: {chamfer_init_fwd:.2f}")
    print(f"  High->Low: {chamfer_init_bwd:.2f}")
    print(f"  Symmetric: {(chamfer_init_fwd + chamfer_init_bwd) / 2:.2f}")

    # OBB alignment (no tree)
    print("\nOBB alignment (no tree):")
    T_obb_low_no_tree = tf.fit_obb_alignment(source_low, baseline_cloud, sample_size=0)
    source_low.transformation = T_obb_low_no_tree
    chamfer_obb_no_tree = tf.chamfer_error(source_low, target_cloud)
    print(f"  Chamfer (Low->High): {chamfer_obb_no_tree:.6f}")
    source_low.transformation = None

    # OBB alignment (with tree)
    print("\nOBB alignment (with tree):")
    T_obb_low_tree = tf.fit_obb_alignment(source_low, target_cloud)
    source_low.transformation = T_obb_low_tree
    chamfer_obb_tree = tf.chamfer_error(source_low, target_cloud)
    print(f"  Chamfer (Low->High): {chamfer_obb_tree:.6f}")

    # ICP refinement - compare Point-to-Point vs Point-to-Plane
    print("\nICP refinement (comparing P2P vs P2L):")

    # Run Point-to-Point ICP (returns delta, compose with initial transform)
    print("\nPoint-to-Point ICP...")
    source_low.transformation = T_obb_low_tree
    t0 = time.perf_counter()
    T_p2p_low_delta = tf.fit_icp_alignment(
        source_low, target_cloud,
        max_iterations=max_iterations,
        n_samples=n_samples,
        k=k,
        min_relative_improvement=min_relative_improvement
    )
    p2p_time_low = (time.perf_counter() - t0) * 1000
    T_p2p_low = T_p2p_low_delta @ T_obb_low_tree  # total = delta @ init
    source_low.transformation = T_p2p_low
    p2p_chamfer_low = tf.chamfer_error(source_low, target_cloud)
    print(f"  Chamfer: {p2p_chamfer_low:.6f}, time: {p2p_time_low:.1f} ms")

    # Run Point-to-Plane ICP (returns delta, compose with initial transform)
    print("\nPoint-to-Plane ICP...")
    source_low.transformation = T_obb_low_tree
    t0 = time.perf_counter()
    T_p2l_low_delta = tf.fit_icp_alignment(
        source_low, (target_cloud, target_normals),
        max_iterations=max_iterations,
        n_samples=n_samples,
        k=k,
        min_relative_improvement=min_relative_improvement
    )
    p2l_time_low = (time.perf_counter() - t0) * 1000
    T_p2l_low = T_p2l_low_delta @ T_obb_low_tree  # total = delta @ init
    source_low.transformation = T_p2l_low
    p2l_chamfer_low = tf.chamfer_error(source_low, target_cloud)
    print(f"  Chamfer: {p2l_chamfer_low:.6f}, time: {p2l_time_low:.1f} ms")

    print("\n--- Summary (Part 4) ---")
    print(f"  Baseline:        {chamfer_baseline_fwd:.6f} (best possible)")
    print(f"  Initial:         {chamfer_init_fwd:.2f} (after transformation)")
    print(f"  OBB (no tree):   {chamfer_obb_no_tree:.6f}")
    print(f"  OBB (with tree): {chamfer_obb_tree:.6f}")
    print(f"  P2P ICP: Chamfer={p2p_chamfer_low:.6f}, time={p2p_time_low:.1f} ms")
    print(f"  P2L ICP: Chamfer={p2l_chamfer_low:.6f}, time={p2l_time_low:.1f} ms")
    if p2l_time_low < p2p_time_low:
        print(f"  Point-to-Plane was {p2p_time_low / p2l_time_low:.1f}x faster!")

    return 0


if __name__ == "__main__":
    main()
