"""
Point cloud alignment example using trueform

Demonstrates alignment between two meshes with different resolutions:
1. Rigid vs OBB alignment (with correspondences)
2. Shuffled points to show rigid alignment fails without correspondences
3. ICP refinement using k-NN alignment

Usage:
    python alignment.py [high_res.stl] [low_res.stl]

Default meshes: dragon-500k.stl and dragon-50k.stl
"""

import sys
import os
import numpy as np
import trueform as tf


def random_rotation_matrix_3d() -> np.ndarray:
    """Generate a random 3D rotation matrix using QR decomposition."""
    # Random matrix
    A = np.random.randn(3, 3).astype(np.float32)
    # QR decomposition gives orthogonal matrix
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = 1, not -1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_transformation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Create a 4x4 homogeneous transformation matrix."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply transformation matrix to points."""
    # points: (N, 3), T: (4, 4)
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    homogeneous = np.hstack([points, ones])  # (N, 4)
    transformed = (T @ homogeneous.T).T  # (N, 4)
    return transformed[:, :3].astype(points.dtype)


def compute_chamfer_symmetric(cloud0: tf.PointCloud, cloud1: tf.PointCloud) -> float:
    """Compute symmetric Chamfer distance."""
    fwd = tf.chamfer_error(cloud0, cloud1)
    bwd = tf.chamfer_error(cloud1, cloud0)
    return (fwd + bwd) / 2


def main():
    # Default data directory
    data_dir = os.path.join(os.path.dirname(
        __file__), '../../benchmarks/data/')

    # Parse command line arguments
    if len(sys.argv) >= 3:
        high_res_path = sys.argv[1]
        low_res_path = sys.argv[2]
    else:
        high_res_path = os.path.join(data_dir, 'dragon-500k.stl')
        low_res_path = os.path.join(data_dir, 'dragon-50k.stl')

    print("=" * 60)
    print("Point Cloud Alignment Example")
    print("=" * 60)

    # Load meshes (read_stl returns (faces, points) tuple)
    print(f"\nLoading high-res mesh: {high_res_path}")
    faces_high, points_high = tf.read_stl(high_res_path)
    print(f"  {len(points_high)} vertices, {len(faces_high)} triangles")

    print(f"\nLoading low-res mesh: {low_res_path}")
    faces_low, points_low = tf.read_stl(low_res_path)
    print(f"  {len(points_low)} vertices, {len(faces_low)} triangles")

    # Create point clouds from mesh vertices
    target_cloud = tf.PointCloud(points_high)
    source_pts_original = points_low.copy()

    # Compute AABB diagonal for scaling
    aabb_min = points_high.min(axis=0)
    aabb_max = points_high.max(axis=0)
    diagonal = np.linalg.norm(aabb_max - aabb_min)
    print(f"\nAABB diagonal: {diagonal:.2f}")

    # Compute centroid of source mesh
    centroid = source_pts_original.mean(axis=0)

    # Create random transformation: rotation around centroid + large translation
    R = random_rotation_matrix_3d()
    translation = np.array(
        [diagonal * 2.5, diagonal * -1.5, diagonal * 2.0], dtype=np.float32)

    # Build transformation: translate to origin, rotate, translate back, then translate far
    T_to_origin = make_transformation(np.eye(3, dtype=np.float32), -centroid)
    T_rotate = make_transformation(R, np.zeros(3, dtype=np.float32))
    T_from_origin = make_transformation(np.eye(3, dtype=np.float32), centroid)
    T_translate = make_transformation(np.eye(3, dtype=np.float32), translation)

    # Combined: T_translate @ T_from_origin @ T_rotate @ T_to_origin
    T_combined = T_translate @ T_from_origin @ T_rotate @ T_to_origin

    # Transform source points
    source_pts_transformed = transform_points(source_pts_original, T_combined)

    # =========================================================================
    # Part 1: Initial alignment with correspondences preserved
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 1: With correspondences (same point order)")
    print("=" * 60)

    source_cloud = tf.PointCloud(source_pts_transformed)

    # Baseline: how different are the meshes due to resolution?
    baseline_cloud = tf.PointCloud(source_pts_original)
    chamfer_baseline = tf.chamfer_error(baseline_cloud, target_cloud)
    print(
        f"\nBaseline Chamfer (resolution difference): {chamfer_baseline:.6f}")

    # Initial error (meshes far apart)
    chamfer_initial = tf.chamfer_error(source_cloud, target_cloud)
    print(f"Initial Chamfer (after transformation): {chamfer_initial:.2f}")

    # Rigid alignment (requires correspondences)
    print("\nRigid alignment (Kabsch):")
    T_rigid = tf.fit_rigid_alignment(source_cloud, baseline_cloud)
    source_cloud.transformation = T_rigid
    chamfer_rigid = tf.chamfer_error(source_cloud, target_cloud)
    print(f"  Chamfer error: {chamfer_rigid:.6f}")
    source_cloud.transformation = None  # Reset for next test

    # OBB alignment (correspondence-free, uses tree for disambiguation)
    print("\nOBB alignment:")
    T_obb = tf.fit_obb_alignment(source_cloud, target_cloud)
    source_cloud.transformation = T_obb
    chamfer_obb = tf.chamfer_error(source_cloud, target_cloud)
    print(f"  Chamfer error: {chamfer_obb:.6f}")

    print("\n--- Summary (Part 1) ---")
    print(f"  Baseline:        {chamfer_baseline:.6f} (best possible)")
    print(f"  Rigid:           {chamfer_rigid:.6f}")
    print(f"  OBB:             {chamfer_obb:.6f}")

    # =========================================================================
    # Part 2: Shuffled points (no correspondences)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 2: Without correspondences (shuffled)")
    print("=" * 60)

    # Shuffle the source points
    shuffle_ids = np.random.permutation(len(source_pts_transformed))
    source_pts_shuffled = source_pts_transformed[shuffle_ids]
    source_shuffled = tf.PointCloud(source_pts_shuffled)

    # Rigid alignment (will FAIL - no correspondences)
    # Fitting shuffled source against unshuffled baseline breaks correspondence
    print("\nRigid alignment (will FAIL - no correspondences):")
    T_rigid_shuffled = tf.fit_rigid_alignment(source_shuffled, baseline_cloud)
    source_shuffled.transformation = T_rigid_shuffled
    chamfer_rigid_shuffled = tf.chamfer_error(source_shuffled, target_cloud)
    print(f"  Chamfer error: {chamfer_rigid_shuffled:.6f} (FAILS)")
    source_shuffled.transformation = None

    # OBB alignment (correspondence-free - still works!)
    print("\nOBB alignment (correspondence-free - still works!):")
    T_obb_shuffled = tf.fit_obb_alignment(source_shuffled, target_cloud)
    source_shuffled.transformation = T_obb_shuffled
    chamfer_obb_shuffled = tf.chamfer_error(source_shuffled, target_cloud)
    print(f"  Chamfer error: {chamfer_obb_shuffled:.6f}")

    print("\n--- Summary (Part 2) ---")
    print(f"  Baseline:        {chamfer_baseline:.6f} (best possible)")
    print(f"  Rigid:           {chamfer_rigid_shuffled:.6f} (FAILS)")
    print(f"  OBB:             {chamfer_obb_shuffled:.6f}")

    # =========================================================================
    # Part 3: ICP refinement
    # =========================================================================
    print("\n" + "=" * 60)
    print("PART 3: ICP refinement")
    print("=" * 60)

    # Start from OBB result (using shuffled source)
    T_accum = T_obb_shuffled.copy()

    # ICP parameters
    max_iters = 50
    n_samples = 1000
    k = 5  # Number of nearest neighbors
    alpha = 0.3  # EMA smoothing factor
    rel_tol = 0.001  # Stop when < 0.1% relative improvement

    subsample_stride = max(1, len(source_pts_shuffled) // n_samples)
    print(f"\nStarting from OBB: Chamfer = {chamfer_obb_shuffled:.6f}")
    print(f"Baseline (best possible): {chamfer_baseline:.6f}")
    print(
        f"Subsampling: ~{n_samples} / {len(source_pts_shuffled)} points per iteration")
    print(f"Using k={k} nearest neighbors")

    ema = 0.0
    ema_prev = 0.0

    print("\nICP iterations:")
    for iter_num in range(max_iters):
        # Random subsample for this iteration
        offset = np.random.randint(0, subsample_stride)
        ids = np.arange(offset, len(source_pts_shuffled),
                        subsample_stride)[:n_samples]
        subsample_pts = source_pts_shuffled[ids]
        subsample_cloud = tf.PointCloud(subsample_pts)
        subsample_cloud.transformation = T_accum

        # Fit k-NN alignment
        T_iter = tf.fit_knn_alignment(subsample_cloud, target_cloud, k=k)

        # Accumulate transformation: T_accum = T_iter @ T_accum
        T_accum = T_iter @ T_accum

        # Evaluate Chamfer error on a different subset
        eval_offset = np.random.randint(0, subsample_stride)
        eval_ids = np.arange(eval_offset, len(
            source_pts_shuffled), subsample_stride)[:n_samples]
        eval_pts = source_pts_shuffled[eval_ids]
        eval_cloud = tf.PointCloud(eval_pts)
        eval_cloud.transformation = T_accum

        chamfer = tf.chamfer_error(eval_cloud, target_cloud)

        # EMA update
        ema_prev = ema
        ema = chamfer if iter_num == 0 else alpha * \
            chamfer + (1.0 - alpha) * ema
        rel_change = 1.0 if iter_num == 0 else (
            ema_prev - ema) / ema if ema > 0 else 0

        print(f"  iter {iter_num}: Chamfer = {chamfer:.6f} (EMA = {ema:.6f})")

        if iter_num > 0 and rel_change < rel_tol:
            break

    print(f"Converged after {iter_num + 1} iterations")

    # Final evaluation on full point cloud
    source_shuffled.transformation = T_accum
    chamfer_final = tf.chamfer_error(source_shuffled, target_cloud)

    print("\n--- Summary (Part 3) ---")
    print(f"  Baseline:        {chamfer_baseline:.6f} (best possible)")
    print(f"  OBB:             {chamfer_obb_shuffled:.6f}")
    print(f"  After ICP:       {chamfer_final:.6f}")

    # =========================================================================
    # Final Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(
        f"\nMeshes: {os.path.basename(high_res_path)} (target) vs {os.path.basename(low_res_path)} (source)")
    print(f"Baseline Chamfer (resolution difference): {chamfer_baseline:.6f}")
    print("\nAlignment pipeline on shuffled points:")
    print(f"  1. Initial (far apart):  {chamfer_initial:.2f}")
    print(f"  2. After OBB alignment:  {chamfer_obb_shuffled:.6f}")
    print(f"  3. After ICP refinement: {chamfer_final:.6f}")
    print(
        f"\nICP achieved {(chamfer_obb_shuffled - chamfer_final) / chamfer_obb_shuffled * 100:.1f}% improvement over OBB")


if __name__ == "__main__":
    main()
