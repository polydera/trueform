#include <trueform/trueform.hpp>

#include <cmath>
#include <iostream>
#include <random>
#include <string>

#ifndef TRUEFORM_DATA_DIR
#define TRUEFORM_DATA_DIR "."
#endif

int main(int argc, char *argv[]) {
  const std::string data_dir =
      std::string(TRUEFORM_DATA_DIR) + "/benchmarks/data/";

  // Default to the dragon mesh
  std::string mesh_path = data_dir + "dragon-500k.stl";
  if (argc > 1) {
    mesh_path = argv[1];
  }

  std::cout << "Loading mesh: " << mesh_path << std::endl;

  // Read the mesh
  auto mesh = tf::read_stl(mesh_path.c_str());
  if (mesh.polygons().size() == 0) {
    std::cerr << "Failed to load mesh or mesh is empty" << std::endl;
    return 1;
  }

  std::cout << "Loaded " << mesh.polygons().size() << " triangles, "
            << mesh.points().size() << " vertices" << std::endl;

  // Compute AABB and diagonal
  auto aabb = tf::aabb_from(mesh.polygons());
  float diagonal = tf::distance(aabb.min, aabb.max);
  std::cout << "AABB diagonal: " << diagonal << std::endl;

  // Build vertex connectivity and create smoothed source mesh
  std::cout << "\nBuilding vertex link..." << std::endl;
  tf::face_membership<int> fm(mesh.polygons());
  tf::vertex_link<int> vlink;
  vlink.build(mesh.polygons(), fm);

  constexpr std::size_t smooth_iters = 200;
  constexpr float smooth_lambda = 0.9f;
  std::cout << "Smoothing mesh (" << smooth_iters
            << " iterations, lambda=" << smooth_lambda << ")..." << std::endl;

  auto smoothed = tf::laplacian_smoothed(mesh.points() | tf::tag(vlink),
                                         smooth_iters, smooth_lambda);

  float smooth_rms = 0.0f;
  for (auto [a, b] : tf::zip(mesh.points(), smoothed.points())) {
    float d = tf::distance(a, b);
    smooth_rms += d * d;
  }
  smooth_rms = std::sqrt(smooth_rms / mesh.points().size());
  std::cout << "Smoothing RMS displacement: " << smooth_rms << " ("
            << (100.0f * smooth_rms / diagonal) << "% of diagonal)"
            << std::endl;

  // Helper to compute max error (applies frame if present)
  auto compute_max_error = [](const auto &A, const auto &B) {
    float max_err = 0.0f;
    for (auto [a, b] : tf::zip(A, B))
      max_err =
          std::max(max_err, tf::distance(tf::transformed(a, tf::frame_of(A)),
                                         tf::transformed(b, tf::frame_of(B))));
    return max_err;
  };

  // Helper to compute RMS error (applies frame if present)
  auto compute_rms_error = [](const auto &A, const auto &B) {
    float sum_sq = 0.0f;
    std::size_t count = 0;
    for (auto [a, b] : tf::zip(A, B)) {
      float d = tf::distance(tf::transformed(a, tf::frame_of(A)),
                             tf::transformed(b, tf::frame_of(B)));
      sum_sq += d * d;
      ++count;
    }
    return std::sqrt(sum_sq / count);
  };

  // Build tree on target for OBB disambiguation and ICP
  tf::aabb_tree<int, float, 3> target_tree(mesh.points(),
                                           tf::config_tree(4, 4));
  auto target_with_tree = mesh.points() | tf::tag(target_tree);

  // Compute normals for point-to-plane ICP
  std::cout << "Computing point normals..." << std::endl;
  auto normals = tf::compute_point_normals(mesh.polygons() | tf::tag(fm));
  auto target_with_normals =
      mesh.points() | tf::tag(target_tree) | tf::tag_normals(normals.unit_vectors());

  // Lambda to create strided sample indices
  auto make_ids = [](std::size_t size, std::size_t offset, std::size_t stride,
                     std::size_t count) {
    return tf::take(tf::make_mapped_range(tf::make_sequence_range(size),
                                          [size, offset, stride](auto id) {
                                            return (offset + id * stride) %
                                                   size;
                                          }),
                    count);
  };

  // =========================================================================
  // Part 1: With correspondences (rigid transformation)
  // =========================================================================
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "=== PART 1: With correspondences ===" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  // Compute centroid of smoothed mesh
  auto centroid = tf::centroid(smoothed.points());

  // Random rotation around centroid + large translation (2.5x diagonal away)
  tf::vector<float, 3> far_translation{diagonal * 2.5f, diagonal * -1.5f,
                                       diagonal * 2.0f};
  auto T1 = tf::transformed(
      tf::random_transformation_at(centroid),
      tf::make_transformation_from_translation(far_translation));

  std::cout
      << "\nTransforming smoothed mesh (rotation around centroid + translation)"
      << std::endl;

  // Create transformed copy of smoothed points
  tf::points_buffer<float, 3> source1;
  source1.allocate(smoothed.size());
  tf::parallel_for_each(tf::zip(smoothed.points(), source1.points()),
                     [&](auto tup) {
                       auto [src, dst] = tup;
                       dst = tf::transformed(src, T1);
                     });

  float initial1 = compute_max_error(mesh.points(), source1.points());
  std::cout << "Initial error: " << initial1 << std::endl;

  std::cout << "\nRigid alignment:" << std::endl;
  auto T_rigid1 = tf::fit_rigid_alignment(source1.points(), mesh.points());
  float rigid1_rms = compute_rms_error(
      mesh.points(), source1.points() | tf::tag(T_rigid1));
  std::cout << "  RMS error: " << rigid1_rms << std::endl;

  std::cout << "\nOBB alignment (no tree):" << std::endl;
  auto T_obb1_no_tree = tf::fit_obb_alignment(source1.points(), mesh.points());
  float obb1_no_tree_rms = compute_rms_error(
      mesh.points(),
      source1.points() | tf::tag(T_obb1_no_tree));
  std::cout << "  RMS error: " << obb1_no_tree_rms << std::endl;

  std::cout << "\nOBB alignment (with tree):" << std::endl;
  auto T_obb1_tree = tf::fit_obb_alignment(source1.points(), target_with_tree);
  float obb1_tree_rms = compute_rms_error(
      mesh.points(), source1.points() | tf::tag(T_obb1_tree));
  std::cout << "  RMS error: " << obb1_tree_rms << std::endl;

  std::cout << "\n--- Summary (Part 1) ---" << std::endl;
  std::cout << "  Ground truth:    " << smooth_rms << std::endl;
  std::cout << "  Rigid:           " << rigid1_rms << std::endl;
  std::cout << "  OBB (no tree):   " << obb1_no_tree_rms << std::endl;
  std::cout << "  OBB (with tree): " << obb1_tree_rms << std::endl;

  // =========================================================================
  // Part 2: Without correspondences (shuffled source)
  // =========================================================================
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "=== PART 2: Without correspondences (shuffled) ==="
            << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  // Create shuffled indices - source2[i] corresponds to
  // mesh.points()[shuffle_ids[i]]
  tf::buffer<int> shuffle_ids;
  shuffle_ids.allocate(source1.size());
  tf::parallel_iota(shuffle_ids, 0);
  std::shuffle(shuffle_ids.begin(), shuffle_ids.end(),
               std::mt19937{std::random_device{}()});

  // Build source2 with shuffled order (source1 is already transformed)
  tf::points_buffer<float, 3> source2;
  source2.allocate(source1.size());
  tf::parallel_copy(tf::make_indirect_range(shuffle_ids, source1.points()),
                    source2.points());

  // For correct RMS: compare source2[i] with mesh.points()[shuffle_ids[i]]
  auto target_shuffled =
      tf::make_points(tf::make_indirect_range(shuffle_ids, mesh.points()));

  std::cout << "\nRigid alignment (will fail - no correspondences):"
            << std::endl;
  auto T_rigid2 = tf::fit_rigid_alignment(source2.points(), mesh.points());
  float rigid2_rms = compute_rms_error(
      target_shuffled, source2.points() | tf::tag(T_rigid2));
  std::cout << "  RMS error: " << rigid2_rms << std::endl;

  std::cout << "\nOBB alignment (no tree - ambiguous):" << std::endl;
  auto T_obb2_no_tree = tf::fit_obb_alignment(source2.points(), mesh.points());
  float obb2_no_tree_rms = compute_rms_error(
      target_shuffled,
      source2.points() | tf::tag(T_obb2_no_tree));
  std::cout << "  RMS error: " << obb2_no_tree_rms << std::endl;

  std::cout << "\nOBB alignment (with tree - disambiguated):" << std::endl;
  auto T_obb2_tree = tf::fit_obb_alignment(source2.points(), target_with_tree);
  float obb2_tree_rms = compute_rms_error(
      target_shuffled, source2.points() | tf::tag(T_obb2_tree));
  std::cout << "  RMS error: " << obb2_tree_rms << std::endl;

  std::cout << "\n--- Summary (Part 2) ---" << std::endl;
  std::cout << "  Ground truth:    " << smooth_rms << std::endl;
  std::cout << "  Rigid:           " << rigid2_rms << " (FAILS)" << std::endl;
  std::cout << "  OBB (no tree):   " << obb2_no_tree_rms
            << " (may be wrong orientation)" << std::endl;
  std::cout << "  OBB (with tree): " << obb2_tree_rms << std::endl;

  // =========================================================================
  // Part 3: ICP refinement - Point-to-Point vs Point-to-Plane
  // =========================================================================
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "=== PART 3: ICP refinement ===" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  // Ground truth is smooth_rms (best achievable)
  std::cout << "Ground truth RMS: " << smooth_rms << std::endl;
  std::cout << "Starting from OBB with tree: RMS = " << obb2_tree_rms
            << std::endl;

  // ICP parameters
  constexpr std::size_t max_iters = 50;
  constexpr std::size_t n_samples = 1000;
  constexpr std::size_t k = 1;
  constexpr float alpha = 0.3f;     // EMA smoothing factor
  constexpr float rel_tol = 0.001f; // stop when < 0.1% relative improvement

  std::size_t subsample_stride =
      std::max(std::size_t(1), source2.size() / n_samples);
  std::cout << "Subsampling: ~" << n_samples << " / " << source2.size()
            << " points per iteration" << std::endl;

  // Generic ICP runner that works with any target type
  auto run_icp = [&](const auto &target, const char *name) {
    auto T_accum = T_obb2_tree;
    float ema = 0.0f;
    float ema_prev = 0.0f;

    std::cout << "\n" << name << " ICP:" << std::endl;
    std::size_t iter = 0;
    for (; iter < max_iters; ++iter) {
      std::size_t offset = tf::random(std::size_t(0), subsample_stride - 1);
      auto ids = make_ids(source2.size(), offset, subsample_stride, n_samples);
      auto subsample =
          tf::make_points(tf::make_indirect_range(ids, source2.points()));

      auto subsample_with_frame = subsample | tf::tag(T_accum);

      auto T_iter = tf::fit_knn_alignment(subsample_with_frame, target, k);

      T_accum = tf::transformed(T_accum, T_iter);

      // Evaluate Chamfer error on a different subset
      std::size_t eval_offset = tf::random(std::size_t(0), subsample_stride - 1);
      auto eval_ids =
          make_ids(source2.size(), eval_offset, subsample_stride, n_samples);
      auto eval_sample =
          tf::make_points(tf::make_indirect_range(eval_ids, source2.points()));
      float chamfer =
          tf::chamfer_error(eval_sample | tf::tag(T_accum), target_with_tree);

      ema_prev = ema;
      ema = (iter == 0) ? chamfer : alpha * chamfer + (1.0f - alpha) * ema;
      float rel_change = (iter == 0) ? 1.0f : (ema_prev - ema) / ema;

      std::cout << "  iter " << iter << ": Chamfer = " << chamfer
                << " (EMA = " << ema << ")" << std::endl;

      if (iter > 0 && rel_change < rel_tol)
        break;
    }
    std::cout << "Converged after " << (iter + 1) << " iterations" << std::endl;

    float final_rms =
        compute_rms_error(target_shuffled, source2.points() | tf::tag(T_accum));
    return std::make_pair(iter + 1, final_rms);
  };

  // Run Point-to-Point ICP
  auto [p2p_iters, p2p_rms] = run_icp(target_with_tree, "Point-to-Point");

  // Run Point-to-Plane ICP
  auto [p2l_iters, p2l_rms] = run_icp(target_with_normals, "Point-to-Plane");

  std::cout << "\n--- ICP Comparison ---" << std::endl;
  std::cout << "  Ground truth RMS:    " << smooth_rms << std::endl;
  std::cout << "  Point-to-Point: " << p2p_iters << " iters, RMS = " << p2p_rms
            << std::endl;
  std::cout << "  Point-to-Plane: " << p2l_iters << " iters, RMS = " << p2l_rms
            << std::endl;
  if (p2l_iters < p2p_iters) {
    std::cout << "  Point-to-Plane converged "
              << (static_cast<float>(p2p_iters) / p2l_iters)
              << "x faster!" << std::endl;
  }

  // =========================================================================
  // Part 4: Different mesh resolutions (no correspondences possible)
  // =========================================================================
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "=== PART 4: Different mesh resolutions ===" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  // Load a lower-resolution version of the mesh
  std::string low_res_path = data_dir + "dragon-50k.stl";
  std::cout << "\nLoading low-res mesh: " << low_res_path << std::endl;

  auto mesh_low = tf::read_stl(low_res_path.c_str());
  if (mesh_low.polygons().size() == 0) {
    std::cerr << "Failed to load low-res mesh, skipping Part 4" << std::endl;
    return 0;
  }

  std::cout << "High-res: " << mesh.points().size() << " vertices" << std::endl;
  std::cout << "Low-res:  " << mesh_low.points().size() << " vertices"
            << std::endl;

  // Build tree on low-res mesh
  tf::aabb_tree<int, float, 3> low_res_tree(mesh_low.points(),
                                            tf::config_tree(4, 4));
  auto low_res_with_tree = mesh_low.points() | tf::tag(low_res_tree);

  // Baseline Chamfer: how different are the meshes due to resolution?
  float chamfer_baseline_fwd =
      tf::chamfer_error(mesh_low.points(), target_with_tree);
  float chamfer_baseline_bwd =
      tf::chamfer_error(mesh.points(), low_res_with_tree);
  std::cout << "\nBaseline Chamfer (aligned, different resolutions):"
            << std::endl;
  std::cout << "  Low→High: " << chamfer_baseline_fwd << std::endl;
  std::cout << "  High→Low: " << chamfer_baseline_bwd << std::endl;
  std::cout << "  Symmetric: "
            << (chamfer_baseline_fwd + chamfer_baseline_bwd) / 2.0f
            << std::endl;

  // Transform low-res mesh far away
  auto centroid_low = tf::centroid(mesh_low.points());
  auto T_low = tf::transformed(
      tf::random_transformation_at(centroid_low),
      tf::make_transformation_from_translation(far_translation));

  tf::points_buffer<float, 3> source_low;
  source_low.allocate(mesh_low.points().size());
  tf::parallel_for_each(tf::zip(mesh_low.points(), source_low.points()),
                     [&](auto tup) {
                       auto [src, dst] = tup;
                       dst = tf::transformed(src, T_low);
                     });

  // Build tree on transformed source for bidirectional chamfer
  tf::aabb_tree<int, float, 3> source_low_tree(source_low.points(),
                                               tf::config_tree(4, 4));
  auto source_low_with_tree = source_low.points() | tf::tag(source_low_tree);

  // Initial Chamfer error (meshes far apart)
  float chamfer_init_fwd =
      tf::chamfer_error(source_low.points(), target_with_tree);
  float chamfer_init_bwd =
      tf::chamfer_error(mesh.points(), source_low_with_tree);
  std::cout << "\nInitial Chamfer error:" << std::endl;
  std::cout << "  Low→High: " << chamfer_init_fwd << std::endl;
  std::cout << "  High→Low: " << chamfer_init_bwd << std::endl;
  std::cout << "  Symmetric: " << (chamfer_init_fwd + chamfer_init_bwd) / 2.0f
            << std::endl;

  // OBB alignment (no tree)
  std::cout << "\nOBB alignment (no tree):" << std::endl;
  auto T_obb_low_no_tree =
      tf::fit_obb_alignment(source_low.points(), mesh.points());
  float chamfer_obb_no_tree = tf::chamfer_error(
      source_low.points() | tf::tag(T_obb_low_no_tree),
      target_with_tree);
  std::cout << "  Chamfer (Low→High): " << chamfer_obb_no_tree << std::endl;

  // OBB alignment (with tree)
  std::cout << "\nOBB alignment (with tree):" << std::endl;
  auto T_obb_low_tree =
      tf::fit_obb_alignment(source_low.points(), target_with_tree);
  float chamfer_obb_tree = tf::chamfer_error(
      source_low.points() | tf::tag(T_obb_low_tree),
      target_with_tree);
  std::cout << "  Chamfer (Low→High): " << chamfer_obb_tree << std::endl;

  // ICP refinement - compare Point-to-Point vs Point-to-Plane
  std::cout << "\nICP refinement (comparing P2P vs P2L):" << std::endl;

  std::size_t subsample_stride_low =
      std::max(std::size_t(1), source_low.size() / n_samples);

  auto run_icp_low = [&](const auto &target, const char *name) {
    auto T_accum_low = T_obb_low_tree;
    float ema_low = 0.0f;
    float ema_low_prev = 0.0f;

    std::cout << "\n" << name << " ICP:" << std::endl;
    std::size_t iter_low = 0;
    for (; iter_low < max_iters; ++iter_low) {
      std::size_t offset = tf::random(std::size_t(0), subsample_stride_low - 1);
      auto ids =
          make_ids(source_low.size(), offset, subsample_stride_low, n_samples);
      auto subsample_low =
          tf::make_points(tf::make_indirect_range(ids, source_low.points()));

      auto subsample_low_with_frame = subsample_low | tf::tag(T_accum_low);

      auto T_iter = tf::fit_knn_alignment(subsample_low_with_frame, target, k);

      T_accum_low = tf::transformed(T_accum_low, T_iter);

      // Evaluate Chamfer error on a different subset
      std::size_t eval_offset =
          tf::random(std::size_t(0), subsample_stride_low - 1);
      auto eval_ids = make_ids(source_low.size(), eval_offset,
                               subsample_stride_low, n_samples);
      auto eval_sample = tf::make_points(
          tf::make_indirect_range(eval_ids, source_low.points()));
      float chamfer = tf::chamfer_error(eval_sample | tf::tag(T_accum_low),
                                        target_with_tree);

      ema_low_prev = ema_low;
      ema_low = (iter_low == 0) ? chamfer
                                : alpha * chamfer + (1.0f - alpha) * ema_low;
      float rel_change =
          (iter_low == 0) ? 1.0f : (ema_low_prev - ema_low) / ema_low;

      std::cout << "  iter " << iter_low << ": Chamfer = " << chamfer
                << " (EMA = " << ema_low << ")" << std::endl;

      if (iter_low > 0 && rel_change < rel_tol)
        break;
    }
    std::cout << "Converged after " << (iter_low + 1) << " iterations"
              << std::endl;

    float chamfer_final = tf::chamfer_error(
        source_low.points() | tf::tag(T_accum_low), target_with_tree);
    return std::make_pair(iter_low + 1, chamfer_final);
  };

  auto [p2p_iters_low, p2p_chamfer_low] =
      run_icp_low(target_with_tree, "Point-to-Point");
  auto [p2l_iters_low, p2l_chamfer_low] =
      run_icp_low(target_with_normals, "Point-to-Plane");

  std::cout << "\n--- Summary (Part 4) ---" << std::endl;
  std::cout << "  Baseline:        " << chamfer_baseline_fwd
            << " (best possible)" << std::endl;
  std::cout << "  Initial:         " << chamfer_init_fwd
            << " (after transformation)" << std::endl;
  std::cout << "  OBB (no tree):   " << chamfer_obb_no_tree << std::endl;
  std::cout << "  OBB (with tree): " << chamfer_obb_tree << std::endl;
  std::cout << "  P2P ICP: " << p2p_iters_low << " iters, Chamfer = "
            << p2p_chamfer_low << std::endl;
  std::cout << "  P2L ICP: " << p2l_iters_low << " iters, Chamfer = "
            << p2l_chamfer_low << std::endl;
  if (p2l_iters_low < p2p_iters_low) {
    std::cout << "  Point-to-Plane converged "
              << (static_cast<float>(p2p_iters_low) / p2l_iters_low)
              << "x faster!" << std::endl;
  }

  return 0;
}
