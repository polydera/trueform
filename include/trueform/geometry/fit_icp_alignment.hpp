/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once

#include "../core/frame_of.hpp"
#include "../core/policy/frame.hpp"
#include "../core/policy/normals.hpp"
#include "../core/views/cyclic_sequence_range.hpp"
#include "../core/views/indirect_range.hpp"
#include "./chamfer_error.hpp"
#include "./fit_knn_alignment.hpp"
#include "./icp_config.hpp"
#include "./icp_state.hpp"

namespace tf {

/// @ingroup geometry_registration
/// @brief Iterative Closest Point (ICP) alignment.
///
/// Iteratively refines a rigid transformation aligning source points X to
/// target points Y. Each iteration:
/// 1. Subsamples X with a varying offset (different points each iteration)
/// 2. Fits a rigid transformation using k-NN correspondences
/// 3. Accumulates the transformation
/// 4. Evaluates Chamfer error on a separate subsample
/// 5. Checks EMA-smoothed convergence
///
/// If Y has normals attached, uses point-to-plane ICP which converges faster.
///
/// Returns a DELTA transformation mapping source world coordinates to target
/// world coordinates. To get the total transformation for source local coords:
/// @code
/// auto delta = fit_icp_alignment(source | tf::tag(T_initial), target);
/// auto total = tf::transformed(T_initial, delta);
/// @endcode
///
/// @param X Source point set (may have frame and/or normals).
/// @param Y Target point set with tree (may have frame and/or normals).
/// @param state Reusable workspace.
/// @param config ICP configuration.
/// @return Rigid transform mapping X world coords to Y world coords (delta).
///
/// @see @ref tf::icp_config
/// @see @ref tf::icp_state
template <typename Policy0, typename Policy1>
auto fit_icp_alignment(const tf::points<Policy0> &X,
                       const tf::points<Policy1> &Y,
                       icp_state<Policy0, Policy1> &state,
                       const icp_config &config = {})
    -> tf::transformation<tf::coordinate_type<Policy0, Policy1>,
                          tf::coordinate_dims_v<Policy0>> {
  using T = tf::coordinate_type<Policy0, Policy1>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(Dims == tf::coordinate_dims_v<Policy1>,
                "Point sets must have the same dimensionality");
  static_assert(tf::has_tree_policy<Policy1>,
                "Target point set Y must have a tree policy attached");

  // Track two transformations:
  // - T_total: includes initial frame, used for world positioning during
  // iterations
  // - T_delta: pure accumulated delta (world-to-world), what we return
  auto T_total = tf::concrete_transformation_of(X);
  auto T_delta = tf::make_identity_transformation<T, Dims>();

  auto X_plain = tf::untag_frame(X);

  tf::knn_alignment_config align_cfg{config.k, config.sigma,
                                     config.outlier_proportion};

  const std::size_t n = X.size();
  const std::size_t n_samples =
      config.n_samples > 0 ? std::min(config.n_samples, n) : n;
  const std::size_t stride = std::max(std::size_t(1), n / n_samples);

  T ema = 0;
  T ema_prev = 0;

  for (std::size_t iter = 0; iter < config.max_iterations; ++iter) {
    std::size_t align_offset = (iter * 17) % stride;
    auto align_ids = make_cyclic_sequence_range(n_samples, n, align_offset);

    auto subsample_base =
        tf::make_points(tf::make_indirect_range(align_ids, X_plain));

    if constexpr (tf::has_normals_policy<std::decay_t<decltype(X_plain)>>) {
      auto subsample =
          subsample_base |
          tf::tag_normals(tf::make_unit_vectors(
              tf::make_indirect_range(align_ids, X_plain.normals()))) |
          tf::tag(T_total);

      auto T_iter = tf::fit_knn_alignment(subsample, Y, state, align_cfg);
      T_total = tf::transformed(T_total, T_iter);
      T_delta = tf::transformed(T_delta, T_iter);
    } else {
      auto subsample = subsample_base | tf::tag(T_total);

      auto T_iter = tf::fit_knn_alignment(subsample, Y, state, align_cfg);
      T_total = tf::transformed(T_total, T_iter);
      T_delta = tf::transformed(T_delta, T_iter);
    }
    if (config.min_relative_improvement > 0) {
      std::size_t eval_offset = (iter * 31 + 13) % stride;
      auto eval_ids = make_cyclic_sequence_range(n_samples, n, eval_offset);
      auto eval_sample =
          tf::make_points(tf::make_indirect_range(eval_ids, X_plain)) |
          tf::tag(T_total);

      T error = tf::chamfer_error(eval_sample, Y, config.outlier_proportion);

      ema_prev = ema;
      ema = (iter == 0)
                ? error
                : config.ema_alpha * error + (T(1) - config.ema_alpha) * ema;

      if (iter > 0) {
        T rel_change = (ema_prev - ema) / ema_prev;
        if (rel_change < T(config.min_relative_improvement))
          break;
      }
    }
  }

  return T_delta;
}

/// @ingroup geometry_registration
/// @brief ICP alignment (allocates state internally).
///
/// @see @ref fit_icp_alignment(const tf::points<Policy0>&, const
/// tf::points<Policy1>&, icp_state<Policy0, Policy1>&, const icp_config&)
template <typename Policy0, typename Policy1>
auto fit_icp_alignment(const tf::points<Policy0> &X,
                       const tf::points<Policy1> &Y,
                       const icp_config &config = {})
    -> tf::transformation<tf::coordinate_type<Policy0, Policy1>,
                          tf::coordinate_dims_v<Policy0>> {
  icp_state<Policy0, Policy1> state;
  return fit_icp_alignment(X, Y, state, config);
}

} // namespace tf
