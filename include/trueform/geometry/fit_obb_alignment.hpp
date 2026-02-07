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

#include "../core/algorithm/reduce.hpp"
#include "../core/frame_of.hpp"
#include "../core/make_rotation.hpp"
#include "../core/obb_from.hpp"
#include "../core/orthogonalize.hpp"
#include "../core/policy/unwrap.hpp"
#include "../core/transformation.hpp"
#include "../core/transformed.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../spatial/neighbor_search.hpp"
#include "../spatial/policy/tree.hpp"

#include <array>

namespace tf {

namespace geometry {
namespace impl {

/// @brief Generate 180° rotation candidates for 2D OBB disambiguation
template <typename T>
auto make_obb_rotations_2d(const tf::obb<T, 2> &obb)
    -> std::array<tf::transformation<T, 2>, 2> {
  auto pivot = obb.center();
  return {tf::make_identity_transformation<T, 2>(),
          tf::make_rotation(tf::deg<T>(T(180)), pivot)};
}

/// @brief Generate 180° rotation candidates for 3D OBB disambiguation
template <typename T>
auto make_obb_rotations_3d(const tf::obb<T, 3> &obb)
    -> std::array<tf::transformation<T, 3>, 4> {
  auto pivot = obb.center();
  return {tf::make_identity_transformation<T, 3>(),
          tf::make_rotation(tf::deg<T>(T(180)), obb.axes[0], pivot),
          tf::make_rotation(tf::deg<T>(T(180)), obb.axes[1], pivot),
          tf::make_rotation(tf::deg<T>(T(180)), obb.axes[2], pivot)};
}

} // namespace impl
} // namespace geometry

/// @ingroup geometry_registration
/// @brief Compute a rigid alignment from X to Y using oriented bounding boxes
/// (OBBs).
///
/// The returned transform T maps points from X into Y:
///
///   y ≈ T(x) = R * x + t
///
/// where R aligns the OBB axes of X to those of Y, and t aligns the OBB
/// centers. No point correspondences are used.
///
/// If the point sets have frames attached, the alignment is computed
/// in world space (i.e., with frames applied).
///
/// @note OBB alignment is inherently ambiguous up to the symmetry group of
/// the bounding box (180° rotations about each axis). If Y has a tree policy
/// attached (3D only), the function resolves the ambiguity by testing all 4
/// orientations and selecting the one with lowest chamfer distance.
///
/// @param X Source point set.
/// @param Y Target point set (with optional tree policy for disambiguation).
/// @param sample_size Number of points to sample for disambiguation (default:
/// 100).
/// @return Rigid transform mapping X -> Y.
template <typename Policy0, typename Policy1>
auto fit_obb_alignment(const tf::points<Policy0> &X_,
                       const tf::points<Policy1> &Y_,
                       std::size_t sample_size = 100) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy0>;
  static_assert(Dims == tf::coordinate_dims_v<Policy1>,
                "Point sets must have the same dimensionality");
  static_assert(Dims == 2 || Dims == 3,
                "Only 2D and 3D point sets are supported");

  // Extract plain points and frames
  const auto &X = X_ | tf::plain();
  const auto &Y = Y_ | tf::plain();
  const auto &tX = tf::frame_of(X_).transformation();
  const auto &tY = tf::frame_of(Y_).transformation();

  // Compute OBBs on plain points and transform to world space
  auto obb0 = tf::transformed(tf::obb_from(X), tX);
  auto obb1 = tf::transformed(tf::obb_from(Y), tY);

  // Build frame matrices A0, A1 with columns = axes[k]
  std::array<std::array<T, Dims>, Dims> A0{};
  std::array<std::array<T, Dims>, Dims> A1{};
  for (std::size_t i = 0; i < Dims; ++i) {
    for (std::size_t k = 0; k < Dims; ++k) {
      A0[i][k] = obb0.axes[k][i];
      A1[i][k] = obb1.axes[k][i];
    }
  }

  // Ensure both frames are right-handed (det = +1)
  auto det_of = [](const auto &M) -> T {
    if constexpr (tf::coordinate_dims_v<Policy0> == 2) {
      return M[0][0] * M[1][1] - M[0][1] * M[1][0];
    } else {
      return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) -
             M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) +
             M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]);
    }
  };

  if (det_of(A0) < T(0))
    for (std::size_t i = 0; i < Dims; ++i)
      A0[i][Dims - 1] = -A0[i][Dims - 1];

  if (det_of(A1) < T(0))
    for (std::size_t i = 0; i < Dims; ++i)
      A1[i][Dims - 1] = -A1[i][Dims - 1];

  // R = A1 * A0^T
  tf::transformation<T, Dims> T_base;
  for (std::size_t i = 0; i < Dims; ++i) {
    for (std::size_t j = 0; j < Dims; ++j) {
      T s = T(0);
      for (std::size_t k = 0; k < Dims; ++k)
        s += A1[i][k] * A0[j][k];
      T_base(i, j) = s;
    }
  }

  // Orthogonalize R to correct numerical drift
  tf::orthogonalize(T_base);

  // t = c1 - R * c0
  const auto c0 = obb0.center();
  const auto c1 = obb1.center();
  for (std::size_t i = 0; i < Dims; ++i) {
    T rc0 = T(0);
    for (std::size_t j = 0; j < Dims; ++j)
      rc0 += T_base(i, j) * c0[j];
    T_base(i, Dims) = c1[i] - rc0;
  }

  // If Y has tree policy, disambiguate using chamfer distance
  if constexpr (tf::has_tree_policy<Policy1>) {
    // Cap sample size at number of points
    sample_size = std::min(sample_size, X_.size());
    std::size_t stride = std::max(std::size_t(1), X_.size() / sample_size);

    auto indices = tf::make_mapped_range(tf::make_sequence_range(sample_size),
                                         [=](auto i) { return i * stride; });
    auto sample = tf::make_indirect_range(indices, X_);

    auto rotations = [&]() {
      if constexpr (tf::coordinate_dims_v<Policy0> == 2) {
        return geometry::impl::make_obb_rotations_2d(obb1);
      } else {
        return geometry::impl::make_obb_rotations_3d(obb1);
      }
    }();

    // Build candidate transforms
    using N_t = std::integral_constant<std::size_t,
        tf::coordinate_dims_v<Policy0> == 2 ? 2 : 4>;
    constexpr std::size_t N = N_t::value;
    std::array<tf::transformation<T, Dims>, N> candidates;
    for (std::size_t i = 0; i < N; ++i)
      candidates[i] = tf::transformed(T_base, rotations[i]);

    // Accumulate chamfer error for all candidates in one pass
    std::array<T, N> init{};
    auto errors = tf::reduce(
        tf::make_mapped_range(
            sample,
            [&](const auto &pt) {
              auto pt_world = tf::transformed(pt, tf::frame_of(X_));
              std::array<T, N_t::value> errs;
              for (std::size_t i = 0; i < N_t::value; ++i) {
                auto query_pt = tf::transformed(pt_world, candidates[i]);
                auto [id, cpt] =
                    tf::neighbor_search(Y_, query_pt);
                errs[i] = cpt.metric;
              }
              return errs;
            }),
        [](auto acc, const auto &e) {
          for (std::size_t i = 0; i < acc.size(); ++i)
            acc[i] += e[i];
          return acc;
        },
        init, tf::checked);

    // Pick candidate with lowest error
    std::size_t best = 0;
    for (std::size_t i = 1; i < N; ++i)
      if (errors[i] < errors[best])
        best = i;

    return candidates[best];
  } else {
    return T_base;
  }
}

} // namespace tf
