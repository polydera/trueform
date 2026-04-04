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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/basis.hpp"
#include "../core/cross.hpp"
#include "../core/curve.hpp"
#include "../core/dot.hpp"
#include "../core/epsilon.hpp"
#include "../core/unit_vector.hpp"
#include "../core/unit_vectors.hpp"
#include "../core/unit_vectors_buffer.hpp"
#include "../core/views/sequence_range.hpp"

namespace tf {

/// @ingroup geometry
/// @brief Compute parallel transport frames along a curve.
///
/// Writes tangents, normals, and binormals into pre-allocated ranges.
/// Detects closed curves (first ~ last point) automatically and
/// applies wrap-around tangents and twist correction.
///
/// @param curve The input curve.
/// @param tangents Output range for tangents (size == curve.size()).
/// @param normals Output range for normals (size == curve.size()).
/// @param binormals Output range for binormals (size == curve.size()).
template <typename CurvePolicy, typename TangentPolicy, typename NormalPolicy,
          typename BinormalPolicy>
void make_curve_frames(const tf::curve<3, CurvePolicy> &curve,
                       tf::unit_vectors<TangentPolicy> &tangents,
                       tf::unit_vectors<NormalPolicy> &normals,
                       tf::unit_vectors<BinormalPolicy> &binormals) {
  using RealT = tf::coordinate_type<CurvePolicy>;
  const auto n = static_cast<std::size_t>(curve.size());
  if (n == 0)
    return;

  if (n == 1) {
    normals[0] = tf::make_unit_vector(
        tf::unsafe, tf::make_vector(RealT{1}, RealT{0}, RealT{0}));
    binormals[0] = tf::make_unit_vector(
        tf::unsafe, tf::make_vector(RealT{0}, RealT{1}, RealT{0}));
    return;
  }

  auto diff = curve[n - 1] - curve[0];
  bool closed = tf::dot(diff, diff) < tf::epsilon<RealT> * tf::epsilon<RealT>;

  auto rodrigues = [](const auto &v, const auto &k, RealT cos_t, RealT sin_t) {
    return v * cos_t + tf::cross(k, v) * sin_t +
           k * (tf::dot(k, v) * (RealT{1} - cos_t));
  };

  if (closed) {
    tf::parallel_for_each(
        tf::make_sequence_range(n),
        [&](std::size_t i) {
          auto prev = (i == 0) ? n - 1 : i - 1;
          auto next = (i + 1) % n;
          tangents[i] = tf::make_unit_vector(curve[next] - curve[prev]);
        },
        tf::checked);
  } else {
    tangents[0] = tf::make_unit_vector(curve[1] - curve[0]);
    tangents[n - 1] = tf::make_unit_vector(curve[n - 1] - curve[n - 2]);
    if (n > 2) {
      tf::parallel_for_each(
          tf::make_sequence_range(std::size_t{1}, n - 1),
          [&](std::size_t i) {
            tangents[i] = tf::make_unit_vector(curve[i + 1] - curve[i - 1]);
          },
          tf::checked);
    }
  }

  auto [basis0, basis1] = tf::make_basis_from_normal(tangents[0]);
  normals[0] = basis0;
  binormals[0] = tf::make_unit_vector(tf::cross(tangents[0], normals[0]));

  for (std::size_t i = 1; i < n; ++i) {
    auto axis = tf::cross(tangents[i - 1], tangents[i]);
    if (tf::dot(axis, axis) > RealT{1e-12}) {
      auto d = std::max(
          RealT{-1}, std::min(RealT{1}, tf::dot(tangents[i - 1], tangents[i])));
      auto theta = std::acos(d);
      normals[i] = tf::make_unit_vector(
          rodrigues(normals[i - 1], tf::make_unit_vector(axis), std::cos(theta),
                    std::sin(theta)));
    } else {
      normals[i] = normals[i - 1];
    }
    binormals[i] = tf::make_unit_vector(tf::cross(tangents[i], normals[i]));
  }

  if (closed && n > 2) {
    auto d = std::max(RealT{-1},
                      std::min(RealT{1}, tf::dot(normals[0], normals[n - 1])));
    auto twist = std::acos(d);
    if (tf::dot(tangents[0], tf::cross(normals[0], normals[n - 1])) > RealT{0})
      twist = -twist;

    auto inv_n = RealT{1} / static_cast<RealT>(n);
    for (std::size_t i = 1; i < n; ++i) {
      auto angle = twist * static_cast<RealT>(i) * inv_n;
      normals[i] = tf::make_unit_vector(
          rodrigues(normals[i], tangents[i], std::cos(angle), std::sin(angle)));
      binormals[i] = tf::make_unit_vector(tf::cross(tangents[i], normals[i]));
    }
  }
}

/// @overload
template <typename CurvePolicy, typename TangentPolicy, typename NormalPolicy,
          typename BinormalPolicy>
void make_curve_frames(const tf::curve<3, CurvePolicy> &curve,
                       tf::unit_vectors<TangentPolicy> &&tangents,
                       tf::unit_vectors<NormalPolicy> &&normals,
                       tf::unit_vectors<BinormalPolicy> &&binormals) {
  make_curve_frames(curve, tangents, normals, binormals);
}

/// @ingroup geometry
/// @brief Compute parallel transport frames along a curve.
///
/// Allocating overload. Returns (tangents, normals, binormals) buffers.
template <typename CurvePolicy>
auto make_curve_frames(const tf::curve<3, CurvePolicy> &curve) {
  using RealT = tf::coordinate_type<CurvePolicy>;
  const auto n = static_cast<std::size_t>(curve.size());

  tf::unit_vectors_buffer<RealT, 3> tangents;
  tf::unit_vectors_buffer<RealT, 3> normals;
  tf::unit_vectors_buffer<RealT, 3> binormals;
  tangents.allocate(n);
  normals.allocate(n);
  binormals.allocate(n);

  make_curve_frames(curve, tangents.unit_vectors(), normals.unit_vectors(),
                    binormals.unit_vectors());

  return std::tuple{std::move(tangents), std::move(normals),
                    std::move(binormals)};
}

} // namespace tf
