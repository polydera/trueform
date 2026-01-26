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
#include "./algorithm/reduce.hpp"
#include "./centroid.hpp"
#include "./points.hpp"
#include "./polygons.hpp"
#include "./segment.hpp"
#include "./segments.hpp"
#include "./unit_vectors.hpp"
#include "./vectors.hpp"

namespace tf {
namespace core {
template <typename Range, std::size_t Dims, typename Policy>
auto covariance_of(const Range &points, const tf::point_like<Dims, Policy> &) {
  using T = tf::coordinate_type<Policy>;
  auto centroid = tf::centroid(tf::make_points(points));
  std::array<std::array<T, Dims>, Dims> cov;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      cov[i][j] = T(0);

  cov = tf::reduce(
      points,
      [&centroid](auto acc, const auto &element) {
        using ElementType = std::decay_t<decltype(element)>;
        using AccType = std::decay_t<decltype(acc)>;

        if constexpr (std::is_same_v<ElementType, AccType>) {
          // Merging two partial covariance matrices
          for (std::size_t i = 0; i < Dims; ++i)
            for (std::size_t j = 0; j < Dims; ++j)
              acc[i][j] += element[i][j];
        } else {
          // Adding a point's contribution
          tf::vector<T, Dims> diff = element - centroid;
          for (std::size_t i = 0; i < Dims; ++i)
            for (std::size_t j = 0; j < Dims; ++j)
              acc[i][j] += diff[i] * diff[j];
        }

        return acc;
      },
      cov, tf::checked);

  auto n = points.size();
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      cov[i][j] /= n + (n == 0);

  return std::make_pair(centroid, cov);
}

template <typename Range, std::size_t Dims, typename Policy>
auto covariance_of(const Range &vectors,
                   const tf::vector_like<Dims, Policy> &) {
  using T = tf::coordinate_type<Policy>;
  auto centroid = tf::centroid(tf::make_vectors(vectors));
  std::array<std::array<T, Dims>, Dims> cov;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      cov[i][j] = T(0);

  cov = tf::reduce(
      vectors,
      [&centroid](auto acc, const auto &element) {
        using ElementType = std::decay_t<decltype(element)>;
        using AccType = std::decay_t<decltype(acc)>;

        if constexpr (std::is_same_v<ElementType, AccType>) {
          // Merging two partial covariance matrices
          for (std::size_t i = 0; i < Dims; ++i)
            for (std::size_t j = 0; j < Dims; ++j)
              acc[i][j] += element[i][j];
        } else {
          // Adding a point's contribution
          tf::vector<T, Dims> diff = element - centroid;
          for (std::size_t i = 0; i < Dims; ++i)
            for (std::size_t j = 0; j < Dims; ++j)
              acc[i][j] += diff[i] * diff[j];
        }

        return acc;
      },
      cov, tf::checked);

  auto n = vectors.size();
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      cov[i][j] /= n + (n == 0);

  return std::make_pair(centroid, cov);
}

template <typename Range, std::size_t Dims, typename Policy>
auto covariance_of(const Range &polygons, const tf::polygon<Dims, Policy> &) {
  using T = tf::coordinate_type<Policy>;

  auto centroid = tf::centroid(tf::make_polygons(polygons));

  auto polygon_data =
      tf::make_mapped_range(polygons, [&centroid](const auto &poly) {
        std::array<std::array<T, Dims>, Dims> cov_contrib;
        for (std::size_t i = 0; i < Dims; ++i)
          for (std::size_t j = 0; j < Dims; ++j)
            cov_contrib[i][j] = T(0);

        for (const auto &pt : poly) {
          tf::vector<T, Dims> diff = pt - centroid;
          for (std::size_t i = 0; i < Dims; ++i)
            for (std::size_t j = 0; j < Dims; ++j)
              cov_contrib[i][j] += diff[i] * diff[j];
        }

        return std::make_pair(poly.size(), cov_contrib);
      });

  std::pair<std::size_t, std::array<std::array<T, Dims>, Dims>> init;
  init.first = 0;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      init.second[i][j] = T(0);

  auto result = tf::reduce(
      polygon_data,
      [](auto acc, const auto &element) {
        acc.first += element.first;
        for (std::size_t i = 0; i < Dims; ++i)
          for (std::size_t j = 0; j < Dims; ++j)
            acc.second[i][j] += element.second[i][j];
        return acc;
      },
      init, tf::checked);

  std::array<std::array<T, Dims>, Dims> cov = result.second;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      cov[i][j] /= result.first;

  return std::make_pair(centroid, cov);
}

template <typename Range, std::size_t Dims, typename Policy>
auto covariance_of(const Range &segments, const tf::segment<Dims, Policy> &) {
  using T = tf::coordinate_type<Policy>;

  auto centroid = tf::centroid(tf::make_segments(segments));

  auto segment_data =
      tf::make_mapped_range(segments, [&centroid](const auto &seg) {
        std::array<std::array<T, Dims>, Dims> cov_contrib;
        for (std::size_t i = 0; i < Dims; ++i)
          for (std::size_t j = 0; j < Dims; ++j)
            cov_contrib[i][j] = T(0);

        for (const auto &pt : seg) {
          tf::vector<T, Dims> diff = pt - centroid;
          for (std::size_t i = 0; i < Dims; ++i)
            for (std::size_t j = 0; j < Dims; ++j)
              cov_contrib[i][j] += diff[i] * diff[j];
        }

        return std::make_pair(seg.size(), cov_contrib);
      });

  std::pair<std::size_t, std::array<std::array<T, Dims>, Dims>> init;
  init.first = 0;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      init.second[i][j] = T(0);

  auto result = tf::reduce(
      segment_data,
      [](auto acc, const auto &element) {
        acc.first += element.first;
        for (std::size_t i = 0; i < Dims; ++i)
          for (std::size_t j = 0; j < Dims; ++j)
            acc.second[i][j] += element.second[i][j];
        return acc;
      },
      init, tf::checked);

  std::array<std::array<T, Dims>, Dims> cov = result.second;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      cov[i][j] /= result.first;

  return std::make_pair(centroid, cov);
}
} // namespace core

/// @ingroup core_properties
/// @brief Compute covariance matrix of a point set.
///
/// Returns the centroid and covariance matrix of the point cloud.
/// Used for PCA-based algorithms like OBB computation.
///
/// @tparam Policy The points policy type.
/// @param pts The point set.
/// @return A pair of (centroid, covariance matrix).
template <typename Policy> auto covariance_of(const tf::points<Policy> &pts) {
  return core::covariance_of(pts, pts[0]);
}

/// @ingroup core_properties
/// @brief Compute covariance matrix of a vector set.
/// @overload
template <typename Policy> auto covariance_of(const tf::vectors<Policy> &vcs) {
  return core::covariance_of(vcs, vcs[0]);
}

/// @ingroup core_properties
/// @brief Compute covariance matrix of a unit vector set.
/// @overload
template <typename Policy>
auto covariance_of(const tf::unit_vectors<Policy> &vcs) {
  return core::covariance_of(tf::make_range(vcs), vcs[0]);
}

/// @ingroup core_properties
/// @brief Compute covariance matrix of segment endpoints.
/// @overload
template <typename Policy>
auto covariance_of(const tf::segments<Policy> &segs) {
  return core::covariance_of(segs, segs[0]);
}

/// @ingroup core_properties
/// @brief Compute covariance matrix of polygon vertices.
/// @overload
template <typename Policy>
auto covariance_of(const tf::polygons<Policy> &polys) {
  return core::covariance_of(polys, polys[0]);
}

} // namespace tf
