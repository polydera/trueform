/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./algorithm/reduce.hpp"
#include "./covariance_of.hpp"
#include "./dot.hpp"
#include "./eigen_of.hpp"
#include "./obb.hpp"
#include "./points.hpp"
#include "./polygons.hpp"
#include "./segments.hpp"

namespace tf {
namespace core {

template <std::size_t Dims, typename T>
struct proj_accum {
  std::array<T, Dims> min_proj;
  std::array<T, Dims> max_proj;
};

template <std::size_t Dims, typename T>
auto make_proj_accum_init() {
  proj_accum<Dims, T> init;
  for (std::size_t i = 0; i < Dims; ++i) {
    init.min_proj[i] = std::numeric_limits<T>::max();
    init.max_proj[i] = -std::numeric_limits<T>::max();
  }
  return init;
}

template <std::size_t Dims, typename T>
auto merge_proj_accum(proj_accum<Dims, T> acc, const proj_accum<Dims, T> &other) {
  for (std::size_t i = 0; i < Dims; ++i) {
    acc.min_proj[i] = std::min(acc.min_proj[i], other.min_proj[i]);
    acc.max_proj[i] = std::max(acc.max_proj[i], other.max_proj[i]);
  }
  return acc;
}

template <typename Range, std::size_t Dims, typename Policy>
auto obb_from(const Range &polygons, const tf::polygon<Dims, Policy> &) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 2 || Dims == 3, "OBB computation implemented for 2D and 3D");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_polygons(polygons));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of(cov);

  tf::obb<T, Dims> box;

  // 3) Axes ordered by largest eigenvalue first
  for (std::size_t k = 0; k < Dims; ++k) {
    box.axes[k] = eigenvectors[Dims - 1 - k];
  }

  // 4) Project all vertices to get min/max along each axis
  auto proj_init = make_proj_accum_init<Dims, T>();

  auto proj_acc = tf::reduce(
      tf::make_mapped_range(polygons,
                            [&](const auto &poly) {
                              auto poly_acc = proj_init;
                              for (const auto &pt : poly) {
                                auto diff = pt - centroid;
                                for (std::size_t i = 0; i < Dims; ++i) {
                                  T p = tf::dot(diff, box.axes[i]);
                                  poly_acc.min_proj[i] = min(poly_acc.min_proj[i], p);
                                  poly_acc.max_proj[i] = max(poly_acc.max_proj[i], p);
                                }
                              }
                              return poly_acc;
                            }),
      merge_proj_accum<Dims, T>,
      proj_init, tf::checked);

  // 5) Store as corner + full extents
  box.origin = centroid;
  for (std::size_t i = 0; i < Dims; ++i) {
    box.origin = box.origin + box.axes[i] * proj_acc.min_proj[i];
    box.extent[i] = max(T(0), proj_acc.max_proj[i] - proj_acc.min_proj[i]);
  }

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto obb_from(const Range &segments, const tf::segment<Dims, Policy> &) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 2 || Dims == 3, "OBB computation implemented for 2D and 3D");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_segments(segments));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of(cov);

  tf::obb<T, Dims> box;

  // 3) Axes ordered by largest eigenvalue first
  for (std::size_t k = 0; k < Dims; ++k) {
    box.axes[k] = eigenvectors[Dims - 1 - k];
  }

  // 4) Project all vertices to get min/max along each axis
  auto proj_init = make_proj_accum_init<Dims, T>();

  auto proj_acc = tf::reduce(
      tf::make_mapped_range(segments,
                            [&](const auto &seg) {
                              auto seg_acc = proj_init;
                              for (const auto &pt : seg) {
                                auto diff = pt - centroid;
                                for (std::size_t i = 0; i < Dims; ++i) {
                                  T p = tf::dot(diff, box.axes[i]);
                                  seg_acc.min_proj[i] = min(seg_acc.min_proj[i], p);
                                  seg_acc.max_proj[i] = max(seg_acc.max_proj[i], p);
                                }
                              }
                              return seg_acc;
                            }),
      merge_proj_accum<Dims, T>,
      proj_init, tf::checked);

  // 5) Store as corner + full extents
  box.origin = centroid;
  for (std::size_t i = 0; i < Dims; ++i) {
    box.origin = box.origin + box.axes[i] * proj_acc.min_proj[i];
    box.extent[i] = max(T(0), proj_acc.max_proj[i] - proj_acc.min_proj[i]);
  }

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto obb_from(const Range &points, const tf::point_like<Dims, Policy> &) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 2 || Dims == 3, "OBB computation implemented for 2D and 3D");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_points(points));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of(cov);

  tf::obb<T, Dims> box;

  // 3) Axes ordered by largest eigenvalue first
  for (std::size_t k = 0; k < Dims; ++k) {
    box.axes[k] = eigenvectors[Dims - 1 - k];
  }

  // 4) Project all points to get min/max along each axis
  auto proj_init = make_proj_accum_init<Dims, T>();

  auto proj_acc = tf::reduce(
      tf::make_mapped_range(points,
                            [&](const auto &pt) {
                              auto diff = pt - centroid;
                              proj_accum<Dims, T> pt_acc;
                              for (std::size_t i = 0; i < Dims; ++i) {
                                T p = tf::dot(diff, box.axes[i]);
                                pt_acc.min_proj[i] = p;
                                pt_acc.max_proj[i] = p;
                              }
                              return pt_acc;
                            }),
      merge_proj_accum<Dims, T>,
      proj_init, tf::checked);

  // 5) Store as corner + full extents
  box.origin = centroid;
  for (std::size_t i = 0; i < Dims; ++i) {
    box.origin = box.origin + box.axes[i] * proj_acc.min_proj[i];
    box.extent[i] = max(T(0), proj_acc.max_proj[i] - proj_acc.min_proj[i]);
  }

  return box;
}

} // namespace core

// Convenience overloads in tf namespace
template <typename Policy>
auto obb_from(const tf::polygons<Policy> &polys) {
  return core::obb_from(polys, polys[0]);
}

template <typename Policy>
auto obb_from(const tf::segments<Policy> &segs) {
  return core::obb_from(segs, segs[0]);
}

template <typename Policy>
auto obb_from(const tf::points<Policy> &pts) {
  return core::obb_from(pts, pts[0]);
}

} // namespace tf
