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

template <typename Range, std::size_t Dims, typename Policy>
auto obb_from(const Range &polygons, const tf::polygon<Dims, Policy> &) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 3, "OBB computation only implemented for 3D");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_polygons(polygons));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of(cov);

  tf::obb<T, 3> box;

  // 3) Axes ordered by largest eigenvalue first
  for (int k = 0; k < 3; ++k) {
    const auto &ev = eigenvectors[2 - k];
    box.axes[k] = ev;
  }

  // 4) Project all vertices to get min/max along each axis
  struct proj_accum {
    T minx, maxx, miny, maxy, minz, maxz;
  };

  proj_accum proj_init{std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max(),
                       std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max(),
                       std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max()};

  auto proj_acc = tf::reduce(
      tf::make_mapped_range(polygons,
                            [&](const auto &poly) {
                              proj_accum poly_acc = proj_init;
                              for (const auto &pt : poly) {
                                auto diff = pt - centroid;
                                T px = tf::dot(diff, box.axes[0]);
                                T py = tf::dot(diff, box.axes[1]);
                                T pz = tf::dot(diff, box.axes[2]);
                                poly_acc.minx = min(poly_acc.minx, px);
                                poly_acc.maxx = max(poly_acc.maxx, px);
                                poly_acc.miny = min(poly_acc.miny, py);
                                poly_acc.maxy = max(poly_acc.maxy, py);
                                poly_acc.minz = min(poly_acc.minz, pz);
                                poly_acc.maxz = max(poly_acc.maxz, pz);
                              }
                              return poly_acc;
                            }),
      [](proj_accum acc, const auto &element) {
        acc.minx = std::min(acc.minx, element.minx);
        acc.maxx = std::max(acc.maxx, element.maxx);
        acc.miny = std::min(acc.miny, element.miny);
        acc.maxy = std::max(acc.maxy, element.maxy);
        acc.minz = std::min(acc.minz, element.minz);
        acc.maxz = std::max(acc.maxz, element.maxz);
        return acc;
      },
      proj_init, tf::checked);

  // 5) Store as corner + full extents
  box.origin = centroid + box.axes[0] * proj_acc.minx +
               box.axes[1] * proj_acc.miny + box.axes[2] * proj_acc.minz;
  box.extent[0] = max(T(0), proj_acc.maxx - proj_acc.minx);
  box.extent[1] = max(T(0), proj_acc.maxy - proj_acc.miny);
  box.extent[2] = max(T(0), proj_acc.maxz - proj_acc.minz);

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto obb_from(const Range &segments, const tf::segment<Dims, Policy> &) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 3, "OBB computation only implemented for 3D");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_segments(segments));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of(cov);

  tf::obb<T, 3> box;

  // 3) Axes ordered by largest eigenvalue first
  for (int k = 0; k < 3; ++k) {
    const auto &ev = eigenvectors[2 - k];
    box.axes[k] = ev;
  }

  // 4) Project all vertices to get min/max along each axis
  struct proj_accum {
    T minx, maxx, miny, maxy, minz, maxz;
  };

  proj_accum proj_init{std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max(),
                       std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max(),
                       std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max()};

  auto proj_acc = tf::reduce(
      tf::make_mapped_range(segments,
                            [&](const auto &seg) {
                              proj_accum seg_acc = proj_init;
                              for (const auto &pt : seg) {
                                auto diff = pt - centroid;
                                T px = tf::dot(diff, box.axes[0]);
                                T py = tf::dot(diff, box.axes[1]);
                                T pz = tf::dot(diff, box.axes[2]);
                                seg_acc.minx = min(seg_acc.minx, px);
                                seg_acc.maxx = max(seg_acc.maxx, px);
                                seg_acc.miny = min(seg_acc.miny, py);
                                seg_acc.maxy = max(seg_acc.maxy, py);
                                seg_acc.minz = min(seg_acc.minz, pz);
                                seg_acc.maxz = max(seg_acc.maxz, pz);
                              }
                              return seg_acc;
                            }),
      [](proj_accum acc, const auto &element) {
        acc.minx = std::min(acc.minx, element.minx);
        acc.maxx = std::max(acc.maxx, element.maxx);
        acc.miny = std::min(acc.miny, element.miny);
        acc.maxy = std::max(acc.maxy, element.maxy);
        acc.minz = std::min(acc.minz, element.minz);
        acc.maxz = std::max(acc.maxz, element.maxz);
        return acc;
      },
      proj_init, tf::checked);

  // 5) Store as corner + full extents
  box.origin = centroid + box.axes[0] * proj_acc.minx +
               box.axes[1] * proj_acc.miny + box.axes[2] * proj_acc.minz;
  box.extent[0] = max(T(0), proj_acc.maxx - proj_acc.minx);
  box.extent[1] = max(T(0), proj_acc.maxy - proj_acc.miny);
  box.extent[2] = max(T(0), proj_acc.maxz - proj_acc.minz);

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto obb_from(const Range &points, const tf::point_like<Dims, Policy> &) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 3, "OBB computation only implemented for 3D");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_points(points));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of(cov);

  tf::obb<T, 3> box;

  // 3) Axes ordered by largest eigenvalue first
  for (int k = 0; k < 3; ++k) {
    const auto &ev = eigenvectors[2 - k];
    box.axes[k] = ev;
  }

  // 4) Project all points to get min/max along each axis
  struct proj_accum {
    T minx, maxx, miny, maxy, minz, maxz;
  };

  proj_accum proj_init{std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max(),
                       std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max(),
                       std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max()};

  auto proj_acc = tf::reduce(
      points,
      [&](proj_accum acc, const auto &pt) {
        auto diff = pt - centroid;
        T px = tf::dot(diff, box.axes[0]);
        T py = tf::dot(diff, box.axes[1]);
        T pz = tf::dot(diff, box.axes[2]);
        acc.minx = min(acc.minx, px);
        acc.maxx = max(acc.maxx, px);
        acc.miny = min(acc.miny, py);
        acc.maxy = max(acc.maxy, py);
        acc.minz = min(acc.minz, pz);
        acc.maxz = max(acc.maxz, pz);
        return acc;
      },
      proj_init, tf::checked);

  // 5) Store as corner + full extents
  box.origin = centroid + box.axes[0] * proj_acc.minx +
               box.axes[1] * proj_acc.miny + box.axes[2] * proj_acc.minz;
  box.extent[0] = max(T(0), proj_acc.maxx - proj_acc.minx);
  box.extent[1] = max(T(0), proj_acc.maxy - proj_acc.miny);
  box.extent[2] = max(T(0), proj_acc.maxz - proj_acc.minz);

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
