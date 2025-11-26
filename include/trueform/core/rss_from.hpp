/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./algorithm/reduce.hpp"
#include "./base/rss_from_impl.hpp"
#include "./covariance_of.hpp"
#include "./dot.hpp"
#include "./eigen_of.hpp"
#include "./points.hpp"
#include "./sqrt.hpp"
#include "./polygons.hpp"
#include "./rss.hpp"
#include "./segments.hpp"

namespace tf {
namespace core {

template <typename Range, std::size_t Dims, typename Policy>
auto rss_from(const Range &polygons, const tf::polygon<Dims, Policy> &) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 3, "RSS computation only implemented for 3D");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_polygons(polygons));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of(cov);

  tf::rss<T, 3> box;

  // 3) Axes ordered by largest eigenvalue first
  for (int k = 0; k < 3; ++k) {
    const auto &ev = eigenvectors[2 - k];
    box.axes[k] = ev;
  }

  // 4) Project all vertices to get z min/max for radius (in centroid frame)
  struct proj_accum {
    T minz, maxz;
  };

  proj_accum proj_init{std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max()};

  auto proj_acc = tf::reduce(
      tf::make_mapped_range(polygons,
                            [&](const auto &poly) {
                              proj_accum poly_acc = proj_init;
                              for (const auto &pt : poly) {
                                auto diff = pt - centroid;
                                T pz = tf::dot(diff, box.axes[2]);
                                poly_acc.minz = min(poly_acc.minz, pz);
                                poly_acc.maxz = max(poly_acc.maxz, pz);
                              }
                              return poly_acc;
                            }),
      [](proj_accum acc, const auto &element) {
        acc.minz = std::min(acc.minz, element.minz);
        acc.maxz = std::max(acc.maxz, element.maxz);
        return acc;
      },
      proj_init, tf::checked);

  // 5) Compute radius and z-center (in centroid frame)
  T r = T(0.5) * (proj_acc.maxz - proj_acc.minz);
  T radsqr = r * r;
  T cz = T(0.5) * (proj_acc.maxz + proj_acc.minz);

  // 6) Compute shrunk x/y bounds (in centroid frame)
  struct rss_accum {
    T minx, maxx, miny, maxy;
  };

  rss_accum rss_init{std::numeric_limits<T>::max(),
                     -std::numeric_limits<T>::max(),
                     std::numeric_limits<T>::max(),
                     -std::numeric_limits<T>::max()};

  auto rss_acc = tf::reduce(
      tf::make_mapped_range(polygons,
                            [&](const auto &poly) {
                              rss_accum poly_rss = rss_init;
                              for (const auto &pt : poly) {
                                auto diff = pt - centroid;
                                T px = tf::dot(diff, box.axes[0]);
                                T py = tf::dot(diff, box.axes[1]);
                                T pz = tf::dot(diff, box.axes[2]);

                                T dz = pz - cz;
                                T dz2 = dz * dz;
                                T shrink =
                                    (dz2 < radsqr) ? tf::sqrt(radsqr - dz2) : T(0);

                                poly_rss.minx = min(poly_rss.minx, px + shrink);
                                poly_rss.maxx = max(poly_rss.maxx, px - shrink);
                                poly_rss.miny = min(poly_rss.miny, py + shrink);
                                poly_rss.maxy = max(poly_rss.maxy, py - shrink);
                              }
                              return poly_rss;
                            }),
      [](rss_accum acc, const auto &element) {
        acc.minx = std::min(acc.minx, element.minx);
        acc.maxx = std::max(acc.maxx, element.maxx);
        acc.miny = std::min(acc.miny, element.miny);
        acc.maxy = std::max(acc.maxy, element.maxy);
        return acc;
      },
      rss_init, tf::checked);

  // 7) Corner handling
  T minx = rss_acc.minx;
  T maxx = rss_acc.maxx;
  T miny = rss_acc.miny;
  T maxy = rss_acc.maxy;

  for (const auto &poly : polygons) {
    for (const auto &pt : poly) {
      auto diff = pt - centroid;
      T px = tf::dot(diff, box.axes[0]);
      T py = tf::dot(diff, box.axes[1]);
      T pz = tf::dot(diff, box.axes[2]);
      impl::update_corners(minx, maxx, miny, maxy, px, py, pz, cz, radsqr);
    }
  }

  // 8) Store as corner + full lengths + radius
  box.origin =
      centroid + box.axes[0] * minx + box.axes[1] * miny + box.axes[2] * cz;
  box.length[0] = max(T(0), maxx - minx);
  box.length[1] = max(T(0), maxy - miny);
  box.radius = r;

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto rss_from(const Range &segments, const tf::segment<Dims, Policy> &) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 3, "RSS computation only implemented for 3D");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_segments(segments));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of(cov);

  tf::rss<T, 3> box;

  // 3) Axes ordered by largest eigenvalue first
  for (int k = 0; k < 3; ++k) {
    const auto &ev = eigenvectors[2 - k];
    box.axes[k] = ev;
  }

  // 4) Project all vertices to get z min/max for radius
  struct proj_accum {
    T minz, maxz;
  };

  proj_accum proj_init{std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max()};

  auto proj_acc = tf::reduce(
      tf::make_mapped_range(segments,
                            [&](const auto &seg) {
                              proj_accum seg_acc = proj_init;
                              for (const auto &pt : seg) {
                                auto diff = pt - centroid;
                                T pz = tf::dot(diff, box.axes[2]);
                                seg_acc.minz = min(seg_acc.minz, pz);
                                seg_acc.maxz = max(seg_acc.maxz, pz);
                              }
                              return seg_acc;
                            }),
      [](proj_accum acc, const auto &element) {
        acc.minz = std::min(acc.minz, element.minz);
        acc.maxz = std::max(acc.maxz, element.maxz);
        return acc;
      },
      proj_init, tf::checked);

  // 5) Compute radius and z-center
  T r = T(0.5) * (proj_acc.maxz - proj_acc.minz);
  T radsqr = r * r;
  T cz = T(0.5) * (proj_acc.maxz + proj_acc.minz);

  // 6) Compute shrunk x/y bounds
  struct rss_accum {
    T minx, maxx, miny, maxy;
  };

  rss_accum rss_init{std::numeric_limits<T>::max(),
                     -std::numeric_limits<T>::max(),
                     std::numeric_limits<T>::max(),
                     -std::numeric_limits<T>::max()};

  auto rss_acc = tf::reduce(
      tf::make_mapped_range(segments,
                            [&](const auto &seg) {
                              rss_accum seg_rss = rss_init;
                              for (const auto &pt : seg) {
                                auto diff = pt - centroid;
                                T px = tf::dot(diff, box.axes[0]);
                                T py = tf::dot(diff, box.axes[1]);
                                T pz = tf::dot(diff, box.axes[2]);

                                T dz = pz - cz;
                                T dz2 = dz * dz;
                                T shrink =
                                    (dz2 < radsqr) ? tf::sqrt(radsqr - dz2) : T(0);

                                seg_rss.minx = min(seg_rss.minx, px + shrink);
                                seg_rss.maxx = max(seg_rss.maxx, px - shrink);
                                seg_rss.miny = min(seg_rss.miny, py + shrink);
                                seg_rss.maxy = max(seg_rss.maxy, py - shrink);
                              }
                              return seg_rss;
                            }),
      [](rss_accum acc, const auto &element) {
        acc.minx = std::min(acc.minx, element.minx);
        acc.maxx = std::max(acc.maxx, element.maxx);
        acc.miny = std::min(acc.miny, element.miny);
        acc.maxy = std::max(acc.maxy, element.maxy);
        return acc;
      },
      rss_init, tf::checked);

  // 7) Corner handling
  T minx = rss_acc.minx;
  T maxx = rss_acc.maxx;
  T miny = rss_acc.miny;
  T maxy = rss_acc.maxy;

  for (const auto &seg : segments) {
    for (const auto &pt : seg) {
      auto diff = pt - centroid;
      T px = tf::dot(diff, box.axes[0]);
      T py = tf::dot(diff, box.axes[1]);
      T pz = tf::dot(diff, box.axes[2]);
      impl::update_corners(minx, maxx, miny, maxy, px, py, pz, cz, radsqr);
    }
  }

  // 8) Store as corner + full lengths + radius
  box.origin =
      centroid + box.axes[0] * minx + box.axes[1] * miny + box.axes[2] * cz;
  box.length[0] = max(T(0), maxx - minx);
  box.length[1] = max(T(0), maxy - miny);
  box.radius = r;

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto rss_from(const Range &points, const tf::point_like<Dims, Policy> &) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 3, "RSS computation only implemented for 3D");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_points(points));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of(cov);

  tf::rss<T, 3> box;

  // 3) Axes ordered by largest eigenvalue first
  for (int k = 0; k < 3; ++k) {
    const auto &ev = eigenvectors[2 - k];
    box.axes[k] = ev;
  }

  // 4) Project all points to get z min/max for radius
  struct proj_accum {
    T minz, maxz;
  };

  proj_accum proj_init{std::numeric_limits<T>::max(),
                       -std::numeric_limits<T>::max()};

  auto proj_acc = tf::reduce(
      points,
      [&](proj_accum acc, const auto &pt) {
        auto diff = pt - centroid;
        T pz = tf::dot(diff, box.axes[2]);
        acc.minz = min(acc.minz, pz);
        acc.maxz = max(acc.maxz, pz);
        return acc;
      },
      proj_init, tf::checked);

  // 5) Compute radius and z-center
  T r = T(0.5) * (proj_acc.maxz - proj_acc.minz);
  T radsqr = r * r;
  T cz = T(0.5) * (proj_acc.maxz + proj_acc.minz);

  // 6) Compute shrunk x/y bounds
  struct rss_accum {
    T minx, maxx, miny, maxy;
  };

  rss_accum rss_init{std::numeric_limits<T>::max(),
                     -std::numeric_limits<T>::max(),
                     std::numeric_limits<T>::max(),
                     -std::numeric_limits<T>::max()};

  auto rss_acc = tf::reduce(
      points,
      [&](rss_accum acc, const auto &pt) {
        auto diff = pt - centroid;
        T px = tf::dot(diff, box.axes[0]);
        T py = tf::dot(diff, box.axes[1]);
        T pz = tf::dot(diff, box.axes[2]);

        T dz = pz - cz;
        T dz2 = dz * dz;
        T shrink = (dz2 < radsqr) ? tf::sqrt(radsqr - dz2) : T(0);

        acc.minx = min(acc.minx, px + shrink);
        acc.maxx = max(acc.maxx, px - shrink);
        acc.miny = min(acc.miny, py + shrink);
        acc.maxy = max(acc.maxy, py - shrink);
        return acc;
      },
      rss_init, tf::checked);

  // 7) Corner handling
  T minx = rss_acc.minx;
  T maxx = rss_acc.maxx;
  T miny = rss_acc.miny;
  T maxy = rss_acc.maxy;

  for (const auto &pt : points) {
    auto diff = pt - centroid;
    T px = tf::dot(diff, box.axes[0]);
    T py = tf::dot(diff, box.axes[1]);
    T pz = tf::dot(diff, box.axes[2]);
    impl::update_corners(minx, maxx, miny, maxy, px, py, pz, cz, radsqr);
  }

  // 8) Store as corner + full lengths + radius
  box.origin =
      centroid + box.axes[0] * minx + box.axes[1] * miny + box.axes[2] * cz;
  box.length[0] = max(T(0), maxx - minx);
  box.length[1] = max(T(0), maxy - miny);
  box.radius = r;

  return box;
}

} // namespace core

// Convenience overloads in tf namespace
template <typename Policy>
auto rss_from(const tf::polygons<Policy> &polys) {
  return core::rss_from(polys, polys[0]);
}

template <typename Policy>
auto rss_from(const tf::segments<Policy> &segs) {
  return core::rss_from(segs, segs[0]);
}

template <typename Policy>
auto rss_from(const tf::points<Policy> &pts) {
  return core::rss_from(pts, pts[0]);
}

} // namespace tf
