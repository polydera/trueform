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
#include "./base/rss_from_impl.hpp"
#include "./covariance_of.hpp"
#include "./dispatch.hpp"
#include "./dot.hpp"
#include "./eigen_of_symmetric.hpp"
#include "./empty_obbrss.hpp"
#include "./obbrss.hpp"
#include "./points.hpp"
#include "./polygons.hpp"
#include "./segments.hpp"
#include "./sqrt.hpp"

namespace tf {
namespace core {

template <typename Range, std::size_t Dims, typename Policy>
auto obbrss_from(const Range &polygons, dispatch_t<tf::polygon<Dims, Policy>>) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 2 || Dims == 3,
                "OBBRSS computation implemented for 2D and 3D only");

  // 1) Covariance + centroid (shared)
  auto [centroid, cov] = tf::covariance_of(tf::make_polygons(polygons));

  // 2) Eigen decomposition (shared)
  auto [eigenvalues, eigenvectors] = tf::eigen_of_symmetric(cov);

  tf::obbrss<T, Dims> box;

  // 3) Axes ordered by largest eigenvalue first (shared)
  for (std::size_t k = 0; k < Dims; ++k) {
    box.axes[k] = eigenvectors[Dims - 1 - k];
  }

  // 4) Project all vertices along radius axis (last axis) to get radius
  struct radius_accum {
    T min_r, max_r;
  };

  radius_accum radius_init{std::numeric_limits<T>::max(),
                           -std::numeric_limits<T>::max()};

  auto radius_acc = tf::reduce(
      tf::make_mapped_range(polygons,
                            [&, &centroid=centroid](const auto &poly) {
                              radius_accum poly_acc = radius_init;
                              for (const auto &pt : poly) {
                                auto diff = pt - centroid;
                                T pr = tf::dot(diff, box.axes[Dims - 1]);
                                poly_acc.min_r = min(poly_acc.min_r, pr);
                                poly_acc.max_r = max(poly_acc.max_r, pr);
                              }
                              return poly_acc;
                            }),
      [](radius_accum acc, const auto &element) {
        acc.min_r = std::min(acc.min_r, element.min_r);
        acc.max_r = std::max(acc.max_r, element.max_r);
        return acc;
      },
      radius_init, tf::checked);

  // 5) Compute radius and center along radius axis
  T r = T(0.5) * (radius_acc.max_r - radius_acc.min_r);
  T radsqr = r * r;
  T cr = T(0.5) * (radius_acc.max_r + radius_acc.min_r);

  if constexpr (Dims == 2) {
    // 2D: Project along both axes for OBB, shrink x for RSS
    struct proj_accum_2d {
      T minx, maxx, miny, maxy;
    };

    proj_accum_2d proj_init{
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};

    struct rss_accum_2d {
      T minx, maxx;
    };

    rss_accum_2d rss_init{std::numeric_limits<T>::max(),
                          -std::numeric_limits<T>::max()};

    auto combined_acc = tf::reduce(
        tf::make_mapped_range(
            polygons,
            [&, &centroid = centroid](const auto &poly) {
              proj_accum_2d poly_proj = proj_init;
              rss_accum_2d poly_rss = rss_init;
              for (const auto &pt : poly) {
                auto diff = pt - centroid;
                T px = tf::dot(diff, box.axes[0]);
                T py = tf::dot(diff, box.axes[1]);

                // OBB bounds
                poly_proj.minx = min(poly_proj.minx, px);
                poly_proj.maxx = max(poly_proj.maxx, px);
                poly_proj.miny = min(poly_proj.miny, py);
                poly_proj.maxy = max(poly_proj.maxy, py);

                // RSS shrunk bounds
                T dy = py - cr;
                T dy2 = dy * dy;
                T shrink = (dy2 < radsqr) ? tf::sqrt(radsqr - dy2) : T(0);
                poly_rss.minx = min(poly_rss.minx, px + shrink);
                poly_rss.maxx = max(poly_rss.maxx, px - shrink);
              }
              return std::make_pair(poly_proj, poly_rss);
            }),
        [](auto acc, const auto &element) {
          acc.first.minx = std::min(acc.first.minx, element.first.minx);
          acc.first.maxx = std::max(acc.first.maxx, element.first.maxx);
          acc.first.miny = std::min(acc.first.miny, element.first.miny);
          acc.first.maxy = std::max(acc.first.maxy, element.first.maxy);
          acc.second.minx = std::min(acc.second.minx, element.second.minx);
          acc.second.maxx = std::max(acc.second.maxx, element.second.maxx);
          return acc;
        },
        std::make_pair(proj_init, rss_init), tf::checked);

    auto proj_acc = combined_acc.first;
    auto rss_acc = combined_acc.second;

    // Build OBBRSS
    box.obb_origin =
        centroid + box.axes[0] * proj_acc.minx + box.axes[1] * proj_acc.miny;
    box.rss_origin = centroid + box.axes[0] * rss_acc.minx + box.axes[1] * cr;
    box.extent[0] = max(T(0), proj_acc.maxx - proj_acc.minx);
    box.extent[1] = max(T(0), proj_acc.maxy - proj_acc.miny);
    box.length[0] = max(T(0), rss_acc.maxx - rss_acc.minx);
    box.radius = r;
  } else {
    // 3D: Project along all axes, shrink x/y, corner handling
    struct proj_accum_3d {
      T minx, maxx, miny, maxy, minz, maxz;
    };

    proj_accum_3d proj_init{
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};

    auto proj_acc = tf::reduce(
        tf::make_mapped_range(polygons,
                              [&, &centroid = centroid](const auto &poly) {
                                proj_accum_3d poly_acc = proj_init;
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
        [](proj_accum_3d acc, const auto &element) {
          acc.minx = std::min(acc.minx, element.minx);
          acc.maxx = std::max(acc.maxx, element.maxx);
          acc.miny = std::min(acc.miny, element.miny);
          acc.maxy = std::max(acc.maxy, element.maxy);
          acc.minz = std::min(acc.minz, element.minz);
          acc.maxz = std::max(acc.maxz, element.maxz);
          return acc;
        },
        proj_init, tf::checked);

    struct rss_accum_3d {
      T minx, maxx, miny, maxy;
    };

    rss_accum_3d rss_init{
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};

    auto rss_acc = tf::reduce(
        tf::make_mapped_range(polygons,
                              [&, &centroid = centroid](const auto &poly) {
                                rss_accum_3d poly_rss = rss_init;
                                for (const auto &pt : poly) {
                                  auto diff = pt - centroid;
                                  T px = tf::dot(diff, box.axes[0]);
                                  T py = tf::dot(diff, box.axes[1]);
                                  T pz = tf::dot(diff, box.axes[2]);

                                  T dz = pz - cr;
                                  T dz2 = dz * dz;
                                  T shrink = (dz2 < radsqr)
                                                 ? tf::sqrt(radsqr - dz2)
                                                 : T(0);

                                  poly_rss.minx = min(poly_rss.minx, px + shrink);
                                  poly_rss.maxx = max(poly_rss.maxx, px - shrink);
                                  poly_rss.miny = min(poly_rss.miny, py + shrink);
                                  poly_rss.maxy = max(poly_rss.maxy, py - shrink);
                                }
                                return poly_rss;
                              }),
        [](rss_accum_3d acc, const auto &element) {
          acc.minx = std::min(acc.minx, element.minx);
          acc.maxx = std::max(acc.maxx, element.maxx);
          acc.miny = std::min(acc.miny, element.miny);
          acc.maxy = std::max(acc.maxy, element.maxy);
          return acc;
        },
        rss_init, tf::checked);

    // Corner handling
    T rss_minx = rss_acc.minx;
    T rss_maxx = rss_acc.maxx;
    T rss_miny = rss_acc.miny;
    T rss_maxy = rss_acc.maxy;

    for (const auto &poly : polygons) {
      for (const auto &pt : poly) {
        auto diff = pt - centroid;
        T px = tf::dot(diff, box.axes[0]);
        T py = tf::dot(diff, box.axes[1]);
        T pz = tf::dot(diff, box.axes[2]);
        impl::update_corners(rss_minx, rss_maxx, rss_miny, rss_maxy, px, py, pz,
                             cr, radsqr);
      }
    }

    // Build OBBRSS
    box.obb_origin = centroid + box.axes[0] * proj_acc.minx +
                     box.axes[1] * proj_acc.miny + box.axes[2] * proj_acc.minz;
    box.rss_origin =
        centroid + box.axes[0] * rss_minx + box.axes[1] * rss_miny +
        box.axes[2] * cr;
    box.extent[0] = max(T(0), proj_acc.maxx - proj_acc.minx);
    box.extent[1] = max(T(0), proj_acc.maxy - proj_acc.miny);
    box.extent[2] = max(T(0), proj_acc.maxz - proj_acc.minz);
    box.length[0] = max(T(0), rss_maxx - rss_minx);
    box.length[1] = max(T(0), rss_maxy - rss_miny);
    box.radius = r;
  }

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto obbrss_from(const Range &segments, dispatch_t<tf::segment<Dims, Policy>>) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 2 || Dims == 3,
                "OBBRSS computation implemented for 2D and 3D only");

  // 1) Covariance + centroid (shared)
  auto [centroid, cov] = tf::covariance_of(tf::make_segments(segments));

  // 2) Eigen decomposition (shared)
  auto [eigenvalues, eigenvectors] = tf::eigen_of_symmetric(cov);

  tf::obbrss<T, Dims> box;

  // 3) Axes ordered by largest eigenvalue first (shared)
  for (std::size_t k = 0; k < Dims; ++k) {
    box.axes[k] = eigenvectors[Dims - 1 - k];
  }

  // 4) Project all vertices along radius axis (last axis) to get radius
  struct radius_accum {
    T min_r, max_r;
  };

  radius_accum radius_init{std::numeric_limits<T>::max(),
                           -std::numeric_limits<T>::max()};

  auto radius_acc = tf::reduce(
      tf::make_mapped_range(segments,
                            [&, &centroid = centroid](const auto &seg) {
                              radius_accum seg_acc = radius_init;
                              for (const auto &pt : seg) {
                                auto diff = pt - centroid;
                                T pr = tf::dot(diff, box.axes[Dims - 1]);
                                seg_acc.min_r = min(seg_acc.min_r, pr);
                                seg_acc.max_r = max(seg_acc.max_r, pr);
                              }
                              return seg_acc;
                            }),
      [](radius_accum acc, const auto &element) {
        acc.min_r = std::min(acc.min_r, element.min_r);
        acc.max_r = std::max(acc.max_r, element.max_r);
        return acc;
      },
      radius_init, tf::checked);

  // 5) Compute radius and center along radius axis
  T r = T(0.5) * (radius_acc.max_r - radius_acc.min_r);
  T radsqr = r * r;
  T cr = T(0.5) * (radius_acc.max_r + radius_acc.min_r);

  if constexpr (Dims == 2) {
    // 2D: Project along both axes for OBB, shrink x for RSS
    struct proj_accum_2d {
      T minx, maxx, miny, maxy;
    };

    proj_accum_2d proj_init{
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};

    struct rss_accum_2d {
      T minx, maxx;
    };

    rss_accum_2d rss_init{std::numeric_limits<T>::max(),
                          -std::numeric_limits<T>::max()};

    auto combined_acc = tf::reduce(
        tf::make_mapped_range(
            segments,
            [&, &centroid = centroid](const auto &seg) {
              proj_accum_2d seg_proj = proj_init;
              rss_accum_2d seg_rss = rss_init;
              for (const auto &pt : seg) {
                auto diff = pt - centroid;
                T px = tf::dot(diff, box.axes[0]);
                T py = tf::dot(diff, box.axes[1]);

                // OBB bounds
                seg_proj.minx = min(seg_proj.minx, px);
                seg_proj.maxx = max(seg_proj.maxx, px);
                seg_proj.miny = min(seg_proj.miny, py);
                seg_proj.maxy = max(seg_proj.maxy, py);

                // RSS shrunk bounds
                T dy = py - cr;
                T dy2 = dy * dy;
                T shrink = (dy2 < radsqr) ? tf::sqrt(radsqr - dy2) : T(0);
                seg_rss.minx = min(seg_rss.minx, px + shrink);
                seg_rss.maxx = max(seg_rss.maxx, px - shrink);
              }
              return std::make_pair(seg_proj, seg_rss);
            }),
        [](auto acc, const auto &element) {
          acc.first.minx = std::min(acc.first.minx, element.first.minx);
          acc.first.maxx = std::max(acc.first.maxx, element.first.maxx);
          acc.first.miny = std::min(acc.first.miny, element.first.miny);
          acc.first.maxy = std::max(acc.first.maxy, element.first.maxy);
          acc.second.minx = std::min(acc.second.minx, element.second.minx);
          acc.second.maxx = std::max(acc.second.maxx, element.second.maxx);
          return acc;
        },
        std::make_pair(proj_init, rss_init), tf::checked);

    auto proj_acc = combined_acc.first;
    auto rss_acc = combined_acc.second;

    // Build OBBRSS
    box.obb_origin =
        centroid + box.axes[0] * proj_acc.minx + box.axes[1] * proj_acc.miny;
    box.rss_origin = centroid + box.axes[0] * rss_acc.minx + box.axes[1] * cr;
    box.extent[0] = max(T(0), proj_acc.maxx - proj_acc.minx);
    box.extent[1] = max(T(0), proj_acc.maxy - proj_acc.miny);
    box.length[0] = max(T(0), rss_acc.maxx - rss_acc.minx);
    box.radius = r;
  } else {
    // 3D: Project along all axes, shrink x/y, corner handling
    struct proj_accum_3d {
      T minx, maxx, miny, maxy, minz, maxz;
    };

    proj_accum_3d proj_init{
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};

    auto proj_acc = tf::reduce(
        tf::make_mapped_range(segments,
                              [&, &centroid = centroid](const auto &seg) {
                                proj_accum_3d seg_acc = proj_init;
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
        [](proj_accum_3d acc, const auto &element) {
          acc.minx = std::min(acc.minx, element.minx);
          acc.maxx = std::max(acc.maxx, element.maxx);
          acc.miny = std::min(acc.miny, element.miny);
          acc.maxy = std::max(acc.maxy, element.maxy);
          acc.minz = std::min(acc.minz, element.minz);
          acc.maxz = std::max(acc.maxz, element.maxz);
          return acc;
        },
        proj_init, tf::checked);

    struct rss_accum_3d {
      T minx, maxx, miny, maxy;
    };

    rss_accum_3d rss_init{
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};

    auto rss_acc = tf::reduce(
        tf::make_mapped_range(segments,
                              [&, &centroid = centroid](const auto &seg) {
                                rss_accum_3d seg_rss = rss_init;
                                for (const auto &pt : seg) {
                                  auto diff = pt - centroid;
                                  T px = tf::dot(diff, box.axes[0]);
                                  T py = tf::dot(diff, box.axes[1]);
                                  T pz = tf::dot(diff, box.axes[2]);

                                  T dz = pz - cr;
                                  T dz2 = dz * dz;
                                  T shrink = (dz2 < radsqr)
                                                 ? tf::sqrt(radsqr - dz2)
                                                 : T(0);

                                  seg_rss.minx = min(seg_rss.minx, px + shrink);
                                  seg_rss.maxx = max(seg_rss.maxx, px - shrink);
                                  seg_rss.miny = min(seg_rss.miny, py + shrink);
                                  seg_rss.maxy = max(seg_rss.maxy, py - shrink);
                                }
                                return seg_rss;
                              }),
        [](rss_accum_3d acc, const auto &element) {
          acc.minx = std::min(acc.minx, element.minx);
          acc.maxx = std::max(acc.maxx, element.maxx);
          acc.miny = std::min(acc.miny, element.miny);
          acc.maxy = std::max(acc.maxy, element.maxy);
          return acc;
        },
        rss_init, tf::checked);

    // Corner handling
    T rss_minx = rss_acc.minx;
    T rss_maxx = rss_acc.maxx;
    T rss_miny = rss_acc.miny;
    T rss_maxy = rss_acc.maxy;

    for (const auto &seg : segments) {
      for (const auto &pt : seg) {
        auto diff = pt - centroid;
        T px = tf::dot(diff, box.axes[0]);
        T py = tf::dot(diff, box.axes[1]);
        T pz = tf::dot(diff, box.axes[2]);
        impl::update_corners(rss_minx, rss_maxx, rss_miny, rss_maxy, px, py, pz,
                             cr, radsqr);
      }
    }

    // Build OBBRSS
    box.obb_origin = centroid + box.axes[0] * proj_acc.minx +
                     box.axes[1] * proj_acc.miny + box.axes[2] * proj_acc.minz;
    box.rss_origin =
        centroid + box.axes[0] * rss_minx + box.axes[1] * rss_miny +
        box.axes[2] * cr;
    box.extent[0] = max(T(0), proj_acc.maxx - proj_acc.minx);
    box.extent[1] = max(T(0), proj_acc.maxy - proj_acc.miny);
    box.extent[2] = max(T(0), proj_acc.maxz - proj_acc.minz);
    box.length[0] = max(T(0), rss_maxx - rss_minx);
    box.length[1] = max(T(0), rss_maxy - rss_miny);
    box.radius = r;
  }

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto obbrss_from(const Range &points, dispatch_t<tf::point_like<Dims, Policy>>) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 2 || Dims == 3,
                "OBBRSS computation implemented for 2D and 3D only");

  // 1) Covariance + centroid (shared)
  auto [centroid, cov] = tf::covariance_of(tf::make_points(points));

  // 2) Eigen decomposition (shared)
  auto [eigenvalues, eigenvectors] = tf::eigen_of_symmetric(cov);

  tf::obbrss<T, Dims> box;

  // 3) Axes ordered by largest eigenvalue first (shared)
  for (std::size_t k = 0; k < Dims; ++k) {
    box.axes[k] = eigenvectors[Dims - 1 - k];
  }

  // 4) Project all points along radius axis (last axis) to get radius
  struct radius_accum {
    T min_r, max_r;
  };

  radius_accum radius_init{std::numeric_limits<T>::max(),
                           -std::numeric_limits<T>::max()};

  auto radius_acc = tf::reduce(
      tf::make_mapped_range(points,
                            [&, &centroid = centroid](const auto &pt) {
                              auto diff = pt - centroid;
                              T pr = tf::dot(diff, box.axes[Dims - 1]);
                              return radius_accum{pr, pr};
                            }),
      [](radius_accum acc, const auto &element) {
        acc.min_r = std::min(acc.min_r, element.min_r);
        acc.max_r = std::max(acc.max_r, element.max_r);
        return acc;
      },
      radius_init, tf::checked);

  // 5) Compute radius and center along radius axis
  T r = T(0.5) * (radius_acc.max_r - radius_acc.min_r);
  T radsqr = r * r;
  T cr = T(0.5) * (radius_acc.max_r + radius_acc.min_r);

  if constexpr (Dims == 2) {
    // 2D: Project along both axes for OBB, shrink x for RSS
    struct proj_accum_2d {
      T minx, maxx, miny, maxy;
    };

    proj_accum_2d proj_init{
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};

    struct rss_accum_2d {
      T minx, maxx;
    };

    rss_accum_2d rss_init{std::numeric_limits<T>::max(),
                          -std::numeric_limits<T>::max()};

    auto combined_acc = tf::reduce(
        tf::make_mapped_range(
            points,
            [&, &centroid = centroid](const auto &pt) {
              auto diff = pt - centroid;
              T px = tf::dot(diff, box.axes[0]);
              T py = tf::dot(diff, box.axes[1]);

              // RSS shrunk bounds
              T dy = py - cr;
              T dy2 = dy * dy;
              T shrink = (dy2 < radsqr) ? tf::sqrt(radsqr - dy2) : T(0);

              return std::make_pair(proj_accum_2d{px, px, py, py},
                                    rss_accum_2d{px + shrink, px - shrink});
            }),
        [](auto acc, const auto &element) {
          acc.first.minx = std::min(acc.first.minx, element.first.minx);
          acc.first.maxx = std::max(acc.first.maxx, element.first.maxx);
          acc.first.miny = std::min(acc.first.miny, element.first.miny);
          acc.first.maxy = std::max(acc.first.maxy, element.first.maxy);
          acc.second.minx = std::min(acc.second.minx, element.second.minx);
          acc.second.maxx = std::max(acc.second.maxx, element.second.maxx);
          return acc;
        },
        std::make_pair(proj_init, rss_init), tf::checked);

    auto proj_acc = combined_acc.first;
    auto rss_acc = combined_acc.second;

    // Build OBBRSS
    box.obb_origin =
        centroid + box.axes[0] * proj_acc.minx + box.axes[1] * proj_acc.miny;
    box.rss_origin = centroid + box.axes[0] * rss_acc.minx + box.axes[1] * cr;
    box.extent[0] = max(T(0), proj_acc.maxx - proj_acc.minx);
    box.extent[1] = max(T(0), proj_acc.maxy - proj_acc.miny);
    box.length[0] = max(T(0), rss_acc.maxx - rss_acc.minx);
    box.radius = r;
  } else {
    // 3D: Project along all axes, shrink x/y, corner handling
    struct proj_accum_3d {
      T minx, maxx, miny, maxy, minz, maxz;
    };

    proj_accum_3d proj_init{
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};

    auto proj_acc = tf::reduce(
        tf::make_mapped_range(points,
                              [&, &centroid = centroid](const auto &pt) {
                                auto diff = pt - centroid;
                                T px = tf::dot(diff, box.axes[0]);
                                T py = tf::dot(diff, box.axes[1]);
                                T pz = tf::dot(diff, box.axes[2]);
                                return proj_accum_3d{px, px, py, py, pz, pz};
                              }),
        [](proj_accum_3d acc, const auto &element) {
          acc.minx = std::min(acc.minx, element.minx);
          acc.maxx = std::max(acc.maxx, element.maxx);
          acc.miny = std::min(acc.miny, element.miny);
          acc.maxy = std::max(acc.maxy, element.maxy);
          acc.minz = std::min(acc.minz, element.minz);
          acc.maxz = std::max(acc.maxz, element.maxz);
          return acc;
        },
        proj_init, tf::checked);

    struct rss_accum_3d {
      T minx, maxx, miny, maxy;
    };

    rss_accum_3d rss_init{
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max(),
        std::numeric_limits<T>::max(), -std::numeric_limits<T>::max()};

    auto rss_acc = tf::reduce(
        tf::make_mapped_range(points,
                              [&, &centroid = centroid](const auto &pt) {
                                auto diff = pt - centroid;
                                T px = tf::dot(diff, box.axes[0]);
                                T py = tf::dot(diff, box.axes[1]);
                                T pz = tf::dot(diff, box.axes[2]);

                                T dz = pz - cr;
                                T dz2 = dz * dz;
                                T shrink =
                                    (dz2 < radsqr) ? tf::sqrt(radsqr - dz2) : T(0);

                                return rss_accum_3d{px + shrink, px - shrink,
                                                     py + shrink, py - shrink};
                              }),
        [](rss_accum_3d acc, const auto &element) {
          acc.minx = std::min(acc.minx, element.minx);
          acc.maxx = std::max(acc.maxx, element.maxx);
          acc.miny = std::min(acc.miny, element.miny);
          acc.maxy = std::max(acc.maxy, element.maxy);
          return acc;
        },
        rss_init, tf::checked);

    // Corner handling
    T rss_minx = rss_acc.minx;
    T rss_maxx = rss_acc.maxx;
    T rss_miny = rss_acc.miny;
    T rss_maxy = rss_acc.maxy;

    for (const auto &pt : points) {
      auto diff = pt - centroid;
      T px = tf::dot(diff, box.axes[0]);
      T py = tf::dot(diff, box.axes[1]);
      T pz = tf::dot(diff, box.axes[2]);
      impl::update_corners(rss_minx, rss_maxx, rss_miny, rss_maxy, px, py, pz,
                           cr, radsqr);
    }

    // Build OBBRSS
    box.obb_origin = centroid + box.axes[0] * proj_acc.minx +
                     box.axes[1] * proj_acc.miny + box.axes[2] * proj_acc.minz;
    box.rss_origin =
        centroid + box.axes[0] * rss_minx + box.axes[1] * rss_miny +
        box.axes[2] * cr;
    box.extent[0] = max(T(0), proj_acc.maxx - proj_acc.minx);
    box.extent[1] = max(T(0), proj_acc.maxy - proj_acc.miny);
    box.extent[2] = max(T(0), proj_acc.maxz - proj_acc.minz);
    box.length[0] = max(T(0), rss_maxx - rss_minx);
    box.length[1] = max(T(0), rss_maxy - rss_miny);
    box.radius = r;
  }

  return box;
}

} // namespace core

/// @ingroup core_primitives
/// @brief Create OBBRSS bounding volume from polygons.
///
/// Computes a tight-fitting OBB-RSS hybrid bounding volume using
/// covariance-based principal component analysis.
///
/// @tparam Policy The polygons' policy type.
/// @param polys The polygon mesh.
/// @return An @ref tf::obbrss bounding volume.
template <typename Policy>
auto obbrss_from(const tf::polygons<Policy> &polys) {
  if (!polys.size())
    return tf::make_empty_obbrss<tf::coordinate_type<Policy>,
                                 tf::coordinate_dims_v<Policy>>();
  return core::obbrss_from(polys, core::dispatch_element(polys));
}

/// @ingroup core_primitives
/// @brief Create OBBRSS bounding volume from segments.
/// @overload
template <typename Policy>
auto obbrss_from(const tf::segments<Policy> &segs) {
  if (!segs.size())
    return tf::make_empty_obbrss<tf::coordinate_type<Policy>,
                                 tf::coordinate_dims_v<Policy>>();
  return core::obbrss_from(segs, core::dispatch_element(segs));
}

/// @ingroup core_primitives
/// @brief Create OBBRSS bounding volume from points.
/// @overload
template <typename Policy>
auto obbrss_from(const tf::points<Policy> &pts) {
  if (!pts.size())
    return tf::make_empty_obbrss<tf::coordinate_type<Policy>,
                                 tf::coordinate_dims_v<Policy>>();
  return core::obbrss_from(pts, core::dispatch_element(pts));
}

} // namespace tf
