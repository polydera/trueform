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
#include "./aabb_like.hpp"
#include "./algorithm/reduce.hpp"
#include "./base/rss_from_impl.hpp"
#include "./covariance_of.hpp"
#include "./dispatch.hpp"
#include "./dot.hpp"
#include "./eigen_of_symmetric.hpp"
#include "./empty_rss.hpp"
#include "./obb_like.hpp"
#include "./points.hpp"
#include "./polygons.hpp"
#include "./rss.hpp"
#include "./segments.hpp"
#include "./sqrt.hpp"
#include "./unit_vector.hpp"
#include "./vector.hpp"

namespace tf {
namespace core {

template <typename Range, std::size_t Dims, typename Policy>
auto rss_from(const Range &polygons, dispatch_t<tf::polygon<Dims, Policy>>) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 2 || Dims == 3,
                "RSS computation implemented for 2D and 3D only");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_polygons(polygons));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of_symmetric(cov);

  tf::rss<T, Dims> box;

  // 3) Axes ordered by largest eigenvalue first
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
                            [&, &centroid = centroid](const auto &poly) {
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
    // 2D: Shrink x bounds only, no corner handling needed
    struct rss_accum_2d {
      T minx, maxx;
    };

    rss_accum_2d rss_init{std::numeric_limits<T>::max(),
                          -std::numeric_limits<T>::max()};

    auto rss_acc = tf::reduce(
        tf::make_mapped_range(polygons,
                              [&, &centroid = centroid](const auto &poly) {
                                rss_accum_2d poly_rss = rss_init;
                                for (const auto &pt : poly) {
                                  auto diff = pt - centroid;
                                  T px = tf::dot(diff, box.axes[0]);
                                  T py = tf::dot(diff, box.axes[1]);

                                  T dy = py - cr;
                                  T dy2 = dy * dy;
                                  T shrink =
                                      (dy2 < radsqr) ? tf::sqrt(radsqr - dy2) : T(0);

                                  poly_rss.minx = min(poly_rss.minx, px + shrink);
                                  poly_rss.maxx = max(poly_rss.maxx, px - shrink);
                                }
                                return poly_rss;
                              }),
        [](rss_accum_2d acc, const auto &element) {
          acc.minx = std::min(acc.minx, element.minx);
          acc.maxx = std::max(acc.maxx, element.maxx);
          return acc;
        },
        rss_init, tf::checked);

    // Store as corner + length + radius
    box.origin = centroid + box.axes[0] * rss_acc.minx + box.axes[1] * cr;
    box.length[0] = max(T(0), rss_acc.maxx - rss_acc.minx);
    box.radius = r;
  } else {
    // 3D: Shrink x/y bounds, then corner handling
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
                                  T shrink =
                                      (dz2 < radsqr) ? tf::sqrt(radsqr - dz2) : T(0);

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
        impl::update_corners(minx, maxx, miny, maxy, px, py, pz, cr, radsqr);
      }
    }

    // Store as corner + full lengths + radius
    box.origin =
        centroid + box.axes[0] * minx + box.axes[1] * miny + box.axes[2] * cr;
    box.length[0] = max(T(0), maxx - minx);
    box.length[1] = max(T(0), maxy - miny);
    box.radius = r;
  }

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto rss_from(const Range &segments, dispatch_t<tf::segment<Dims, Policy>>) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 2 || Dims == 3,
                "RSS computation implemented for 2D and 3D only");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_segments(segments));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of_symmetric(cov);

  tf::rss<T, Dims> box;

  // 3) Axes ordered by largest eigenvalue first
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
    // 2D: Shrink x bounds only, no corner handling needed
    struct rss_accum_2d {
      T minx, maxx;
    };

    rss_accum_2d rss_init{std::numeric_limits<T>::max(),
                          -std::numeric_limits<T>::max()};

    auto rss_acc = tf::reduce(
        tf::make_mapped_range(segments,
                              [&, &centroid = centroid](const auto &seg) {
                                rss_accum_2d seg_rss = rss_init;
                                for (const auto &pt : seg) {
                                  auto diff = pt - centroid;
                                  T px = tf::dot(diff, box.axes[0]);
                                  T py = tf::dot(diff, box.axes[1]);

                                  T dy = py - cr;
                                  T dy2 = dy * dy;
                                  T shrink =
                                      (dy2 < radsqr) ? tf::sqrt(radsqr - dy2) : T(0);

                                  seg_rss.minx = min(seg_rss.minx, px + shrink);
                                  seg_rss.maxx = max(seg_rss.maxx, px - shrink);
                                }
                                return seg_rss;
                              }),
        [](rss_accum_2d acc, const auto &element) {
          acc.minx = std::min(acc.minx, element.minx);
          acc.maxx = std::max(acc.maxx, element.maxx);
          return acc;
        },
        rss_init, tf::checked);

    // Store as corner + length + radius
    box.origin = centroid + box.axes[0] * rss_acc.minx + box.axes[1] * cr;
    box.length[0] = max(T(0), rss_acc.maxx - rss_acc.minx);
    box.radius = r;
  } else {
    // 3D: Shrink x/y bounds, then corner handling
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
                                  T shrink =
                                      (dz2 < radsqr) ? tf::sqrt(radsqr - dz2) : T(0);

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
        impl::update_corners(minx, maxx, miny, maxy, px, py, pz, cr, radsqr);
      }
    }

    // Store as corner + full lengths + radius
    box.origin =
        centroid + box.axes[0] * minx + box.axes[1] * miny + box.axes[2] * cr;
    box.length[0] = max(T(0), maxx - minx);
    box.length[1] = max(T(0), maxy - miny);
    box.radius = r;
  }

  return box;
}

template <typename Range, std::size_t Dims, typename Policy>
auto rss_from(const Range &points, dispatch_t<tf::point_like<Dims, Policy>>) {
  using std::max;
  using std::min;
  using T = tf::coordinate_type<Policy>;

  static_assert(Dims == 2 || Dims == 3,
                "RSS computation implemented for 2D and 3D only");

  // 1) Covariance + centroid
  auto [centroid, cov] = tf::covariance_of(tf::make_points(points));

  // 2) Eigen decomposition
  auto [eigenvalues, eigenvectors] = tf::eigen_of_symmetric(cov);

  tf::rss<T, Dims> box;

  // 3) Axes ordered by largest eigenvalue first
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
    // 2D: Shrink x bounds only, no corner handling needed
    struct rss_accum_2d {
      T minx, maxx;
    };

    rss_accum_2d rss_init{std::numeric_limits<T>::max(),
                          -std::numeric_limits<T>::max()};

    auto rss_acc = tf::reduce(
        tf::make_mapped_range(points,
                              [&, &centroid = centroid](const auto &pt) {
                                auto diff = pt - centroid;
                                T px = tf::dot(diff, box.axes[0]);
                                T py = tf::dot(diff, box.axes[1]);

                                T dy = py - cr;
                                T dy2 = dy * dy;
                                T shrink =
                                    (dy2 < radsqr) ? tf::sqrt(radsqr - dy2) : T(0);

                                return rss_accum_2d{px + shrink, px - shrink};
                              }),
        [](rss_accum_2d acc, const auto &element) {
          acc.minx = std::min(acc.minx, element.minx);
          acc.maxx = std::max(acc.maxx, element.maxx);
          return acc;
        },
        rss_init, tf::checked);

    // Store as corner + length + radius
    box.origin = centroid + box.axes[0] * rss_acc.minx + box.axes[1] * cr;
    box.length[0] = max(T(0), rss_acc.maxx - rss_acc.minx);
    box.radius = r;
  } else {
    // 3D: Shrink x/y bounds, then corner handling
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
          acc.minx = min(acc.minx, element.minx);
          acc.maxx = max(acc.maxx, element.maxx);
          acc.miny = min(acc.miny, element.miny);
          acc.maxy = max(acc.maxy, element.maxy);
          return acc;
        },
        rss_init, tf::checked);

    // Corner handling
    T minx = rss_acc.minx;
    T maxx = rss_acc.maxx;
    T miny = rss_acc.miny;
    T maxy = rss_acc.maxy;

    for (const auto &pt : points) {
      auto diff = pt - centroid;
      T px = tf::dot(diff, box.axes[0]);
      T py = tf::dot(diff, box.axes[1]);
      T pz = tf::dot(diff, box.axes[2]);
      impl::update_corners(minx, maxx, miny, maxy, px, py, pz, cr, radsqr);
    }

    // Store as corner + full lengths + radius
    box.origin =
        centroid + box.axes[0] * minx + box.axes[1] * miny + box.axes[2] * cr;
    box.length[0] = max(T(0), maxx - minx);
    box.length[1] = max(T(0), maxy - miny);
    box.radius = r;
  }

  return box;
}

} // namespace core

/// @ingroup core_primitives
/// @brief Create RSS bounding volume from polygons.
///
/// Computes a tight-fitting rectangle swept sphere using
/// covariance-based principal component analysis.
///
/// @tparam Policy The polygons' policy type.
/// @param polys The polygon mesh.
/// @return An @ref tf::rss bounding volume.
template <typename Policy> auto rss_from(const tf::polygons<Policy> &polys) {
  if (!polys.size())
    return tf::make_empty_rss<tf::coordinate_type<Policy>,
                              tf::coordinate_dims_v<Policy>>();
  return core::rss_from(polys, core::dispatch_element(polys));
}

/// @ingroup core_primitives
/// @brief Create RSS bounding volume from segments.
/// @overload
template <typename Policy> auto rss_from(const tf::segments<Policy> &segs) {
  if (!segs.size())
    return tf::make_empty_rss<tf::coordinate_type<Policy>,
                              tf::coordinate_dims_v<Policy>>();
  return core::rss_from(segs, core::dispatch_element(segs));
}

/// @ingroup core_primitives
/// @brief Create RSS bounding volume from points.
/// @overload
template <typename Policy> auto rss_from(const tf::points<Policy> &pts) {
  if (!pts.size())
    return tf::make_empty_rss<tf::coordinate_type<Policy>,
                              tf::coordinate_dims_v<Policy>>();
  return core::rss_from(pts, core::dispatch_element(pts));
}

/// @ingroup core_primitives
/// @brief Convert an OBB to an RSS approximation.
///
/// In 3D: Creates an RSS by placing the rectangle at the midplane of axes[2].
/// The rectangle has dimensions extent[0] x extent[1], and radius =
/// extent[2]/2. This tightly bounds the OBB (slightly larger due to spherical
/// caps).
///
/// In 2D: Creates an RSS (stadium) by placing the segment at the midline of
/// axes[1]. The segment has length extent[0], and radius = extent[1]/2.
template <std::size_t Dims, typename Policy>
auto rss_from(const tf::obb_like<Dims, Policy> &obb) {
  static_assert(Dims == 2 || Dims == 3,
                "rss_from(obb) is implemented for 2D and 3D only.");
  using T = tf::coordinate_type<Policy>;

  if constexpr (Dims == 2) {
    // 2D: Origin at start of segment, centered on radius axis
    auto origin = obb.origin + obb.axes[1] * (obb.extent[1] * T(0.5));
    return tf::make_rss_like(origin, obb.axes, std::array<T, 1>{obb.extent[0]},
                             obb.extent[1] * T(0.5));
  } else {
    // 3D: Origin at corner of rectangle, centered on radius axis
    auto origin = obb.origin + obb.axes[2] * (obb.extent[2] * T(0.5));
    return tf::make_rss_like(origin, obb.axes,
                             std::array<T, 2>{obb.extent[0], obb.extent[1]},
                             obb.extent[2] * T(0.5));
  }
}

/// @ingroup core_primitives
/// @brief Convert an AABB to an RSS approximation.
///
/// In 3D: Creates an RSS with identity axes, rectangle at the z-midplane.
/// The rectangle has dimensions (max-min) in x and y, radius = (max-min)/2 in
/// z.
///
/// In 2D: Creates an RSS (stadium) with identity axes, segment at the
/// y-midline. The segment has length (max-min) in x, radius = (max-min)/2 in y.
template <std::size_t Dims, typename Policy>
auto rss_from(const tf::aabb_like<Dims, Policy> &aabb) {
  static_assert(Dims == 2 || Dims == 3,
                "rss_from(aabb) is implemented for 2D and 3D only.");
  using T = tf::coordinate_type<Policy>;

  if constexpr (Dims == 2) {
    std::array<tf::unit_vector<T, 2>, 2> axes{
        tf::make_unit_vector(tf::unsafe,
                             tf::make_vector(std::array<T, 2>{T(1), T(0)})),
        tf::make_unit_vector(tf::unsafe,
                             tf::make_vector(std::array<T, 2>{T(0), T(1)}))};

    tf::point<T, 2> origin{aabb.min[0],
                           (aabb.min[1] + aabb.max[1]) * T(0.5)};

    std::array<T, 1> length{aabb.max[0] - aabb.min[0]};
    T radius = (aabb.max[1] - aabb.min[1]) * T(0.5);

    return tf::make_rss_like(origin, axes, length, radius);
  } else {
    std::array<tf::unit_vector<T, 3>, 3> axes{
        tf::make_unit_vector(
            tf::unsafe, tf::make_vector(std::array<T, 3>{T(1), T(0), T(0)})),
        tf::make_unit_vector(
            tf::unsafe, tf::make_vector(std::array<T, 3>{T(0), T(1), T(0)})),
        tf::make_unit_vector(
            tf::unsafe, tf::make_vector(std::array<T, 3>{T(0), T(0), T(1)}))};

    tf::point<T, 3> origin{aabb.min[0], aabb.min[1],
                           (aabb.min[2] + aabb.max[2]) * T(0.5)};

    std::array<T, 2> length{aabb.max[0] - aabb.min[0],
                            aabb.max[1] - aabb.min[1]};
    T radius = (aabb.max[2] - aabb.min[2]) * T(0.5);

    return tf::make_rss_like(origin, axes, length, radius);
  }
}

} // namespace tf
