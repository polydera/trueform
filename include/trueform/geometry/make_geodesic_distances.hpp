/*
 * Copyright (c) 2026 XLAB
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
#include "../core/buffer.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/resolved_output_real.hpp"
#include "./geodesic/geodesic_distances_within.hpp"
#include <limits>
#include <type_traits>

namespace tf {

/// @ingroup geometry
/// @brief The geodesic distance from the nearest seed to every point.
///
/// A label-correcting front over the triangle-unfolding update: a seed's
/// ring is reached at its Euclidean distance, and the front runs from there
/// through @ref tf::geometry::propagate_geodesic_distances. A front that
/// outgrows 200 points on more than one thread splits into one front per
/// thread that run at once (@ref tf::geometry::propagate_geodesic_fronts).
/// On a planar mesh it is the straight line; on a curved one it beats the
/// shortest edge path.
///
/// A front that never splits answers the same distances on every run. A
/// split one answers what the threads' schedule settles on: each revision
/// still declines only improvements below
/// @ref tf::geometry::geodesic_revision_tolerance, so runs differ by at most
/// about that share per edge the front crossed, and a point that close to a
/// radius may fall on either side of it. Where two seeds' fronts meet, which
/// seed states a point is the schedule's too, and there the seam answers the
/// edge from either side: at a seam the answer is the edge path, so a sliver
/// there costs up to its longest edge.
///
/// Every reached point stands at least the straight line from the seed that
/// stated its distance, checked under `tf::core::verification`.
///
/// A seed stands at zero and a point no front reaches — another piece of the
/// mesh — answers infinity. Distances are measured in the polygons' own
/// coordinates; a tagged frame does not enter them. The face membership and
/// the vertex link are read where tagged and built otherwise. The front runs
/// in double precision and the distances are stated in the resolved output
/// type as they are read out.
///
/// @tparam OutputCoordinateType The type the distances are stated in;
/// `tf::none` states the mesh's own.
/// @param polygons The mesh.
/// @param seeds The ids the distance is measured from.
/// @return One distance per point.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Range>
auto make_geodesic_distances(const tf::polygons<Policy> &polygons,
                             const Range &seeds)
    -> tf::buffer<tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>> {
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>;
  static_assert(std::is_floating_point_v<RealOut>,
                "a geodesic distance is stated in a floating-point type");
  return tf::geometry::geodesic_distances_within<RealOut>(
      polygons, seeds, std::numeric_limits<double>::infinity(), tf::none);
}

/// @ingroup geometry
/// @brief The geodesic distance from the nearest seed to every point within
/// `radius`; a point past it answers infinity.
/// @overload
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename Range>
auto make_geodesic_distances(const tf::polygons<Policy> &polygons,
                             const Range &seeds,
                             tf::coordinate_type<Policy> radius)
    -> tf::buffer<tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>> {
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>;
  static_assert(std::is_floating_point_v<RealOut>,
                "a geodesic distance is stated in a floating-point type");
  return tf::geometry::geodesic_distances_within<RealOut>(
      polygons, seeds, double(radius), tf::none);
}

} // namespace tf
