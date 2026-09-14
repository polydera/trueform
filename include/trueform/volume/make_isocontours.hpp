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
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/curves_buffer.hpp"
#include "../core/edges.hpp"
#include "../core/none.hpp"
#include "../core/point_like.hpp"
#include "../core/points_buffer.hpp"
#include "../core/range.hpp"
#include "../core/resolved_output_real.hpp"
#include "../core/views/blocked_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./impl/emit_in_stated_frame.hpp"
#include "./impl/flying_edges_2d.hpp"
#include "./make_volume_slice.hpp"
#include "./volume.hpp"
#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup volume
/// @brief Isocontours of a 2D volume, as curves in the frame the grid states.
///
/// The level set `{ p : field(p) == iso }` of the grid's bilinear field,
/// connected into polylines in the frame the grid states — the same product
/// @ref tf::make_isocontours yields for a scalar field on a mesh, one
/// dimension down. Each crossing is
/// generated exactly once, so the points are welded and the connection is
/// topological rather than a coordinate search.
///
/// @tparam Index The index type for the output curves (default: int).
/// @tparam OutputCoordinateType The coordinate type of the emitted curves
///         (default: the grid's coordinate type — floating, refused
///         otherwise).
/// @param grid The 2D scalar volume.
/// @param isovalues The isovalues to contour, e.g. `tf::make_range(values)`.
/// @return A `curves_buffer` of connected 2D polylines; empty if no isovalue
///         is crossed.
template <typename Index = int, typename OutputCoordinateType = tf::none_t,
          typename Policy, typename Iterator, std::size_t N>
auto make_isocontours(const tf::volume<Policy> &grid,
                      const tf::range<Iterator, N> &isovalues) {
  static_assert(tf::coordinate_dims_v<Policy> == 2,
                "isocontours are the level set of a 2D volume; a 3D one is "
                "asked for its isosurface, or for a slice to contour");
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>;
  static_assert(std::is_floating_point_v<RealOut>,
                "the extraction decides in a floating type: state "
                "OutputCoordinateType, or give the grid a floating "
                "coordinate type");
  tf::points_buffer<RealOut, 2> points;
  tf::buffer<Index> segments;
  for (const auto &iso : isovalues)
    volume_detail::flying_edges_2d(grid, static_cast<RealOut>(iso), points,
                                   segments);

  tf::curves_buffer<Index, RealOut, 2> cb;
  if (segments.size() == 0)
    return cb;
  cb.paths_buffer() = tf::connect_edges_to_paths(
      tf::make_edges(tf::make_blocked_range<2>(segments)));
  cb.points_buffer() = std::move(points);
  tf::volume_detail::emit_in_stated_frame(cb.points(), grid);
  return cb;
}

/// @ingroup volume
/// @brief Isocontours of a 2D volume at one isovalue.
/// @overload
template <typename Index = int, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_isocontours(const tf::volume<Policy> &grid,
                      tf::resolved_output_real_t<
                          OutputCoordinateType, tf::coordinate_type<Policy>>
                          isovalue) {
  const auto one = std::array<
      tf::resolved_output_real_t<OutputCoordinateType,
                                 tf::coordinate_type<Policy>>,
      1>{isovalue};
  return tf::make_isocontours<Index, OutputCoordinateType>(
      grid, tf::make_range(one.data(), one.data() + 1));
}

/// @ingroup volume
/// @brief Isocontours of a 3D volume on an oriented slice plane, as 3D curves.
///
/// The composition `make_isocontours(make_volume_slice(...))` plus the lift
/// back into the frame the volume states, and the one producer of that lifted
/// product — the plane inputs stay local-space: the slice's 2D frame has its
/// origin at `plane_origin` and axes
/// `u`, `v`, so a contour point p2 becomes
///
///     p3 = plane_origin + p2.x * u + p2.y * v
///
/// A caller who wants the contours in the slice's own 2D frame composes the
/// two entries instead and keeps the 2D curves.
///
/// Contours terminate cleanly at the volume boundary: slice nodes outside the
/// volume's box receive a "far outside" sentinel value above the field's
/// maximum, so no isovalue at or below that maximum is crossed there.
///
/// @tparam Index The index type for the output curves (default: int).
/// @tparam OutputCoordinateType The coordinate type of the emitted curves
///         (default: the volume's coordinate type — floating, refused
///         otherwise).
/// @param vol The 3D scalar volume.
/// @param plane_origin Local-space position of slice grid node (0, 0).
/// @param u Unit 3D direction of the slice grid's first (i) axis.
/// @param v Unit 3D direction of the slice grid's second (j) axis.
/// @param dims2 Number of slice grid nodes along u and v.
/// @param spacing2 Slice grid step along u and v, in the type the call decides
///        in — a braced pair of numbers.
/// @param isovalues The isovalues to contour, e.g. `tf::make_range(values)`.
/// @return A `curves_buffer` of connected polylines in the frame the volume
///         states; empty if no isovalue is crossed.
template <typename Index = int, typename OutputCoordinateType = tf::none_t,
          typename Policy, typename P0, typename P1, typename P2,
          typename Iterator, std::size_t N>
auto make_isocontours(
    const tf::volume<Policy> &vol, const tf::point_like<3, P0> &plane_origin,
    const tf::point_like<3, P1> &u, const tf::point_like<3, P2> &v,
    std::array<int, 2> dims2,
    std::array<tf::resolved_output_real_t<OutputCoordinateType,
                                          tf::coordinate_type<Policy>>,
               2>
        spacing2,
    const tf::range<Iterator, N> &isovalues) {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "a slice is a plane through a 3D volume");
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>;
  static_assert(std::is_floating_point_v<RealOut>,
                "the extraction decides in a floating type: state "
                "OutputCoordinateType, or give the grid a floating "
                "coordinate type");
  auto slice = tf::make_volume_slice<RealOut>(vol, plane_origin, u, v, dims2,
                                              spacing2);
  auto flat =
      tf::make_isocontours<Index, RealOut>(slice.volume(), isovalues);

  tf::curves_buffer<Index, RealOut, 3> cb;
  if (flat.points_buffer().size() == 0)
    return cb;
  cb.paths_buffer() = std::move(flat.paths_buffer());

  const auto og = plane_origin.template as<RealOut>();
  const auto au = u.template as<RealOut>();
  const auto av = v.template as<RealOut>();
  const auto &p2 = flat.points_buffer();
  cb.points_buffer().allocate(p2.size());
  auto p3 = cb.points();
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t{0}, p2.size()), [&](std::size_t i) {
        const auto flat_point = p2[i];
        auto lifted = p3[i];
        for (int d = 0; d < 3; ++d)
          lifted[d] = og[d] + flat_point[0] * au[d] + flat_point[1] * av[d];
      });
  tf::volume_detail::emit_in_stated_frame(cb.points(), vol);
  return cb;
}

} // namespace tf
