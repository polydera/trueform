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

#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_copy_blocked.hpp"
#include "../../core/coordinate_dims.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/polygons.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/resolved_output_real.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/take.hpp"
#include "./mesh_triangulation.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::arrangement {

/// The mesh a triangulation states, in coordinates.
///
/// THE CORNERS NEED NO REMAP. The identity space is the input's, so a corner
/// below `n_original_points()` IS the input vertex id and one past it is
/// exactly the row of `created_points()` that follows the originals — the
/// triangles are the faces, verbatim.
///
/// The points are the input's own, then this build's mints. An original goes
/// through `transformed(pt, frame)` and a mint through the converter, which is
/// the arrangement's own emission law: a minted point cannot be expressed in
/// the untransformed space without inverting the frame, so both sides of the
/// table live in ONE space. On an untagged form the frame is the identity and
/// the copy is the input's own points.
///
/// A mesh that needed no resolution mints nothing, so the point table is the
/// input's and the whole materialization is one copy.
template <typename OutputCoordinateType = tf::none_t, typename Index,
          typename RealT, typename Int, std::size_t Dims, typename Faces,
          typename Policy>
auto materialize_mesh_triangulation(
    const mesh_triangulation<Index, RealT, Int, Dims, Faces> &triangulation,
    const tf::polygons<Policy> &polygons) {
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, RealT>;
  static_assert(tf::coordinate_dims_v<Policy> == Dims,
                "the polygons' dimension is the triangulation's");

  tf::polygons_buffer<Index, RealOut, Dims, 3> out;
  const auto triangles = triangulation.triangles();
  out.faces_buffer().allocate(triangles.size());
  tf::parallel_copy_blocked(triangles, out.faces());

  const auto n_original = std::size_t(triangulation.n_original_points());
  const auto created = triangulation.created_points();
  out.points_buffer().allocate(n_original + created.size());
  const auto frame = tf::frame_of(polygons);
  const auto &converter = triangulation.converter();
  // the lattice is three-dimensional whatever the input is, so a mint of a
  // two-dimensional mesh answers in the plane it was minted on
  const auto minted = tf::make_mapped_range(
      created, [](const auto &point) {
        tf::point<Int, Dims> lattice;
        for (std::size_t axis = 0; axis < Dims; ++axis)
          lattice[axis] = point[axis];
        return lattice;
      });
  if constexpr (std::is_integral_v<RealOut>) {
    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            polygons.points(),
            [frame, &converter](const auto &point) {
              return converter.convert(tf::transformed(point, frame))
                  .template as<RealOut>();
            })),
        tf::take(out.points_buffer(), n_original));
    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            minted,
            [](const auto &point) { return point.template as<RealOut>(); })),
        tf::drop(out.points_buffer(), n_original));
  } else {
    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            polygons.points(),
            [frame](const auto &point) {
              return tf::transformed(point, frame);
            })),
        tf::take(out.points_buffer(), n_original));
    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            minted,
            [&converter](const auto &point) {
              return converter.deconvert(point);
            })),
        tf::drop(out.points_buffer(), n_original));
  }
  return out;
}

} // namespace tf::arrangement
