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
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/point_like.hpp"
#include "../core/resolved_output_real.hpp"
#include "../core/views/sequence_range.hpp"
#include "./impl/far_outside_value.hpp"
#include "./impl/sample_multilinear.hpp"
#include "./volume.hpp"
#include "./volume_buffer.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup volume
/// @brief Resample a 3D volume onto an oriented plane, as a 2D volume.
///
/// The slice of a volume is a volume one dimension down, and that is what this
/// returns: the field multilinearly sampled at each node of a 2D grid embedded
/// in the 3D volume's LOCAL space — a stated frame does not move the plane,
/// and the slice itself carries no embedding to emit through. Node (i, j)
/// samples the volume at
///
///     plane_origin + i * spacing2[0] * u + j * spacing2[1] * v
///
/// so the slice's own 2D frame has its origin at (0, 0) and axes `u`, `v`
/// (unit 3D vectors, typically orthonormal). @ref tf::make_isocontours reads
/// that 2D volume directly.
///
/// Sample points outside the volume's box receive a sentinel "far outside"
/// value — the field's maximum sample plus the maximum voxel spacing — instead
/// of a clamped boundary sample. Any isovalue at or below the field maximum is
/// therefore never crossed between an in-grid and an out-of-grid node's
/// interpolated neighbourhood, so contours extracted from the slice terminate
/// cleanly at the volume boundary.
///
/// @tparam OutputCoordinateType The type the slice is resampled and stored
///         in (default: the volume's coordinate type — floating, refused
///         otherwise).
/// @param vol The 3D scalar volume.
/// @param plane_origin Local-space position of slice grid node (0, 0).
/// @param u Unit 3D direction of the slice grid's first (i) axis.
/// @param v Unit 3D direction of the slice grid's second (j) axis.
/// @param dims2 Number of slice grid nodes along u and v.
/// @param spacing2 Slice grid step along u and v, in the type the call decides
///        in — a braced pair of numbers.
/// @return A 2D @ref tf::volume_buffer holding the resampled field.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename P0, typename P1, typename P2>
auto make_volume_slice(
    const tf::volume<Policy> &vol, const tf::point_like<3, P0> &plane_origin,
    const tf::point_like<3, P1> &u, const tf::point_like<3, P2> &v,
    std::array<int, 2> dims2,
    std::array<tf::resolved_output_real_t<OutputCoordinateType,
                                          tf::coordinate_type<Policy>>,
               2>
        spacing2) {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "a slice is a plane through a 3D volume");
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>;
  static_assert(std::is_floating_point_v<RealOut>,
                "the slice decides in a floating type: state "
                "OutputCoordinateType, or give the grid a floating "
                "coordinate type");
  tf::point<RealOut, 2> slice_origin;
  slice_origin[0] = RealOut{0};
  slice_origin[1] = RealOut{0};
  tf::point<RealOut, 2> slice_spacing;
  slice_spacing[0] = spacing2[0];
  slice_spacing[1] = spacing2[1];

  tf::volume_buffer<RealOut, RealOut, 2> out(
      {dims2[0] < 0 ? 0 : dims2[0], dims2[1] < 0 ? 0 : dims2[1]},
      slice_spacing, slice_origin);
  if (out.voxel_count() == 0)
    return out;

  const RealOut outside =
      tf::volume_detail::far_outside_value<RealOut>(vol);

  if (vol.voxel_count() == 0) {
    tf::parallel_fill(out.samples_buffer(), outside);
    return out;
  }

  const auto og = plane_origin.template as<RealOut>();
  const auto au = u.template as<RealOut>();
  const auto av = v.template as<RealOut>();
  auto slice = out.volume();

  tf::parallel_for_each(tf::make_sequence_range(0, dims2[1]), [&](int j) {
    const RealOut b = static_cast<RealOut>(j) * spacing2[1];
    for (int i = 0; i < dims2[0]; ++i) {
      const RealOut a = static_cast<RealOut>(i) * spacing2[0];
      tf::point<RealOut, 3> p;
      for (int d = 0; d < 3; ++d)
        p[d] = og[d] + a * au[d] + b * av[d];
      slice(i, j) = volume_detail::sample_multilinear(vol, p, outside);
    }
  });
  return out;
}

} // namespace tf
