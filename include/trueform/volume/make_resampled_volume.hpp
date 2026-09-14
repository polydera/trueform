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
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/point_like.hpp"
#include "../core/policy/frame.hpp"
#include "../core/resolved_output_real.hpp"
#include "../core/transformed.hpp"
#include "../core/views/sequence_range.hpp"
#include "./impl/far_outside_value.hpp"
#include "./impl/grid_sample_count.hpp"
#include "./impl/sample_multilinear.hpp"
#include "./volume.hpp"
#include "./volume_buffer.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup volume
/// @brief Resample a scalar field onto a stated grid, as a new field.
///
/// The regrid of the module: target node (i, ...) sits at
/// `origin + index * spacing` and takes the field's multilinear value there.
/// An untagged volume regrids in its own local space with clamp-to-edge
/// reads: past the domain the nearest boundary sample answers, continuing
/// the field outward. A posed volume (`| tf::tag(frame)`) resamples THROUGH
/// its pose: the target grid is world-space, each node maps through the
/// frame's inverse into the field's local space, and a node outside the
/// posed domain takes the far-outside sentinel — strictly above every
/// sample, so no level set of the field is crossed against it. Two reads,
/// two contracts; @ref tf::make_boolean consumes the posed one for every
/// operand.
///
/// @tparam OutputCoordinateType The sample and coordinate type of the
///         resampled field (default: the volume's coordinate type —
///         floating, refused otherwise).
/// @param vol The scalar volume.
/// @param dims Number of samples along each axis of the target grid.
/// @param spacing Physical size of one target grid step along each axis.
/// @param origin Position of target sample (0, ..., 0) — local-space for an
///        untagged volume, world-space for a posed one.
/// @return A `volume_buffer` holding the resampled field.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename P0, typename P1>
auto make_resampled_volume(
    const tf::volume<Policy> &vol,
    std::array<int, tf::coordinate_dims_v<Policy>> dims,
    const tf::point_like<tf::coordinate_dims_v<Policy>, P0> &spacing,
    const tf::point_like<tf::coordinate_dims_v<Policy>, P1> &origin) {
  constexpr std::size_t Dims = tf::coordinate_dims_v<Policy>;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType,
                                             tf::coordinate_type<Policy>>;
  static_assert(std::is_floating_point_v<RealOut>,
                "the resample decides in a floating type: state "
                "OutputCoordinateType, or give the grid a floating "
                "coordinate type");
  tf::volume_buffer<RealOut, RealOut, Dims> out(
      dims, spacing.template as<RealOut>(), origin.template as<RealOut>());
  if (out.voxel_count() == 0)
    return out;
  auto target = out.volume();
  const int nx = dims[0];
  const auto rows = static_cast<std::ptrdiff_t>(
      tf::volume_detail::grid_sample_count(dims) /
      std::size_t(nx == 0 ? 1 : nx));
  const auto sweep = [&](auto &&sample_at) {
    tf::parallel_for_each(
        tf::make_sequence_range(rows), [&](std::ptrdiff_t row) {
          std::array<int, tf::coordinate_dims_v<Policy>> at{};
          auto rest = row;
          for (std::size_t d = 1; d < tf::coordinate_dims_v<Policy>; ++d) {
            at[d] = int(rest % dims[d]);
            rest /= dims[d];
          }
          for (at[0] = 0; at[0] < nx; ++at[0])
            target(at) = sample_at(target.template point_at<RealOut>(at));
        });
  };
  if constexpr (tf::has_frame_policy<Policy>) {
    const RealOut outside =
        tf::volume_detail::far_outside_value<RealOut>(vol);
    const auto &inverse = vol.frame().inverse_transformation();
    sweep([&](const auto &node) {
      return tf::volume_detail::sample_multilinear(
          vol, tf::transformed(node, inverse), outside);
    });
  } else {
    sweep([&](const auto &node) {
      return tf::volume_detail::sample_multilinear(vol, node);
    });
  }
  return out;
}

} // namespace tf
