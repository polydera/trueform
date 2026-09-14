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
#include "../core/point.hpp"
#include "../core/views/sequence_range.hpp"
#include "./impl/grid_sample_count.hpp"
#include "./volume_buffer.hpp"
#include <array>
#include <cstddef>

namespace tf {

/// @ingroup volume
/// @brief Sample the signed distance field of a sphere onto a regular grid.
///
/// Each sample holds `distance(point, center) - radius`: negative inside the
/// sphere, positive outside, zero on the surface. Extracting the isosurface at
/// value 0 recovers the sphere.
///
/// This is the simplest exact SDF generator and doubles as a fixture for the
/// isosurface pipeline (no mesh voxelization required to produce a valid field).
///
/// The distance to a centre is the same fact at any number of axes, so the
/// grid's own dimensionality decides the field's: a 2D `dims` states a disc.
///
///
/// @tparam RealType The floating-point type for samples and coordinates.
/// @param dims Number of samples along each axis.
/// @param spacing Physical size of one grid step along each axis.
/// @param origin Local-space position of sample (0, ..., 0).
/// @param center Sphere center, in the volume's local frame.
/// @param radius Sphere radius.
/// @return A `volume_buffer<RealType, RealType, Dims>` holding the sphere
///         SDF.
template <typename RealType, std::size_t Dims>
auto make_sphere_sdf(std::array<int, Dims> dims,
                     tf::point<RealType, Dims> spacing,
                     tf::point<RealType, Dims> origin,
                     tf::point<RealType, Dims> center, RealType radius)
    -> tf::volume_buffer<RealType, RealType, Dims> {
  tf::volume_buffer<RealType, RealType, Dims> out(dims, spacing, origin);
  auto vol = out.volume();
  const int nx = dims[0];
  const auto rows = static_cast<std::ptrdiff_t>(
      tf::volume_detail::grid_sample_count(dims) /
      std::size_t(nx == 0 ? 1 : nx));

  // The sample row is the carrier, as it is for the extractors: one sweep of
  // the first axis is a tight serial kernel, and the rows of a grid fill the
  // machine at any shape.
  tf::parallel_for_each(
      tf::make_sequence_range(rows), [&](std::ptrdiff_t row) {
        std::array<int, Dims> at{};
        auto rest = row;
        for (std::size_t d = 1; d < Dims; ++d) {
          at[d] = int(rest % dims[d]);
          rest /= dims[d];
        }
        for (at[0] = 0; at[0] < nx; ++at[0])
          vol(at) = (vol.point_at(at) - center).length() - radius;
      });
  return out;
}

} // namespace tf
