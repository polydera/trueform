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
#include "../core/algorithm/parallel_contains.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/coordinate_dims.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/point_like.hpp"
#include "../core/policy/unwrap.hpp"
#include "../core/polygons.hpp"
#include "../core/range.hpp"
#include "../core/resolved_output_real.hpp"
#include "../core/views/sequence_range.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../spatial/aabb_tree.hpp"
#include "../spatial/neighbor_search.hpp"
#include "../spatial/policy/tree.hpp"
#include "../spatial/tree_config.hpp"
#include "./impl/band_selection.hpp"
#include "./impl/compose_mesh_sdf.hpp"
#include "./impl/mask_ids.hpp"
#include "./impl/measure_band.hpp"
#include "./impl/mesh_sdf_lattice.hpp"
#include "./impl/parity_inside.hpp"
#include "./impl/seeded_distance_sweep.hpp"
#include "./impl/shell_walk.hpp"
#include "./mesh_sdf_config.hpp"
#include "./volume_buffer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace tf {

/// @ingroup volume
/// @brief Sample the signed distance field of a closed mesh onto a regular
/// grid.
///
/// The magnitude is the distance to the nearest point of the surface; the
/// sign is exact crossing parity on the integer lattice — negative inside by
/// winding, so an inverted shell inverts its field and nested shells deepen
/// it. `polygons` must be a closed surface; an open mesh has no inside, and
/// @ref tf::make_outer_shell repairs one. A form carrying the `tree` policy
/// is queried through it, a form without gets one built for the call. The
/// grid samples the field in the frame the form states, and a sample on the
/// surface takes the perturbation's sign at distance zero.
///
/// Under `mesh_sdf_mode::banded` only the samples within `config.band` voxels
/// of the surface are measured; the far field is propagated by a seeded
/// distance sweep. The band is exact, the sign is exact everywhere, and on
/// grids whose lines resolve the surface the far field is within about one
/// voxel (99th percentile under it, shrinking with resolution; the deep
/// interior near the medial axis may locally undershoot by a few voxels).
/// A feature no grid line meets is invisible to the banded discovery and
/// its neighbourhood takes the far field's answer; measurement applies to
/// what the shell states — the surface past the grid.
///
/// @tparam OutputCoordinateType The sample and coordinate type of the field
///         (default: the mesh's own coordinate type).
/// @param polygons The closed surface mesh, optionally tree-tagged and framed.
/// @param dims Number of samples along x, y, z.
/// @param spacing Physical size of one voxel step along x, y, z.
/// @param origin Position of sample (0, 0, 0), in the frame the form states.
/// @param config The magnitude mode; default measures every sample.
/// @return A `volume_buffer<RealOut>` holding the mesh SDF.
template <typename OutputCoordinateType = tf::none_t, typename Policy,
          typename P0, typename P1>
auto make_mesh_sdf(const tf::polygons<Policy> &polygons,
                   std::array<int, 3> dims,
                   const tf::point_like<3, P0> &spacing,
                   const tf::point_like<3, P1> &origin,
                   const mesh_sdf_config &config = {}) {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "the parity sign is a closed surface's, in three dimensions");
  using InReal = tf::coordinate_type<Policy>;

  if constexpr (!tf::has_tree_policy<Policy>) {
    using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
    tf::aabb_tree<Index, InReal, 3> tree;
    tree.build(polygons, tf::config_tree(4, 12));
    return tf::make_mesh_sdf<OutputCoordinateType>(
        polygons | tf::tag(tree), dims, spacing, origin, config);
  } else {
    using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InReal>;
    using Int = tf::exact::resolve_int_type<tf::none_t, InReal>;

    tf::volume_buffer<RealOut> out(dims, spacing.template as<RealOut>(),
                                   origin.template as<RealOut>());
    auto vol = out.volume();
    if (vol.voxel_count() == 0)
      return out;

    const bool banded =
        config.mode == mesh_sdf_mode::banded && polygons.size() > 0;
    tf::buffer<char> inside;
    tf::volume_detail::band_discovery discovery;
    std::array<bool, 3> flips{};
    if (polygons.size() > 0) {
      const auto lattice =
          tf::volume_detail::make_mesh_sdf_lattice<Int, InReal>(
              polygons, dims, spacing, origin);
      flips = lattice.flips;
      const std::array<const tf::buffer<Int> *, 3> samples{
          &lattice.axes[0], &lattice.axes[1], &lattice.axes[2]};
      if (banded)
        tf::volume_detail::parity_inside<Int>(polygons.faces(), lattice.points,
                                              samples, lattice.flips, inside,
                                              &discovery);
      else
        tf::volume_detail::parity_inside<Int>(polygons.faces(), lattice.points,
                                              samples, lattice.flips, inside);
    } else {
      inside.allocate(vol.voxel_count());
      std::fill(inside.begin(), inside.end(), char(0));
    }

    auto dpolygons = tf::wrap_map(polygons, [](auto &&x) {
      return tf::core::make_polygons(x.faces(),
                                     x.points().template as<double>());
    });
    const int nx = dims[0], ny = dims[1];
    const auto rows = static_cast<std::ptrdiff_t>(ny) * dims[2];
    const auto exact_field = [&] {
      // positions read in double for the query; the base entry read them in
      // the output type — at most one float ulp apart
      tf::parallel_for_each(
          tf::make_sequence_range(rows), [&](std::ptrdiff_t row) {
            const int y = int(row % ny), z = int(row / ny);
            const auto *in = inside.begin() + row * nx;
            for (int x = 0; x < nx; ++x) {
              const auto proj = tf::neighbor_search(
                  dpolygons, vol.template point_at<double>(x, y, z));
              const double d = proj ? std::sqrt(proj.metric()) : 0.0;
              vol(x, y, z) = static_cast<RealOut>(in[x] ? -d : d);
            }
          });
    };
    if (!banded) {
      exact_field();
      return out;
    }

    if (!tf::parallel_contains(tf::make_range(discovery.seeds),
                               [](char c) { return c != 0; })) {
      exact_field();
      return out;
    }

    const std::array<std::ptrdiff_t, 3> dims3{nx, ny, dims[2]};
    const bool leaves_grid = tf::parallel_contains(
        tf::make_range(discovery.shell), [](char c) { return c != 0; });
    const auto measured_mask = tf::volume_detail::band_measured_mask(
        discovery, flips, dims3, config.band, leaves_grid);

    const auto ids = tf::volume_detail::mask_ids(measured_mask);
    tf::buffer<RealOut> measured;
    tf::volume_detail::measure_band(
        dpolygons, vol, ids,
        std::abs(static_cast<double>(static_cast<InReal>(spacing[0]))),
        measured);

    // A seed's own distance bounds the composite's undershoot, so the
    // sub-voxel far field allows only sub-voxel seeds: a crossing seed is one
    // by construction, and a shell sample seeds only where the surface sits
    // within one step past the boundary — farther out it stays a measurement,
    // which the crossing seeds already answer.
    const double h_max =
        std::max({std::abs(static_cast<double>(spacing[0])),
                  std::abs(static_cast<double>(spacing[1])),
                  std::abs(static_cast<double>(spacing[2]))});
    const RealOut shell_reach2 = static_cast<RealOut>(h_max * h_max);
    tf::buffer<RealOut> field;
    field.allocate(vol.voxel_count());
    tf::parallel_fill(field, std::numeric_limits<RealOut>::infinity());
    tf::parallel_for_each(
        tf::make_sequence_range(std::ptrdiff_t(ids.size())),
        [&](std::ptrdiff_t bi) {
          const auto i = ids[std::size_t(bi)];
          const bool seed_here =
              discovery.seeds[std::size_t(i)] ||
              (discovery.shell[std::size_t(i)] &&
               measured[std::size_t(bi)] <= shell_reach2);
          if (seed_here)
            field[std::size_t(i)] = measured[std::size_t(bi)];
        });
    tf::volume_detail::seeded_distance_sweep(
        field, dims3,
        {static_cast<double>(spacing[0]) * static_cast<double>(spacing[0]),
         static_cast<double>(spacing[1]) * static_cast<double>(spacing[1]),
         static_cast<double>(spacing[2]) * static_cast<double>(spacing[2])});
    if (leaves_grid) {
      const std::array<double, 3> h3{std::abs(double(spacing[0])),
                                     std::abs(double(spacing[1])),
                                     std::abs(double(spacing[2]))};
      const auto owned_ids = tf::volume_detail::shell_walk(
          discovery, measured_mask, ids, measured, field, dims3, h3);
      if (owned_ids.size() > 0) {
        tf::buffer<RealOut> owned_measured;
        tf::volume_detail::measure_band(
            dpolygons, vol, owned_ids,
            std::abs(static_cast<double>(static_cast<InReal>(spacing[0]))),
            owned_measured);
        tf::parallel_for_each(
            tf::make_sequence_range(std::ptrdiff_t(owned_ids.size())),
            [&](std::ptrdiff_t bi) {
              field[std::size_t(owned_ids[std::size_t(bi)])] =
                  owned_measured[std::size_t(bi)];
            });
      }
    }
    tf::volume_detail::compose_mesh_sdf(vol, inside, field, ids, measured);
    return out;
  }
}

} // namespace tf
