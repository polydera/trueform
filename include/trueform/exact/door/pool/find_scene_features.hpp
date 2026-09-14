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

#include "./scene_targets.hpp"

#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/small_vector.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../canonical_plane.hpp"
#include "../../meta.hpp"
#include "../direction_grid_steps.hpp"
#include "../placement_tables.hpp"
#include "../quantize_direction.hpp"

#include <array>
#include <cstddef>

namespace tf::exact::door::pool {

/// What every original vertex says about itself, read in one walk of its own
/// incidence: whether it is a FEATURE, and the LEAST exact plane of its own
/// incident faces.
///
/// A vertex is flat while its incident faces round onto ONE direction at the
/// pooling pitch, so two faces differing by less than that resolution leave
/// it flat and a wall whose halves are not exactly coplanar can still be
/// pooled. A feature keeps the rank-3/2 cascade.
///
/// THE PITCH IS THE PARTITION'S, NOT THE DOOR'S. A door name is quantized at
/// the band itself, because the reach that admits it is the band against the
/// name's own length and both scale with the lattice. A direction class is
/// dimensionless (@ref tf::exact::door::direction_grid_steps), so reading
/// flatness off the door's grid would make the same physical tolerance a
/// different question at each width — every sub-band wobble a crease on the
/// wider lattice, and no vertex left to pool.
///
/// The target source is a plane the vertex already stands on, which an
/// incident-normal sum is not, and it is what puts the vertex's own target
/// into the same currency as the faces'.
///
/// The incidence is the tables' — the one producer of which faces a vertex
/// belongs to.
template <typename Index, typename Int, typename RealType>
auto find_scene_features(
    const placement_tables<Index, Int, RealType> &tables,
    const tf::buffer<tf::exact::canonical_plane<Int>> &face_plane,
    const tf::buffer<int> &face_name, Int tolerance, scene_targets &targets)
    -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  struct local_t {
    tf::small_vector<std::array<T2, 3>, 16> incident;
  };

  const auto steps = direction_grid_steps<Int>(T1(tolerance));
  tf::buffer<std::array<T1, 3>> direction;
  direction.allocate(face_plane.size());
  tf::parallel_for_each(
      tf::make_sequence_range(face_plane.size()),
      [&face_plane, &direction, steps](std::size_t f) {
        direction[f] = {T1(0), T1(0), T1(0)};
        quantize_direction<Int>(static_cast<double>(face_plane[f][0]),
                                static_cast<double>(face_plane[f][1]),
                                static_cast<double>(face_plane[f][2]), steps,
                                direction[f]);
      },
      tf::checked);

  const auto n_points = tables.points.size();
  targets.feature.allocate(n_points);
  targets.source.allocate(n_points);

  tf::parallel_for_each(
      tf::make_sequence_range(n_points),
      [&tables, &direction, &face_name, &targets](std::size_t v,
                                                   local_t &local) {
        local.incident.clear();
        int source = -1;
        for (auto k = tables.incidence_offsets[v];
             k < tables.incidence_offsets[v + 1]; ++k) {
          const auto f = std::size_t(tables.incidence[std::size_t(k)]);
          const auto &normal = direction[f];
          if (normal[0] != 0 || normal[1] != 0 || normal[2] != 0)
            local.incident.push_back(
                {T2(normal[0]), T2(normal[1]), T2(normal[2])});
          const int at = face_name[f];
          if (at >= 0 && (source < 0 || at < source))
            source = at;
        }
        targets.source[v] = source;

        char feature = 0;
        for (std::size_t a = 0; a + 1 < local.incident.size() && !feature; ++a)
          for (std::size_t b = a + 1; b < local.incident.size(); ++b) {
            const auto &p = local.incident[a];
            const auto &q = local.incident[b];
            if (p[1] * q[2] - p[2] * q[1] != T2(0) ||
                p[2] * q[0] - p[0] * q[2] != T2(0) ||
                p[0] * q[1] - p[1] * q[0] != T2(0)) {
              feature = 1;
              break;
            }
          }
        targets.feature[v] = feature;
      },
      local_t{}, tf::checked);
}

} // namespace tf::exact::door::pool
