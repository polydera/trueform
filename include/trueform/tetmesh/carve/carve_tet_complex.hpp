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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/cdt/dt3/tet_facet_is_protected.hpp"
#include "../../topology/cdt/dt3/tet_facet_key.hpp"
#include "../../topology/cdt/dt3/tet_orient3d_sites.hpp"
#include "../../topology/cdt/dt3/tet_shared_owners.hpp"
#include "./label_tet_regions.hpp"
#include "./tet_carving.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tbb/parallel_sort.h>
#include <tuple>

namespace tf::tetmesh {

/// Carve the published complex `owner` by the winding of its constraint
/// triangles `faces`, site slots each, the faces each protected facet
/// covers read off the owner lane. A facet's cell on
/// the negative side of a covering face is inside it, so crossing the
/// facet from that cell to the other lowers the winding by one per face
/// of the facet's stack, coincident faces of opposite winding cancelling. The regions the
/// unprotected facets join take one winding each, read from the unbounded
/// region by a walk over the crossings.
template <typename Owner, typename Faces>
auto carve_tet_complex(const Owner &owner, const Faces &faces,
                       tet_carving<typename Owner::index_type> &carving)
    -> void {
  using Index = typename Owner::index_type;
  struct crossing {
    Index from;
    Index to;
    int step;
  };
  auto &regions = carving.regions;
  const std::size_t n_regions = label_tet_regions(owner, regions);
  const std::size_t n_finite = owner._n_finite;
  tf::buffer<crossing> crossings;
  tf::sequenced_generate(
      tf::make_sequence_range(n_finite), crossings,
      [&](std::size_t cell, auto &out) {
        const auto corners = owner._corners[cell];
        for (std::size_t slot = 0; slot < 4; ++slot) {
          const auto beyond = std::size_t(owner._neighbors[cell][slot]);
          if ((beyond < n_finite && beyond < cell) ||
              !tf::topology::cdt::dt3::tet_facet_is_protected(owner, cell,
                                                              slot))
            continue;
          const Index who = owner._owners[4 * cell + slot];
          if (who == Owner::none)
            continue;
          const auto side_of = [&faces, &owner, &corners, slot](Index f) {
            const auto &face = faces[std::size_t(f)];
            const int side = tf::topology::cdt::dt3::tet_orient3d_sites(
                owner, Index(face[0]), Index(face[1]), Index(face[2]),
                corners[slot]);
            return side < 0 ? -1 : 1;
          };
          int step = 0;
          if (who == Owner::shared_owner) {
            const auto stack = tf::topology::cdt::dt3::tet_shared_owners(
                owner, tf::topology::cdt::dt3::tet_facet_key(corners, slot));
            for (auto at = stack.first; at != stack.second; ++at)
              step += side_of(at->second);
          } else {
            step = side_of(who);
          }
          const Index a = regions[cell], b = regions[beyond];
          out.push_back({a, b, step});
          out.push_back({b, a, -step});
        }
      },
      tf::checked);
  tbb::parallel_sort(
      crossings.begin(), crossings.end(), [](const auto &x, const auto &y) {
        return std::tie(x.from, x.to, x.step) < std::tie(y.from, y.to, y.step);
      });
  tf::buffer<int> winding;
  winding.allocate(n_regions);
  tf::buffer<char> seen;
  seen.allocate(n_regions);
  std::fill(seen.begin(), seen.end(), char(0));
  carving.consistent = true;
  tf::buffer<Index> queue;
  const Index outside =
      regions[n_finite < std::size_t(owner._corners.size()) ? n_finite
                                                            : std::size_t(0)];
  winding[std::size_t(outside)] = 0;
  seen[std::size_t(outside)] = 1;
  queue.push_back(outside);
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const Index region = queue[head];
    const auto range = std::equal_range(
        crossings.begin(), crossings.end(), crossing{region, Index(0), 0},
        [](const auto &x, const auto &y) { return x.from < y.from; });
    for (auto at = range.first; at != range.second; ++at) {
      const int next = winding[std::size_t(region)] + at->step;
      if (seen[std::size_t(at->to)]) {
        carving.consistent =
            carving.consistent && winding[std::size_t(at->to)] == next;
        continue;
      }
      seen[std::size_t(at->to)] = 1;
      winding[std::size_t(at->to)] = next;
      queue.push_back(at->to);
    }
  }
  for (std::size_t r = 0; r < n_regions; ++r)
    carving.consistent = carving.consistent && seen[r];
  carving.winding.allocate(n_finite);
  tf::parallel_for_each(
      tf::make_sequence_range(n_finite),
      [&carving, &winding, &regions](std::size_t cell) {
        carving.winding[cell] = winding[std::size_t(regions[cell])];
      },
      tf::checked);
}

} // namespace tf::tetmesh
