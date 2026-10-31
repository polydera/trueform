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

#include "../../core/buffer.hpp"
#include "./plane_def_respan.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_identity_collapse.hpp"
#include "./plane_tables.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// The groups a restatement changes, ascending: a group of a reached plane
/// holding an instance of a dying face or naming a `moved` identity, and
/// every group whose key a statement or a changed group's restated live
/// instance lands on, closed. `moved` is `{kind, id}` ascending in the
/// merges' own currency, `merges` the closed table.
template <typename Index, typename Int, typename VertexOffsets,
          typename IsDying>
auto touch_restated_plane_groups(const plane_tables<Index, Int> &tables,
                                 const tf::buffer<std::array<Index, 3>> &merges,
                                 const VertexOffsets &vertex_offsets,
                                 const tf::buffer<std::array<Index, 2>> &moved,
                                 const tf::buffer<Index> &planes,
                                 const IsDying &is_dying,
                                 const tf::buffer<plane_edge_def<Index>> &rows)
    -> tf::buffer<Index> {
  using def_t = plane_edge_def<Index>;
  const auto defs = tables.edge_defs();
  const auto names_moved = [&](const def_t &def) {
    const auto probe = [&](std::int16_t tag, Index id) {
      const std::array<Index, 2> key =
          tag < 0 ? std::array<Index, 2>{Index(1), id}
                  : std::array<Index, 2>{Index(0),
                                         vertex_offsets[std::size_t(tag)] + id};
      return std::binary_search(moved.begin(), moved.end(), key);
    };
    return probe(def.point_tag_0, def.point_0) ||
           probe(def.point_tag_1, def.point_1);
  };
  const auto landing = [&](const def_t &parent) {
    const auto ends = plane_merged_endpoints(merges, vertex_offsets, parent);
    auto def = parent;
    plane_piece_def(def, {Index(ends.lo_tag), ends.lo_id},
                    {Index(ends.hi_tag), ends.hi_id}, parent, Index(1));
    return find_plane_canon_group(tables, def);
  };

  tf::buffer<char> changed;
  changed.allocate_and_initialize(std::size_t(tables.n_canon()), char(0));
  tf::buffer<Index> touched;
  const auto touch = [&](Index group) {
    if (group != Index(-1) && !changed[std::size_t(group)]) {
      changed[std::size_t(group)] = char(1);
      touched.push_back(group);
    }
  };
  for (const auto plane : planes)
    if (plane < tables.n_planes())
      for (const auto row : tables.plane_edges(plane)) {
        const auto &def = defs[std::size_t(row)];
        if (!changed[std::size_t(def.id)] &&
            (is_dying(def.face) || names_moved(def)))
          touch(def.id);
      }
  for (const auto &row : rows)
    touch(landing(row));
  // a moved identity can bring a changed group's key onto a standing one
  for (std::size_t at = 0; at < touched.size(); ++at)
    for (const auto &def : tables.canon_group(touched[at]))
      if (!is_dying(def.face))
        touch(landing(def));
  std::sort(touched.begin(), touched.end());
  return touched;
}

} // namespace tf::intersect::graph
