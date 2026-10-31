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

#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./find_touched_group_planes.hpp"
#include "./plane_tables.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// The planes of `tables` holding a row that names a `moved` identity,
/// ascending. `moved` is `{kind, id}` in the merges' own currency, ascending;
/// a group's rows share its key, so each key is searched in it once and the
/// groups that name one state their carriers through `plane_of_face`.
template <typename Index, typename Int, typename VertexOffsets,
          typename PlaneOfFace>
auto find_moved_identity_planes(const plane_tables<Index, Int> &tables,
                                const VertexOffsets &vertex_offsets,
                                const tf::buffer<std::array<Index, 2>> &moved,
                                const PlaneOfFace &plane_of_face)
    -> tf::buffer<Index> {
  tf::buffer<Index> groups;
  if (moved.size() == 0)
    return groups;
  const auto names_moved = [&](std::int16_t tag, Index id) {
    const std::array<Index, 2> key =
        tag < 0 ? std::array<Index, 2>{Index(1), id}
                : std::array<Index, 2>{Index(0),
                                       vertex_offsets[std::size_t(tag)] + id};
    return std::binary_search(moved.begin(), moved.end(), key);
  };
  tf::sequenced_generate(
      tf::make_sequence_range(tables.n_canon()), groups,
      [&](Index group, tf::buffer<Index> &out) {
        const auto span = tables.canon_group(group);
        if (span.size() == 0)
          return;
        const auto &def = span[0];
        if (names_moved(def.point_tag_0, def.point_0) ||
            names_moved(def.point_tag_1, def.point_1))
          out.push_back(group);
      },
      tf::checked);
  return find_touched_group_planes(tables, groups, plane_of_face);
}

} // namespace tf::intersect::graph
