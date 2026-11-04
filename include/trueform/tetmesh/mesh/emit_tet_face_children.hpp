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
#include "../../core/algorithm/generate_offset_blocks.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../topology/cdt/dt3/recovery/bucket_tet_records.hpp"
#include "../../topology/cdt/dt3/recovery/tet_face_status.hpp"
#include "../../topology/cdt/dt3/recovery/tet_plane_normal.hpp"
#include "../../topology/cdt/dt3/tet_facet_is_protected.hpp"
#include "../../topology/cdt/dt3/tet_facet_key.hpp"
#include "../../topology/cdt/dt3/tet_orient2d_sites.hpp"
#include "../../topology/cdt/dt3/tet_shared_owners.hpp"
#include "../../topology/cdt/dt3/tet_site_position.hpp"
#include "../../topology/cdt/dt3/tet_state.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::tetmesh {

/// Block `f` of `children` holds the output triangles that cover facet `f`
/// of `faces`, wound as it is: the lane facets the owner lane gives it, a
/// shared facet to both its faces, when `status` says it was recovered, and
/// none otherwise; `point_of` maps a site to its output point. The facets
/// are gathered once from the lane and bucketed by face, each face's run in
/// key order. A lane facet's corners are stored ascending, so each is wound
/// by its projected turn against the facet's first triangle on the facet's
/// own plane.
template <typename Index, typename Owner, typename Faces, typename Status,
          typename PointOf>
auto emit_tet_face_children(
    const Owner &owner, const Faces &faces, const Status &status,
    const PointOf &point_of,
    tf::offset_block_buffer<Index, std::array<Index, 3>> &children) -> void {
  using Int = typename Owner::int_type;
  using record_t = std::pair<Index, std::array<Index, 3>>;
  namespace dt3 = tf::topology::cdt::dt3;
  const std::size_t rows = std::min(std::size_t(owner._corners.size()),
                                    owner._protected_slots.size());
  tf::buffer<record_t> records;
  tf::sequenced_generate(
      tf::make_sequence_range(rows), records,
      [&owner](std::size_t cell, auto &out) {
        if (owner._states[cell] == dt3::tet_state::dead ||
            !owner._protected_slots[cell])
          return;
        for (std::size_t slot = 0; slot < 4; ++slot) {
          if (!dt3::tet_facet_is_protected(owner, Index(cell), slot) ||
              std::size_t(owner._neighbors[cell][slot]) < cell)
            continue;
          const Index who = owner._owners[4 * cell + slot];
          if (who == Owner::none)
            continue;
          const auto key = dt3::tet_facet_key(owner._corners[cell], slot);
          if (who != Owner::shared_owner) {
            out.push_back({who, key});
            continue;
          }
          const auto stack = dt3::tet_shared_owners(owner, key);
          for (auto at = stack.first; at != stack.second; ++at)
            out.push_back({at->second, key});
        }
      },
      tf::checked);
  tf::buffer<record_t> scratch;
  tf::buffer<Index> runs;
  dt3::bucket_tet_records(
      records, faces.size(), runs, scratch,
      [](const record_t &record) { return record.first; },
      [](const record_t &x, const record_t &y) { return x.second < y.second; });
  children.data_buffer().clear();
  tf::generate_offset_blocks(
      tf::make_sequence_range(faces.size()), children.offsets_buffer(),
      children.data_buffer(),
      [&](std::size_t f, auto &out) {
        if (status[f] != dt3::tet_face_status::recovered)
          return;
        const auto &first = faces[f][0];
        const auto axes =
            tf::exact::projection_axes<Int>(dt3::tet_plane_normal<Int>(
                dt3::tet_shallow_point(owner, Index(first[0])),
                dt3::tet_shallow_point(owner, Index(first[1])),
                dt3::tet_shallow_point(owner, Index(first[2]))));
        const int facing = dt3::tet_orient2d_sites(
            owner, axes, Index(first[0]), Index(first[1]), Index(first[2]));
        for (Index r = runs[f]; r < runs[f + 1]; ++r) {
          auto key = records[std::size_t(r)].second;
          if (dt3::tet_orient2d_sites(owner, axes, key[0], key[1], key[2]) !=
              facing)
            std::swap(key[1], key[2]);
          out.push_back({point_of[std::size_t(key[0])],
                         point_of[std::size_t(key[1])],
                         point_of[std::size_t(key[2])]});
        }
      },
      tf::checked);
}

} // namespace tf::tetmesh
