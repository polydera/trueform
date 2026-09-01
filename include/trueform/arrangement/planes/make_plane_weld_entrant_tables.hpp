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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/make_supported_plane_frame.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "../../intersect/graph/plane_def_respan.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::arrangement {

/// State newly reached source faces as exact PA-local plane tables.
///
/// Each source face contributes one descriptor, one exact final frame and one
/// plane block. `expand_side` lifts its raw corner cycle through the immutable
/// LA's published pieces; the PA weld table is then applied once, so a
/// promoted plane enters triangulation in final identity and any collapsed
/// side is absent.
template <typename Index, typename Int, typename Entrants,
          typename VertexOffsets, typename FaceOffsets, typename GetPoint,
          typename ApplyToForm, typename ExpandSide>
auto make_plane_weld_entrant_tables(
    const Entrants &entrants, Index face_base,
    const tf::buffer<std::array<Index, 3>> &merges,
    const VertexOffsets &vertex_offsets, const FaceOffsets &face_offsets,
    const GetPoint &get_point, const ApplyToForm &apply_to_form,
    const ExpandSide &expand_side,
    tf::buffer<tf::intersect::graph::face_descriptor<Index>> &descriptors,
    tf::buffer<tf::exact::plane_frame<Int>> &frames,
    tf::intersect::graph::plane_tables<Index, Int> &tables) -> void {
  using def_t = tf::intersect::graph::plane_edge_def<Index>;
  using descriptor_t = tf::intersect::graph::face_descriptor<Index>;

  descriptors.clear();
  frames.clear();
  tables.defs().clear();
  tables.def_offsets().clear();
  tables.edges().offsets_buffer().clear();
  tables.edges().data_buffer().clear();
  tables.n_canon() = 0;
  if (entrants.size() == 0)
    return;

  descriptors.allocate(entrants.size());
  frames.allocate(entrants.size());
  tf::offset_block_buffer<Index, def_t> blocks;
  tf::generate_offset_blocks(
      tf::enumerate(entrants), blocks, [&](auto pair, tf::buffer<def_t> &out) {
        auto &&[position, flat_face] = pair;
        const auto tag = tf::exact::tag_of_flat_vertex(face_offsets, flat_face);
        const auto object = flat_face - face_offsets[std::size_t(tag)];
        const auto at = std::size_t(position);
        const auto stamp = face_base + Index(position);
        const auto def_begin = out.size();
        descriptors[at] = descriptor_t{short(tag), object};
        apply_to_form(tag, [&](const auto &form) {
          const auto face = form.faces()[object];
          for (std::size_t side = 0; side < face.size(); ++side)
            expand_side(
                tf::intersect::graph::make_plane_boundary_side_def<Index>(
                    std::int16_t(tag), face, side, Index(-1), stamp, object),
                out);

          // an entrant's own corners may be degenerate where its
          // definitions are not, so both are offered
          frames[at] = tf::exact::make_supported_plane_frame<Int>(
              [&](const auto &consider) {
                for (const auto corner : face)
                  consider(get_point(std::int16_t(tag), Index(corner)));
                for (auto definition = def_begin; definition < out.size();
                     ++definition) {
                  const auto &def = out[definition];
                  consider(get_point(def.point_tag_0, def.point_0));
                  consider(get_point(def.point_tag_1, def.point_1));
                }
              });
        });
      });

  tables.defs() = std::move(blocks.data_buffer());
  tables.edges().offsets_buffer() = std::move(blocks.offsets_buffer());
  const auto n_defs = Index(tables.defs().size());
  tables.edges().data_buffer().allocate(std::size_t(n_defs));
  tables.def_offsets().allocate(std::size_t(n_defs) + 1);
  tf::parallel_for_each(
      tf::make_sequence_range(n_defs + Index(1)),
      [&](Index row) {
        tables.def_offsets()[std::size_t(row)] = row;
        if (row != n_defs)
          tables.edges().data_buffer()[std::size_t(row)] = row;
      },
      tf::checked);
  tables.n_canon() = tf::intersect::graph::fuse_plane_defs(
      merges, vertex_offsets, tables.defs(), tables.def_offsets(),
      tables.edges());
}

} // namespace tf::arrangement
