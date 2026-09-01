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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../exact/make_supported_plane_frame.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/plane_frame_winding.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "./face_descriptor.hpp"
#include "./plane_edge_def.hpp"
#include "./plane_graph.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace tf::intersect::graph {

/// An entrant face enters WHOLE: its corner cycle is its loop, so every
/// side is a boundary definition of a face of its own — stamped past
/// the graph's descriptors, on a plane of its own appended past the
/// graph's planes, and stated in RAW keys like every definition the
/// merge has not touched.
///
/// A side whose key names a canonical group the arrangement CUT is one
/// more instance of that group and rides its piece emission; every
/// other side is a whole part that joins a group of its key or founds
/// one. Both channels are the respan's; nothing here splits anything.
template <typename Index, typename Int, typename GetPoint, typename ApplyToForm,
          typename FaceOffsets, typename Entrants, typename SplitRoots>
auto state_uncut_entrants(
    const plane_graph<Index, Int> &g, const GetPoint &get_point,
    const ApplyToForm &apply_to_form, const FaceOffsets &face_offsets,
    const Entrants &entrants, const SplitRoots &split_edge,
    tf::buffer<face_descriptor<Index>> &entrant_descriptors,
    tf::buffer<tf::exact::plane_frame<Int>> &entrant_planes,
    tf::buffer<std::int8_t> &entrant_orientations,
    tf::buffer<plane_edge_def<Index>> &parts,
    tf::buffer<Index> &instance_offsets,
    tf::buffer<plane_edge_def<Index>> &instances) -> void {
  using def_t = plane_edge_def<Index>;
  // one side of one entrant face: the split group it rides, or -1 for
  // a whole part. The definition already names its face and its side,
  // so the record carries its own order and no aggregation order is
  // load bearing.
  struct side_t {
    Index group;
    def_t def;
  };
  // the cut roots are stated in ascending order, so membership is one
  // binary search of them
  auto split_group_of = [&](Index canon) -> Index {
    const auto at =
        std::lower_bound(split_edge.begin(), split_edge.end(), canon);
    return at != split_edge.end() && *at == canon
               ? Index(at - split_edge.begin())
               : Index(-1);
  };

  // the entrant tables are dense in the discovery's own order, so the
  // face carrier writes its descriptor and its plane by position
  entrant_descriptors.allocate(entrants.size());
  entrant_planes.allocate(entrants.size());
  entrant_orientations.allocate(entrants.size());
  const auto stamp_base = Index(g.descriptors().size());
  tf::buffer<side_t> sides;
  tf::generic_generate(
      tf::enumerate(entrants), sides,
      [&](auto pair, tf::buffer<side_t> &out) {
        auto &&[index, entrant] = pair;
        const auto position = std::size_t(index);
        const auto flat = Index(entrant);
        const auto tag = tf::exact::tag_of_flat_vertex(face_offsets, flat);
        const auto object = flat - face_offsets[std::size_t(tag)];
        const auto stamp = stamp_base + Index(position);
        entrant_descriptors[position] = {Index(tag), object};
        apply_to_form(tag, [&](const auto &form) {
          const auto face = form.faces()[object];
          const auto position_of = [&](auto corner) {
            return get_point(std::int16_t(tag), Index(corner));
          };
          entrant_planes[position] = tf::exact::make_supported_plane_frame<Int>(
              [&](const auto &consider) {
                for (const auto corner : face)
                  consider(position_of(corner));
              });
          entrant_orientations[position] = tf::exact::plane_frame_winding(
              entrant_planes[position], face, position_of);
          for (std::size_t s = 0; s < face.size(); ++s) {
            const def_t def = make_plane_boundary_side_def<Index>(
                std::int16_t(tag), face, s, Index(-1), stamp, object);
            const auto canon = find_plane_canon_group(g, def);
            out.push_back(
                {canon == Index(-1) ? Index(-1) : split_group_of(canon), def});
          }
        });
      },
      tf::checked);
  // the riders group by their root and the whole parts lead, each run
  // in its faces' own order — one sort of records that carry the key
  std::sort(sides.begin(), sides.end(), [](const side_t &x, const side_t &y) {
    return std::tie(x.group, x.def.face, x.def.side) <
           std::tie(y.group, y.def.face, y.def.side);
  });
  std::size_t n_parts = 0;
  while (n_parts < sides.size() && sides[n_parts].group == Index(-1))
    ++n_parts;
  parts.allocate(n_parts);
  for (std::size_t i = 0; i < n_parts; ++i)
    parts[i] = sides[i].def;
  if (n_parts == sides.size())
    return;
  instances.allocate(sides.size() - n_parts);
  instance_offsets.allocate(split_edge.size() + 1);
  std::fill(instance_offsets.begin(), instance_offsets.end(), Index(0));
  for (auto i = n_parts; i < sides.size(); ++i) {
    instances[i - n_parts] = sides[i].def;
    ++instance_offsets[std::size_t(sides[i].group) + 1];
  }
  for (std::size_t i = 1; i < instance_offsets.size(); ++i)
    instance_offsets[i] += instance_offsets[i - 1];
}

} // namespace tf::intersect::graph
