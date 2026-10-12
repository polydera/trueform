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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/canonical_plane.hpp"
#include "../../exact/make_supported_plane_frame.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/plane_frame_winding.hpp"
#include "./face_descriptor.hpp"
#include "./name_plane_carriers.hpp"
#include "./plane_graph.hpp"
#include "./seat_named_planes.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// THE ENTRANCE OBEYS THE IDENTITY: the local arrangement's entrants are
/// seated by @ref tf::intersect::graph::seat_named_planes, the cut planes
/// standing.
///
/// Per entrant, `entrant_plane_of` is its plane and `entrant_orientations`
/// its winding in that plane's frame; `entrant_planes` holds the frames of
/// the new planes, read off their smallest entrant; `seated` lists, per
/// plane of the whole carrier space, the entrants it holds. Returns the
/// number of new planes.
template <typename Index, typename Int, typename ApplyToForm, typename GetPoint>
auto seat_entrant_planes(
    const plane_graph<Index, Int> &g,
    const tf::buffer<face_descriptor<Index>> &entrant_descriptors,
    const ApplyToForm &apply_to_form, const GetPoint &get_point,
    tf::buffer<Index> &entrant_plane_of,
    tf::buffer<tf::exact::plane_frame<Int>> &entrant_planes,
    tf::buffer<std::int8_t> &entrant_orientations,
    tf::offset_block_buffer<Index, Index> &seated) -> Index {
  using name_t = tf::exact::canonical_plane<Int>;
  const auto n_entrants = Index(entrant_descriptors.size());
  const auto n_cut = g.n_planes();
  tf::buffer<name_t> names;
  names.allocate(std::size_t(n_entrants));
  tf::parallel_for_each(
      tf::make_sequence_range(n_entrants),
      [&](Index e) {
        const auto &d = entrant_descriptors[std::size_t(e)];
        names[std::size_t(e)] = name_plane_face<Index, Int>(
            Index(d.tag), d.object, apply_to_form, get_point);
      },
      tf::checked(64));

  tf::buffer<Index> run_first;
  const auto n_new = seat_named_planes<Index, Int>(
      names, g.plane_names(), n_cut, entrant_plane_of, run_first);

  const auto position_reader = [&](Index e) {
    const auto &d = entrant_descriptors[std::size_t(e)];
    return [&get_point, tag = d.tag](auto corner) {
      return get_point(std::int16_t(tag), Index(corner));
    };
  };
  entrant_planes.allocate(std::size_t(n_new));
  tf::parallel_for_each(
      tf::make_sequence_range(n_new),
      [&](Index r) {
        const auto first = run_first[std::size_t(r)];
        const auto &d = entrant_descriptors[std::size_t(first)];
        const auto position_of = position_reader(first);
        apply_to_form(d.tag, [&](const auto &form) {
          const auto face = form.faces()[d.object];
          entrant_planes[std::size_t(r)] =
              tf::exact::make_supported_plane_frame<Int>(
                  [&](const auto &consider) {
                    for (const auto corner : face)
                      consider(position_of(corner));
                  });
        });
      },
      tf::checked);
  entrant_orientations.allocate(std::size_t(n_entrants));
  tf::parallel_for_each(
      tf::make_sequence_range(n_entrants),
      [&](Index e) {
        const auto plane = entrant_plane_of[std::size_t(e)];
        const auto &frame = plane < n_cut
                                ? g.frame(plane)
                                : entrant_planes[std::size_t(plane - n_cut)];
        const auto &d = entrant_descriptors[std::size_t(e)];
        const auto position_of = position_reader(e);
        apply_to_form(d.tag, [&](const auto &form) {
          entrant_orientations[std::size_t(e)] = tf::exact::plane_frame_winding(
              frame, form.faces()[d.object], position_of);
        });
      },
      tf::checked);

  // per plane the entrants it holds, by counts and one prefix
  const auto n_planes = n_cut + n_new;
  auto &offsets = seated.offsets_buffer();
  auto &data = seated.data_buffer();
  offsets.allocate(std::size_t(n_planes) + 1);
  std::fill(offsets.begin(), offsets.end(), Index(0));
  for (Index e = 0; e < n_entrants; ++e)
    ++offsets[std::size_t(entrant_plane_of[std::size_t(e)]) + 1];
  for (std::size_t p = 1; p < offsets.size(); ++p)
    offsets[p] += offsets[p - 1];
  data.allocate(std::size_t(n_entrants));
  tf::buffer<Index> cursor;
  cursor.allocate(std::size_t(n_planes));
  for (Index p = 0; p < n_planes; ++p)
    cursor[std::size_t(p)] = offsets[std::size_t(p)];
  for (Index e = 0; e < n_entrants; ++e)
    data[std::size_t(cursor[std::size_t(entrant_plane_of[std::size_t(e)])]++)] =
        e;
  return n_new;
}

} // namespace tf::intersect::graph
