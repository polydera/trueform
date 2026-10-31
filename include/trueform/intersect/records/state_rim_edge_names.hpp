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
#include "../../core/range.hpp"
#include "../../exact/door/within_band_of_edge_span.hpp"
#include "../../exact/input_lattice_reader.hpp"
#include "../../exact/tag_of_flat_vertex.hpp"
#include "../../topology/topo_type.hpp"
#include "../../topology/vertex_id_in_face.hpp"
#include "../classify/intersection_payload.hpp"
#include "./tagged_intersection.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::intersect {

/// THE EDGE LANDING IS A NAME BESIDE ITS POSITION. The band put each rim
/// vertex of `names` at the lattice point nearest its projection onto
/// another form's edge; the rounding leaves it within a unit of the edge,
/// so the exact tier is told what the band did: one raw record per name,
/// in the exact vertex-on-edge shape — the vertex at its slot of a face of
/// its own, against the named face's edge slot, the vertex's projection
/// onto the edge as the parameter, every position read where the lattice
/// placed it. A vertex the rounding put on an end of the edge coincides
/// with that corner and the gate states it, so it names nothing here.
/// Answers how many names were stated.
template <typename Index, typename Int, typename RealType, typename ApplyToForm>
auto state_rim_edge_names(
    tf::range<const std::array<Index, 3> *, tf::dynamic_size> names,
    const tf::buffer<Index> &vertex_offsets,
    const tf::buffer<Index> &face_offsets,
    const tf::exact::input_lattice_reader<Index, RealType, Int, ApplyToForm>
        &reader,
    const ApplyToForm &apply_to_form, Int band,
    tf::buffer<tagged_intersection<Index>> &records,
    tf::buffer<tf::exact::edge_fractions<Int, Index>> &parameters)
    -> std::size_t {
  std::size_t stated = 0;
  for (const auto &name : names) {
    const auto flat = name[0], slot = name[2];
    const auto vertex_tag = tf::exact::tag_of_flat_vertex(vertex_offsets, flat);
    const auto face_tag = tf::exact::tag_of_flat_vertex(face_offsets, name[1]);
    const auto id = flat - vertex_offsets[std::size_t(vertex_tag)];
    const auto object_other = name[1] - face_offsets[std::size_t(face_tag)];
    const auto corner_base = vertex_offsets[std::size_t(face_tag)];
    Index from = Index(0), to = Index(0);
    apply_to_form(int(face_tag), [&from, &to, object_other, slot,
                                  corner_base](const auto &form) {
      const auto corners = form.faces()[object_other];
      const auto n = Index(corners.size());
      from = corner_base + Index(corners[slot]);
      to = corner_base + Index(corners[(slot + 1) % n]);
    });
    const auto at = reader.point_of_flat(flat);
    const auto a = reader.point_of_flat(from);
    const auto b = reader.point_of_flat(to);
    if (at == a || at == b)
      continue;
    tf::exact::door::edge_span_distance<Int> distance{};
    if (!tf::exact::door::within_band_of_edge_span<Int>(at, a, b, band,
                                                        distance))
      continue;
    Index object = Index(0), target = Index(0);
    apply_to_form(int(vertex_tag), [&object, &target, id](const auto &form) {
      object = Index(form.face_membership()[id][0]);
      target = Index(tf::vertex_id_in_face(id, form.faces()[object]));
    });
    records.push_back({std::int16_t(vertex_tag),
                       std::int16_t(face_tag),
                       object,
                       object_other,
                       {target, tf::topo_type::vertex},
                       {slot, tf::topo_type::edge},
                       Index(parameters.size())});
    parameters.push_back(tf::exact::make_vertex_edge_payload<Index, Int>(
        {from, a}, {to, b}, at, flat));
    ++stated;
  }
  return stated;
}

} // namespace tf::intersect
