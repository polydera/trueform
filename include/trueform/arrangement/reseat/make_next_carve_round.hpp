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
#include "./make_reseat_delta_forms.hpp"
#include "./reseat_pieces.hpp"
#include "./reseat_targets.hpp"
#include <cstddef>

namespace tf::arrangement::reseat {

/// A delta's targets in carve language: its faces and edges named by the
/// corners its forms were built from, each found on the source face its
/// delta face lies in.
template <typename Index, typename Int, typename DeltaForms>
auto make_next_carve_round(const reseat_targets<Index, Int> &targets,
                           const reseat_pieces<Index, Int> &pieces,
                           const standing_pieces<Index> &standing,
                           const reseat_delta_forms<Index, Int> &forms,
                           const DeltaForms &delta_forms)
    -> carve_round<Index, Int> {
  carve_round<Index, Int> round;
  const auto corner_of = [&](Index tag, Index vertex) {
    return forms.vertices[std::size_t(tag)][std::size_t(vertex)];
  };
  const auto source_of = [&](Index tag, Index face) -> Index {
    const auto &o = forms.faces[std::size_t(tag)][std::size_t(face)];
    if (o.role == delta_role::piece)
      return pieces.pieces[std::size_t(o.id)].source;
    return standing.source_of_key(o.tag, o.id);
  };
  for (const auto &e : targets.edges)
    round.edges.push_back({e.tag, source_of(e.tag, e.face),
                           corner_of(e.tag, e.a), corner_of(e.tag, e.b),
                           e.point});
  for (const auto &f : targets.faces)
    delta_forms(f.tag, [&](const auto &form) {
      const auto face = form.faces()[f.face];
      round.faces.push_back(
          {f.tag,
           source_of(f.tag, f.face),
           {corner_of(f.tag, Index(face[0])), corner_of(f.tag, Index(face[1])),
            corner_of(f.tag, Index(face[2]))},
           f.point});
    });
  return round;
}

} // namespace tf::arrangement::reseat
