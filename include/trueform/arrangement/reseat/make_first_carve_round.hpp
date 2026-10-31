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
#include "./reseat_pieces.hpp"
#include "./reseat_targets.hpp"
#include <cstdint>

namespace tf::arrangement::reseat {

/// The world's first round in carve language: its targets name source edges
/// and source faces, each found on the source face it names.
template <typename Index, typename Int, typename ApplyToForm>
auto make_first_carve_round(const reseat_targets<Index, Int> &targets,
                            const ApplyToForm &apply_to_form)
    -> carve_round<Index, Int> {
  carve_round<Index, Int> round;
  for (const auto &e : targets.edges)
    round.edges.push_back({e.tag,
                           e.face,
                           {std::int16_t(e.tag), e.a},
                           {std::int16_t(e.tag), e.b},
                           e.point});
  for (const auto &f : targets.faces)
    apply_to_form(f.tag, [&](const auto &form) {
      const auto face = form.faces()[f.face];
      round.faces.push_back(
          {f.tag,
           f.face,
           {piece_corner<Index>{std::int16_t(f.tag), Index(face[0])},
            piece_corner<Index>{std::int16_t(f.tag), Index(face[1])},
            piece_corner<Index>{std::int16_t(f.tag), Index(face[2])}},
           f.point});
    });
  return round;
}

} // namespace tf::arrangement::reseat
