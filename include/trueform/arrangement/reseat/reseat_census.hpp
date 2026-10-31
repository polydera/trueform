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
#include "./reseat_refusal.hpp"
#include <cstddef>

namespace tf::arrangement::reseat {

/// What one carve reached and what its statement retired.
struct carve_reach_census {
  /// World faces the stated carve restates whole.
  std::size_t restated = 0;
  /// Faces the stated carve's delta held only to complete the restated rows.
  std::size_t neighbours = 0;
  /// Delta worlds the carve built over its rounds and widens, and the faces
  /// they held, summed.
  std::size_t delta_rounds = 0;
  std::size_t delta_faces = 0;
  /// Times the reach widened.
  std::size_t widens = 0;
  /// Welds of the stated carve both of whose identities stood in the world
  /// before it: a standing identity retired into another standing one. A
  /// carve mint retiring the original it lands on is not foreign.
  std::size_t foreign_retirements = 0;

  auto operator+=(const carve_reach_census &o) -> carve_reach_census & {
    restated += o.restated;
    neighbours += o.neighbours;
    delta_rounds += o.delta_rounds;
    delta_faces += o.delta_faces;
    widens += o.widens;
    foreign_retirements += o.foreign_retirements;
    return *this;
  }
};

/// What the re-seat did to the build: the carves it stated, what they
/// reached summed over every carve, stated or refused, and why the last one
/// it could not state was refused.
struct reseat_census {
  std::size_t carves = 0;
  carve_reach_census reach;
  reseat_refusal refusal = reseat_refusal::none;
};

} // namespace tf::arrangement::reseat
