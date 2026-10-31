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

#include <cstddef>

namespace tf::arrangement {

/// What one plane arrangement counted: the carriers it triangulated, the
/// product it emitted, the rounds a refusal cost it, and everything the
/// recovery waves stated. A task-local census sums into the round's, and the
/// round's into the build's; `planes`, `triangles` and `failed_planes` are
/// the build's own, stated where it holds them and never summed.
struct plane_arrangement_census {
  std::size_t planes = 0;
  std::size_t stacks = 0;
  std::size_t constraints = 0;
  std::size_t triangles = 0;
  std::size_t dead_triangles = 0;
  std::size_t stack_regions = 0;
  std::size_t rounds = 0;
  std::size_t rebuilt_planes = 0;
  std::size_t refusals = 0;
  std::size_t crossings = 0;
  std::size_t landings = 0;
  /// Points the rim law landed: a cut end the rounding carried across its
  /// own face's side, or a face corner it carried across the cut.
  std::size_t rim_landings = 0;
  /// Cuts whose face covers neither side that the rim law could not land:
  /// the sliver is wider than the lattice's rounding, and the plane emits the
  /// cut in no slot.
  std::size_t uncovered_cuts = 0;
  std::size_t collisions = 0;
  std::size_t splits = 0;
  std::size_t splits_out_of_span = 0;
  std::size_t splits_on_endpoint = 0;
  std::size_t failed_planes = 0;
  std::size_t stalled_planes = 0;
  /// Carriers whose refusal budget ran out, so the wave stopped asking them.
  std::size_t spent_planes = 0;
  /// Carriers the convex family answered: one simple ring it could prove
  /// convex, so no triangulation was built for them.
  std::size_t fanned_planes = 0;
  /// Refined planes the refinement producer could not emit; they carry the
  /// stock kernel's triangulation instead.
  std::size_t refined_fallbacks = 0;
  /// Source faces a wave's own splits reached and this arrangement promoted.
  std::size_t entrant_planes = 0;
  /// Source faces a wave reached whose shared side names a group this
  /// arrangement had already taken or retired: the join would have to grow a
  /// closed span, so the face was declined and never offered again.
  std::size_t declined_entrants = 0;

  auto operator+=(const plane_arrangement_census &o)
      -> plane_arrangement_census & {
    stacks += o.stacks;
    constraints += o.constraints;
    dead_triangles += o.dead_triangles;
    stack_regions += o.stack_regions;
    rounds += o.rounds;
    rebuilt_planes += o.rebuilt_planes;
    refusals += o.refusals;
    crossings += o.crossings;
    landings += o.landings;
    rim_landings += o.rim_landings;
    uncovered_cuts += o.uncovered_cuts;
    collisions += o.collisions;
    splits += o.splits;
    splits_out_of_span += o.splits_out_of_span;
    splits_on_endpoint += o.splits_on_endpoint;
    stalled_planes += o.stalled_planes;
    spent_planes += o.spent_planes;
    fanned_planes += o.fanned_planes;
    refined_fallbacks += o.refined_fallbacks;
    entrant_planes += o.entrant_planes;
    declined_entrants += o.declined_entrants;
    return *this;
  }
};

} // namespace tf::arrangement
