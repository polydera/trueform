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
#include "../../intersect/graph/plane_uncut_entrants.hpp"
#include "./plane_wave_answered.hpp"

namespace tf::arrangement {

/// The source faces a WAVE'S own statements reach that no tier of this
/// arrangement has named.
///
/// A group still the world's holds only the faces the cut world named, so a
/// split of one that lands on an ORIGINAL SIDE reaches every face holding
/// that edge, and an original this wave RETIRES reaches every face incident
/// to it — including faces that own no definition row anywhere.
/// @ref tf::intersect::graph::discover_uncut_entrants is the one producer of
/// both facts, for this tier exactly as for the cut world's.
///
/// NOT IN THE GROUP IS NOT THE SAME AS NOT IN THE WORLD. A face can hold the
/// edge, own a definition somewhere, and still hold no instance of the group
/// this split names — a degenerate carrier is one way — so the group's own
/// span cannot decide who is new. @ref tf::arrangement::plane_wave_answered
/// is this tier's authority on that, and the producer asks it as it emits.
///
/// The caller owns that mask and this SEEDS it, once — a build that never
/// reaches an entrance never pays for the pass, which is the grain every
/// structure a wave needs is built at.
template <typename Index, typename World, typename Retired, typename Roots,
          typename FaceOffsets, typename ApplyToForm>
auto discover_plane_wave_entrants(const World &world,
                                  const Retired &retired_originals,
                                  const Roots &roots,
                                  plane_wave_answered<Index> &answered,
                                  const FaceOffsets &face_offsets,
                                  const ApplyToForm &apply_to_form,
                                  tf::buffer<Index> &entrants) -> void {
  entrants.clear();
  if (!answered.seeded())
    answered.seed(world, Index(face_offsets.size()) - Index(1), apply_to_form);
  tf::intersect::graph::discover_uncut_entrants(
      world, retired_originals, roots, world.vertex_offsets(), face_offsets,
      apply_to_form,
      [&answered](Index tag, Index object) {
        return answered.answered(tag, object);
      },
      entrants);
}

} // namespace tf::arrangement
