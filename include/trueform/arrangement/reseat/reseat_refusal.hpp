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
#include <cstdint>

namespace tf::arrangement::reseat {

/// Why a carve was not stated. A refused carve leaves the world as built.
enum class reseat_refusal : std::uint8_t {
  none,
  /// the targets placed no piece
  no_pieces,
  /// the carve's rounds ended with targets still standing
  rounds_capped,
  /// a round stated more targets than it was given, after a round that did
  /// the same
  diverged,
  /// the faces the carve must restate kept growing past the widening budget
  reach_unbounded,
  /// a promoted face found no delta face to name its plane
  piece_unplaced,
  /// a seat corner resolved to no identity
  seat_unresolved,
  /// a split root the world states has no delta root over the same ends
  root_unmatched,
  /// a root's surviving identities differ in count between world and delta
  root_survivors,
  /// a root's moved identity lands on an original
  root_onto_original,
  /// an identity the carve moved has no place in the delta
  moved_unplaced,
  /// a piece names a plane only the plane arrangement's promotion holds
  plane_promoted,
  /// a name the carve reads answers outside the currency it reads in
  name_foreign
};

} // namespace tf::arrangement::reseat
