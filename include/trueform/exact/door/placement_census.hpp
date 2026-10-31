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

namespace tf::exact::door {

/// What the band did, in its own terms. A refusal is the law working — a
/// plane its own corners could not follow stays exact, a vertex no plane
/// could hold within the band keeps its input point — so it is counted
/// and published, never silent.
struct placement_census {
  /// THE BAND THAT ACTED, in world units: the request taken to the
  /// smallest actionable tolerance containing it and the widest the pair
  /// search can carry. A caller reads here what was HONOURED, never what
  /// was asked, and zero exactly when no band ran.
  double effective_tolerance = 0.0;
  /// The pool: its committed planes, the faces that answer to one, and
  /// among those the faces a corner of which could not stand on its pool's
  /// plane within the band, so they keep their exact planes.
  std::int64_t pools = 0;
  std::int64_t pooled_faces = 0;
  std::int64_t released_pool_faces = 0;
  /// The vertices on boundary edges, the only candidates for a landing.
  std::int64_t rim_vertices = 0;
  /// Rim vertices exactly on another form's face: no gap, no landing.
  std::int64_t on_face = 0;
  /// Rim vertices a rim edge of which already crosses another form's edge:
  /// the input states the meeting, and the vertex is left where it is.
  std::int64_t crossing = 0;
  /// Per feature, the rim vertices whose nearest feature of another form
  /// within the band it is, and of those the ones that did not land on it.
  std::int64_t corner_targets = 0;
  std::int64_t corner_refused = 0;
  std::int64_t edge_targets = 0;
  std::int64_t edge_refused = 0;
  std::int64_t face_targets = 0;
  std::int64_t face_refused = 0;
  /// Faces a rim vertex lands inside of that had no pool plane: those the
  /// grid committed to a plane their corners could follow, and those it
  /// could not.
  std::int64_t target_planes = 0;
  std::int64_t target_planes_refused = 0;
  /// Vertices that owed the committed planes of their own faces and could
  /// not stand on all of them within the band.
  std::int64_t ride_refused = 0;
  /// Vertices that moved.
  std::int64_t moved = 0;
};

} // namespace tf::exact::door
