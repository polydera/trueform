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
  /// Pooled faces a corner of which could not stand on its pool's plane
  /// within the band, so they keep their exact planes.
  std::int64_t released_pool_faces = 0;
  /// `(vertex, face)` pairs the discovery admitted.
  std::int64_t contacts = 0;
  /// Planes the contacts pulled together, each one committed plane or none.
  std::int64_t groups = 0;
  /// Groups committed to the plane a face of theirs already stands on.
  std::int64_t exact_commits = 0;
  /// Groups committed to a small-normal plane their corners moved onto.
  std::int64_t moved_commits = 0;
  /// Groups whose corners no plane could hold within the band: their faces
  /// keep their own exact planes and are no drop target.
  std::int64_t uncommittable = 0;
  /// Vertices that owed a committed plane and moved onto it.
  std::int64_t moved = 0;
  /// Vertices that owed committed planes and kept their input point,
  /// because no lattice point on all of them stood within the band.
  std::int64_t refused = 0;
  /// Contacts whose face committed no plane, so their vertex was not
  /// pulled by them.
  std::int64_t unheld_contacts = 0;
  /// The exact identifications the input states between forms and the
  /// placement kept: a vertex on a closed face, two edges crossing.
  std::int64_t kept_on_face = 0;
  std::int64_t kept_edges = 0;
  /// Rounds the placement ran before every kept identification held, and
  /// the vertices pinned to their input points so that it would.
  std::int64_t rounds = 0;
  std::int64_t pinned = 0;
};

} // namespace tf::exact::door
