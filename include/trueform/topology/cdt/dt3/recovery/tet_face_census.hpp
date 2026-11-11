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

namespace tf::topology::cdt::dt3 {

/// What face recovery did, stated so no path runs unseen: the facets found
/// covering faces before any flip; the passes the uncovered faces took, the
/// piercing edges flips removed, the points placed, on the frame or below
/// it, and the points `unplaced`: no cell of the complex held the point the
/// face's lattice named, or the append withdrew its site, so the pass left
/// that piercing edge for the next, and the points `on_piece`: the frame's
/// point lay on a piece of a segment, which no face point divides, so the
/// pass left that edge too; the refusals in all, across every round a face
/// is due in, and by kind —
/// a rim segment refused, a segment piercing the face, a point with no place
/// on the face's lattice to the deep lattice's cap, the site budget, a
/// piercing edge that neither flips nor a point removed — and, of the last,
/// `standing`: the faces whose lattice point a site already held at every
/// depth to that cap.
struct tet_face_census {
  std::size_t covered = 0;
  std::size_t passes = 0;
  std::size_t flipped = 0;
  std::size_t points = 0;
  std::size_t unplaced = 0;
  std::size_t on_piece = 0;
  std::size_t refused = 0;
  std::size_t open_rim = 0;
  std::size_t crossed = 0;
  std::size_t off_lattice = 0;
  std::size_t budget = 0;
  std::size_t capped = 0;
  std::size_t pierced = 0;
  std::size_t standing = 0;
};

} // namespace tf::topology::cdt::dt3
