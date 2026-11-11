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

namespace tf::topology::cdt::dt3 {

/// A piece of the rim of a region of a face's plane — the face's own, or
/// what its cover leaves of it — between the site slots `lo` < `hi`, and the
/// side of the line from `lo` to `hi` the region lies on: the orient2d sign,
/// on the face's projection, of a point of its plane inside it beside the
/// piece.
template <typename Index> struct tet_rim_piece {
  Index lo;
  Index hi;
  int side;
};

/// The order a rim is searched in, by its edge: `lo`, then `hi`.
template <typename Index>
auto operator<(const tet_rim_piece<Index> &a, const tet_rim_piece<Index> &b)
    -> bool {
  return a.lo < b.lo || (a.lo == b.lo && a.hi < b.hi);
}

/// The order a rim is stored in: its edge, then `side`, so the two pieces of
/// a slit — a segment its region lies on both sides of — stand in one order,
/// and a search by the edge finds both.
template <typename Index>
auto tet_rim_piece_precedes(const tet_rim_piece<Index> &a,
                            const tet_rim_piece<Index> &b) -> bool {
  return a < b || (!(b < a) && a.side < b.side);
}

} // namespace tf::topology::cdt::dt3
