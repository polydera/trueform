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
#include "../../../../exact/vertex.hpp"
#include <cmath>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Offer `site` as the encroacher of the piece from `a` to `b`: kept in
/// `best` when it stands inside the piece's diametral sphere and sees the
/// piece under a wider angle than the one held, `cosine` its angle's, the
/// smaller name winning a tie.
template <typename Index, typename Coord>
auto offer_tet_segment_encroacher(const tf::exact::pt3<Coord> &a,
                                  const tf::exact::pt3<Coord> &b,
                                  const tf::exact::pt3<Coord> &pt, Index site,
                                  Index &best, double &cosine) -> void {
  double dot = 0, aa = 0, bb = 0;
  for (std::size_t k = 0; k < 3; ++k) {
    const double da = double(a[k]) - double(pt[k]);
    const double db = double(b[k]) - double(pt[k]);
    dot += da * db;
    aa += da * da;
    bb += db * db;
  }
  if (!(dot < 0))
    return;
  const double c = dot / std::sqrt(aa * bb);
  if (best == Index(-1) || c < cosine || (c == cosine && site < best)) {
    best = site;
    cosine = c;
  }
}

} // namespace tf::topology::cdt::dt3
