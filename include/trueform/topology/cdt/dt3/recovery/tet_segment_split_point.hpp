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

/// Where along the piece from `a` to `b` its encroacher `r` asks the split,
/// as a fraction from `a`, the encroacher being the site inside the
/// diametral sphere that sees the piece widest. The end nearer the
/// encroacher decides: an acute corner there splits at the encroacher's own
/// radius about it, so the corner piece is never encroached by its
/// encroacher and equal radii on the corner's chains never encroach one
/// another; any other end splits at the encroacher's projection. A split
/// leaving the encroacher nearer it than the far end, or a projection off
/// the piece, takes the midpoint, `0.5`.
template <typename Coord>
auto tet_segment_split_point(const tf::exact::pt3<Coord> &a,
                             const tf::exact::pt3<Coord> &b,
                             const tf::exact::pt3<Coord> &r, bool acute_a,
                             bool acute_b) -> double {
  double pa[3], pb[3], pr[3];
  for (std::size_t k = 0; k < 3; ++k) {
    pa[k] = double(a[k]);
    pb[k] = double(b[k]);
    pr[k] = double(r[k]);
  }
  const auto distance = [](const double *x, const double *y) {
    double d = 0;
    for (std::size_t k = 0; k < 3; ++k)
      d += (x[k] - y[k]) * (x[k] - y[k]);
    return std::sqrt(d);
  };
  const double ra = distance(pa, pr), rb = distance(pb, pr);
  const bool from_a = ra <= rb;
  if (from_a ? acute_a : acute_b) {
    const double t = (from_a ? ra : rb) / distance(pa, pb);
    if (!(t > 0 && t < 1))
      return 0.5;
    double p[3];
    for (std::size_t k = 0; k < 3; ++k)
      p[k] = from_a ? pa[k] + t * (pb[k] - pa[k]) : pb[k] + t * (pa[k] - pb[k]);
    const double to_far_end = distance(p, from_a ? pb : pa);
    return distance(p, pr) > to_far_end ? 0.5 : from_a ? t : 1 - t;
  }
  double along = 0, length = 0;
  for (std::size_t k = 0; k < 3; ++k) {
    along += (pr[k] - pa[k]) * (pb[k] - pa[k]);
    length += (pb[k] - pa[k]) * (pb[k] - pa[k]);
  }
  const double t = along / length;
  if (!(t > 0 && t < 1))
    return 0.5;
  double p[3];
  for (std::size_t k = 0; k < 3; ++k)
    p[k] = pa[k] + t * (pb[k] - pa[k]);
  const double reach = distance(p, pr);
  return reach > distance(p, pa) || reach > distance(p, pb) ? 0.5 : t;
}

} // namespace tf::topology::cdt::dt3
