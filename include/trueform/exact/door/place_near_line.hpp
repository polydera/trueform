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

#include "../../core/point.hpp"
#include "../meta.hpp"
#include "./quantized_plane.hpp"
#include "./round_to_wide.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace tf::exact::door {

/// The lattice point nearest the real line of two plane names, through
/// the point of that line nearest `p`. It is what rank 2 states when no
/// lattice point lies on both planes at all — the pair is still the
/// feature the vertex belongs to, and the certificate decides whether
/// the answer is close enough to keep.
template <typename Int>
auto place_near_line(const quantized_plane<Int> &a,
                     const quantized_plane<Int> &b,
                     const tf::point<Int, 3> &p,
                     std::array<typename tf::exact::meta<Int>::T1, 3> &out)
    -> bool {
  const double na[3] = {static_cast<double>(a.normal[0]),
                        static_cast<double>(a.normal[1]),
                        static_cast<double>(a.normal[2])};
  const double nb[3] = {static_cast<double>(b.normal[0]),
                        static_cast<double>(b.normal[1]),
                        static_cast<double>(b.normal[2])};
  const double aa = na[0] * na[0] + na[1] * na[1] + na[2] * na[2];
  const double bb = nb[0] * nb[0] + nb[1] * nb[1] + nb[2] * nb[2];
  const double ab = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2];
  const double det = aa * bb - ab * ab;
  if (!(det > 0.0))
    return false;
  const double at[3] = {static_cast<double>(p[0]), static_cast<double>(p[1]),
                        static_cast<double>(p[2])};
  const double ra = static_cast<double>(a.offset) -
                    (na[0] * at[0] + na[1] * at[1] + na[2] * at[2]);
  const double rb = static_cast<double>(b.offset) -
                    (nb[0] * at[0] + nb[1] * at[1] + nb[2] * at[2]);
  const double la = (ra * bb - rb * ab) / det;
  const double lb = (rb * aa - ra * ab) / det;
  for (std::size_t k = 0; k < 3; ++k) {
    const double q = at[k] + la * na[k] + lb * nb[k];
    if (!std::isfinite(q))
      return false;
    out[k] = round_to_wide<Int>(q);
  }
  return true;
}

} // namespace tf::exact::door
