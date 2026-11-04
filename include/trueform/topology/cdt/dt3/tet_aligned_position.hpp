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
#include "../../../core/config/verify.hpp"
#include "../../../exact/vertex.hpp"
#include "./tet_position.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The position `p` read `depth` bits below the frame, `depth` at least its
/// own, in `Wide`: `floor 2^depth + residual 2^(depth - its depth)`.
template <typename Wide, typename Coord>
auto tet_aligned_position(const tet_position<Coord> &p, int depth)
    -> tf::exact::pt3<Wide> {
  if constexpr (tf::core::verification)
    tf::core::verify(depth >= p.depth,
                     "an aligned read reads a position at its depth or below");
  tf::exact::pt3<Wide> out;
  for (std::size_t k = 0; k < 3; ++k)
    out[k] = Wide(p.floor[k]) * (Wide(1) << depth) +
             Wide(p.residual[k]) * (Wide(1) << (depth - p.depth));
  return out;
}

} // namespace tf::topology::cdt::dt3
