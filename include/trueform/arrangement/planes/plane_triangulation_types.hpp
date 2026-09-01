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

namespace tf::arrangement {

inline constexpr int ring_bucket_bits = 8;
inline constexpr std::uint32_t ring_buckets = std::uint32_t(1)
                                              << ring_bucket_bits;

/// Coplanar currency at triangle grain. A dead triangle is its survivor's
/// exact twin — the same triangulation triangle re-emitted under another
/// covering member — and the survivor is the minimal-tag covering member.
template <typename Index> struct coplanar_descriptor {
  Index survivor; // twin triangle, in the stream the descriptor lives in
  char opposing;  // the two members' windings differ
};

/// One row per exposed face slot: the face AND the plane it was
/// triangulated on. Under bands a carrier's elected plane is not
/// necessarily the face's own supporting plane. `plane == -1` = uncut, so
/// the face's own plane is the reference.
template <typename Index> struct exposed_descriptor {
  short tag;
  Index object;
  Index plane;
};

} // namespace tf::arrangement
