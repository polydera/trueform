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

namespace tf {

/// @ingroup fill
/// @brief The dyadic depth every hole-filling split parameter is stated at.
///
/// It is the scale the whole arc shares: the protection tier halves a
/// subedge at most this many times, the planar refiner's own split depth is
/// the same, and the lattice a group works on carries exactly this many bits
/// of scale, so every split point is an exact lattice point.
inline constexpr int hole_split_depth = 6;

/// @ingroup fill
/// @brief The denominator a split parameter's numerator is stated over.
inline constexpr int hole_split_scale = 1 << tf::hole_split_depth;

/// @ingroup fill
/// @brief One split of an original boundary edge.
///
/// The edge is named in its canonical direction — `v0 < v1` — and the
/// parameter runs from `v0`, so two carriers of one edge place the same
/// point from the same record. `point` is the flat identity the canonical
/// merge resolved the record to, which the patch and the carrying face's
/// plan both name.
///
/// @tparam Index The index type of the mesh the split is stated against.
template <typename Index> struct hole_split {
  /// The carrying triangle this split subdivides.
  Index face;
  /// The original edge's lower vertex.
  Index v0;
  /// The original edge's higher vertex.
  Index v1;
  /// The point the record resolved to.
  Index point;
  /// The numerator over @ref tf::hole_split_scale, measured from `v0`.
  std::uint8_t parameter;
};

} // namespace tf
