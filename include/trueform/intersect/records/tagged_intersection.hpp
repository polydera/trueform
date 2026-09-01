/*
 * Copyright (c) 2025 XLAB
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

#include "../../topology/topo_id.hpp"
#include <cstdint>
#include <tuple>

namespace tf::intersect {

/// Record flag: the record's PAIR is exactly coplanar (all vertex signs
/// zero, both ways). Decided once, at the pair call, on original exact
/// coordinates, and carried as a pair table until
/// @ref tf::intersect::distribute_coplanar_flags — the one producer of
/// this bit — stamps it on the records the fan finally holds.
inline constexpr std::uint8_t coplanar_pair_flag = 1;

template <typename Index> struct tagged_intersection {
  // Form indices; 16 bits keep the record at 36 bytes with the flags
  // byte included — the pairwise stages are memory-bound over these
  // buffers. Hard ceiling: 32767 forms (construction narrows with
  // short(tag), which wraps beyond that).
  std::int16_t tag;
  std::int16_t tag_other;
  Index object;
  Index object_other;
  tf::topo_id<Index> target;
  tf::topo_id<Index> target_other;
  Index id;
  /// Pair-call metadata; excluded from ordering and identity below —
  /// consumers reduce it per group (any-of), so relative order of
  /// equal-tuple records with differing flags cannot matter.
  std::uint8_t flags = 0;

  auto key() const { return std::make_tuple(tag, object); }

  friend auto operator<(const tagged_intersection &a,
                        const tagged_intersection &b) -> bool {
    return std::make_tuple(a.tag, a.object, a.tag_other, a.object_other,
                           a.target, a.target_other, a.id) <
           std::make_tuple(b.tag, b.object, b.tag_other, b.object_other,
                           b.target, b.target_other, b.id);
  }

  friend auto operator==(const tagged_intersection &a,
                         const tagged_intersection &b) -> bool {
    return std::make_tuple(a.tag, a.object, a.tag_other, a.object_other,
                           a.target, a.target_other, a.id) ==
           std::make_tuple(b.tag, b.object, b.tag_other, b.object_other,
                           b.target, b.target_other, b.id);
  }
};

} // namespace tf::intersect
