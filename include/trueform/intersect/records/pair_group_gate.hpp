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

#include "../../core/range.hpp"
#include "../../core/views/slice.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::intersect {

/// What a (face, face) group can still state, asked before the fan
/// writes it.
///
/// Two facts are read per group and no others: the chords its points
/// bound, and — for an exactly coplanar pair — the row that pools the
/// two faces into one plane. A group's points are contained in what BOTH
/// faces were delivered, because every record of a group delivers its
/// point to both sides; so fewer than two shared deliveries proves the
/// group holds at most one point, and a one-point group bounds no chord.
/// The containment is one-sided, so a pair this admits may still hold a
/// single point: that is one singleton group, which falls through every
/// branch of the extractor and states only the loop vertex its face's
/// own deliveries already state.
template <typename Index> struct pair_group_gate {
  using ids_t = tf::range<const Index *, tf::dynamic_size>;

  ids_t face_offsets;
  ids_t delivered_offsets;
  ids_t delivered_points;
  ids_t coplanar_offsets;
  ids_t coplanar_partners;

  /// One face's whole answer, read once and reused across the product.
  struct side {
    Index flat;
    ids_t points;
    ids_t coplanar;
  };

  auto side_of(std::int16_t tag, Index object) const -> side {
    const Index flat = face_offsets[std::size_t(tag)] + object;
    return {flat, block(delivered_offsets, delivered_points, flat),
            block(coplanar_offsets, coplanar_partners, flat)};
  }

  auto keeps(const side &a, const side &b) const -> bool {
    auto i = a.points.begin(), ie = a.points.end();
    auto j = b.points.begin(), je = b.points.end();
    int shared = 0;
    while (i != ie && j != je) {
      if (*i < *j)
        ++i;
      else if (*j < *i)
        ++j;
      else {
        if (++shared == 2)
          return true;
        ++i;
        ++j;
      }
    }
    for (const auto partner : a.coplanar)
      if (partner == b.flat)
        return true;
    return false;
  }

private:
  static auto block(const ids_t &offsets, const ids_t &data, Index flat)
      -> ids_t {
    return tf::slice(data, std::size_t(offsets[std::size_t(flat)]),
                     std::size_t(offsets[std::size_t(flat) + 1]));
  }
};

} // namespace tf::intersect
