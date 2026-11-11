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
#include "./tet_membership_like.hpp"

namespace tf {

/// @ingroup topology_connectivity
/// @brief Outputs every tet other than `tet_id` holding the facet
/// `(v0, v1, v2)`.
///
/// The intersection of the three corners' sorted memberships.
template <typename Index, typename Policy, typename Iterator>
auto tet_facet_neighbors(const tf::tet_membership_like<Policy> &blink,
                         Index tet_id, const Index &v0, const Index &v1,
                         const Index &v2, Iterator out) {
  const auto &range0 = blink[v0];
  auto it0 = range0.begin();
  auto end0 = range0.end();
  const auto &range1 = blink[v1];
  auto it1 = range1.begin();
  auto end1 = range1.end();
  const auto &range2 = blink[v2];
  auto it2 = range2.begin();
  auto end2 = range2.end();
  while ((it0 != end0) & (it1 != end1) & (it2 != end2)) {
    Index a = Index(*it0);
    Index b = Index(*it1);
    Index c = Index(*it2);
    if ((a == b) & (b == c)) {
      if (a != tet_id)
        *out++ = a;
      ++it0;
      ++it1;
      ++it2;
    } else {
      Index lowest = a < b ? (a < c ? a : c) : (b < c ? b : c);
      if (a > lowest)
        ++it0;
      if (b > lowest)
        ++it1;
      if (c > lowest)
        ++it2;
    }
  }
  return out;
}

} // namespace tf
