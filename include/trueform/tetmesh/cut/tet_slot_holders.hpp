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
#include "../../core/buffer.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/faces.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../topology/face_membership.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <tbb/parallel_sort.h>

namespace tf::tetmesh {

/// The triangles holding each site slot, as a standing face membership over
/// the first `covered` triangles and the `appended` ones past them as sorted
/// `{slot, triangle}` pairs, so a drain states only what it appended. It
/// lives the build and only grows: a retired triangle stays named, and a
/// reader asks the surface whether a holder is live.
template <typename Index> struct tet_slot_holders {
  tf::face_membership<Index> standing;
  std::size_t covered = 0;
  std::size_t slots = 0;
  tf::buffer<std::array<Index, 2>> appended;
};

/// State `holders` for the slot triangles `faces` over `n_slots` slots: the
/// triangles past the standing membership join the appended pairs, and the
/// membership is stated afresh over them all once the appended pairs
/// outnumber the triangles it covers.
template <typename Index, typename Faces>
auto state_tet_slot_holders(const Faces &faces, std::size_t n_slots,
                            tet_slot_holders<Index> &holders) -> void {
  const std::size_t n = std::size_t(faces.size());
  if (holders.covered == 0 || 3 * (n - holders.covered) > holders.covered) {
    holders.standing.build(
        tf::make_faces(tf::make_blocked_range<3>(
            tf::make_range(faces[0].data(), faces[0].data() + 3 * n))),
        n_slots);
    holders.covered = n;
    holders.slots = n_slots;
    holders.appended.clear();
    return;
  }
  const std::size_t known = holders.appended.size() / 3 + holders.covered;
  for (std::size_t t = known; t < n; ++t)
    for (std::size_t k = 0; k < 3; ++k)
      holders.appended.push_back({Index(faces[t][k]), Index(t)});
  tbb::parallel_sort(holders.appended.begin(), holders.appended.end());
  if constexpr (tf::core::verification)
    tf::core::verify(holders.appended.size() == 3 * (n - holders.covered),
                     "the holders name every triangle of the surface, each "
                     "at its three slots");
}

/// Visit every triangle `holders` names at `slot`.
template <typename Index, typename Visit>
auto for_each_tet_slot_holder(const tet_slot_holders<Index> &holders,
                              Index slot, const Visit &visit) -> void {
  if (std::size_t(slot) < holders.slots)
    for (auto u : holders.standing[std::size_t(slot)])
      visit(Index(u));
  for (auto at =
           std::lower_bound(holders.appended.begin(), holders.appended.end(),
                            std::array<Index, 2>{slot, Index(-1)});
       at != holders.appended.end() && (*at)[0] == slot; ++at)
    visit((*at)[1]);
}

} // namespace tf::tetmesh
