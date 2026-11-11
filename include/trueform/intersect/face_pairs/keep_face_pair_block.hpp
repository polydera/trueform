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
#include "../../core/aabb.hpp"
#include "../../core/config/verify.hpp"
#include "../../core/intersects.hpp"
#include "./face_pair_search.hpp"

#include <cstddef>

namespace tf::intersect {

/// Whether two exact boxes meet, closed on every side as
/// @ref tf::intersects states it, read without a branch: a leaf block
/// tests every candidate and keeps a small fraction, so a branch per axis
/// is a misprediction per pair.
template <typename Int>
auto exact_boxes_meet(const tf::aabb<Int, 3> &a, const tf::aabb<Int, 3> &b)
    -> int {
  return int(a.min[0] <= b.max[0]) & int(b.min[0] <= a.max[0]) &
         int(a.min[1] <= b.max[1]) & int(b.min[1] <= a.max[1]) &
         int(a.min[2] <= b.max[2]) & int(b.min[2] <= a.max[2]);
}

/// THE BLOCK'S KEPT PAIRS. A leaf block states `n0 * n1` candidates and
/// its box filter keeps a small fraction of them, so the filter writes
/// every candidate's position and advances past it only when the boxes
/// meet: the kept pairs come out row-major — the order the pair loops
/// visit them — with no branch on the verdict.
///
/// `skip_mirror` is the diagonal block's own rule: it visits each unordered
/// pair once, so row `i` starts at `i + 1`.
template <typename Index, typename Int>
void keep_face_pair_block(face_pair_workspace<Index, Int> &ws,
                          bool skip_mirror) {
  const auto n0 = ws.n0();
  const auto n1 = ws.n1();
  ws.kept_pairs.allocate(skip_mirror ? n0 * (n0 - 1) / 2 : n0 * n1);
  std::size_t n_kept = 0;
  for (std::size_t i = 0; i < n0; ++i) {
    const auto box0 = ws.ibox0[i];
    for (std::size_t j = skip_mirror ? i + 1 : 0; j < n1; ++j) {
      ws.kept_pairs[n_kept] = {int(i), int(j)};
      n_kept += std::size_t(exact_boxes_meet(box0, ws.ibox1[j]));
    }
  }
  ws.kept_pairs.allocate(n_kept);
  if constexpr (tf::core::verification) {
    std::size_t at = 0;
    for (std::size_t i = 0; i < n0; ++i)
      for (std::size_t j = skip_mirror ? i + 1 : 0; j < n1; ++j) {
        if (!tf::intersects(ws.ibox0[i], ws.ibox1[j]))
          continue;
        tf::core::verify(at < n_kept && ws.kept_pairs[at][0] == int(i) &&
                             ws.kept_pairs[at][1] == int(j),
                         "the kept pairs are the candidates whose boxes meet, "
                         "row-major");
        ++at;
      }
    tf::core::verify(at == n_kept, "every kept pair's boxes meet");
  }
}

} // namespace tf::intersect
