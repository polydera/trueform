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

#include "../../core/intersects.hpp"
#include "../classify/face_plane_info.hpp"
#include "./face_pair_search.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::intersect {

/// THE BLOCK'S KEPT PAIRS, AND THE PLANES THOSE PAIRS READ.
///
/// A leaf block states `n0 * n1` candidates and its box filter keeps a
/// small fraction of them. The filter is a handful of integer comparisons;
/// a face's supporting plane is a wide-integer cross. So the cheap fact is
/// stated first and stated once — the pair loops read `pair_kept` instead of
/// retesting the boxes — and only a face some kept pair reaches is planed. A
/// face no pair reaches holds no plane and is never asked for one.
///
/// `skip_mirror` is the diagonal block's own rule: it visits each unordered
/// pair once, so the reach is counted the same way.
template <typename Index, typename Int, typename Payload>
void prepare_face_pair_block(face_pair_workspace<Index, Int, Payload> &ws,
                             bool skip_mirror) {
  const auto n0 = ws.n0();
  const auto n1 = ws.n1();
  ws.pair_kept.allocate(n0 * n1);
  ws.reached0.allocate(n0);
  ws.reached1.allocate(n1);
  std::fill(ws.reached0.begin(), ws.reached0.end(), false);
  std::fill(ws.reached1.begin(), ws.reached1.end(), false);
  for (std::size_t i = 0; i < n0; ++i)
    for (std::size_t j = skip_mirror ? i + 1 : 0; j < n1; ++j) {
      const bool kept = tf::intersects(ws.ibox0[i], ws.ibox1[j]);
      ws.pair_kept[i * n1 + j] = kept;
      if (kept) {
        ws.reached0[i] = true;
        ws.reached1[j] = true;
      }
    }
  ws.fp0.allocate(n0);
  for (std::size_t i = 0; i < n0; ++i)
    if (ws.reached0[i])
      ws.fp0[i] = tf::exact::make_face_plane(ws.face0(i));
  ws.fp1.allocate(n1);
  for (std::size_t j = 0; j < n1; ++j)
    if (ws.reached1[j])
      ws.fp1[j] = tf::exact::make_face_plane(ws.face1(j));
}

} // namespace tf::intersect
