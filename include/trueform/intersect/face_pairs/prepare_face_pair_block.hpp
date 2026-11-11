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

#include "../classify/face_plane_info.hpp"
#include "./face_pair_search.hpp"
#include "./keep_face_pair_block.hpp"

#include <algorithm>
#include <cstddef>

namespace tf::intersect {

/// THE BLOCK'S KEPT PAIRS, AND THE PLANES THOSE PAIRS READ, for a search
/// that gathers its faces per block. A face's supporting plane is a
/// wide-integer cross, so only a face some kept pair of the block reaches
/// is planed; a face the block's own pairs do not reach holds no plane
/// here.
template <typename Index, typename Int>
void prepare_face_pair_block(face_pair_workspace<Index, Int> &ws,
                             bool skip_mirror) {
  keep_face_pair_block(ws, skip_mirror);
  const auto n0 = ws.n0();
  const auto n1 = ws.n1();
  ws.reached0.allocate(n0);
  ws.reached1.allocate(n1);
  std::fill(ws.reached0.begin(), ws.reached0.end(), false);
  std::fill(ws.reached1.begin(), ws.reached1.end(), false);
  for (const auto &kept : ws.kept_pairs) {
    ws.reached0[std::size_t(kept[0])] = true;
    ws.reached1[std::size_t(kept[1])] = true;
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
