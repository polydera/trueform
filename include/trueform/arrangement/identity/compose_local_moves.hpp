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
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include "../planes/plane_flat_weld.hpp"
#include "./extent_shift.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace tf::arrangement {

/// The rows a diff moved in `L` (`local_rows`, closed), in row order, then the
/// plane-tier folds they stand behind, appended to `welds` in the next value's
/// flat currency and their planes to `dirty_planes`; a refused carve
/// (`carved` false) states no fold onto an original.
template <typename Index, typename LocalRows>
auto compose_local_moves(const LocalRows &local_rows,
                         const diff_extents<Index> &extents, Index n_flat,
                         tf::buffer<std::array<Index, 3>> &&carve_moved,
                         const tf::buffer<std::array<Index, 3>> &census_moved,
                         const tf::buffer<plane_flat_weld<Index>> &plane_welds,
                         bool carved, tf::buffer<plane_flat_weld<Index>> &welds,
                         tf::buffer<Index> &dirty_planes) -> void {
  const auto states = [&](const plane_flat_weld<Index> &weld) {
    return carved || weld.flat_0 >= n_flat;
  };
  for (const auto p : extents.planes.gap())
    dirty_planes.push_back(p);
  for (const auto &weld : plane_welds)
    if (weld.plane != Index(-1) && states(weld))
      dirty_planes.push_back(extents.planes.forward(weld.plane));

  welds.reserve(carve_moved.size() + census_moved.size() + plane_welds.size());
  auto moved = std::move(carve_moved);
  for (const auto &row : census_moved)
    if (tf::intersect::graph::plane_merge_target(local_rows, row[0], row[1]) ==
        row[2])
      moved.push_back(row);
  std::sort(moved.begin(), moved.end());
  for (const auto &row : moved)
    welds.push_back({row[0] == Index(0) ? row[1] : n_flat + row[1],
                     n_flat + row[2], Index(-1)});

  for (const auto &weld : plane_welds) {
    if constexpr (tf::core::verification)
      tf::core::verify(weld.flat_1 >= n_flat + extents.created.prior,
                       "a plane-tier weld folds a created identity of the "
                       "plane arrangement");
    if (!states(weld))
      continue;
    const auto member = extents.created.forward(weld.flat_1 - n_flat);
    Index survivor;
    if (weld.flat_0 < n_flat) {
      survivor = tf::intersect::graph::plane_merge_target(local_rows, Index(0),
                                                          weld.flat_0);
      if constexpr (tf::core::verification)
        tf::core::verify(survivor != Index(-1),
                         "a minted survivor's original chases to its mint");
    } else {
      const auto id = weld.flat_0 - n_flat;
      if (id < extents.created.prior) {
        const auto target =
            tf::intersect::graph::plane_merge_target(local_rows, Index(1), id);
        survivor = target == Index(-1) ? id : target;
      } else
        survivor = extents.created.forward(id);
    }
    welds.push_back({n_flat + survivor, n_flat + member,
                     extents.planes.forward(weld.plane)});
  }
}

} // namespace tf::arrangement
