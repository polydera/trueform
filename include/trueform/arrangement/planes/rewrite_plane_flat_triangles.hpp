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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../intersect/graph/plane_identity_collapse.hpp"
#include <array>

namespace tf::arrangement {

/// Rewrite every retained triangle corner through one closed weld table.
/// Triangle identity is the largest independent carrier; an empty table does
/// no work and a target always speaks the created suffix of flat space.
template <typename Index, typename Triangles>
auto rewrite_plane_flat_triangles(
    Triangles &triangles, const tf::buffer<std::array<Index, 3>> &merges,
    Index n_flat) -> void {
  if (merges.size() == 0)
    return;
  tf::parallel_for_each(
      triangles,
      [&](auto &triangle) {
        for (auto &corner : triangle) {
          const auto source = corner < n_flat ? Index(0) : Index(1);
          const auto id = corner < n_flat ? corner : corner - n_flat;
          const auto target =
              tf::intersect::graph::plane_merge_target(merges, source, id);
          if (target != Index(-1))
            corner = n_flat + target;
        }
      },
      tf::checked);
}

} // namespace tf::arrangement
