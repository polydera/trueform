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
#include "../../core/algorithm/reduce.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/cdt/dt3/tet_site_position.hpp"
#include "../../topology/cdt/dt3/tetrahedralization_owner.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// The sites of `owner` standing below the frame and the deepest of them,
/// `{0, 0}` at once while every site has stood on it.
template <typename Index, typename Coord, typename Int>
auto count_tet_deep_sites(
    const tf::topology::cdt::dt3::tetrahedralization_owner<Index, Coord, Int>
        &owner) -> std::array<std::size_t, 2> {
  namespace dt3 = tf::topology::cdt::dt3;
  if (!dt3::tet_deepest_depth(owner))
    return {0, 0};
  return tf::reduce(
      tf::make_mapped_range(
          tf::make_sequence_range(dt3::tet_site_count(owner)),
          [&owner](std::size_t site) {
            const auto depth =
                std::size_t(dt3::tet_site_depth(owner, Index(site)));
            return std::array<std::size_t, 2>{std::size_t(depth != 0), depth};
          }),
      [](const std::array<std::size_t, 2> &a,
         const std::array<std::size_t, 2> &b) {
        return std::array<std::size_t, 2>{a[0] + b[0], std::max(a[1], b[1])};
      },
      std::array<std::size_t, 2>{0, 0}, tf::checked);
}

} // namespace tf::tetmesh
