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
#include "../../topology/cdt/dt3/tet_orient3d_sites.hpp"
#include <cstddef>
#include <functional>

namespace tf::tetmesh {

/// The finite cells `keep` admits of `owner` whose four sites are coplanar,
/// positive only symbolically.
template <typename Owner, typename Keep>
auto count_flat_tet_cells(const Owner &owner, const Keep &keep)
    -> std::size_t {
  return tf::reduce(
      tf::make_mapped_range(
          tf::make_sequence_range(owner._n_finite),
          [&owner, &keep](std::size_t cell) {
            if (!keep(cell))
              return std::size_t(0);
            const auto corners = owner._corners[cell];
            return std::size_t(tf::topology::cdt::dt3::tet_orient3d_sites(
                                   owner, corners[0], corners[1], corners[2],
                                   corners[3]) == 0);
          }),
      std::plus<>{}, std::size_t(0), tf::checked);
}

} // namespace tf::tetmesh
