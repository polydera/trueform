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
#include "../regions/tet_regions.hpp"
#include <cstddef>
#include <functional>

namespace tf::tetmesh {

/// The cells `regions` winds negatively: a closed surface wound inward
/// encloses only these.
template <typename Index>
auto count_negative_tet_cells(const tet_regions<Index> &regions)
    -> std::size_t {
  return tf::reduce(tf::make_mapped_range(
                        regions.winding,
                        [](int winding) { return std::size_t(winding < 0); }),
                    std::plus<>{}, std::size_t(0), tf::checked);
}

} // namespace tf::tetmesh
