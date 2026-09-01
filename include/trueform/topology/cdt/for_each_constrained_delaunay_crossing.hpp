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
#include "../../core/range.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner, typename F>
auto for_each_constrained_delaunay_crossing(const Owner &owner, F &&function)
    -> void {
  using Index = typename Owner::index_type;
  for (const auto &record : owner._incremental_crossings) {
    const std::array<Index, 2> parents{record.id_a, record.id_b};
    const std::size_t count = record.id_b == Owner::none ? 1 : 2;
    function(record.point,
             tf::make_range(parents.data(), parents.data() + count));
  }
}

} // namespace tf::topology::cdt
