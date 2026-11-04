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
#include "../../../../exact/vertex.hpp"
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// `pt` read on the projection `axes` names.
template <typename Coord>
auto tet_projected_point(const tf::exact::pt3<Coord> &pt,
                         const std::pair<int, int> &axes)
    -> tf::exact::pt2<Coord> {
  return {pt[std::size_t(axes.first)], pt[std::size_t(axes.second)]};
}

} // namespace tf::topology::cdt::dt3
