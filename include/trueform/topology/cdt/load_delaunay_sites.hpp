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
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner, typename Points, typename Converter>
auto load_delaunay_sites(Owner &owner, const Points &points,
                         const Converter &converter) -> void {
  using Index = typename Owner::index_type;
  owner._sites.allocate(points.size());
  for (std::size_t i = 0; i < points.size(); ++i) {
    const auto point = converter(points[i]);
    owner._sites[i] = {point[0], point[1], static_cast<Index>(i), Index(0)};
  }
}

} // namespace tf::topology::cdt
