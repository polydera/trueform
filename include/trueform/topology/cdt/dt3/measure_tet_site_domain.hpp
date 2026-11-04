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
#include "../../../core/algorithm/reduce.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/mapped_range.hpp"
#include "../../../exact/vertex.hpp"
#include "./tet_site_domain.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Measure the floors `floors` on the halved lattice: a position's floor
/// keys its insertion order as well as the position itself. At least one
/// lies there.
template <bool Parallel = false, typename Coord, typename Floors>
auto measure_tet_site_domain(const Floors &floors) -> tet_site_domain<Coord> {
  using Domain = tet_site_domain<Coord>;
  const auto domains =
      tf::make_mapped_range(floors, [](const tf::exact::pt3<Coord> &p) {
        Domain domain;
        for (std::size_t axis = 0; axis < 3; ++axis) {
          domain.minimum[axis] = Domain::half(p[axis]);
          domain.span[axis] = 0;
        }
        return domain;
      });
  const auto merge = [](const Domain &a, const Domain &b) {
    return a.merged(b);
  };
  if constexpr (Parallel)
    return tf::reduce(domains, merge, domains[0], tf::checked);
  else {
    auto domain = domains[0];
    for (std::size_t i = 1; i < domains.size(); ++i)
      domain = domain.merged(domains[i]);
    return domain;
  }
}

} // namespace tf::topology::cdt::dt3
