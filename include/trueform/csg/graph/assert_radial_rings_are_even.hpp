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
#include "../../core/config/verify.hpp"
#include "./make_plane_radial_fans.hpp"
#include "./ring_domain_parity_is_odd.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// Around every admitted ring every domain occurs an even number of times
/// (@ref ring_domain_parity_is_odd): a certified ring's sectors consume
/// facing sides in pairs, and a ring its exact configuration withholds is
/// one whose components the certified rings around it already pair.
template <typename Index, typename Labels, typename ExposedOfRow>
auto assert_radial_rings_are_even(const Labels &labels,
                                  const plane_radial_fans<Index> &fans,
                                  const ExposedOfRow &exposed_of_row,
                                  const tf::buffer<char> &valid,
                                  const tf::buffer<Index> &domain_of_side)
    -> void {
  bool even = true;
  for (std::size_t fan = 0; fan < fans.pieces.size() && even; ++fan)
    even =
        !valid[fan] || !ring_domain_parity_is_odd(labels, fans, exposed_of_row,
                                                  domain_of_side, fan);
  tf::core::verify(even,
                   "a domain stands an odd number of times around a ring");
}

} // namespace tf::csg::graph
