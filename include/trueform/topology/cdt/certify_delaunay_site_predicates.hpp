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
#include "../../exact/meta.hpp"
#include "./delaunay_site_domain.hpp"

namespace tf::topology::cdt {

template <typename Owner, typename Int>
auto certify_delaunay_site_predicates(Owner &owner,
                                      const delaunay_site_domain<Int> &domain)
    -> void {
  using Wide = typename tf::exact::meta<Int>::unsigned_T1;
  const auto limit = (Wide(1) << tf::exact::meta<Int>::t1_bits) - Wide(1);
  const auto span_x = Wide(domain.span_x);
  const auto span_y = Wide(domain.span_y);

  owner._orientation_fits_t1 =
      span_x == Wide(0) || span_y <= limit / (Wide(2) * span_x);

  if (span_x != Wide(0) && span_x > limit / span_x) {
    owner._incircle_inner_fits_t1 = false;
    return;
  }
  const auto x_squared = span_x * span_x;
  owner._incircle_inner_fits_t1 =
      span_y == Wide(0) || span_y <= (limit - x_squared) / span_y;
}

} // namespace tf::topology::cdt
