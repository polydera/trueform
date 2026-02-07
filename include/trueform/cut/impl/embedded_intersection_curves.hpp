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
#include "../../intersect/types/tagged_intersections.hpp"
#include "../tagged_cut_faces.hpp"
#include "./make_boolean_common.hpp"
#include "./make_full_arrangement_ids.hpp"

namespace tf::cut {

template <typename Index, typename Policy0, typename RealT,
          std::size_t Dims>
auto embedded_intersection_curves(
    const tf::polygons<Policy0> &polygons0,
    const tf::intersect::tagged_intersections<Index, RealT, Dims> &ibp,
    const tf::tagged_cut_faces<Index> &tcf) {
  auto pai = make_full_arrangement_ids<Index>(
      polygons0.size(), tcf.mapped_loops0().size(), tcf.descriptors0());

  return make_boolean_common(polygons0,
                             tf::make_points(ibp.intersection_points()), pai,
                             tcf.descriptors0(), tcf.mapped_loops0(), 0,
                             tf::direction::forward);
}

} // namespace tf::cut
