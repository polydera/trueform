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

#include "../../intersect/intersections_within_polygons.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../cut_graph.hpp"
#include "../face_cuts.hpp"

namespace tf::cut::dispatch {

template <typename Index, typename RealType, typename Int, typename Policy>
auto build_self_pipeline(const tf::polygons<Policy> &p) {
  tf::intersections_within_polygons<Index, RealType, Int> iwp;
  iwp.build(p, tf::intersect_mode::primitives);

  auto &conv = iwp.converter();
  auto apply_to_face = [&](int, Index object, const auto &f) {
    f(p.faces()[object]);
  };
  auto get_mesh_point = [&](int, Index id) -> tf::point<Int, 3> {
    return conv.convert(p.points()[id]);
  };

  tf::intersection_graph<Index, Int> ig;
  ig.build(iwp, apply_to_face, get_mesh_point);

  tf::face_cuts<Index, Int> fc;
  fc.build(ig, apply_to_face, get_mesh_point);

  tf::cut_graph<Index> cg;
  cg.build(fc, ig, p);

  return std::make_tuple(std::move(iwp), std::move(ig), std::move(fc),
                         std::move(cg));
}

} // namespace tf::cut::dispatch
