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

#include "../../core/frame_of.hpp"
#include "../../core/transformed.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../../intersect/intersections_between_polygons.hpp"
#include "../cut_graph.hpp"
#include "../face_cuts.hpp"

namespace tf::cut::dispatch {

template <typename Index, typename RealType, typename Int, typename Policy0,
          typename Policy1>
auto build_exact_pipeline(const tf::polygons<Policy0> &p0,
                          const tf::polygons<Policy1> &p1,
                          tf::intersect_mode mode) {
  tf::intersections_between_polygons<Index, RealType, Int> ibp;
  ibp.build(p0, p1, mode);

  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    if (tag == 0)
      f(p0.faces()[object]);
    else
      f(p1.faces()[object]);
  };
  auto get_mesh_point = [&](int tag, Index id) -> tf::point<Int, 3> {
    if (tag == 0)
      return conv.convert(tf::transformed(p0.points()[id], tf::frame_of(p0)));
    else
      return conv.convert(tf::transformed(p1.points()[id], tf::frame_of(p1)));
  };

  tf::intersection_graph<Index, Int> ig;
  ig.build(ibp, apply_to_face, get_mesh_point, mode);

  tf::face_cuts<Index, Int> fc;
  fc.build(ig, apply_to_face, get_mesh_point);

  tf::cut_graph<Index> cg;
  cg.build(fc, static_cast<Index>(ig.points().size()));

  return std::make_tuple(std::move(ibp), std::move(ig), std::move(fc),
                         std::move(cg));
}

} // namespace tf::cut::dispatch
