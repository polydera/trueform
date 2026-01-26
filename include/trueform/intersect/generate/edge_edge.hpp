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
#include "../../core/buffer.hpp"
#include "../polygon/edge_edge.hpp"
#include "../types/intersection_id.hpp"
#include "../types/tagged_intersection.hpp"

namespace tf::intersect::generate {

template <typename Handle0, typename Handle1, typename Index, typename T,
          std::size_t Dims>
auto edge_edge(const Handle0 &handle0, const Handle1 &handle1,
               tf::buffer<tagged_intersection<Index>> &intersections,
               tf::buffer<intersection_id<Index>> &intersection_ids,
               tf::buffer<tf::point<T, Dims>> &points) {
  tf::intersect::polygon::edge_edge(
      [&](Index e0_0, Index e0_1, Index e1_0, Index e1_1, const auto &pt) {
        Index id = points.size();
        points.push_back(pt);
        intersection_ids.push_back(intersection_id<Index>{}.make_edge_edge(
            handle0.polygon.indices()[e0_0], handle0.polygon.indices()[e0_1],
            handle1.polygon.indices()[e1_0], handle1.polygon.indices()[e1_1],
            id));
        intersections.push_back({Index(0),
                                 Index(handle0.id),
                                 Index(handle1.id),
                                 {e0_0, tf::topo_type::edge},
                                 {e1_0, tf::topo_type::edge},
                                 id});
      },
      handle0, handle1);
}
} // namespace tf::intersect::generate
