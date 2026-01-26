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
#include "../polygon/self_vertex_vertex.hpp"
#include "../types/intersection.hpp"
#include "../types/intersection_id.hpp"

namespace tf::intersect::generate {
template <typename Handle0, typename Handle1, typename Index, typename T,
          std::size_t Dims>
auto self_vertex_vertex(const Handle0 &handle0, const Handle1 &handle1,
                        tf::buffer<intersection<Index>> &intersections,
                        tf::buffer<intersection_id<Index>> &intersection_ids,
                        tf::buffer<tf::point<T, Dims>> &points) {
  tf::intersect::polygon::self_vertex_vertex(
      [&](Index sub_id0, Index sub_id1) {
        Index id = points.size();
        points.push_back(handle0.polygon[sub_id0]);
        intersection_ids.push_back(intersection_id<Index>{}.make_vertex_vertex(
            handle0.polygon.indices()[sub_id0],
            handle1.polygon.indices()[sub_id1], id));
        intersections.push_back(make_canonical_intersection<Index>(
            Index(handle0.id), Index(handle1.id),
            {sub_id0, tf::topo_type::vertex}, {sub_id1, tf::topo_type::vertex},
            id));
      },
      handle0, handle1);
}
} // namespace tf::intersect::generate
