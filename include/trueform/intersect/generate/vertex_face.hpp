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
#include "../polygon/vertex_face.hpp"
#include "../types/intersection_id.hpp"
#include "../types/tagged_intersection.hpp"

namespace tf::intersect::generate {

template <typename Handle0, typename Handle1, typename Index, typename T,
          std::size_t Dims>
auto vertex_face(const Handle0 &handle0, const Handle1 &handle1,
                 tf::buffer<tagged_intersection<Index>> &intersections,
                 tf::buffer<intersection_id<Index>> &intersection_ids,
                 tf::buffer<tf::point<T, Dims>> &points) {
  tf::intersect::polygon::vertex_face(
      [&](Index sub_v_id, bool ordering) {
        Index id = points.size();
        if (!ordering) {
          intersection_ids.push_back(intersection_id<Index>{}.make_vertex_face(
              handle0.polygon.indices()[sub_v_id], handle1.id, id));
          points.push_back(handle0.polygon[sub_v_id]);
          intersections.push_back({Index(0),
                                   Index(handle0.id),
                                   Index(handle1.id),
                                   {sub_v_id, tf::topo_type::vertex},
                                   {Index(handle1.id), tf::topo_type::face},
                                   id});
        } else {
          intersection_ids.push_back(intersection_id<Index>{}.make_face_vertex(
              handle0.id, handle1.polygon.indices()[sub_v_id], id));
          points.push_back(handle1.polygon[sub_v_id]);
          intersections.push_back({Index(0),
                                   Index(handle0.id),
                                   Index(handle1.id),
                                   {Index(handle0.id), tf::topo_type::face},
                                   {sub_v_id, tf::topo_type::vertex},
                                   id});
        }
      },
      handle0, handle1);
}
} // namespace tf::intersect::generate
