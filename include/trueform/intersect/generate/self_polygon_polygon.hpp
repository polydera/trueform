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
#include "./self_edge_edge.hpp"
#include "./self_edge_face.hpp"
#include "./self_vertex_edge.hpp"
#include "./self_vertex_face.hpp"
#include "./self_vertex_vertex.hpp"

namespace tf::intersect::generate {
template <typename Handle0, typename Handle1, typename Index, typename T,
          std::size_t Dims>
auto self_polygon_polygon(const Handle0 &handle0, const Handle1 &handle1,
                          tf::buffer<intersection<Index>> &intersections,
                          tf::buffer<intersection_id<Index>> &intersection_ids,
                          tf::buffer<tf::point<T, Dims>> &points) {
  generate::self_vertex_vertex(handle0, handle1, intersections,
                               intersection_ids, points);
  generate::self_vertex_edge(handle0, handle1, intersections, intersection_ids,
                             points);
  generate::self_edge_edge(handle0, handle1, intersections, intersection_ids,
                           points);
  generate::self_vertex_face(handle0, handle1, intersections, intersection_ids,
                             points);
  generate::self_edge_face(handle0, handle1, intersections, intersection_ids,
                           points);
}
} // namespace tf::intersect::generate
