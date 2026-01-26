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
#include "./edge_edge.hpp"
#include "./edge_face.hpp"
#include "./vertex_edge.hpp"
#include "./vertex_face.hpp"
#include "./vertex_vertex.hpp"

namespace tf::intersect::generate {
template <typename Handle0, typename Handle1, typename Index, typename T,
          std::size_t Dims>
auto polygon_polygon(const Handle0 &handle0, const Handle1 &handle1,
                     tf::buffer<tagged_intersection<Index>> &intersections,
                     tf::buffer<intersection_id<Index>> &intersection_ids,
                     tf::buffer<tf::point<T, Dims>> &points) {
  generate::vertex_vertex(handle0, handle1, intersections, intersection_ids,
                          points);
  generate::vertex_edge(handle0, handle1, intersections, intersection_ids,
                        points);
  generate::edge_edge(handle0, handle1, intersections, intersection_ids,
                      points);
  generate::vertex_face(handle0, handle1, intersections, intersection_ids,
                        points);
  generate::edge_face(handle0, handle1, intersections, intersection_ids,
                      points);
}
} // namespace tf::intersect::generate
