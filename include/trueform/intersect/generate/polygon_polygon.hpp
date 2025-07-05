/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
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
                     tf::buffer<intersection<Index>> &intersections,
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
