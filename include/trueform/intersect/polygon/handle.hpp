/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf::intersect::polygon {

template <typename Range0, typename Range1> struct representation {
  Range0 vertex;
  Range1 edge;
};

template <typename Polygon, typename Index, typename Range0, typename Range1>
struct handle {
  Polygon polygon;
  Index id;
  polygon::representation<Range0, Range1> representation;
};

template <typename Polygon, typename Index, typename Range0, typename Range1>
auto make_handle(const Polygon &poly, Index id,
                 const Range0 &vertex_representation,
                 const Range1 &edge_representation) {
  return handle<Polygon, Index, Range0, Range1>{
      poly, id, {vertex_representation, edge_representation}};
}
} // namespace tf::intersect::polygon
