/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/curves_buffer.hpp"
#include "../intersect/intersections_within_polygons.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./cut_faces.hpp"
#include "./impl/embedded_self_intersection_curves.hpp"
#include "./return_curves.hpp"
namespace tf {
template <typename Policy>
auto embedded_self_intersection_curves(const tf::polygons<Policy> &_polygons) {
  using Index = std::common_type_t<typename Policy::index_type>;
  tf::intersections_within_polygons<Index, double, 3> iwp;
  iwp.build(tf::make_form(_polygons));
  tf::cut_faces<Index> cf;
  cf.build(_polygons, iwp);
  return tf::cut::embedded_self_intersection_curves<Index>(
      _polygons, tf::make_points(iwp.intersection_points()), cf.descriptors(),
      cf.mapped_loops());
}

template <typename Policy>
auto embedded_self_intersection_curves(const tf::polygons<Policy> &_polygons,
                                       tf::return_curves_t) {
  using Index = std::common_type_t<typename Policy::index_type>;
  tf::intersections_within_polygons<Index, double, 3> iwp;
  iwp.build(tf::make_form(_polygons));
  tf::cut_faces<Index> cf;
  cf.build(_polygons, iwp);
  auto res = tf::cut::embedded_self_intersection_curves<Index>(
      _polygons, tf::make_points(iwp.intersection_points()), cf.descriptors(),
      cf.mapped_loops());
  auto ie = tf::make_mapped_range(cf.intersection_edges(), [](auto e) {
    return std::array<Index, 2>{e[0].id, e[1].id};
  });
  auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
  tf::curves_buffer<Index, tf::coordinate_type<Policy>, 3> cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(iwp.intersection_points().size());
  tf::parallel_copy(iwp.intersection_points(), cb.points());
  return std::make_tuple(std::move(res), std::move(cb));
}
} // namespace tf
