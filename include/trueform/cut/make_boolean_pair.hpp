/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/curves_buffer.hpp"
#include "../intersect/intersections_between_polygons.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./boolean_op.hpp"
#include "./impl/make_boolean_pair.hpp"
#include "./return_curves.hpp"
#include "./tagged_cut_faces.hpp"
namespace tf {
template <typename Policy0, typename Policy1>
auto make_boolean_pair(const tf::polygons<Policy0> _polygons0,
                       const tf::polygons<Policy1> &_polygons1,
                       tf::boolean_op op) {
  using Index =
      std::common_type_t<typename Policy0::index_type, typename Policy1::index_type>;
  tf::intersections_between_polygons<Index, double, 3> ibp;
  ibp.build(tf::make_form(_polygons0), tf::make_form(_polygons1));
  tf::tagged_cut_faces<Index> tcf;
  tcf.build(_polygons0, _polygons1, ibp);
  return tf::cut::make_boolean_pair<int>(_polygons0, _polygons1, ibp, tcf,
                                         tf::cut::make_boolean_op_spec(op));
}

template <typename Policy0, typename Policy1>
auto make_boolean_pair(const tf::polygons<Policy0> _polygons0,
                       const tf::polygons<Policy1> &_polygons1,
                       tf::boolean_op op, tf::return_curves_t) {
  using Index =
      std::common_type_t<typename Policy0::index_type, typename Policy1::index_type>;
  tf::intersections_between_polygons<Index, double, 3> ibp;
  ibp.build(tf::make_form(_polygons0), tf::make_form(_polygons1));
  tf::tagged_cut_faces<Index> tcf;
  tcf.build(_polygons0, _polygons1, ibp);
  auto res = tf::cut::make_boolean_pair<int>(_polygons0, _polygons1, ibp, tcf,
                                             tf::cut::make_boolean_op_spec(op));
  auto ie = tf::make_mapped_range(tcf.intersection_edges(), [](auto e) {
    return std::array<Index, 2>{e[0].id, e[1].id};
  });
  auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
  tf::curves_buffer<Index, tf::coordinate_type<Policy0, Policy1>, 3> cb;
  cb.paths_buffer() = std::move(paths);
  cb.points_buffer().allocate(ibp.intersection_points().size());
  tf::parallel_copy(ibp.intersection_points(), cb.points());
  return std::make_tuple(std::move(res.first), std::move(res.second),
                         std::move(cb));
}
} // namespace tf
