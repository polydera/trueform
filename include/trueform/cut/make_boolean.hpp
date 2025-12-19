/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/curves_buffer.hpp"
#include "../intersect/intersections_between_polygons.hpp"
#include "../topology/connect_edges_to_paths.hpp"
#include "./boolean_op.hpp"
#include "./impl/dispatch.hpp"
#include "./impl/make_boolean.hpp"
#include "./return_curves.hpp"
#include "./tagged_cut_faces.hpp"

namespace tf {

template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [op](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(tf::make_form(p0), tf::make_form(p1));
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        return tf::cut::make_boolean<int>(p0, p1, ibp, tcf,
                                          tf::cut::make_boolean_op_spec(op));
      });
}

template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_index_map_t) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [op](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(tf::make_form(p0), tf::make_form(p1));
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        return tf::cut::make_boolean<int>(p0, p1, ibp, tcf,
                                          tf::cut::make_boolean_op_spec(op),
                                          tf::return_index_map);
      });
}

template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_curves_t) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [op](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(tf::make_form(p0), tf::make_form(p1));
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        auto res = tf::cut::make_boolean<int>(
            p0, p1, ibp, tcf, tf::cut::make_boolean_op_spec(op));
        auto ie = tf::make_mapped_range(tcf.intersection_edges(), [](auto e) {
          return std::array<Index, 2>{e[0].id, e[1].id};
        });
        auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
        tf::curves_buffer<Index,
                          tf::coordinate_type<std::decay_t<decltype(p0)>,
                                              std::decay_t<decltype(p1)>>,
                          3>
            cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ibp.intersection_points().size());
        tf::parallel_copy(ibp.intersection_points(), cb.points());
        return std::make_tuple(std::move(res.first), std::move(res.second),
                               std::move(cb));
      });
}

template <typename Policy0, typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_curves_t, tf::return_index_map_t) {
  return cut::impl::boolean_dispatch(
      _polygons0, _polygons1, [op](const auto &p0, const auto &p1) {
        using Index =
            std::common_type_t<typename std::decay_t<decltype(p0)>::index_type,
                               typename std::decay_t<decltype(p1)>::index_type>;
        tf::intersections_between_polygons<Index, double, 3> ibp;
        ibp.build(tf::make_form(p0), tf::make_form(p1));
        tf::tagged_cut_faces<Index> tcf;
        tcf.build(p0, p1, ibp);
        auto [res_mesh, res_labels, res_im] = tf::cut::make_boolean<int>(
            p0, p1, ibp, tcf, tf::cut::make_boolean_op_spec(op),
            tf::return_index_map);
        auto ie = tf::make_mapped_range(tcf.intersection_edges(), [](auto e) {
          return std::array<Index, 2>{e[0].id, e[1].id};
        });
        auto paths = tf::connect_edges_to_paths(tf::make_edges(ie));
        tf::curves_buffer<Index,
                          tf::coordinate_type<std::decay_t<decltype(p0)>,
                                              std::decay_t<decltype(p1)>>,
                          3>
            cb;
        cb.paths_buffer() = std::move(paths);
        cb.points_buffer().allocate(ibp.intersection_points().size());
        tf::parallel_copy(ibp.intersection_points(), cb.points());
        return std::make_tuple(std::move(res_mesh), std::move(res_labels),
                               std::move(cb), std::move(res_im));
      });
}

} // namespace tf
