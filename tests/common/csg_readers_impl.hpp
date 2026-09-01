/**
 * @file csg_readers_impl.hpp
 * @brief The read tier's bodies
 *
 * Included ONLY by a combination's builder translation unit.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "csg_readers.hpp"

#include <trueform/core/none.hpp>
#include <trueform/csg/make_boolean.hpp>
#include <trueform/csg/make_csg_domains.hpp>
#include <trueform/csg/make_csg_mesh.hpp>
#include <trueform/csg/make_outer_shell.hpp>

namespace tf::test {

template <typename Graph>
auto csg_mesh_of(const Graph &graph) -> csg_mesh_t<Graph> {
  return tf::make_csg_mesh<tf::none_t>(graph);
}

template <typename Graph>
auto csg_mesh_of(const Graph &graph, const tf::csg::selection_t &selection)
    -> csg_mesh_t<Graph> {
  return tf::make_csg_mesh<tf::none_t>(graph, selection);
}

template <typename Graph>
auto csg_mesh_with_source_ids_of(const Graph &graph)
    -> std::tuple<csg_mesh_t<Graph>, csg_labels_t<Graph>, csg_labels_t<Graph>> {
  return tf::make_csg_mesh<tf::none_t>(graph, tf::return_source_ids);
}

template <typename Graph>
auto csg_mesh_with_source_ids_of(const Graph &graph,
                                 const tf::csg::selection_t &selection)
    -> std::tuple<csg_mesh_t<Graph>, csg_labels_t<Graph>, csg_labels_t<Graph>> {
  return tf::make_csg_mesh<tf::none_t>(graph, selection, tf::return_source_ids);
}

template <typename Graph>
auto csg_mesh_with_index_map_of(const Graph &graph)
    -> std::tuple<csg_mesh_t<Graph>, csg_index_map_t<Graph>> {
  return tf::make_csg_mesh<tf::none_t>(graph, tf::return_index_map);
}

template <typename Graph>
auto csg_mesh_with_index_map_of(const Graph &graph,
                                const tf::csg::selection_t &selection)
    -> std::tuple<csg_mesh_t<Graph>, csg_index_map_t<Graph>> {
  return tf::make_csg_mesh<tf::none_t>(graph, selection, tf::return_index_map);
}

template <typename Graph>
auto csg_domains_of(const Graph &graph) -> csg_domains_t<Graph> {
  return tf::make_csg_domains<tf::none_t>(graph);
}

template <typename Graph>
auto csg_domains_of(const Graph &graph, tf::domain_config config)
    -> csg_domains_t<Graph> {
  return tf::make_csg_domains<tf::none_t>(graph, config);
}

template <typename Graph>
auto csg_domains_of(const Graph &graph, const tf::csg::selection_t &selection)
    -> csg_domains_t<Graph> {
  return tf::make_csg_domains<tf::none_t>(graph, selection);
}

template <typename Graph>
auto csg_domains_of(const Graph &graph, const tf::csg::selection_t &selection,
                    tf::domain_config config) -> csg_domains_t<Graph> {
  return tf::make_csg_domains<tf::none_t>(graph, selection, config);
}

template <typename Form0, typename Form1>
auto boolean_of(const Form0 &form0, const Form1 &form1, tf::boolean_op op)
    -> boolean_result_t<Form0, Form1> {
  return tf::make_boolean<tf::none_t>(form0, form1, op);
}

template <typename Form0, typename Form1>
auto boolean_with_index_map_of(const Form0 &form0, const Form1 &form1,
                               tf::boolean_op op)
    -> boolean_index_map_result_t<Form0, Form1> {
  return tf::make_boolean<tf::none_t>(form0, form1, op, tf::return_index_map);
}

template <typename Graph>
auto outer_shell_of(const Graph &graph) -> csg_mesh_t<Graph> {
  return tf::make_outer_shell<tf::none_t>(graph);
}

} // namespace tf::test
