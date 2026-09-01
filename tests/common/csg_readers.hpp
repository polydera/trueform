/**
 * @file csg_readers.hpp
 * @brief The suite's compiled read tier over a built csg graph
 *
 * Declared with explicit structural return types, spelled from the graph
 * type's own member typedefs, so a test TU compiles a call and emits no
 * reader.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "csg_builders.hpp"

#include <trueform/arrangement/mesh_arrangement_index_map.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/memory.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/coordinate_type.hpp>
#include <trueform/csg/boolean_op.hpp>
#include <trueform/csg/expression.hpp>
#include <trueform/csg/graph/chosen_sides_for.hpp>
#include <trueform/topology/domain_config.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <utility>

namespace tf::test {

/// The mesh a read of `Graph` emits: the graph's own index and input real,
/// at the graph's own face arity.
template <typename Graph>
using csg_mesh_t = tf::polygons_buffer<typename Graph::index_type,
                                       typename Graph::input_real_type, 3,
                                       Graph::face_static_size>;

template <typename Graph>
using csg_labels_t = tf::buffer<typename Graph::index_type>;

template <typename Graph>
using csg_index_map_t =
    tf::mesh_arrangement_index_map<typename Graph::index_type>;

/// A domain cell's arity follows the input: an all-triangle input keeps the
/// static blocked<3>, any other arity gives each cell a dynamic face buffer.
template <typename Graph>
using csg_domain_mesh_t = tf::polygons_buffer<
    typename Graph::index_type, typename Graph::input_real_type, 3,
    (Graph::face_static_size == 3 ? std::size_t(3) : tf::dynamic_size)>;

template <typename Graph>
using csg_domains_t = std::pair<tf::core::std_vector<csg_domain_mesh_t<Graph>>,
                                csg_labels_t<Graph>>;

/// The whole arrangement mesh, no selection.
template <typename Graph>
auto csg_mesh_of(const Graph &graph) -> csg_mesh_t<Graph>;

/// The result mesh of a selection.
template <typename Graph>
auto csg_mesh_of(const Graph &graph, const tf::csg::selection_t &selection)
    -> csg_mesh_t<Graph>;

/// The whole arrangement mesh plus per output face its provenance.
template <typename Graph>
auto csg_mesh_with_source_ids_of(const Graph &graph)
    -> std::tuple<csg_mesh_t<Graph>, csg_labels_t<Graph>, csg_labels_t<Graph>>;

/// The result mesh plus per output face its provenance.
template <typename Graph>
auto csg_mesh_with_source_ids_of(const Graph &graph,
                                 const tf::csg::selection_t &selection)
    -> std::tuple<csg_mesh_t<Graph>, csg_labels_t<Graph>, csg_labels_t<Graph>>;

/// The whole arrangement mesh plus the point and face map.
template <typename Graph>
auto csg_mesh_with_index_map_of(const Graph &graph)
    -> std::tuple<csg_mesh_t<Graph>, csg_index_map_t<Graph>>;

/// The result mesh plus the point and face map back to the operands.
template <typename Graph>
auto csg_mesh_with_index_map_of(const Graph &graph,
                                const tf::csg::selection_t &selection)
    -> std::tuple<csg_mesh_t<Graph>, csg_index_map_t<Graph>>;

/// One watertight mesh per kept domain.
template <typename Graph>
auto csg_domains_of(const Graph &graph) -> csg_domains_t<Graph>;

template <typename Graph>
auto csg_domains_of(const Graph &graph, tf::domain_config config)
    -> csg_domains_t<Graph>;

template <typename Graph>
auto csg_domains_of(const Graph &graph, const tf::csg::selection_t &selection)
    -> csg_domains_t<Graph>;

template <typename Graph>
auto csg_domains_of(const Graph &graph, const tf::csg::selection_t &selection,
                    tf::domain_config config) -> csg_domains_t<Graph>;

/// The index a form names its vertices with.
template <typename Form>
using form_index_t =
    std::decay_t<decltype(std::declval<const Form &>().faces()[0][0])>;

/// The public pairwise wrapper, compiled once per operand combination. The
/// mesh is the pair graph's own, so the arity is the pair's, not either
/// operand's.
template <typename Form0, typename Form1>
using boolean_result_t =
    std::tuple<csg_mesh_t<pair_csg_graph_t<Form0, Form1>>,
               tf::buffer<std::int8_t>, tf::buffer<form_index_t<Form0>>>;

template <typename Form0, typename Form1>
auto boolean_of(const Form0 &form0, const Form1 &form1, tf::boolean_op op)
    -> boolean_result_t<Form0, Form1>;

template <typename Form0, typename Form1>
using boolean_index_map_result_t =
    std::tuple<csg_mesh_t<pair_csg_graph_t<Form0, Form1>>,
               tf::buffer<std::int8_t>, tf::buffer<form_index_t<Form0>>,
               csg_index_map_t<pair_csg_graph_t<Form0, Form1>>>;

template <typename Form0, typename Form1>
auto boolean_with_index_map_of(const Form0 &form0, const Form1 &form1,
                               tf::boolean_op op)
    -> boolean_index_map_result_t<Form0, Form1>;

/// The boundary between the unbounded universe and everything the forms
/// enclose.
template <typename Graph>
auto outer_shell_of(const Graph &graph) -> csg_mesh_t<Graph>;

} // namespace tf::test
