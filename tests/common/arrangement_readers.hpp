/**
 * @file arrangement_readers.hpp
 * @brief The suite's compiled read tier over a built arrangement graph
 *
 * Declared with explicit structural return types, spelled from the graph
 * type's own member typedefs, so a test TU compiles a call and emits no
 * reader.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#pragma once

#include "arrangement_builders.hpp"

#include <trueform/core/buffer.hpp>
#include <trueform/core/curves_buffer.hpp>
#include <trueform/core/polygons_buffer.hpp>

#include <cstddef>
#include <tuple>

namespace tf::test {

/// The mesh a read of `Graph` emits. An arrangement keeps every uncut face
/// as it stood and adds triangles for the cut ones, so an all-triangle
/// input stays blocked<3> and any other arity gives jagged faces.
template <typename Graph>
using arrangement_mesh_t = tf::polygons_buffer<
    typename Graph::index_type, typename Graph::input_real_type, 3,
    (Graph::face_static_size == 3 ? std::size_t(3) : tf::dynamic_size)>;

template <typename Graph>
using arrangement_curves_t =
    tf::curves_buffer<typename Graph::index_type,
                      typename Graph::input_real_type, 3>;

/// The whole arrangement materialised.
template <typename Graph>
auto arrangement_mesh_of(const Graph &graph) -> arrangement_mesh_t<Graph>;

/// The seam polylines read off the arrangement.
template <typename Graph>
auto arrangement_curves_of(const Graph &graph) -> arrangement_curves_t<Graph>;

template <typename Graph>
using arrangement_labels_t = tf::buffer<typename Graph::index_type>;

/// The N-mesh decomposition, compiled once per operand combination.
template <typename Forms>
using mesh_arrangements_result_t =
    std::tuple<arrangement_mesh_t<range_arrangement_t<Forms>>,
               arrangement_labels_t<range_arrangement_t<Forms>>,
               arrangement_labels_t<range_arrangement_t<Forms>>>;

template <typename Forms>
using mesh_arrangements_curves_result_t =
    std::tuple<arrangement_mesh_t<range_arrangement_t<Forms>>,
               arrangement_labels_t<range_arrangement_t<Forms>>,
               arrangement_labels_t<range_arrangement_t<Forms>>,
               arrangement_curves_t<range_arrangement_t<Forms>>>;

template <typename Forms>
auto mesh_arrangements_of(Forms forms, tf::arrangement_config config)
    -> mesh_arrangements_result_t<Forms>;

template <typename Forms>
auto mesh_arrangements_with_curves_of(Forms forms,
                                      tf::arrangement_config config)
    -> mesh_arrangements_curves_result_t<Forms>;

/// One mesh split at its self-intersection curves.
template <typename Form>
using polygon_arrangements_result_t =
    std::tuple<arrangement_mesh_t<self_arrangement_t<Form>>,
               arrangement_labels_t<self_arrangement_t<Form>>>;

template <typename Form>
using polygon_arrangements_curves_result_t =
    std::tuple<arrangement_mesh_t<self_arrangement_t<Form>>,
               arrangement_labels_t<self_arrangement_t<Form>>,
               arrangement_curves_t<self_arrangement_t<Form>>>;

template <typename Form>
auto polygon_arrangements_of(const Form &form, tf::arrangement_config config)
    -> polygon_arrangements_result_t<Form>;

template <typename Form>
auto polygon_arrangements_with_curves_of(const Form &form,
                                         tf::arrangement_config config)
    -> polygon_arrangements_curves_result_t<Form>;

} // namespace tf::test
