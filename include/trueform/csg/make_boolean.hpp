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
#include "../arrangement/arrangement_config.hpp"
#include "../arrangement/return_curves.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/buffer.hpp"
#include "../core/resolved_output_real.hpp"
#include "../reindex/return_index_map.hpp"
#include "../reindex/return_source_ids.hpp"
#include "./boolean_op.hpp"
#include "./expression/make_boolean_expr.hpp"
#include "./graph/make_boolean_labels.hpp"
#include "./make_csg_graph.hpp"
#include "./make_csg_mesh.hpp"
#include "./make_intersection_curves.hpp"

namespace tf {

/// @ingroup csg
/// @brief Perform boolean operations on two meshes.
///
/// Computes union, intersection, or difference of two polygon meshes.
/// Uses exact integer predicates for robust classification.
///
/// A boolean is one arrangement of the two operands answering one
/// expression against it (`merge` is `op(0) | op(1)`, and so on), so it
/// is the two-operand case of @ref tf::make_csg_graph +
/// @ref tf::make_csg_mesh. Build the graph directly when you want
/// several results from the same pair, or more than two operands: the
/// arrangement is the expensive part and it is then paid once.
///
/// `sheets` lists the operands (`0`, `1`) that bound no volume and are
/// to act as oriented separators — a fault, a horizon, a cutting plane.
/// A sheet still cuts and is still cut, and its bit means the side of
/// its normal, so `difference` and `intersection` against it yield the
/// two capped halves. This is a declaration of intent, not a property
/// read off the mesh: a closed solid carrying a non-manifold flap is
/// open by inspection and a volume by intent, and only the caller knows
/// which. Operands not listed are volumes, with open fragments fused.
///
/// A return-shape tag is always the LAST argument, after `sheets` and
/// `config`, as everywhere in the library.
///
/// @tparam Policy0 The policy type of the first mesh.
/// @tparam Policy1 The policy type of the second mesh.
/// @param _polygons0 The first mesh @ref tf::polygons (or tagged form).
/// @param _polygons1 The second mesh @ref tf::polygons (or tagged form).
/// @param op The @ref tf::boolean_op to perform.
/// @param sheets Operand ids to treat as sheets.
/// @return Tuple of (@ref tf::polygons_buffer, labels buffer).
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy0, typename Policy1, typename Iter, std::size_t N>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::range<Iter, N> sheets,
                  tf::arrangement_config config = {}) {
  auto graph = tf::make_csg_graph<Int>(_polygons0, _polygons1, sheets, config);
  using InputReal = typename decltype(graph)::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;
  auto [mesh, tag_labels, face_labels] = tf::make_csg_mesh<RealOut>(
      graph, tf::csg::make_boolean_expr(op), tf::return_source_ids);
  auto labels = tf::csg::graph::make_boolean_labels(tag_labels);
  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    return std::make_tuple(std::move(mesh), std::move(labels),
                           std::move(face_labels), graph.converter());
  } else {
    return std::make_tuple(std::move(mesh), std::move(labels),
                           std::move(face_labels));
  }
}

/// @ingroup csg
/// @brief No-sheets overload: both operands are volumes.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::arrangement_config config = {}) {
  return make_boolean<Int, OutputCoordinateType>(_polygons0, _polygons1, op,
                                                 tf::csg::no_sheets(), config);
}

/// @ingroup csg
/// @brief Perform boolean operations with a @ref tf::mesh_arrangement_index_map
///        relating every output point and face back to the two operands.
///
/// The same map @ref tf::make_csg_mesh returns, and what
/// @ref tf::make_stitch_index_map reads to carry the operands' topology
/// and trees onto the result.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1, typename Iter, std::size_t N>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::range<Iter, N> sheets, tf::arrangement_config config,
                  tf::return_index_map_t) {
  auto graph = tf::make_csg_graph<Int>(_polygons0, _polygons1, sheets, config);
  using Index = typename decltype(graph)::index_type;
  using InputReal = typename decltype(graph)::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;
  auto [mesh, imap] = tf::make_csg_mesh<RealOut>(
      graph, tf::csg::make_boolean_expr(op), tf::return_index_map);
  auto labels = tf::csg::graph::make_boolean_labels(imap.face_tag_labels);
  tf::buffer<Index> face_labels;
  face_labels.allocate(imap.face_labels.size());
  tf::parallel_copy(imap.face_labels, face_labels);
  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    return std::make_tuple(std::move(mesh), std::move(labels),
                           std::move(face_labels), std::move(imap),
                           graph.converter());
  } else {
    return std::make_tuple(std::move(mesh), std::move(labels),
                           std::move(face_labels), std::move(imap));
  }
}

/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1, typename Iter, std::size_t N>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::range<Iter, N> sheets, tf::return_index_map_t) {
  return make_boolean<Int, OutputCoordinateType>(
      _polygons0, _polygons1, op, sheets, tf::arrangement_config{},
      tf::return_index_map);
}

/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::arrangement_config config, tf::return_index_map_t) {
  return make_boolean<Int, OutputCoordinateType>(_polygons0, _polygons1, op,
                                                 tf::csg::no_sheets(), config,
                                                 tf::return_index_map);
}

/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_index_map_t) {
  return make_boolean<Int, OutputCoordinateType>(
      _polygons0, _polygons1, op, tf::csg::no_sheets(),
      tf::arrangement_config{}, tf::return_index_map);
}

/// @ingroup csg
/// @brief Perform boolean operations with curve output.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1, typename Iter, std::size_t N>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::range<Iter, N> sheets, tf::arrangement_config config,
                  tf::return_curves_t) {
  auto graph = tf::make_csg_graph<Int>(_polygons0, _polygons1, sheets, config);
  using InputReal = typename decltype(graph)::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;
  auto [mesh, tag_labels, face_labels] = tf::make_csg_mesh<RealOut>(
      graph, tf::csg::make_boolean_expr(op), tf::return_source_ids);
  auto labels = tf::csg::graph::make_boolean_labels(tag_labels);
  auto cb = tf::make_intersection_curves<RealOut>(graph);
  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    return std::make_tuple(std::move(mesh), std::move(labels),
                           std::move(face_labels), std::move(cb),
                           graph.converter());
  } else {
    return std::make_tuple(std::move(mesh), std::move(labels),
                           std::move(face_labels), std::move(cb));
  }
}

/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1, typename Iter, std::size_t N>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::range<Iter, N> sheets, tf::return_curves_t) {
  return make_boolean<Int, OutputCoordinateType>(
      _polygons0, _polygons1, op, sheets, tf::arrangement_config{},
      tf::return_curves);
}

/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::arrangement_config config, tf::return_curves_t) {
  return make_boolean<Int, OutputCoordinateType>(_polygons0, _polygons1, op,
                                                 tf::csg::no_sheets(), config,
                                                 tf::return_curves);
}

/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_curves_t) {
  return make_boolean<Int, OutputCoordinateType>(
      _polygons0, _polygons1, op, tf::csg::no_sheets(),
      tf::arrangement_config{}, tf::return_curves);
}

/// @ingroup csg
/// @brief Perform boolean operations with curves and a
///        @ref tf::mesh_arrangement_index_map.
/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1, typename Iter, std::size_t N>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::range<Iter, N> sheets, tf::arrangement_config config,
                  tf::return_curves_t, tf::return_index_map_t) {
  auto graph = tf::make_csg_graph<Int>(_polygons0, _polygons1, sheets, config);
  using Index = typename decltype(graph)::index_type;
  using InputReal = typename decltype(graph)::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;
  auto [mesh, imap] = tf::make_csg_mesh<RealOut>(
      graph, tf::csg::make_boolean_expr(op), tf::return_index_map);
  auto labels = tf::csg::graph::make_boolean_labels(imap.face_tag_labels);
  tf::buffer<Index> face_labels;
  face_labels.allocate(imap.face_labels.size());
  tf::parallel_copy(imap.face_labels, face_labels);
  auto cb = tf::make_intersection_curves<RealOut>(graph);
  if constexpr (!std::is_integral_v<InputReal> &&
                std::is_integral_v<RealOut>) {
    return std::make_tuple(std::move(mesh), std::move(labels),
                           std::move(face_labels), std::move(cb),
                           std::move(imap), graph.converter());
  } else {
    return std::make_tuple(std::move(mesh), std::move(labels),
                           std::move(face_labels), std::move(cb),
                           std::move(imap));
  }
}

/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1, typename Iter, std::size_t N>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::range<Iter, N> sheets, tf::return_curves_t,
                  tf::return_index_map_t) {
  return make_boolean<Int, OutputCoordinateType>(
      _polygons0, _polygons1, op, sheets, tf::arrangement_config{},
      tf::return_curves, tf::return_index_map);
}

/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::arrangement_config config, tf::return_curves_t,
                  tf::return_index_map_t) {
  return make_boolean<Int, OutputCoordinateType>(
      _polygons0, _polygons1, op, tf::csg::no_sheets(), config,
      tf::return_curves, tf::return_index_map);
}

/// @overload
template <typename Int = tf::none_t,
          typename OutputCoordinateType = tf::none_t, typename Policy0,
          typename Policy1>
auto make_boolean(const tf::polygons<Policy0> &_polygons0,
                  const tf::polygons<Policy1> &_polygons1, tf::boolean_op op,
                  tf::return_curves_t, tf::return_index_map_t) {
  return make_boolean<Int, OutputCoordinateType>(
      _polygons0, _polygons1, op, tf::csg::no_sheets(),
      tf::arrangement_config{}, tf::return_curves, tf::return_index_map);
}

} // namespace tf
