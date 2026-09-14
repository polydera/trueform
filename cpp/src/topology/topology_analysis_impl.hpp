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

#include "../core/carrier_arrays.hpp"
#include "../core/materialized_mesh.hpp"

#include "trueform/cpp/core/detail/non_deduced.hpp"
#include "trueform/cpp/topology/cell_membership.hpp"
#include "trueform/cpp/topology/euler_characteristic.hpp"
#include "trueform/cpp/topology/face_link.hpp"
#include "trueform/cpp/topology/is_closed.hpp"
#include "trueform/cpp/topology/is_manifold.hpp"
#include "trueform/cpp/topology/k_rings.hpp"
#include "trueform/cpp/topology/manifold_edge_link.hpp"
#include "trueform/cpp/topology/neighborhoods.hpp"
#include "trueform/cpp/topology/non_manifold_edges.hpp"
#include "trueform/cpp/topology/orient_faces_consistently.hpp"
#include "trueform/cpp/topology/vertex_link.hpp"

#include "trueform/core/algorithm/parallel_transform.hpp"
#include "trueform/core/policy/frame.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/topology.hpp"
#include "trueform/topology/policy/manifold_edge_link.hpp"
#include "trueform/topology/policy/vertex_link.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace tf::cpp {
namespace detail {

template <typename Index>
auto require_id_count(Index n_ids, const char *operation) -> std::size_t {
  if (n_ids < Index{0})
    throw std::invalid_argument(std::string(operation) +
                                ": n_ids must be nonnegative");
  if (n_ids >= static_cast<Index>(std::numeric_limits<int>::max()))
    throw std::length_error(std::string(operation) +
                            ": n_ids exceeds the supported range");
  return static_cast<std::size_t>(n_ids);
}

template <typename Index>
auto require_fixed_cells(const nd_array<Index> &cells, const char *operation,
                         int required_arity = 0) -> int {
  if (!cells.is_valid() || cells.ndim() != 2)
    throw std::invalid_argument(std::string(operation) +
                                ": cells must be a two-dimensional array");
  const auto arity = cells.shape_at(1);
  if ((required_arity != 0 && arity != required_arity) ||
      (required_arity == 0 && arity != 2 && arity != 3))
    throw std::invalid_argument(std::string(operation) +
                                ": unsupported fixed cell arity");
  return arity;
}

template <typename Index, typename Range>
auto require_cell_domain(const Range &indices, std::size_t n_ids,
                         const char *operation) -> void {
  const auto limit = static_cast<Index>(n_ids);
  for (const auto point_id : indices)
    if (point_id < Index{0} || point_id >= limit)
      throw std::out_of_range(std::string(operation) +
                              ": cell point index out of range");
}

template <typename Index, typename Faces>
auto require_link_inputs(
    const Faces &faces,
    const offset_blocked_buffer<Index, Index> &face_membership,
    const char *operation) -> tf::face_membership<Index> {
  const auto point_count = carrier::require_offset_blocks(
      face_membership, carrier::any_count, static_cast<int>(faces.size()),
      "cell membership");
  for (const auto face : faces)
    require_cell_domain<Index>(face, point_count, operation);

  const auto membership_data = face_membership.data();
  std::size_t reference_count = 0;
  for (const auto face : faces)
    reference_count += face.size();
  if (membership_data.length() != reference_count)
    throw std::invalid_argument(std::string(operation) +
                                ": cell membership does not match cells");

  tf::face_membership<Index> expected_membership;
  expected_membership.build(faces, point_count, reference_count);
  const auto supplied_membership = face_membership.make_range();

  tf::buffer<std::size_t> generations;
  tf::buffer<std::size_t> balances;
  generations.allocate(faces.size());
  balances.allocate(faces.size());
  for (auto &generation : generations)
    generation = 0;

  for (std::size_t point_id = 0; point_id < point_count; ++point_id) {
    const auto expected = expected_membership[point_id];
    const auto supplied = supplied_membership[point_id];
    if (expected.size() != supplied.size())
      throw std::invalid_argument(std::string(operation) +
                                  ": cell membership does not match cells");

    const auto generation = point_id + 1;
    for (const auto face_id : expected) {
      const auto id = static_cast<std::size_t>(face_id);
      if (generations[id] != generation) {
        generations[id] = generation;
        balances[id] = 0;
      }
      ++balances[id];
    }
    for (const auto face_id : supplied) {
      const auto id = static_cast<std::size_t>(face_id);
      if (generations[id] != generation || balances[id] == 0)
        throw std::invalid_argument(std::string(operation) +
                                    ": cell membership does not match cells");
      --balances[id];
    }
    for (const auto face_id : expected)
      if (balances[static_cast<std::size_t>(face_id)] != 0)
        throw std::invalid_argument(std::string(operation) +
                                    ": cell membership does not match cells");
  }
  // Keep the supplied blocks untouched. Order-sensitive callers use them;
  // sorted-intersection link builders use this incidence-built carrier.
  return expected_membership;
}

template <typename Index>
auto require_fixed_link_faces(const nd_array<Index> &faces,
                              const char *operation) -> void {
  static_cast<void>(require_fixed_cells(faces, operation, 3));
}

template <typename Index>
auto require_dynamic_link_faces(
    const offset_blocked_buffer<Index, Index> &faces) -> void {
  // what a corner may name is the membership's count, and the two are asked
  // for each other once they are both here
  static_cast<void>(carrier::require_offset_blocks(faces, carrier::any_count,
                                                   carrier::any_count,
                                                   "dynamic faces", Index{3}));
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto is_closed(const mesh<Index, Real, Dims, Ngon> &value) -> bool {
  return tf::is_closed(value.faces(), value.face_membership());
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto is_open(const mesh<Index, Real, Dims, Ngon> &value) -> bool {
  return !cpp::is_closed(value);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto is_manifold(const mesh<Index, Real, Dims, Ngon> &value) -> bool {
  return tf::is_manifold(value.faces(), value.face_membership());
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto is_non_manifold(const mesh<Index, Real, Dims, Ngon> &value) -> bool {
  return !cpp::is_manifold(value);
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto euler_characteristic(const mesh<Index, Real, Dims, Ngon> &value)
    -> std::int32_t {
  return static_cast<std::int32_t>(tf::euler_characteristic(
      value.polygons() | tf::tag(value.manifold_edge_link())));
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto non_manifold_edges(const mesh<Index, Real, Dims, Ngon> &value)
    -> nd_array<Index> {
  auto edges =
      tf::make_non_manifold_edges(value.faces(), value.face_membership());
  const auto count = static_cast<int>(edges.size());
  return nd_array<Index>::from_buffer(std::move(edges.data_buffer()),
                                      {count, 2});
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto k_rings(const mesh<Index, Real, Dims, Ngon> &value, std::int32_t k,
             bool inclusive) -> offset_blocked_buffer<Index, Index> {
  if (k <= 0)
    throw std::invalid_argument("k_rings: k must be positive");
  return offset_blocked_buffer<Index, Index>::from_buffer(tf::make_k_rings(
      value.vertex_link(), static_cast<std::size_t>(k), inclusive));
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighborhoods(const mesh<Index, Real, Dims, Ngon> &value, Real radius,
                   bool inclusive) -> offset_blocked_buffer<Index, Index> {
  if (std::isnan(radius) || radius <= Real{0})
    throw std::invalid_argument(
        "neighborhoods: radius must be positive and not NaN");
  auto points =
      value.points() | tf::tag(value.vertex_link()) | tf::tag(value.frame());
  return offset_blocked_buffer<Index, Index>::from_buffer(
      tf::make_neighborhoods(points, radius, inclusive));
}

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto cell_membership(const nd_array<Index> &cells_array,
                     detail::non_deduced_t<Index> n_ids)
    -> offset_blocked_buffer<Index, Index> {
  const auto count = detail::require_id_count(n_ids, "cell_membership");
  const auto arity =
      detail::require_fixed_cells(cells_array, "cell_membership");
  detail::require_cell_domain<Index>(cells_array.make_range(), count,
                                     "cell_membership");

  if (arity == 2) {
    auto cells =
        tf::make_edges(tf::make_blocked_range<2>(cells_array.make_range()));
    tf::edge_membership<Index> membership;
    membership.build(cells, count);
    return offset_blocked_buffer<Index, Index>::from_buffer(std::move(
        static_cast<tf::offset_block_buffer<Index, Index> &>(membership)));
  }

  auto cells = tf::make_faces<3>(cells_array.make_range());
  tf::face_membership<Index> membership;
  membership.build(cells, count, cells_array.length());
  return offset_blocked_buffer<Index, Index>::from_buffer(std::move(
      static_cast<tf::offset_block_buffer<Index, Index> &>(membership)));
}

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto cell_membership(const offset_blocked_buffer<Index, Index> &cells_array,
                     detail::non_deduced_t<Index> n_ids)
    -> offset_blocked_buffer<Index, Index> {
  const auto count = detail::require_id_count(n_ids, "cell_membership");
  static_cast<void>(carrier::require_offset_blocks(
      cells_array, carrier::any_count, static_cast<int>(count), "dynamic cells",
      Index{3}));
  const auto data = cells_array.data();

  auto cells = tf::make_faces(cells_array.make_range());
  tf::face_membership<Index> membership;
  membership.build(cells, count, data.length());
  return offset_blocked_buffer<Index, Index>::from_buffer(std::move(
      static_cast<tf::offset_block_buffer<Index, Index> &>(membership)));
}

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto vertex_link_edges(const nd_array<Index> &edges_array,
                       detail::non_deduced_t<Index> n_ids)
    -> offset_blocked_buffer<Index, Index> {
  const auto count = detail::require_id_count(n_ids, "vertex_link_edges");
  static_cast<void>(
      detail::require_fixed_cells(edges_array, "vertex_link_edges", 2));
  detail::require_cell_domain<Index>(edges_array.make_range(), count,
                                     "vertex_link_edges");

  auto edges =
      tf::make_edges(tf::make_blocked_range<2>(edges_array.make_range()));
  tf::vertex_link<Index> links;
  links.build(edges, count);
  return offset_blocked_buffer<Index, Index>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<Index, Index> &>(links)));
}

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto vertex_link_faces(
    const nd_array<Index> &faces_array,
    const offset_blocked_buffer<Index, Index> &cell_membership)
    -> offset_blocked_buffer<Index, Index> {
  detail::require_fixed_link_faces(faces_array, "vertex_link_faces");
  auto faces = tf::make_faces<3>(faces_array.make_range());
  static_cast<void>(detail::require_link_inputs<Index>(faces, cell_membership,
                                                       "vertex_link_faces"));
  auto membership = tf::make_face_membership_like(cell_membership.make_range());
  tf::vertex_link<Index> links;
  links.build(faces, membership);
  return offset_blocked_buffer<Index, Index>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<Index, Index> &>(links)));
}

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto vertex_link_faces(
    const offset_blocked_buffer<Index, Index> &faces_array,
    const offset_blocked_buffer<Index, Index> &cell_membership)
    -> offset_blocked_buffer<Index, Index> {
  detail::require_dynamic_link_faces(faces_array);
  auto faces = tf::make_faces(faces_array.make_range());
  static_cast<void>(detail::require_link_inputs<Index>(faces, cell_membership,
                                                       "vertex_link_faces"));
  auto membership = tf::make_face_membership_like(cell_membership.make_range());
  tf::vertex_link<Index> links;
  links.build(faces, membership);
  return offset_blocked_buffer<Index, Index>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<Index, Index> &>(links)));
}

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto manifold_edge_link(
    const nd_array<Index> &faces_array,
    const offset_blocked_buffer<Index, Index> &face_membership)
    -> nd_array<Index> {
  detail::require_fixed_link_faces(faces_array, "manifold_edge_link");
  auto faces = tf::make_faces<3>(faces_array.make_range());
  auto membership = detail::require_link_inputs<Index>(faces, face_membership,
                                                       "manifold_edge_link");
  tf::manifold_edge_link<Index, 3> links;
  links.build(faces, membership);

  tf::buffer<Index> data;
  data.allocate(links.data_buffer().size());
  tf::parallel_transform(
      links.data_buffer(), data,
      [](const auto &peer) { return peer.face_peer; }, tf::checked);
  return nd_array<Index>::from_buffer(std::move(data),
                                      {faces_array.shape_at(0), 3});
}

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto manifold_edge_link(
    const offset_blocked_buffer<Index, Index> &faces_array,
    const offset_blocked_buffer<Index, Index> &face_membership)
    -> offset_blocked_buffer<Index, Index> {
  detail::require_dynamic_link_faces(faces_array);
  auto faces = tf::make_faces(faces_array.make_range());
  auto membership = detail::require_link_inputs<Index>(faces, face_membership,
                                                       "manifold_edge_link");
  tf::manifold_edge_link<Index, tf::dynamic_size> links;
  links.build(faces, membership);

  tf::offset_block_buffer<Index, Index> output;
  output.offsets_buffer() = std::move(links.offsets_buffer());
  output.data_buffer().allocate(links.data_buffer().size());
  tf::parallel_transform(
      links.data_buffer(), output.data_buffer(),
      [](const auto &peer) { return peer.face_peer; }, tf::checked);
  return offset_blocked_buffer<Index, Index>::from_buffer(std::move(output));
}

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto face_link(const nd_array<Index> &faces_array,
               const offset_blocked_buffer<Index, Index> &face_membership)
    -> offset_blocked_buffer<Index, Index> {
  detail::require_fixed_link_faces(faces_array, "face_link");
  auto faces = tf::make_faces<3>(faces_array.make_range());
  auto membership =
      detail::require_link_inputs<Index>(faces, face_membership, "face_link");
  tf::face_link<Index> links;
  links.build(faces, membership);
  return offset_blocked_buffer<Index, Index>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<Index, Index> &>(links)));
}

template <typename Index, std::enable_if_t<is_supported_index_v<Index>, int>>
auto face_link(const offset_blocked_buffer<Index, Index> &faces_array,
               const offset_blocked_buffer<Index, Index> &face_membership)
    -> offset_blocked_buffer<Index, Index> {
  detail::require_dynamic_link_faces(faces_array);
  auto faces = tf::make_faces(faces_array.make_range());
  auto membership =
      detail::require_link_inputs<Index>(faces, face_membership, "face_link");
  tf::face_link<Index> links;
  links.build(faces, membership);
  return offset_blocked_buffer<Index, Index>::from_buffer(
      std::move(static_cast<tf::offset_block_buffer<Index, Index> &>(links)));
}

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto orient_faces_consistently(const mesh<Index, Real, Dims, Ngon> &value)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon> {
  // the link is the source mesh's own, and asking for it is what refuses a
  // face these points do not have — before a copy is allocated for it
  const auto edge_link = value.manifold_edge_link();
  auto polygons = detail::materialized(value);
  auto oriented = polygons.polygons() | tf::tag(edge_link);
  tf::orient_faces_consistently(oriented);
  return polygons;
}

} // namespace tf::cpp
