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

#include "trueform/cpp/topology/connect_edges_to_paths.hpp"
#include "trueform/cpp/topology/label_connected_components.hpp"

#include "trueform/core/buffer.hpp"
#include "trueform/core/edges.hpp"
#include "trueform/topology/connect_edges_to_paths.hpp"
#include "trueform/topology/label_connected_components.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace tf::cpp::detail {

template <typename Index>
auto require_connectivity_value(Index peer, Index size, bool allow_missing)
    -> void {
  if ((allow_missing && peer == Index{-1}) || (peer >= Index{0} && peer < size))
    return;
  throw std::out_of_range(
      "label_connected_components: peer index out of range");
}

template <typename Index>
auto require_component_hint(Index expected_number_of_components) -> void {
  if (expected_number_of_components <= Index{0})
    throw std::invalid_argument(
        "label_connected_components: expected number of components must be "
        "positive");
}

template <typename Index>
auto label_offset_blocked_connected_components(
    const offset_blocked_buffer<Index, Index> &connectivity,
    Index expected_number_of_components) -> connected_components_result<Index> {
  if (!connectivity.is_valid())
    throw std::invalid_argument(
        "label_connected_components: connectivity must be valid");
  require_component_hint(expected_number_of_components);
  const auto size = static_cast<Index>(connectivity.size());
  const auto range = connectivity.make_range();
  for (const auto block : range)
    for (const auto peer : block)
      require_connectivity_value(peer, size, false);

  tf::connected_component_labels<Index> labels;
  labels.labels.allocate(static_cast<std::size_t>(size));
  auto apply = [&range](Index id, const auto &fn) {
    for (const auto peer : range[id])
      fn(peer);
  };
  labels.n_components = tf::label_connected_components<Index>(
      labels.labels, apply, expected_number_of_components);
  return {nd_array<Index>::from_buffer(std::move(labels.labels),
                                       {static_cast<int>(size)}),
          static_cast<Index>(labels.n_components)};
}

template <typename Index>
auto label_dense_connected_components(const nd_array<Index> &connectivity,
                                      Index expected_number_of_components)
    -> connected_components_result<Index> {
  if (!connectivity.is_valid() || connectivity.ndim() != 2)
    throw std::invalid_argument(
        "label_connected_components: connectivity must have shape [N, K]");
  require_component_hint(expected_number_of_components);
  const auto size = static_cast<Index>(connectivity.shape_at(0));
  const auto width = static_cast<std::size_t>(connectivity.shape_at(1));
  for (const auto peer : connectivity)
    require_connectivity_value(peer, size, true);

  tf::connected_component_labels<Index> labels;
  labels.labels.allocate(static_cast<std::size_t>(size));
  auto apply = [&connectivity, width](Index id, const auto &fn) {
    const auto offset = static_cast<std::size_t>(id) * width;
    for (std::size_t column = 0; column < width; ++column) {
      const auto peer = connectivity[offset + column];
      if (peer >= Index{0})
        fn(peer);
    }
  };
  labels.n_components = tf::label_connected_components<Index>(
      labels.labels, apply, expected_number_of_components);
  return {nd_array<Index>::from_buffer(std::move(labels.labels),
                                       {static_cast<int>(size)}),
          static_cast<Index>(labels.n_components)};
}

/// The edges are an identity table, so a path names the points the caller
/// handed in and nothing is renumbered.
template <typename Index>
auto connect_edges_to_paths_impl(const nd_array<Index> &edges)
    -> offset_blocked_buffer<Index, Index> {
  if (!edges.is_valid() || edges.ndim() != 2 || edges.shape_at(1) != 2)
    throw std::invalid_argument(
        "connect_edges_to_paths: edges must have shape [N, 2]");
  for (const auto point : edges)
    if (point < Index{0})
      throw std::out_of_range(
          "connect_edges_to_paths: point index must be nonnegative");
  auto edge_range = tf::make_edges(edges.make_range());
  auto paths = tf::connect_edges_to_paths(edge_range);
  return offset_blocked_buffer<Index, Index>::from_buffer(std::move(paths));
}

} // namespace tf::cpp::detail
