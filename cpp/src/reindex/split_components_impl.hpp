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

#include "trueform/cpp/reindex/split_into_components.hpp"

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/segments_buffer.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/reindex/split_into_components.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace tf::cpp {
namespace detail {

template <typename Index, typename Real, std::size_t Dims>
auto require_split_axes() -> void {
  static_assert(std::is_same_v<Real, float> || std::is_same_v<Real, double>,
                "split_into_components Real must be float or double");
  static_assert(is_supported_index_v<Index>,
                "split_into_components Index must be int32 or int64");
  static_assert(Dims == 2 || Dims == 3,
                "split_into_components Dims must be two or three");
}

inline auto require_split_labels(const nd_array<std::int32_t> &labels,
                                 std::size_t number_of_elements) -> void {
  if (!labels.is_valid())
    throw std::invalid_argument("split_into_components: labels must be valid");
  if (labels.ndim() != 1)
    throw std::invalid_argument(
        "split_into_components: labels must be one-dimensional");
  if (labels.length() != number_of_elements)
    throw std::invalid_argument("split_into_components: labels size mismatch");
}

/// The components are core's own buffers already, so a split is the pieces
/// moved into the vector that hands them back.
template <typename Component, typename Components, typename Labels>
auto make_split_components_result(Components &&components, Labels &&labels)
    -> split_components_result<Component> {
  std::vector<Component> result;
  result.reserve(components.size());
  for (auto &component : components)
    result.push_back(std::move(component));
  const auto label_count = static_cast<int>(labels.size());
  return {std::move(result),
          nd_array<std::int32_t>::from_buffer(std::forward<Labels>(labels),
                                              {label_count})};
}

} // namespace detail

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto split_into_components(const mesh<Index, Real, Dims, Ngon> &value,
                           const nd_array<std::int32_t> &labels)
    -> split_components_result<tf::polygons_buffer<Index, Real, Dims, Ngon>> {
  detail::require_split_axes<Index, Real, Dims>();
  value.require_indices();
  detail::require_split_labels(labels, value.number_of_faces());
  auto [components, component_labels] =
      tf::split_into_components(value.polygons(), labels.make_range());
  return detail::make_split_components_result<
      tf::polygons_buffer<Index, Real, Dims, Ngon>>(
      std::move(components), std::move(component_labels));
}

template <typename Index, typename Real, std::size_t Dims>
auto split_into_components(const edge_mesh<Index, Real, Dims> &value,
                           const nd_array<std::int32_t> &labels)
    -> split_components_result<tf::segments_buffer<Index, Real, Dims>> {
  detail::require_split_axes<Index, Real, Dims>();
  value.require_indices();
  detail::require_split_labels(labels, value.number_of_edges());
  auto [components, component_labels] =
      tf::split_into_components(value.segments(), labels.make_range());
  return detail::make_split_components_result<
      tf::segments_buffer<Index, Real, Dims>>(std::move(components),
                                              std::move(component_labels));
}

} // namespace tf::cpp
