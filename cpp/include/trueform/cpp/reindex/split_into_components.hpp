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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/segments_buffer.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tf::cpp {

/// @brief Owning split result for a mesh or edge-mesh carrier.
template <typename Component> struct split_components_result {
  std::vector<Component> components;
  nd_array<std::int32_t> labels;
};

/// @brief Split a fixed- or dynamic-connectivity typed mesh by int32 labels.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto split_into_components(const mesh<Index, Real, Dims, Ngon> &value,
                           const nd_array<std::int32_t> &labels)
    -> split_components_result<tf::polygons_buffer<Index, Real, Dims, Ngon>>;

/// @brief Split a typed edge mesh by int32 labels.
template <typename Index, typename Real, std::size_t Dims>
auto split_into_components(const edge_mesh<Index, Real, Dims> &value,
                           const nd_array<std::int32_t> &labels)
    -> split_components_result<tf::segments_buffer<Index, Real, Dims>>;

#define TF_CPP_REINDEX_EXTERN_SPLIT_COMPONENTS_AT(Index, Real, Dims, Ngon)     \
  extern template auto split_into_components<Index, Real, Dims, Ngon>(         \
      const mesh<Index, Real, Dims, Ngon> &,                                   \
      const nd_array<std::int32_t> &)                                          \
      -> split_components_result<tf::polygons_buffer<Index, Real, Dims, Ngon>>

#define TF_CPP_REINDEX_EXTERN_SPLIT_COMPONENTS(Index, Real, Dims)              \
  extern template auto split_into_components<Index, Real, Dims>(               \
      const edge_mesh<Index, Real, Dims> &,                                    \
      const nd_array<std::int32_t> &)                                          \
      -> split_components_result<tf::segments_buffer<Index, Real, Dims>>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(
    TF_CPP_REINDEX_EXTERN_SPLIT_COMPONENTS_AT)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS(TF_CPP_REINDEX_EXTERN_SPLIT_COMPONENTS)

#undef TF_CPP_REINDEX_EXTERN_SPLIT_COMPONENTS
#undef TF_CPP_REINDEX_EXTERN_SPLIT_COMPONENTS_AT

} // namespace tf::cpp
