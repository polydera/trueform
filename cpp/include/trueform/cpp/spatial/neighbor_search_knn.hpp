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

#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace tf::cpp {

/// @brief Up to k nearest source elements for a query.
template <typename Index, typename Real, std::size_t Dims = 3>
struct neighbor_knn_result {
  nd_array<Index> element_ids; // [count]
  nd_array<Real> points;       // [count, Dims]
  nd_array<Real> distances;    // [count]
};

/// @brief Up to k nearest source elements for each query.
template <typename Index, typename Real, std::size_t Dims = 3>
struct neighbor_knn_batch_result {
  nd_array<Index> element_ids;   // [N, k], padded with -1
  nd_array<Real> points;         // [N, k, Dims]
  nd_array<Real> distances;      // [N, k]
  nd_array<std::int32_t> counts; // [N]
};

/// @brief Find up to k nearest cells for one primitive query.
///
/// Result ids retain each carrier's own index type; point-cloud ids are
/// int32.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighbor_search_knn(const mesh<Index, Real, Dims, Ngon> &form,
                         const primitive<Real, Dims> &query, int k,
                         Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_knn_result<Index, Real, Dims>;

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighbor_search_knn_batch(
    const mesh<Index, Real, Dims, Ngon> &form,
    const primitive<Real, Dims> &queries, int k,
    Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_knn_batch_result<Index, Real, Dims>;

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search_knn(const edge_mesh<Index, Real, Dims> &form,
                         const primitive<Real, Dims> &query, int k,
                         Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_knn_result<Index, Real, Dims>;

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search_knn_batch(
    const edge_mesh<Index, Real, Dims> &form,
    const primitive<Real, Dims> &queries, int k,
    Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_knn_batch_result<Index, Real, Dims>;

template <typename Real, std::size_t Dims>
auto neighbor_search_knn(const point_cloud<Real, Dims> &form,
                         const primitive<Real, Dims> &query, int k,
                         Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_knn_result<std::int32_t, Real, Dims>;

template <typename Real, std::size_t Dims>
auto neighbor_search_knn_batch(
    const point_cloud<Real, Dims> &form, const primitive<Real, Dims> &queries,
    int k, Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_knn_batch_result<std::int32_t, Real, Dims>;

} // namespace tf::cpp
