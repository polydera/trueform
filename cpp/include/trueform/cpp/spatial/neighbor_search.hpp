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

/// @brief Nearest source element for a primitive query.
template <typename Index, typename Real, std::size_t Dims = 3>
struct neighbor_result {
  Index element_id;
  Real distance2;
  nd_array<Real> point; // [Dims]
};

/// @brief Nearest source elements for a primitive batch.
template <typename Index, typename Real, std::size_t Dims = 3>
struct neighbor_batch_result {
  nd_array<Index> element_ids; // [N]
  nd_array<Real> points;       // [N, Dims]
  nd_array<Real> distances;    // [N]
};

/// @brief Nearest pair between two forms.
template <typename Index0, typename Index1, typename Real, std::size_t Dims = 3>
struct neighbor_pair_result {
  Index0 element_id0;
  Index1 element_id1;
  Real distance2;
  nd_array<Real> point0; // [Dims]
  nd_array<Real> point1; // [Dims]
};

/// @brief Find the nearest cell for one primitive query.
///
/// A form-pair search requires matching precision and dimensions. Result ids
/// retain each carrier's own index type; point-cloud ids are int32.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighbor_search(const mesh<Index, Real, Dims, Ngon> &form,
                     const primitive<Real, Dims> &query,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_result<Index, Real, Dims>;

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighbor_search_batch(const mesh<Index, Real, Dims, Ngon> &form,
                           const primitive<Real, Dims> &queries,
                           Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_batch_result<Index, Real, Dims>;

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search(const edge_mesh<Index, Real, Dims> &form,
                     const primitive<Real, Dims> &query,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_result<Index, Real, Dims>;

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search_batch(const edge_mesh<Index, Real, Dims> &form,
                           const primitive<Real, Dims> &queries,
                           Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_batch_result<Index, Real, Dims>;

template <typename Real, std::size_t Dims>
auto neighbor_search(const point_cloud<Real, Dims> &form,
                     const primitive<Real, Dims> &query,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_result<std::int32_t, Real, Dims>;

template <typename Real, std::size_t Dims>
auto neighbor_search_batch(const point_cloud<Real, Dims> &form,
                           const primitive<Real, Dims> &queries,
                           Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_batch_result<std::int32_t, Real, Dims>;

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0, std::size_t Ngon1>
auto neighbor_search(const mesh<Index0, Real, Dims, Ngon0> &a,
                     const mesh<Index1, Real, Dims, Ngon1> &b,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_pair_result<Index0, Index1, Real, Dims>;

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0>
auto neighbor_search(const mesh<Index0, Real, Dims, Ngon0> &a,
                     const edge_mesh<Index1, Real, Dims> &b,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_pair_result<Index0, Index1, Real, Dims>;

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon0>
auto neighbor_search(const mesh<Index, Real, Dims, Ngon0> &a,
                     const point_cloud<Real, Dims> &b,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_pair_result<Index, std::int32_t, Real, Dims>;

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon1>
auto neighbor_search(const edge_mesh<Index0, Real, Dims> &a,
                     const mesh<Index1, Real, Dims, Ngon1> &b,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_pair_result<Index0, Index1, Real, Dims>;

template <typename Index0, typename Real, typename Index1, std::size_t Dims>
auto neighbor_search(const edge_mesh<Index0, Real, Dims> &a,
                     const edge_mesh<Index1, Real, Dims> &b,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_pair_result<Index0, Index1, Real, Dims>;

template <typename Index, typename Real, std::size_t Dims>
auto neighbor_search(const edge_mesh<Index, Real, Dims> &a,
                     const point_cloud<Real, Dims> &b,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_pair_result<Index, std::int32_t, Real, Dims>;

template <typename Real, typename Index, std::size_t Dims, std::size_t Ngon1>
auto neighbor_search(const point_cloud<Real, Dims> &a,
                     const mesh<Index, Real, Dims, Ngon1> &b,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_pair_result<std::int32_t, Index, Real, Dims>;

template <typename Real, typename Index, std::size_t Dims>
auto neighbor_search(const point_cloud<Real, Dims> &a,
                     const edge_mesh<Index, Real, Dims> &b,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_pair_result<std::int32_t, Index, Real, Dims>;

template <typename Real, std::size_t Dims>
auto neighbor_search(const point_cloud<Real, Dims> &a,
                     const point_cloud<Real, Dims> &b,
                     Real radius = std::numeric_limits<Real>::infinity())
    -> neighbor_pair_result<std::int32_t, std::int32_t, Real, Dims>;

} // namespace tf::cpp
