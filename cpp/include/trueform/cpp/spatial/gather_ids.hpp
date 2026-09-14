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

#include "trueform/cpp/core/common_index.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::cpp {

/// @brief Gather mesh face IDs that exactly intersect a runtime primitive.
///
/// The primitive must be scalar and have the same precision and dimension as
/// the form. The returned owner has shape [N] and preserves the mesh index
/// dtype. Result ordering is unspecified.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto gather_ids(const mesh<Index, Real, Dims, Ngon> &form,
                const primitive<Real, Dims> &query) -> nd_array<Index>;

/// @brief Gather edge IDs that exactly intersect a runtime primitive.
template <typename Index, typename Real, std::size_t Dims>
auto gather_ids(const edge_mesh<Index, Real, Dims> &form,
                const primitive<Real, Dims> &query) -> nd_array<Index>;

/// @brief Gather point-cloud point IDs that exactly intersect a primitive.
template <typename Real, std::size_t Dims>
auto gather_ids(const point_cloud<Real, Dims> &form,
                const primitive<Real, Dims> &query) -> nd_array<std::int32_t>;

/// @brief Gather the intersecting id pairs of two forms, one row per pair.
template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0, std::size_t Ngon1>
auto gather_ids(const mesh<Index0, Real, Dims, Ngon0> &a,
                const mesh<Index1, Real, Dims, Ngon1> &b)
    -> nd_array<common_index_t<Index0, Index1>>;

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon0>
auto gather_ids(const mesh<Index0, Real, Dims, Ngon0> &a,
                const edge_mesh<Index1, Real, Dims> &b)
    -> nd_array<common_index_t<Index0, Index1>>;

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon0>
auto gather_ids(const mesh<Index, Real, Dims, Ngon0> &a,
                const point_cloud<Real, Dims> &b) -> nd_array<Index>;

template <typename Index0, typename Real, typename Index1, std::size_t Dims,
          std::size_t Ngon1>
auto gather_ids(const edge_mesh<Index0, Real, Dims> &a,
                const mesh<Index1, Real, Dims, Ngon1> &b)
    -> nd_array<common_index_t<Index0, Index1>>;

template <typename Index0, typename Real, typename Index1, std::size_t Dims>
auto gather_ids(const edge_mesh<Index0, Real, Dims> &a,
                const edge_mesh<Index1, Real, Dims> &b)
    -> nd_array<common_index_t<Index0, Index1>>;

template <typename Index, typename Real, std::size_t Dims>
auto gather_ids(const edge_mesh<Index, Real, Dims> &a,
                const point_cloud<Real, Dims> &b) -> nd_array<Index>;

template <typename Real, typename Index, std::size_t Dims, std::size_t Ngon1>
auto gather_ids(const point_cloud<Real, Dims> &a,
                const mesh<Index, Real, Dims, Ngon1> &b) -> nd_array<Index>;

template <typename Real, typename Index, std::size_t Dims>
auto gather_ids(const point_cloud<Real, Dims> &a,
                const edge_mesh<Index, Real, Dims> &b) -> nd_array<Index>;

template <typename Real, std::size_t Dims>
auto gather_ids(const point_cloud<Real, Dims> &a,
                const point_cloud<Real, Dims> &b) -> nd_array<std::int32_t>;

} // namespace tf::cpp
