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
#include "trueform/cpp/core/point_cloud.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp::detail {

/// The three carriers a tree is built over. An operation admitting fewer than
/// all three states its own acceptance beside itself.
template <typename T> struct is_spatial_carrier : std::false_type {};
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
struct is_spatial_carrier<cpp::mesh<Index, Real, Dims, Ngon>> : std::true_type {
};
template <typename Index, typename Real, std::size_t Dims>
struct is_spatial_carrier<cpp::edge_mesh<Index, Real, Dims>> : std::true_type {
};
template <typename Real, std::size_t Dims>
struct is_spatial_carrier<cpp::point_cloud<Real, Dims>> : std::true_type {};

} // namespace tf::cpp::detail
