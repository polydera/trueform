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

#include "trueform/cpp/clean/supported.hpp"
#include "trueform/cpp/core/index_map.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/point_cloud.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief Owning result of point cleaning with a source-to-result index map.
template <typename Index, typename Real> struct cleaned_points_result {
  nd_array<Real> points; // [N, Dims]
  index_map<Index> point_map;
};

/// @brief Remove duplicate points from a [Dims] or [N, Dims] array.
///
/// Dims must be stated because nd_array shape is runtime data; it defaults to
/// three. The result is detached from the input. Points carry no primitives
/// and no references, so both configuration booleans are accepted and ignored.
template <typename Real, std::size_t Dims = 3>
auto cleaned_points(const nd_array<Real> &points, Real tolerance = Real{},
                    bool remove_duplicate_primitives = true,
                    bool remove_unreferenced_points = true) -> nd_array<Real>;

template <typename Real, std::size_t Dims,
          std::enable_if_t<!is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points(const nd_array<Real> &, Real = Real{}, bool = true,
                    bool = true) -> void = delete;

/// @brief Remove duplicate points and return their source-to-result index map.
template <typename Real, std::size_t Dims = 3>
auto cleaned_points_with_map(const nd_array<Real> &points,
                             Real tolerance = Real{},
                             bool remove_duplicate_primitives = true,
                             bool remove_unreferenced_points = true)
    -> cleaned_points_result<default_index_t, Real>;

template <typename Real, std::size_t Dims,
          std::enable_if_t<!is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points_with_map(const nd_array<Real> &, Real = Real{}, bool = true,
                             bool = true) -> void = delete;

/// @brief Clean the local point storage of a 2D or 3D point cloud.
///
/// The cloud's placement is not applied. The result is an owning point array.
/// The carrier states its own axes, so a caller states nothing.
///
/// The axes are admitted in the template parameter list, so an unsupported
/// explicit spelling fails there rather than naming a carrier that refuses.
template <typename Real, std::size_t Dims,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points(const point_cloud<Real, Dims> &value,
                    Real tolerance = Real{},
                    bool remove_duplicate_primitives = true,
                    bool remove_unreferenced_points = true) -> nd_array<Real>;

/// @brief Clean a point cloud and return the point source-to-result index map.
template <typename Real, std::size_t Dims,
          std::enable_if_t<is_supported_cleaned_points_v<Real, Dims>, int> = 0>
auto cleaned_points_with_map(const point_cloud<Real, Dims> &value,
                             Real tolerance = Real{},
                             bool remove_duplicate_primitives = true,
                             bool remove_unreferenced_points = true)
    -> cleaned_points_result<default_index_t, Real>;

#define TF_CPP_EXTERN_CLEAN_POINTS(Real, Dims)                                 \
  extern template auto cleaned_points<Real, Dims>(                             \
      const nd_array<Real> &, Real, bool, bool) -> nd_array<Real>;             \
  extern template auto cleaned_points_with_map<Real, Dims>(                    \
      const nd_array<Real> &, Real, bool, bool)                                \
      -> cleaned_points_result<default_index_t, Real>;                         \
  extern template auto cleaned_points<Real, Dims>(                             \
      const point_cloud<Real, Dims> &, Real, bool, bool) -> nd_array<Real>;    \
  extern template auto cleaned_points_with_map<Real, Dims>(                    \
      const point_cloud<Real, Dims> &, Real, bool, bool)                       \
      -> cleaned_points_result<default_index_t, Real>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_CLEAN_POINTS)

#undef TF_CPP_EXTERN_CLEAN_POINTS

} // namespace tf::cpp
