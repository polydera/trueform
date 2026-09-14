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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>
#include <type_traits>
#include <variant>

namespace tf::cpp {

/// @brief Closest point pair in operand order and its squared distance.
template <typename Real> struct closest_metric_point_pair_result {
  nd_array<Real> point0;
  nd_array<Real> point1;
  Real distance2;
};

/// @brief Batched closest point pairs in operand order and squared distances.
template <typename Real> struct closest_metric_point_pair_batch_result {
  nd_array<Real> points0;
  nd_array<Real> points1;
  nd_array<Real> distances;
};

/// @brief Compute the closest point pair between runtime primitives using
/// scalar, broadcast, or pairwise batch semantics.
///
/// point0 lies on the first operand and point1 lies on the second. Mixed
/// float/double inputs are computed and returned in double precision.
template <typename Real0, typename Real1, std::size_t Dims>
auto closest_metric_point_pair(const primitive<Real0, Dims> &a,
                               const primitive<Real1, Dims> &b)
    -> std::variant<
        closest_metric_point_pair_result<std::common_type_t<Real0, Real1>>,
        closest_metric_point_pair_batch_result<
            std::common_type_t<Real0, Real1>>>;

#define TF_CPP_EXTERN_CLOSEST_METRIC_POINT_PAIR(Real0, Real1, Dims)            \
  extern template auto closest_metric_point_pair<Real0, Real1, Dims>(          \
      const primitive<Real0, Dims> &, const primitive<Real1, Dims> &)          \
      ->std::variant<                                                          \
          closest_metric_point_pair_result<std::common_type_t<Real0, Real1>>,  \
          closest_metric_point_pair_batch_result<                              \
              std::common_type_t<Real0, Real1>>>

TF_CPP_MATRIX_FOR_EACH_REAL_PAIR_DIMS(TF_CPP_EXTERN_CLOSEST_METRIC_POINT_PAIR)

#undef TF_CPP_EXTERN_CLOSEST_METRIC_POINT_PAIR

} // namespace tf::cpp
