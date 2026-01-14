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
#include "../coordinate_type.hpp"
#include "../unit_vector_like.hpp"

namespace tf::core {
template <std::size_t Dims, typename Policy> struct plane {
  using coordinate_type = tf::coordinate_type<Policy>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;
  using normal_type = tf::unit_vector_like<Dims, Policy>;

  plane() = default;
  plane(const tf::unit_vector_like<Dims, Policy> &normal, coordinate_type d)
      : normal{normal}, d{d} {}

  template <typename Policy1>
  auto operator=(const plane<Dims, Policy1> &other) -> std::enable_if_t<
      std::is_assignable_v<tf::unit_vector_like<Dims, Policy> &,
                           tf::unit_vector_like<Dims, Policy1>> &&
          std::is_assignable_v<coordinate_type &,
                               typename Policy1::coordinate_type>,
      plane &> {
    normal = other.normal;
    d = other.d;
    return *this;
  }

  tf::unit_vector_like<Dims, Policy> normal;
  coordinate_type d;
};
template <std::size_t Dims, typename Policy>
auto make_plane(const unit_vector_like<Dims, Policy> &normal,
                tf::coordinate_type<Policy> d) {
  return plane<Dims, Policy>{normal, d};
}
} // namespace tf::core
