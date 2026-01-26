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
#include "./point_like.hpp"
#include <array>

namespace tf {

/// @ingroup core_primitives
/// @brief 2D wedge defined by an origin and two arm endpoints.
///
/// Represents a triangular region in 2D space. Inherits from
/// std::array for structured bindings support: `auto [origin, arm0, arm1] = wedge;`
///
/// @tparam Policy The policy type for the points.
template <typename Policy>
class wedge : public std::array<point_like<2, Policy>, 3> {
  using base_type = std::array<point_like<2, Policy>, 3>;

public:
  using point_type = point_like<2, Policy>;

  wedge() = default;

  wedge(const point_type &origin, const point_type &arm0,
        const point_type &arm1)
      : base_type{origin, arm0, arm1} {}
};

/// @ingroup core_primitives
/// @brief Create a 2D wedge from origin and two arm endpoints.
///
/// @tparam Policy The policy type of the points.
/// @param origin The wedge origin point.
/// @param arm0 First arm endpoint.
/// @param arm1 Second arm endpoint.
/// @return A @ref tf::wedge.
template <typename Policy>
auto make_wedge(const point_like<2, Policy> &origin,
                const point_like<2, Policy> &arm0,
                const point_like<2, Policy> &arm1) -> wedge<Policy> {
  return wedge<Policy>(origin, arm0, arm1);
}

} // namespace tf
