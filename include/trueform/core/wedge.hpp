/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./point_like.hpp"
#include <array>

namespace tf {

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

template <typename Policy>
auto make_wedge(const point_like<2, Policy> &origin,
                const point_like<2, Policy> &arm0,
                const point_like<2, Policy> &arm1) -> wedge<Policy> {
  return wedge<Policy>(origin, arm0, arm1);
}

} // namespace tf
