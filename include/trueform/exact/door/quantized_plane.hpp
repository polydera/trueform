/*
 * Copyright (c) 2026 XLAB
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

#include "../meta.hpp"

#include <array>
#include <cstddef>

namespace tf::exact::door {

/// A plane name on the door's own grid: the direction quantized to `K`
/// steps of the unit sphere, gcd-reduced and sign-canonical, and the
/// offset on that direction's own step. Two faces name one plane
/// exactly when the quadruples are equal, so a vertex's candidate list
/// is deduplicated by value and a placement is a pure function of the
/// names it stands on.
///
/// `K = T` is the door's own step: it puts the meet of three names on a
/// lattice point near the vertex that stated them.
template <typename Int> struct quantized_plane {
  using wide_type = typename tf::exact::meta<Int>::T1;

  std::array<wide_type, 3> normal{};
  wide_type offset{};

  auto operator==(const quantized_plane &o) const -> bool {
    return normal[0] == o.normal[0] && normal[1] == o.normal[1] &&
           normal[2] == o.normal[2] && offset == o.offset;
  }
  auto operator!=(const quantized_plane &o) const -> bool {
    return !(*this == o);
  }
  auto operator<(const quantized_plane &o) const -> bool {
    for (std::size_t k = 0; k < 3; ++k)
      if (normal[k] != o.normal[k])
        return normal[k] < o.normal[k];
    return offset < o.offset;
  }
};

} // namespace tf::exact::door
