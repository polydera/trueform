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

#include "../../exact/meta.hpp"

#include <array>
#include <tuple>

namespace tf::arrangement {

/// One transient sizing-ring split on a source form's original edge.
/// Endpoints are normalized `u < v`; `parameter` is in the `u -> v` frame.
template <typename Index, typename Int> struct plane_refinement_physical_split {
  Index tag, u, v;
  typename tf::exact::meta<Int>::param_type parameter;
};

template <typename Index, typename Int>
auto operator<(const plane_refinement_physical_split<Index, Int> &a,
               const plane_refinement_physical_split<Index, Int> &b) -> bool {
  return std::tie(a.tag, a.u, a.v, a.parameter) <
         std::tie(b.tag, b.u, b.v, b.parameter);
}

template <typename Index, typename Int>
auto operator==(const plane_refinement_physical_split<Index, Int> &a,
                const plane_refinement_physical_split<Index, Int> &b) -> bool {
  return std::tie(a.tag, a.u, a.v, a.parameter) ==
         std::tie(b.tag, b.u, b.v, b.parameter);
}

template <typename Index, typename Int>
auto plane_refinement_physical_edge(
    const plane_refinement_physical_split<Index, Int> &split)
    -> std::array<Index, 3> {
  return {split.tag, split.u, split.v};
}

} // namespace tf::arrangement
