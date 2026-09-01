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

#include "../../intersect/graph/plane_identity_names.hpp"

#include <tuple>

namespace tf::arrangement {

/// A PA identity name keeps the structural feature name used by LA and the
/// exact parameter needed by a one-edge refinement birth. Stock recovery
/// names carry zero. Keeping the parameter in its native type avoids folding
/// a 125-bit f64-lane parameter into the 32-bit Index fields of the feature.
template <typename Index, typename Param> struct plane_recovery_name {
  tf::intersect::graph::plane_name<Index> feature{};
  Param parameter = Param(0);

  plane_recovery_name() = default;
  plane_recovery_name(const tf::intersect::graph::plane_name<Index> &value,
                      Param exact_parameter = Param(0))
      : feature(value), parameter(exact_parameter) {}
};

template <typename Index, typename Param>
auto operator==(const plane_recovery_name<Index, Param> &x,
                const plane_recovery_name<Index, Param> &y) -> bool {
  return x.feature == y.feature && x.parameter == y.parameter;
}

template <typename Index, typename Param>
auto operator!=(const plane_recovery_name<Index, Param> &x,
                const plane_recovery_name<Index, Param> &y) -> bool {
  return !(x == y);
}

template <typename Index, typename Param>
auto operator<(const plane_recovery_name<Index, Param> &x,
               const plane_recovery_name<Index, Param> &y) -> bool {
  return std::tie(x.feature, x.parameter) < std::tie(y.feature, y.parameter);
}

template <typename Index, typename Param>
auto make_plane_recovery_name(
    const tf::intersect::graph::plane_name<Index> &feature,
    Param parameter = Param(0)) -> plane_recovery_name<Index, Param> {
  return {feature, parameter};
}

} // namespace tf::arrangement
