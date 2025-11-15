/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <optional>
#include <trueform/core/ray_config.hpp>
#include <tuple>

namespace tf::py {

/**
 * Helper to create ray_config from optional Python tuple (min_t, max_t)
 */
template <typename RealT>
auto make_ray_config_from_optional(
    std::optional<std::tuple<RealT, RealT>> opt_config) -> tf::ray_config<RealT> {
  if (opt_config) {
    auto [min_t, max_t] = *opt_config;
    return tf::ray_config<RealT>{min_t, max_t};
  }
  return tf::ray_config<RealT>{};  // Default: min_t=0, max_t=max
}

} // namespace tf::py
