/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../tree_ray_info.hpp"
#include <limits>

namespace tf::spatial {
template <typename Index, typename InfoT> struct tree_ray_info {
  using real_t = typename InfoT::real_t;

  tree_ray_info() = default;
  tree_ray_info(real_t min_t, real_t max_t) : _min_t{min_t}, _max_t{max_t} {}

  auto min_t() const -> real_t { return _min_t; }

  auto max_t() const -> real_t { return _max_t; }

  auto hit_t() const -> real_t { return _ray_info.info.t; }

  auto info() const -> const tf::tree_ray_info<Index, InfoT> & {
    return _ray_info;
  }

  auto update(Index id, InfoT info) -> real_t {
    if (char(info) & char(info.t >= min_t()) & char(info.t <= _max_t)) {
      _ray_info = {id, info};
      _max_t = info.t;
    }
    return _max_t;
  }

  real_t _min_t = 0;
  real_t _max_t = std::numeric_limits<real_t>::max();
  tf::tree_ray_info<Index, InfoT> _ray_info;
};
} // namespace tf::implementation
