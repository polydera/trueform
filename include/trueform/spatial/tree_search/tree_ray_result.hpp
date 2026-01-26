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
#include "../tree_ray_info.hpp"
#include <limits>

namespace tf::spatial {

template <typename Index, typename InfoT>
struct tree_ray_result {
  using real_t = typename InfoT::real_t;

  tree_ray_result() = default;
  tree_ray_result(real_t min_t, real_t max_t) : _min_t{min_t}, _max_t{max_t} {}

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

} // namespace tf::spatial
