/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../../core/algorithm/assign_if.hpp"
#include "../../core/local_value.hpp"
#include <atomic>
#include <limits>

namespace tf::spatial {

template <typename TreeInfo> class local_tree_metric_result {
public:
  using real_t = typename TreeInfo::real_t;
  local_tree_metric_result(real_t metric)
      : _aabb_min{metric}, _aabb_max{metric}, _best{metric} {
    TreeInfo val;
    val.metric(metric);
    _info.reset(val);
  }

  local_tree_metric_result()
      : local_tree_metric_result{std::numeric_limits<real_t>::max()} {}

  auto update_aabb_min(real_t val) {
    return tf::assign_if(_aabb_min, val, std::less<>{});
  }
  auto update_aabb_max(real_t val) {
    return tf::assign_if(_aabb_max, val, std::less<>{});
  }

  auto reject_aabbs(real_t val) const {
    return val > _best.load(std::memory_order_acquire) ||
           val > _aabb_max.load(std::memory_order_acquire);
  }

  auto update(typename TreeInfo::element_t c_element,
              const typename TreeInfo::info_t &c_point) -> bool {
    if (tf::assign_if(_best, c_point.metric, std::less<>{})) {
      // assignment is thread_local
      *_info = {c_element, c_point};
      return c_point.metric < std::numeric_limits<real_t>::epsilon();
    }
    return false;
  }

  auto metric() const { return _best.load(std::memory_order_acquire); }

  auto info() const {
    return _info.aggregate([](const auto &x0, const auto &x1) {
      if (x0.metric() < x1.metric())
        return x0;
      return x1;
    });
  }

  tf::local_value<TreeInfo> _info;
  std::atomic<real_t> _aabb_min;
  std::atomic<real_t> _aabb_max;
  std::atomic<real_t> _best;
};

} // namespace tf::implementation
