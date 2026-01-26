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

#include "../../core/algorithm/assign_if.hpp"
#include "../../core/epsilon.hpp"
#include "../../core/local_value.hpp"
#include <atomic>
#include <limits>

namespace tf::spatial {

template <typename TreeInfo> class local_tree_metric_result {
public:
  using real_t = typename TreeInfo::real_t;
  local_tree_metric_result(real_t metric) : _bv_max{metric}, _best{metric} {
    TreeInfo val;
    val.metric(metric);
    _info.reset(val);
  }

  local_tree_metric_result()
      : local_tree_metric_result{std::numeric_limits<real_t>::max()} {}

  auto update_bv_max(real_t val) {
    return tf::assign_if(_bv_max, val, std::less<>{});
  }

  auto reject_bvs(real_t val) const {
    return val > _best.load(std::memory_order_acquire) ||
           val > _bv_max.load(std::memory_order_acquire);
  }

  auto update(typename TreeInfo::element_t c_element,
              const typename TreeInfo::info_t &c_point) -> bool {
    if (tf::assign_if(_best, c_point.metric, std::less<>{})) {
      // assignment is thread_local
      *_info = {c_element, c_point};
      return c_point.metric < tf::epsilon2<real_t>;
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
  std::atomic<real_t> _bv_max;
  std::atomic<real_t> _best;
};

} // namespace tf::spatial
