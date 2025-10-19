/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../../core/epsilon.hpp"

namespace tf::spatial {

template <typename TreeInfo> class tree_metric_result {
public:
  using real_t = typename TreeInfo::real_t;
  tree_metric_result() = default;
  tree_metric_result(real_t metric) { info.metric(metric); }

  auto update(typename TreeInfo::element_t c_element,
              const typename TreeInfo::info_t &c_point) -> bool {
    if (c_point.metric < metric()) {
      info = {c_element, c_point};
    }
    return metric() < tf::epsilon2<real_t>;
  }

  auto metric() { return info.metric(); }

  TreeInfo info;
};

} // namespace tf::spatial
