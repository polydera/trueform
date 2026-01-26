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
