/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/core.hpp"
#include "trueform/python/core/closest_metric_point_pair.hpp"
#include "trueform/python/core/distance.hpp"
#include "trueform/python/core/intersects.hpp"
#include "trueform/python/core/point_cloud.hpp"
#include "trueform/python/core/ray_cast.hpp"

namespace tf::py {

auto register_core(nanobind::module_ &m) -> void {
  // Register core components
  register_point_cloud(m);
  register_core_closest_metric_point_pair(m);
  register_core_ray_cast(m);
  register_core_intersects(m);
  register_core_distance(m);
}

} // namespace tf::py
