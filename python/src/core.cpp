/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */

#include "trueform/python/core.hpp"

namespace tf::py {

auto register_core(nanobind::module_ &m) -> void {
  // Register core components
  register_point_cloud(m);
  register_mesh(m);
  register_edge_mesh(m);
  register_offset_blocked_array(m);
  register_core_closest_metric_point_pair(m);
  register_core_ray_cast(m);
  register_core_intersects(m);
  register_core_distance(m);
  register_core_distance_field(m);
}

} // namespace tf::py
