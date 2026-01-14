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

#include "trueform/python/core.hpp"

namespace tf::py {

auto register_core(nanobind::module_ &m) -> void {
  // Create core submodule
  auto core_module = m.def_submodule("core", "Core operations");

  // Register core components to submodule
  register_offset_blocked_array(core_module);
  register_core_closest_metric_point_pair(core_module);
  register_core_ray_cast(core_module);
  register_core_intersects(core_module);
  register_core_distance(core_module);
  register_core_distance_field(core_module);
}

} // namespace tf::py
