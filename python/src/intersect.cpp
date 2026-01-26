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

#include "trueform/python/intersect.hpp"

namespace tf::py {

auto register_intersect(nanobind::module_ &m) -> void {
  // Create intersect submodule
  auto intersect_module = m.def_submodule("intersect", "Intersection operations");

  // Register intersect components to submodule
  register_intersect_isocontours(intersect_module);
  register_intersect_intersection_curves(intersect_module);
  register_intersect_self_intersection_curves(intersect_module);
}

} // namespace tf::py
