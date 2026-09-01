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

#include "trueform/python/arrangement.hpp"

namespace tf::py {

auto register_arrangement(nanobind::module_ &m) -> void {
  // Create arrangement submodule
  auto arrangement_module =
      m.def_submodule("arrangement", "Arrangement operations");

  // Register arrangement components to submodule
  register_arrangement_mesh_arrangements(arrangement_module);
  register_arrangement_polygon_arrangement(arrangement_module);
}

} // namespace tf::py
