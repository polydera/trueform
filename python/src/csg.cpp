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

#include "trueform/python/csg.hpp"

namespace tf::py {

auto register_csg(nanobind::module_ &m) -> void {
  auto csg = m.def_submodule("csg", "CSG graph: build once, query many");
  register_csg_graph_int3float3d(csg);
  register_csg_graph_int3double3d(csg);
  register_csg_graph_int643float3d(csg);
  register_csg_graph_int643double3d(csg);
  register_outer_shell_int3float3d(csg);
  register_outer_shell_int3double3d(csg);
  register_outer_shell_int643float3d(csg);
  register_outer_shell_int643double3d(csg);
  register_csg_boolean(csg);
}

} // namespace tf::py
