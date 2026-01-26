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
#include <nanobind/nanobind.h>
#include <trueform/python/geometry.hpp>

namespace tf::py {

auto register_geometry_module(nanobind::module_ &m) -> void {
  auto geometry_module = m.def_submodule("geometry", "Geometry operations");

  register_fit_rigid_alignment(geometry_module);
  register_fit_obb_alignment(geometry_module);
  register_fit_knn_alignment(geometry_module);
  register_chamfer_error(geometry_module);
  register_triangulated(geometry_module);
  register_principal_curvatures(geometry_module);
  register_ensure_positive_orientation(geometry_module);
  register_make_mesh_primitives(geometry_module);
  register_measurements(geometry_module);
}

} // namespace tf::py
