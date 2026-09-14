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
#include <trueform/python/volume/make_mesh_sdf_impl.hpp>

namespace tf::py {

auto register_volume_make_mesh_sdf_intdyndouble3d(nanobind::module_ &m) -> void {
  def_make_mesh_sdf<int, double, tf::dynamic_size>(m, "make_mesh_sdf_intdyndouble3d");
}

} // namespace tf::py
