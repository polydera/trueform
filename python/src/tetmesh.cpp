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

#include "trueform/python/tetmesh.hpp"

namespace tf::py {

auto register_tetmesh(nanobind::module_ &m) -> void {
  auto tetmesh_module = m.def_submodule("tetmesh", "Tetrahedral meshing");
  register_tet_mesh_float(tetmesh_module);
  register_tet_mesh_double(tetmesh_module);
}

} // namespace tf::py
