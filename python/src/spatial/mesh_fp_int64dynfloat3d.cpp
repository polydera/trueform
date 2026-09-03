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
 * Author: Ziga Sajovic
 */

#include "trueform/python/spatial/form_prim_dispatch.hpp"
#include "trueform/python/spatial/signed_distance.hpp"
#include <trueform/python/spatial/mesh.hpp>

namespace tf::py {

auto register_mesh_fp_int64dynfloat3d(nanobind::module_ &m) -> void {
  register_form_prim_ops<mesh_wrapper<std::int64_t, float, tf::dynamic_size, 3>,
                         3, float>(m, "mesh", "int64dynfloat3d");
  register_mesh_signed_distance<
      mesh_wrapper<std::int64_t, float, tf::dynamic_size, 3>, 3, float>(
      m, "int64dynfloat3d");
}

} // namespace tf::py
