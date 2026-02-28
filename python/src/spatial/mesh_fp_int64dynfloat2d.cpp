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
#include <trueform/python/spatial/mesh.hpp>

namespace tf::py {

auto register_mesh_fp_int64dynfloat2d(nanobind::module_ &m) -> void {
  register_form_prim_ops<mesh_wrapper<std::int64_t, float, tf::dynamic_size, 2>,
                         2, float>(m, "mesh", "int64dynfloat2d");
}

} // namespace tf::py
