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
#include <trueform/python/spatial/edge_mesh.hpp>

namespace tf::py {

auto register_edge_mesh_fp_int64double2d(nanobind::module_ &m) -> void {
  register_form_prim_ops<edge_mesh_wrapper<std::int64_t, double, 2>, 2, double>(
      m, "edge_mesh", "int64double2d");
}

} // namespace tf::py
