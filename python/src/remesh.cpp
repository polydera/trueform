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
#include <trueform/python/remesh.hpp>

namespace tf::py {

auto register_remesh(nanobind::module_ &m) -> void {
  auto remesh_module = m.def_submodule("remesh", "Remesh operations");

  register_decimated(remesh_module);
  register_isotropic_remeshed(remesh_module);
  register_simplified(remesh_module);
}

} // namespace tf::py
