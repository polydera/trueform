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

#include "trueform/python/iso.hpp"

namespace tf::py {

auto register_iso(nanobind::module_ &m) -> void {
  // Create iso submodule
  auto iso_module = m.def_submodule("iso", "Scalar field operations");

  // Register iso components to submodule
  register_iso_isocontours(iso_module);
  register_iso_isobands(iso_module);
}

} // namespace tf::py
