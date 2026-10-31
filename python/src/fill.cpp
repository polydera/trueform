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

#include "trueform/python/fill.hpp"

namespace tf::py {

auto register_fill(nanobind::module_ &m) -> void {
  auto fill_module = m.def_submodule("fill", "Hole filling");
  register_fill_holes(fill_module);
}

} // namespace tf::py
