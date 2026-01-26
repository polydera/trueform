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

#include "trueform/python/cut.hpp"

namespace tf::py {

auto register_cut(nanobind::module_ &m) -> void {
  // Create cut submodule
  auto cut_module = m.def_submodule("cut", "Cut operations");

  // Register cut components to submodule
  register_cut_isobands(cut_module);
  register_cut_boolean(cut_module);
  register_cut_embedded_self_intersection_curves(cut_module);
}

} // namespace tf::py
