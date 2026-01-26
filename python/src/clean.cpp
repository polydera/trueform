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

#include "trueform/python/clean.hpp"

namespace tf::py {

auto register_clean(nanobind::module_ &m) -> void {
  // Create clean submodule
  auto clean_module = m.def_submodule("clean", "Cleaning operations");

  // Register clean components to submodule
  register_clean_cleaned(clean_module);
}

} // namespace tf::py
