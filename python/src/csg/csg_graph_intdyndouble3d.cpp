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

#include "trueform/python/csg/csg_graph_impl.hpp"

namespace tf::py {

auto register_csg_graph_intdyndouble3d(nanobind::module_ &m) -> void {
  register_csg_graph<int, double, tf::dynamic_size>(
      m, "CsgGraph_intdyndouble3d");
}

} // namespace tf::py
